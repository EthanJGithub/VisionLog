"""LangGraph multi-agent chatbot over the detections database.

An intent router classifies each question and dispatches to a specialized agent — so vague or
meta questions don't get a brittle single text-to-SQL guess:

    classify_intent ─┬─ analytics ──► author_sql ─► guard ─► execute ─► answer ─► END
                     │                    ▲___________│(unsafe)____│(db error)   (self-correct,
                     │                                                            up to MAX_ATTEMPTS)
                     ├─ overview  ──► run PRE-DEFINED safe aggregates → synthesize ─► END
                     ├─ schema    ──► describe what's queryable + distinct classes ─► END
                     └─ out_of_scope ► polite decline ─────────────────────────────► END

Design notes (mirrors the CredAgent pattern: deterministic, auditable LangGraph transitions):
- LLM is Groq via langchain-groq. The router has a keyword fallback so a flaky/odd LLM reply
  still routes sensibly. No silent fallback for availability: if GROQ_API_KEY is unset the API
  returns 503.
- Read-only: every query (authored OR pre-defined) passes the guard in src/agent/schema.py, so
  the agent can never mutate the DB. The overview agent uses fixed, audited queries (not
  LLM-authored), which is why "what's been detected" is robust.
- Empty-DB aware: instead of an unhelpful "no matching data", it tells the user to log some first.
"""
from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from sqlalchemy import text

from src import store
from src.agent.schema import SCHEMA_DESCRIPTION, UnsafeSQLError, sanitize_sql

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_ATTEMPTS = 3
INTENTS = ("analytics", "overview", "schema", "gallery", "out_of_scope")

# Pre-defined, audited aggregate queries for the overview agent (no LLM SQL authoring → robust).
OVERVIEW_QUERIES: dict[str, str] = {
    "runs": "SELECT COUNT(*) AS n FROM sources",
    "detections_total": "SELECT COUNT(*) AS n FROM detections",
    "objects_total": "SELECT COUNT(*) AS n FROM (SELECT DISTINCT source_id, track_id FROM detections) t",
    "by_class": (
        "SELECT class_label, COUNT(*) AS objects FROM "
        "(SELECT DISTINCT source_id, track_id, class_label FROM detections) t "
        "GROUP BY class_label ORDER BY objects DESC LIMIT 10"
    ),
}

_EMPTY_DB_MSG = (
    "No detections have been logged yet — upload a video or run the live webcam/stream first, "
    "then ask me again."
)


def available() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        return False
    try:
        import langgraph  # noqa: F401
        import langchain_groq  # noqa: F401
        return True
    except Exception:
        return False


class ChatState(TypedDict, total=False):
    question: str
    query_embedding: list[float]
    intent: str
    sql: str
    safe_sql: str
    rows: list[dict[str, Any]]
    crops: list[dict[str, Any]]
    error: str | None
    attempts: int
    answer: str


def _llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])


def _json_safe(v: Any) -> Any:
    return v if isinstance(v, (int, float, str, type(None))) else str(v)


def _run_sql(sql: str) -> list[dict[str, Any]]:
    """Execute a guarded SELECT and return rows as dicts."""
    with store.get_engine().connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, (_json_safe(v) for v in r))) for r in result.fetchall()]


def _detections_count() -> int | None:
    try:
        return _run_sql(sanitize_sql("SELECT COUNT(*) AS n FROM detections"))[0]["n"]
    except Exception:
        return None


# --- intent routing ------------------------------------------------------------------
def _normalize_intent(raw: str) -> str | None:
    r = (raw or "").strip().lower()
    return next((i for i in INTENTS if i in r), None)


def _keyword_intent(question: str) -> str:
    """Deterministic fallback router (used if the LLM is unavailable or replies oddly)."""
    q = question.lower()
    if any(w in q for w in (
        "what can you", "what can i", "can i ask", "what questions", "what data", "what column",
        "what field", "what class", "which class", "help", "schema", "what do you know",
        "what kind of question", "how do i ask",
    )):
        return "schema"
    if any(w in q for w in (
        "show me", "show the", "show all", "let me see", "see the", "picture", "image",
        "thumbnail", "gallery", "look like", "looked like", "what did", "visual", "snapshot",
    )):
        return "gallery"
    if any(w in q for w in (
        "how many", "count", "average", "avg", "most", "top", "highest", "lowest", "per ",
        "by class", "total number", "which source", "which run",
    )):
        return "analytics"
    if any(w in q for w in (
        "overview", "summary", "summarize", "what's been", "what has been", "what's getting",
        "what is getting", "tell me about", "what's happening", "describe", "anything interesting",
        "what did you", "what have you",
    )):
        return "overview"
    return "overview"  # vague questions get a robust overview, not a brittle SQL guess


def classify_intent(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    q = state["question"]
    sys = SystemMessage(content=(
        "Classify the user's question about an object-detection database into exactly ONE label:\n"
        "- analytics: a specific quantitative question answerable by a SQL query (counts, "
        "averages, top-N, filters, per-class or per-run stats).\n"
        "- overview: a broad/vague question asking what's in the data or for a summary "
        "(e.g. 'what has been detected', 'give me an overview').\n"
        "- schema: a meta question about what the data contains or what can be asked "
        "(columns, classes, capabilities, help).\n"
        "- gallery: the user wants to SEE the objects — images, pictures, thumbnails, or what "
        "something looked like (e.g. 'show me the trucks', 'what did the people look like').\n"
        "- out_of_scope: not about the detection data at all.\n"
        "Reply with ONLY the label."
    ))
    intent = None
    try:
        intent = _normalize_intent(_llm().invoke([sys, HumanMessage(content=q)]).content)
    except Exception:
        intent = None
    return {"intent": intent or _keyword_intent(q)}


def _route_intent(state: ChatState) -> str:
    return state.get("intent") or "overview"


# --- analytics agent (self-correcting text-to-SQL) -----------------------------------
def author_sql(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    retry = ""
    if state.get("error"):
        retry = (
            f"\nYour previous attempt failed: {state['error']}\n"
            f"Previous SQL:\n{state.get('sql', '')}\nFix it."
        )
    sys = SystemMessage(content=(
        "You are a precise SQL author. Given the schema, write ONE read-only SQL SELECT query "
        "that runs on BOTH PostgreSQL and SQLite, answering the question. Output ONLY SQL, no "
        "prose. Hard rule: never use COUNT(DISTINCT a, b) (multi-column COUNT DISTINCT is "
        "invalid in PostgreSQL/SQLite); count distinct combinations with a subquery as shown "
        "in the schema notes.\n\n" + SCHEMA_DESCRIPTION
    ))
    human = HumanMessage(content=f"Question: {state['question']}{retry}")
    sql = _llm().invoke([sys, human]).content
    return {"sql": sql, "attempts": state.get("attempts", 0) + 1, "error": None}


def guard(state: ChatState) -> ChatState:
    try:
        return {"safe_sql": sanitize_sql(state["sql"]), "error": None}
    except UnsafeSQLError as exc:
        return {"error": f"unsafe SQL: {exc}"}


def execute(state: ChatState) -> ChatState:
    try:
        return {"rows": _run_sql(state["safe_sql"]), "error": None}
    except Exception as exc:  # surface DB error to the author for self-correction
        return {"error": f"execution error: {str(exc)[:200]}"}


def answer(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    if state.get("error") and not state.get("rows"):
        return {"answer": f"I couldn't answer that reliably ({state['error']})."}
    rows = state.get("rows", [])
    if not rows and _detections_count() == 0:
        return {"answer": _EMPTY_DB_MSG}
    sys = SystemMessage(content=(
        "Answer the user's question about object-detection data conversationally and accurately, "
        "using ONLY the SQL result rows provided. Be concise. If the rows are empty, say no "
        "matching data was found for that specific question. Do not invent numbers."
    ))
    human = HumanMessage(content=(
        f"Question: {state['question']}\nSQL: {state.get('safe_sql','')}\n"
        f"Rows (JSON, up to 200): {json.dumps(rows[:200], default=str)}"
    ))
    return {"answer": _llm().invoke([sys, human]).content}


# --- overview agent (pre-defined aggregates → synthesis) -----------------------------
def overview(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    facts: dict[str, Any] = {}
    used: list[str] = []
    try:
        for key, sql in OVERVIEW_QUERIES.items():
            safe = sanitize_sql(sql)
            used.append(safe)
            facts[key] = _run_sql(safe)
    except Exception as exc:
        return {"answer": f"I couldn't read the data ({str(exc)[:150]}).", "error": str(exc)}

    total = (facts.get("detections_total") or [{}])[0].get("n", 0)
    if not total:
        return {"answer": _EMPTY_DB_MSG, "sql": "; ".join(used), "rows": []}

    sys = SystemMessage(content=(
        "You summarize object-detection data for a user. Using ONLY the provided facts, give a "
        "short, friendly overview: number of runs, total objects and detections, and the most "
        "common classes with their counts. Do not invent numbers."
    ))
    human = HumanMessage(content=(
        f"Question: {state['question']}\nFacts (JSON): {json.dumps(facts, default=str)}"
    ))
    return {
        "answer": _llm().invoke([sys, human]).content,
        "sql": "; ".join(used),
        "rows": facts.get("by_class", []),
    }


# --- schema/help agent ---------------------------------------------------------------
def schema_help(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        classes = _run_sql(
            sanitize_sql("SELECT DISTINCT class_label FROM detections ORDER BY class_label LIMIT 100")
        )
    except Exception:
        classes = []
    labels = [c["class_label"] for c in classes]
    sys = SystemMessage(content=(
        "You explain what an object-detection database holds and what the user can ask. Be "
        "concise and concrete. They can ask for: counts of detections, unique object (track) "
        "counts, top/most-common classes, per-class or per-run breakdowns, and average "
        "confidence. Use ONLY the provided class list for examples."
    ))
    human = HumanMessage(content=(
        f"Question: {state['question']}\n"
        f"Tracked classes so far: {labels or '(none logged yet)'}\n"
        "Tables: sources (one row per run: kind, filename, fps, model_version, conf_threshold, "
        "created_at) and detections (track_id, class_label, confidence, bbox, frame_number, "
        "ts_seconds, source_id)."
    ))
    return {"answer": _llm().invoke([sys, human]).content, "rows": classes}


# --- gallery agent (visual recall — returns object thumbnails) -----------------------
_CLASS_SYNONYMS = {"people": "person", "persons": "person", "ppl": "person", "guy": "person",
                   "guys": "person", "folks": "person"}


def _wanted_class(question: str, labels: set[str]) -> str | None:
    """Find which detected class the user means — handling multi-word labels, simple plurals
    ('cars'→'car', 'buses'→'bus') and synonyms ('people'→'person')."""
    q = question.lower()
    # direct substring (covers multi-word labels like "hard hat", "license plate")
    direct = next((lbl for lbl in sorted(labels, key=len, reverse=True) if lbl in q), None)
    if direct:
        return direct
    import re
    forms: set[str] = set()
    for t in re.findall(r"[a-z]+", q):
        forms.add(_CLASS_SYNONYMS.get(t, t))
        if t.endswith("es"):
            forms.add(t[:-2])
        if t.endswith("s"):
            forms.add(t[:-1])
    return next((lbl for lbl in labels if lbl in forms), None)


def gallery(state: ChatState) -> ChatState:
    from collections import Counter

    crops = store.get_object_crops(limit=200)
    if not crops:
        return {"answer": (
            "No object snapshots have been captured yet — run a detection (upload/webcam/stream) "
            "first, then I can show you the objects that passed."
        ), "crops": []}

    labels = {c["class_label"].lower() for c in crops}
    wanted = _wanted_class(state["question"], labels)

    # Semantic visual search when the client supplied a CLIP query embedding ("the red truck",
    # "a person in white"): rank crops by cosine similarity (optionally within the named class).
    qvec = state.get("query_embedding")
    semantic = False
    if qvec:
        ranked = store.search_object_crops(qvec, class_label=wanted, limit=30)
        if ranked:
            crops, semantic = ranked, True
    if not semantic:
        crops = [c for c in crops if not wanted or c["class_label"].lower() == wanted]

    counts = Counter(c["class_label"] for c in crops)
    summary = ", ".join(f"{n} {lbl}" for lbl, n in counts.most_common())
    scope = f" matching '{wanted}'" if wanted else ""
    lead = "Here are the closest matches" if semantic else "Here are the objects that passed"
    return {"answer": f"{lead}{scope}: {summary}.", "crops": crops[:60]}


# --- out-of-scope agent --------------------------------------------------------------
def out_of_scope(state: ChatState) -> ChatState:
    return {
        "answer": (
            "I can only answer questions about the object-detection data logged here — e.g. which "
            "classes were detected, how many of each, per-run or per-class stats, and confidence. "
            'Try: "what\'s been detected?" or "how many cars?"'
        )
    }


# --- routing helpers (analytics self-correction) -------------------------------------
def _after_guard(state: ChatState) -> str:
    if state.get("error"):
        return "retry" if state.get("attempts", 0) < MAX_ATTEMPTS else "answer"
    return "execute"


def _after_execute(state: ChatState) -> str:
    if state.get("error"):
        return "retry" if state.get("attempts", 0) < MAX_ATTEMPTS else "answer"
    return "answer"


_graph = None


def _build():
    from langgraph.graph import END, StateGraph

    g = StateGraph(ChatState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("author_sql", author_sql)
    g.add_node("guard", guard)
    g.add_node("execute", execute)
    g.add_node("answer", answer)
    g.add_node("overview", overview)
    g.add_node("schema_help", schema_help)
    g.add_node("gallery", gallery)
    g.add_node("out_of_scope", out_of_scope)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges("classify_intent", _route_intent, {
        "analytics": "author_sql",
        "overview": "overview",
        "schema": "schema_help",
        "gallery": "gallery",
        "out_of_scope": "out_of_scope",
    })
    g.add_edge("author_sql", "guard")
    g.add_conditional_edges("guard", _after_guard,
                            {"execute": "execute", "retry": "author_sql", "answer": "answer"})
    g.add_conditional_edges("execute", _after_execute,
                            {"answer": "answer", "retry": "author_sql"})
    for terminal in ("answer", "overview", "schema_help", "gallery", "out_of_scope"):
        g.add_edge(terminal, END)
    return g.compile()


def ask(question: str, query_embedding: list[float] | None = None) -> dict[str, Any]:
    """Run the agent graph and return the answer + SQL/rows/crops/intent for transparency."""
    global _graph
    if _graph is None:
        _graph = _build()
    state: dict[str, Any] = {"question": question, "attempts": 0}
    if query_embedding:
        state["query_embedding"] = query_embedding
    final = _graph.invoke(state)
    return {
        "answer": final.get("answer", ""),
        "intent": final.get("intent"),
        "sql": final.get("safe_sql") or final.get("sql"),
        "rows": (final.get("rows") or [])[:50],
        "crops": (final.get("crops") or [])[:60],
        "attempts": final.get("attempts", 0),
        "error": final.get("error"),
    }
