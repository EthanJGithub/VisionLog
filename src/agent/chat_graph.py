"""LangGraph multi-agent text-to-SQL chatbot over the detections database.

Graph (each node = an agent role), with a validation + self-correction loop:

    author_sql ──► guard ──►(ok)──► execute ──►(ok)──► answer ──► END
        ▲            │(unsafe)          │(db error)
        └────────────┴──────────────────┘   (retry up to MAX_ATTEMPTS with the error)

Mirrors the CredAgent pattern (deterministic, auditable LangGraph state transitions). LLM is
Groq via langchain-groq. Read-only: the guard (src/agent/schema.py) permits only single SELECT
statements, so the agent can never mutate the DB. No silent fallback: if GROQ_API_KEY is unset
the API returns 503 rather than degrading.
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
    sql: str
    safe_sql: str
    rows: list[dict[str, Any]]
    error: str | None
    attempts: int
    answer: str


def _llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])


# --- agent nodes ---------------------------------------------------------------------
def author_sql(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    retry = ""
    if state.get("error"):
        retry = (
            f"\nYour previous attempt failed: {state['error']}\n"
            f"Previous SQL:\n{state.get('sql', '')}\nFix it."
        )
    sys = SystemMessage(content=(
        "You are a precise SQL author. Given the schema, write ONE read-only SQL SELECT "
        "query (SQLite/Postgres compatible) that answers the question. Output ONLY SQL, no "
        "prose.\n\n" + SCHEMA_DESCRIPTION
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
        with store.get_engine().connect() as conn:
            result = conn.execute(text(state["safe_sql"]))
            cols = list(result.keys())
            rows = [dict(zip(cols, (_json_safe(v) for v in r))) for r in result.fetchall()]
        return {"rows": rows, "error": None}
    except Exception as exc:  # surface DB error to the author for self-correction
        return {"error": f"execution error: {str(exc)[:200]}"}


def answer(state: ChatState) -> ChatState:
    from langchain_core.messages import HumanMessage, SystemMessage

    if state.get("error") and state.get("rows") is None:
        return {"answer": f"I couldn't answer that reliably ({state['error']})."}
    rows = state.get("rows", [])
    sys = SystemMessage(content=(
        "Answer the user's question about object-detection data conversationally and "
        "accurately, using ONLY the SQL result rows provided. Be concise. If the rows are "
        "empty, say no matching data was found. Do not invent numbers."
    ))
    human = HumanMessage(content=(
        f"Question: {state['question']}\nSQL: {state.get('safe_sql','')}\n"
        f"Rows (JSON, up to 200): {json.dumps(rows[:200], default=str)}"
    ))
    return {"answer": _llm().invoke([sys, human]).content}


def _json_safe(v: Any) -> Any:
    return v if isinstance(v, (int, float, str, type(None))) else str(v)


# --- routing -------------------------------------------------------------------------
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
    g.add_node("author_sql", author_sql)
    g.add_node("guard", guard)
    g.add_node("execute", execute)
    g.add_node("answer", answer)
    g.set_entry_point("author_sql")
    g.add_edge("author_sql", "guard")
    g.add_conditional_edges("guard", _after_guard, {"execute": "execute", "retry": "author_sql", "answer": "answer"})
    g.add_conditional_edges("execute", _after_execute, {"answer": "answer", "retry": "author_sql"})
    g.add_edge("answer", END)
    return g.compile()


def ask(question: str) -> dict[str, Any]:
    """Run the agent graph and return the answer + the SQL/rows for transparency."""
    global _graph
    if _graph is None:
        _graph = _build()
    final = _graph.invoke({"question": question, "attempts": 0})
    return {
        "answer": final.get("answer", ""),
        "sql": final.get("safe_sql") or final.get("sql"),
        "rows": (final.get("rows") or [])[:50],
        "attempts": final.get("attempts", 0),
        "error": final.get("error"),
    }
