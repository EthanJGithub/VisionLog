import { useState } from "react";

// Match vocabulary classes to typed terms: exact, whole-word, or (for terms >=4 chars) prefix.
// Whole-word/prefix avoids over-matching (e.g. "car" must not select "cardboard box").
function matchClasses(classes, query) {
  const terms = query.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean);
  const set = new Set();
  for (const c of classes) {
    const lc = c.toLowerCase();
    const words = lc.split(/\s+/);
    if (terms.some((t) => lc === t || words.includes(t) || (t.length >= 4 && lc.startsWith(t)))) {
      set.add(c);
    }
  }
  const unknown = terms.filter(
    (t) => !classes.some((c) => {
      const lc = c.toLowerCase();
      return lc === t || lc.split(/\s+/).includes(t) || (t.length >= 4 && lc.startsWith(t));
    })
  );
  return { set, terms, unknown };
}

/**
 * Open-vocab "prompting": type class names to detect (turns everything else off), or toggle
 * chips. Collapsible so the long vocabulary doesn't dominate the panel.
 */
export default function ClassFilter({ classes, enabled, onToggle, onAll, onNone, onSet }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  if (!classes?.length) return null;

  function applyQuery(q) {
    setQuery(q);
    const { set, terms } = matchClasses(classes, q);
    if (!terms.length) onAll?.();          // blank = detect everything
    else onSet?.(set);                     // narrow to exactly the typed classes
  }

  const { unknown } = matchClasses(classes, query);

  return (
    <div className="classfilter">
      <div className="classfilter-head">
        <span className="muted">Detect:</span>
        <input
          className="cf-search"
          type="text"
          placeholder="type classes to detect — e.g. truck, car  (blank = all)"
          value={query}
          onChange={(e) => applyQuery(e.target.value)}
          aria-label="Type which classes to detect"
        />
        <span className="muted cf-count">{enabled.size}/{classes.length}</span>
        <button className="link" type="button" onClick={() => setOpen((o) => !o)}>
          {open ? "hide list" : "show list"}
        </button>
      </div>

      {unknown.length > 0 && (
        <p className="muted cf-unknown">Not in this vocabulary: {unknown.join(", ")}</p>
      )}

      {open && (
        <>
          <div className="classfilter-head">
            <button className="link" type="button" onClick={() => { setQuery(""); onAll?.(); }}>all</button>
            <button className="link" type="button" onClick={() => { setQuery(""); onNone?.(); }}>none</button>
          </div>
          <div className="chipwrap">
            {classes.map((c) => (
              <button
                key={c}
                type="button"
                className={`fchip ${enabled.has(c) ? "fchip-on" : ""}`}
                onClick={() => onToggle(c)}
              >
                {c}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
