"""Plain-English reasons for each flagged week.

Verdicts and matched notes are fixed before this step. The LLM only phrases them, in one batched
call, and each reason is checked (right verdict, cites the matched note, cites no other note)
before use. Anything that fails the check, or any run without an API key, uses the template.
"""
import json
import re

from notes import window

SYSTEM = """You write the 'reason' column of a freight cost audit: one or two plain-English sentences per row.
The verdict and matched note are already decided. Do not change them, and use only the facts given.
- Begin with exactly the given prefix ("Justified:" or "Unexplained:"), then, for every row, give the size of
  the rise, e.g. "cost is 35.5% above its own history and 21.0% above similar routes".
- Justified: name the matched note id, say what it reports and the dates it covers.
- Unexplained: say no note supports the rise. If notes are listed under considered_notes, name them and say
  briefly why they don't apply; if none are listed, say no note covers this route around this week.
  End by recommending a human review.
- Never mention a note id that is not given for that row. Never suggest causes the notes don't state.
- Don't use the words "considered notes", "prefix" or "row key" in the text.
Return JSON {"reasons": {"<row key>": "<reason>"}}."""


def reasons(cases, llm):
    """cases: [(row, supporting_fact_or_None, [(fact, why_not), ...])] -> ({row key: reason}, template_count)."""
    templates = {row_key(r): template(r, s, rej) for r, s, rej in cases}
    if not llm.available or not cases:
        return templates, len(cases)
    payload = [_payload(r, s, rej) for r, s, rej in cases]
    drafted = llm.json(SYSTEM, json.dumps(payload, indent=1)).get("reasons", {})
    final, fallbacks = {}, 0
    for r, s, rej in cases:
        k = row_key(r)
        text = drafted.get(k)
        if isinstance(text, str) and is_grounded(text, s, rej):
            final[k] = text.strip()
        else:
            final[k], fallbacks = templates[k], fallbacks + 1
    return final, fallbacks


def is_grounded(text, supporting, rejected):
    cited = set(re.findall(r"N\d{3}", text))
    allowed = {f["note_id"] for f, _ in rejected} | ({supporting["note_id"]} if supporting else set())
    if supporting:
        return text.startswith("Justified:") and supporting["note_id"] in cited and cited <= allowed
    return text.startswith("Unexplained:") and cited <= allowed


def template(row, supporting, rejected):
    if supporting:
        return (f"Justified: matches note {supporting['note_id']} dated {supporting['date']}, which says: "
                f"\"{supporting['evidence']}\" It applies to {row.route} {_span(supporting)}, "
                f"covering the week of {row.week_of:%Y-%m-%d}.")
    if rejected:
        f, why_not = rejected[0]
        return (f"Unexplained: no note supports this rise. The closest note, {f['note_id']} ({f['date']}), "
                f"{why_not}, so it is not a justification. Flagged for human review.")
    return ("Unexplained: no matching note found for this route or date range. "
            "The cost rise looks unexplained and is worth a human review.")


def _span(fact):
    start, end = window(fact)
    if fact["end"]:
        return f"from {start:%Y-%m-%d} to {end:%Y-%m-%d}"
    return f"from {start:%Y-%m-%d} onward (open-ended, counted until {end:%Y-%m-%d} while the 8-week baseline catches up)"


def row_key(row):
    return f"{row.route}|{row.week_of:%Y-%m-%d}"


def _payload(row, supporting, rejected):
    note = lambda f: {"note_id": f["note_id"], "date": f["date"], "text": f["text"]}
    item = {"row_key": row_key(row), "prefix": "Justified:" if supporting else "Unexplained:",
            "route": row.route, "week_of": f"{row.week_of:%Y-%m-%d}",
            "cost_per_tonne_km": round(row.cost_per_tonne_km, 2),
            "vs_own_history_pct": round(row.own_pct, 1), "vs_similar_routes_pct": round(row.peer_pct, 1)}
    if supporting:
        item["matched_note"] = {**note(supporting), "applies": _span(supporting)}
    item["considered_notes"] = [{**note(f), "why_not": why} for f, why in rejected]
    return item
