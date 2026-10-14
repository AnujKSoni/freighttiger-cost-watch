"""Context notes -> structured facts (one LLM call, cached) -> rule-based matching.

The LLM reads every note once and says which dataset routes it affects, which dates it covers,
whether it reports a cost increase, and quotes the sentence that says so. Each field is checked
against the note text; a note that fails any check is marked unusable and can never justify a
flag. Matching a flagged week to a note is then plain route/date logic, so the verdict path has
no LLM in it and a justified verdict cannot cite a note that does not apply.
"""
import hashlib
import json
import re
from datetime import timedelta

import pandas as pd

from analytics import HISTORY_WEEKS, PEER_GAP_PCT
from llm import MODEL

PROMPT_VERSION = "v2"
NEAR_DAYS = 7               # notes this close to a week are reported as "considered" in reasons
MAX_MAGNITUDE_STRETCH = 2   # ponytail: a "+5-7%" note can't explain a +30% jump; 2x allows weekly noise

SYSTEM = """You turn short logistics notes into structured facts. Use only what each note says; never guess.
Return JSON {"notes": [...]} with one object per note, in the given order:
{
 "note_id": the id given,
 "routes": dataset routes (exact names from the list provided) the note's effect applies to.
     "All Routes" means every dataset route, unless the note says the affected routes are not in this dataset; then [].
 "start": "YYYY-MM-DD", first day the effect applies (the note date unless the text says otherwise),
 "end": "YYYY-MM-DD" last day the effect applies, or null when open-ended.
     The text gives an end ("from X to Y", "passable again by Z") -> the last affected day.
     A one-week event ("festival week", "for about a week") -> start + 6 days.
     A lasting change with no end ("starting this week", "this quarter") -> null.
     Use null whenever cost_effect is not "increase".
 "cost_effect": "increase" only if the note says freight/transport costs or rates went UP on those routes.
     "none" if it says costs were not (significantly) affected, were absorbed, or that things were stable, normal, without disruption.
     "decrease" if it says costs or conditions improved.
 "magnitude_pct": [low, high] if the note states a percentage change, else null,
 "evidence": an exact verbatim quote from the note that supports cost_effect
}"""


def load_index(notes_path, routes, llm, cache_path, fresh=False):
    """Structured note facts, rebuilt only when the notes file, prompt, model or routes change."""
    raw_notes = open(notes_path, "rb").read()
    key = hashlib.sha256(raw_notes + f"|{PROMPT_VERSION}|{MODEL}|{routes}".encode()).hexdigest()
    if not fresh and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["key"] == key:
            return cached["notes"], "cached"
    if not llm.available:
        raise SystemExit("Notes index is missing or stale and GROQ_API_KEY is not set.")

    notes = pd.read_csv(notes_path)
    listing = "\n".join(f"{n.note_id} | date {n.date} | applies_to: {n.applies_to} | {n.note}"
                        for n in notes.itertuples())
    raw = llm.json(SYSTEM, f"Dataset routes: {', '.join(routes)}\n\nNotes:\n{listing}")
    facts = _validate(raw, notes, routes)
    cache_path.write_text(json.dumps({"key": key, "model": MODEL, "notes": facts}, indent=1), encoding="utf-8")
    return facts, "rebuilt"


def _validate(raw, notes, routes):
    items = raw if isinstance(raw, list) else raw.get("notes", [])  # model sometimes drops the wrapper
    by_id = {n.get("note_id"): n for n in items if isinstance(n, dict)}
    facts = []
    for note in notes.itertuples():
        n = by_id.get(note.note_id, {})
        start, end = _date(n.get("start")), _date(n.get("end"))
        problems = []
        if not n:
            problems.append("missing from LLM output")
        if not set(n.get("routes") or []) <= set(routes):
            problems.append("unknown route")
        if start is None or abs((start - pd.Timestamp(note.date)).days) > 31:
            problems.append("start date not near note date")
        if n.get("end") and (end is None or start is None or end < start):
            problems.append("bad end date")
        if n.get("cost_effect") not in ("increase", "none", "decrease"):
            problems.append("bad cost_effect")
        if not n.get("evidence") or _squash(n["evidence"]) not in _squash(note.note):
            problems.append("evidence is not a quote from the note")
        if n.get("magnitude_pct") and not all(f"{x:g}" in note.note for x in n["magnitude_pct"]):
            problems.append("magnitude not stated in note")
        facts.append({
            "note_id": note.note_id, "date": note.date, "applies_to": note.applies_to, "text": note.note,
            "routes": sorted(n.get("routes") or []), "start": n.get("start"),
            # An end date only matters for notes that can justify a rise; dropping it elsewhere
            # removes run-to-run wobble on fields that never affect a verdict.
            "end": n.get("end") if n.get("cost_effect") == "increase" else None,
            "cost_effect": "unusable" if problems else n["cost_effect"],
            "magnitude_pct": n.get("magnitude_pct"), "evidence": n.get("evidence"), "problems": problems,
        })
    return facts


def window(fact):
    """Dates a note can explain. An open-ended change (e.g. diesel up from a date) only explains a
    rise until the 8-week own-history baseline has absorbed it; after that a new rise needs a new reason."""
    start = pd.Timestamp(fact["start"])
    end = pd.Timestamp(fact["end"]) if fact["end"] else start + timedelta(weeks=HISTORY_WEEKS, days=-1)
    return start, end


def match(row, facts):
    """-> (supporting fact or None, [(fact, why_not), ...] for nearby notes on this route that don't support it)."""
    week_start, week_end = row.week_of, row.week_of + timedelta(days=6)
    supporting, rejected = [], []
    for f in facts:
        if row.route not in f["routes"]:
            continue
        start, end = window(f)
        if start > week_end + timedelta(days=NEAR_DAYS) or end < week_start - timedelta(days=NEAR_DAYS):
            continue
        why_not = _why_not(f, row, start, end, week_start, week_end)
        if why_not:
            rejected.append((f, why_not))
        else:
            supporting.append(f)
    # Most specific first (route-specific beats All Routes), then most recent, then id.
    supporting.sort(key=lambda f: (len(f["routes"]), -pd.Timestamp(f["start"]).value, f["note_id"]))
    return (supporting[0] if supporting else None), rejected


def _why_not(f, row, start, end, week_start, week_end):
    if f["cost_effect"] == "unusable":
        return "could not be verified against its own text"
    if f["cost_effect"] != "increase":
        return "does not report a cost increase"
    if start > week_end:
        return f"only starts on {start:%Y-%m-%d}, after this week"
    if end < week_start:
        return f"only covers up to {end:%Y-%m-%d}, before this week"
    if f["magnitude_pct"] and row.own_pct > MAX_MAGNITUDE_STRETCH * max(f["magnitude_pct"]):
        lo, hi = f["magnitude_pct"]
        return f"reports a {lo:g}-{hi:g}% rise, far smaller than this week's {row.own_pct:+.1f}%"
    if len(f["routes"]) > 1 and row.peer_pct >= PEER_GAP_PCT:
        return "applies to similar routes too, so it cannot explain the gap against them"
    return None


def _date(s):
    try:
        return pd.Timestamp(s) if s else None
    except (ValueError, TypeError):
        return None


def _squash(s):
    return re.sub(r"\s+", " ", s).strip().lower()
