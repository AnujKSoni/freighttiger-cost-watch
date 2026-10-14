"""Eval harness: python eval.py  (run after run.py; exits 1 on any failure)

1. Core math   - recompute every route-week from the raw CSV in plain Python and compare.
2. Sample rows - the 3 rows in sample_output_format_v2.csv must come out identical.
3. Note index  - LLM-extracted note facts vs hand labels (eval/notes_gold.json).
4. Labeled set - expected verdict + note for hand-labelled weeks (eval/labeled_cases.csv).
5. Grounding   - every justified row is re-checked against the HAND labels (not the LLM's
                 reading): right route, right window, note says costs rose. Reasons cite it.
"""
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

from analytics import HISTORY_WEEKS

ROOT = Path(__file__).parent
DATA, OUT, EVAL = ROOT / "data", ROOT / "output", ROOT / "eval"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


def core_math():
    cost, tonne_km, route_type = defaultdict(float), defaultdict(float), {}
    with open(DATA / "shipment_records.csv", newline="") as f:
        for r in csv.DictReader(f):
            day = dt.date.fromisoformat(r["shipment_date"])
            key = (f"{r['origin']}-{r['destination']}", day - dt.timedelta(days=day.weekday()))
            cost[key] += float(r["freight_cost_inr"])
            tonne_km[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            route_type[key[0]] = r["route_type"]
    cpk = {k: cost[k] / tonne_km[k] for k in cost}

    weekly = pd.read_csv(OUT / "weekly_costs.csv")
    worst = 0.0
    for row in weekly.itertuples():
        week = dt.date.fromisoformat(row.week_of[:10])
        past = [cpk[(row.route, week - dt.timedelta(weeks=i))] for i in range(1, HISTORY_WEEKS + 1)
                if (row.route, week - dt.timedelta(weeks=i)) in cpk]
        peers = [v for (r, w), v in cpk.items() if w == week and r != row.route and route_type[r] == route_type[row.route]]
        for got, want in [(row.cost_per_tonne_km, cpk[(row.route, week)]),
                          (row.own_baseline, mean(past) if past else float("nan")),
                          (row.peer_baseline, mean(peers) if peers else float("nan"))]:
            if not (pd.isna(got) and pd.isna(want)):
                worst = max(worst, abs(got - want) / want)
    check("core math: cost/tonne-km, own baseline, peer baseline for all route-weeks",
          len(weekly) == len(cpk) and worst < 1e-5, f"{len(weekly)} route-weeks, max relative error {worst:.1e}")


def sample_rows(flags):
    # The given sample has an unquoted comma in one `reason`, so pandas can't parse it;
    # read it raw and keep the 7 graded columns before `reason`.
    with open(DATA / "sample_output_format_v2.csv", newline="") as f:
        header, *lines = list(csv.reader(f))
    cols = header[:7]
    want = pd.DataFrame([line[:7] for line in lines], columns=cols)
    got = want[["route", "week_of"]].merge(flags, how="left", on=["route", "week_of"])[cols].astype(str)
    ok = got.equals(want)
    check("sample rows reproduced (numbers, verdict, note)", ok, "" if ok else f"\n{got}\nvs\n{want}")
    check("output columns match contract", list(flags.columns) == header)


def note_index(gold, all_routes):
    facts = {f["note_id"]: f for f in json.loads((OUT / "notes_index.json").read_text(encoding="utf-8"))}
    misses = []
    for nid, g in gold.items():
        f = facts[nid]
        want_routes = all_routes if g["routes"] == "ALL" else sorted(g["routes"])
        if f["routes"] != want_routes: misses.append(f"{nid} routes")
        if f["start"] != g["start"]: misses.append(f"{nid} start")
        if (f["cost_effect"] == "increase") != g["increase"]: misses.append(f"{nid} cost_effect")
        if g["increase"] and f["end"] != g["end"]: misses.append(f"{nid} end")
        if f["problems"]: misses.append(f"{nid} failed validation: {f['problems']}")
    check("note index vs hand labels", not misses, f"{len(gold)} notes, {len(misses)} mismatches {misses or ''}")


def labeled_cases(flags):
    cases = pd.read_csv(EVAL / "labeled_cases.csv", keep_default_na=False)
    got = {(r.route, r.week_of): (r.flagged, r.matched_note_id) for r in flags.itertuples()}
    wrong = [f"{c.route} {c.week_of}: want {c.expected_flagged}/{c.expected_note or '-'}, "
             f"got {'/'.join(got.get((c.route, c.week_of), ('No', ''))) or '-'}"
             for c in cases.itertuples()
             if got.get((c.route, c.week_of), ("No", "")) != (c.expected_flagged, c.expected_note)]
    check("labeled cases", not wrong, f"{len(cases) - len(wrong)}/{len(cases)} correct {wrong or ''}")


def grounding(flags, gold, all_routes):
    bad = []
    for r in flags.itertuples():
        cited = set(re.findall(r"N\d{3}", r.reason))
        if r.flagged == "Yes":
            if r.matched_note_id or not r.reason.startswith("Unexplained:"):
                bad.append(f"{r.route} {r.week_of}: unexplained row claims a note")
            continue
        g = gold[r.matched_note_id]
        week = pd.Timestamp(r.week_of)
        start = pd.Timestamp(g["start"])
        end = pd.Timestamp(g["end"]) if g["end"] else start + pd.Timedelta(weeks=HISTORY_WEEKS, days=-1)
        routes = all_routes if g["routes"] == "ALL" else g["routes"]
        if not (g["increase"] and r.route in routes and start <= week + pd.Timedelta(days=6) and end >= week):
            bad.append(f"{r.route} {r.week_of}: {r.matched_note_id} does not apply (fabricated)")
        if not r.reason.startswith("Justified:") or r.matched_note_id not in cited:
            bad.append(f"{r.route} {r.week_of}: reason doesn't cite {r.matched_note_id}")
    justified = (flags["flagged"] != "Yes").sum()
    check("grounding: justified rows hold up against hand labels; reasons cite their note",
          not bad, f"{justified} justified rows checked {bad or ''}")


if __name__ == "__main__":
    gold = {k: v for k, v in json.loads((EVAL / "notes_gold.json").read_text()).items() if not k.startswith("_")}
    flags = pd.read_csv(OUT / "flags.csv", keep_default_na=False, dtype={"cost_per_tonne_km": str})
    all_routes = sorted(pd.read_csv(OUT / "weekly_costs.csv")["route"].unique())
    core_math()
    sample_rows(flags)
    note_index(gold, all_routes)
    labeled_cases(flags)
    grounding(flags, gold, all_routes)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)
