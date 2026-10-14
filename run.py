"""Run the full pipeline.

    python run.py                  # uses cached notes index if it matches the notes file
    python run.py --fresh-index    # re-read the notes with the LLM (needed for the repro check)
    python run.py --no-llm         # no API calls: cached index + template reasons
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import analytics
import explain
import notes
from llm import LLM

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--fresh-index", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(api_key="" if args.no_llm else None)

    weekly = analytics.weekly_costs(analytics.load_shipments(DATA / "shipment_records.csv"))
    facts, index_status = notes.load_index(DATA / "context_notes.csv", sorted(weekly["route"].unique()),
                                           llm, ROOT / "notes_index.json", fresh=args.fresh_index)

    cases = [(r, *notes.match(r, facts)) for r in weekly[weekly["anomalous"]].itertuples()]
    reasons, template_count = explain.reasons(cases, llm)

    rows = [{
        "route": r.route,
        "week_of": f"{r.week_of:%Y-%m-%d}",
        "cost_per_tonne_km": round(r.cost_per_tonne_km, 2),
        "vs_own_history": f"{r.own_pct:+.1f}% vs this route's past average",
        "vs_similar_routes": f"{r.peer_pct:+.1f}% vs similar-length routes this week",
        "flagged": "No (justified)" if supporting else "Yes",
        "matched_note_id": supporting["note_id"] if supporting else "",
        "reason": reasons[explain.row_key(r)],
    } for r, supporting, _ in cases]
    contract = list(pd.read_csv(DATA / "sample_output_format_v2.csv", nrows=0).columns)
    result = pd.DataFrame(rows, columns=contract).sort_values(["week_of", "route"])
    result.to_csv(out / "flags.csv", index=False)
    weekly.to_csv(out / "weekly_costs.csv", index=False, float_format="%.6f")
    (out / "notes_index.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    log = {**llm.cost_log(), "notes_index": index_status, "reasons_from_template": template_count,
           "shipments": int(weekly["shipments"].sum()), "route_weeks": len(weekly), "rows_out": len(result)}
    (out / "cost_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    print(result.drop(columns="reason").to_string(index=False))
    print(f"\n{len(result)} rows -> {out / 'flags.csv'}  "
          f"({(result['flagged'] == 'Yes').sum()} unexplained, {(result['flagged'] != 'Yes').sum()} justified)")
    print(f"notes index: {index_status} | LLM calls: {llm.calls}, tokens in/out: {llm.input_tokens}/{llm.output_tokens}, "
          f"~${log['cost_usd_at_published_rate']}")


if __name__ == "__main__":
    main()
