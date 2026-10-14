"""Reproducibility check: python repro.py

Three untouched runs, each re-reading the notes with the LLM (--fresh-index), so the check
covers the LLM step too and not only the cache. Everything except the free-text `reason`
column must match across runs. Writes runs/repro_report.md.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RUNS = 3
DECISION_FIELDS = ["note_id", "routes", "start", "end", "cost_effect", "magnitude_pct"]


def main():
    for i in range(1, RUNS + 1):
        subprocess.run([sys.executable, "run.py", "--fresh-index", "--out", f"runs/run{i}"],
                       cwd=ROOT, check=True, capture_output=True)
    dirs = [ROOT / "runs" / f"run{i}" for i in range(1, RUNS + 1)]
    flags = [pd.read_csv(d / "flags.csv", keep_default_na=False, dtype=str) for d in dirs]
    graded = [f.drop(columns="reason") for f in flags]
    index = [[{k: n[k] for k in DECISION_FIELDS} for n in json.loads((d / "notes_index.json").read_text())] for d in dirs]
    logs = [json.loads((d / "cost_log.json").read_text()) for d in dirs]

    same_flags = all(g.equals(graded[0]) for g in graded[1:])
    same_index = all(ix == index[0] for ix in index[1:])
    same_reasons = all(f["reason"].equals(flags[0]["reason"]) for f in flags[1:])

    lines = ["# Reproducibility check", "",
             f"{RUNS} untouched runs of `python run.py --fresh-index` (notes re-read by the LLM each time).", "",
             "| run | rows | unexplained | justified | LLM calls | tokens in | tokens out | cost (USD) |",
             "|---|---|---|---|---|---|---|---|"]
    for i, (f, log) in enumerate(zip(flags, logs), 1):
        lines.append(f"| {i} | {len(f)} | {(f['flagged'] == 'Yes').sum()} | {(f['flagged'] != 'Yes').sum()} | "
                     f"{log['llm_calls']} | {log['input_tokens']} | {log['output_tokens']} | {log['cost_usd_at_published_rate']} |")
    lines += ["",
              f"- Flags, numbers, verdicts and matched_note_id identical across runs: **{'YES' if same_flags else 'NO'}**",
              f"- LLM-extracted note facts (routes, dates, cost effect) identical: **{'YES' if same_index else 'NO'}**",
              f"- Reason wording identical (allowed to differ): {'yes' if same_reasons else 'no'}"]
    if not same_flags:
        for i, g in enumerate(graded[1:], 2):
            if not g.equals(graded[0]):
                lines += ["", f"## run1 vs run{i}", "```", graded[0].compare(g).to_string(), "```"]
    report = "\n".join(lines) + "\n"
    (ROOT / "runs" / "repro_report.md").write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(0 if same_flags and same_index else 1)


if __name__ == "__main__":
    main()
