"""Ask the assistant about its own findings (stretch goal).

    python ask.py "why did Chennai-Bangalore get pricier in March 2025?"

Answers come only from the latest output/flags.csv, monthly route averages and the notes.
"""
import sys
from pathlib import Path

import pandas as pd

from llm import LLM

ROOT = Path(__file__).parent

SYSTEM = """You answer questions about a freight cost audit. Use ONLY the data given: the flagged weeks
(with verdicts, matched notes and reasons), monthly average cost per tonne-km per route, and the context notes.
- Verdicts in the flagged table are final; report them, don't re-judge them.
- Cite route, week_of and note ids for every claim. Say plainly if the data doesn't answer the question.
- Never offer a cause that isn't in a note. Keep it to a short paragraph."""


def context():
    flags = pd.read_csv(ROOT / "output" / "flags.csv")
    weekly = pd.read_csv(ROOT / "output" / "weekly_costs.csv", parse_dates=["week_of"])
    monthly = (weekly.groupby(["route", weekly["week_of"].dt.strftime("%Y-%m")])["cost_per_tonne_km"]
                     .mean().round(2).unstack(0))
    notes = pd.read_csv(ROOT / "data" / "context_notes.csv")
    return (f"FLAGGED WEEKS\n{flags.to_csv(index=False)}\n"
            f"MONTHLY AVG COST PER TONNE-KM (month of week_of)\n{monthly.to_csv()}\n"
            f"CONTEXT NOTES\n{notes.to_csv(index=False)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8")  # model output uses non-ASCII dashes; Windows console is cp1252
    llm = LLM()
    if not llm.available:
        raise SystemExit("Set GROQ_API_KEY to ask questions.")
    print(llm.text(SYSTEM, f"{context()}\nQUESTION: {' '.join(sys.argv[1:])}"))
    print(f"\n[{llm.input_tokens} in / {llm.output_tokens} out tokens]")
