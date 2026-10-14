# Reproducibility check

3 untouched runs of `python run.py --fresh-index` (notes re-read by the LLM each time).

| run | rows | unexplained | justified | LLM calls | tokens in | tokens out | cost (USD) |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 16 | 4 | 2 | 4841 | 2360 | 0.002142 |
| 2 | 20 | 16 | 4 | 2 | 4841 | 2496 | 0.002224 |
| 3 | 20 | 16 | 4 | 2 | 4841 | 2438 | 0.002189 |

- Flags, numbers, verdicts and matched_note_id identical across runs: **YES**
- LLM-extracted note facts (routes, dates, cost effect) identical: **YES**
- Reason wording identical (allowed to differ): no
