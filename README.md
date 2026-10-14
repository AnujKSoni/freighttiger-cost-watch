# FreightTiger cost watch

This assistant flags routes whose freight cost per tonne-km is rising unusually. For each flag it checks the context notes for a real reason, and it marks the flag **justified** only when a note actually covers that route and that week.

Output: [`output/flags.csv`](output/flags.csv) has 20 flagged route-weeks: 16 unexplained and 4 justified. It matches the columns of `sample_output_format_v2.csv`.

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...          # PowerShell: $env:GROQ_API_KEY="..."
python run.py --fresh-index      # one full run: re-reads the notes with the LLM and writes output/
python eval.py                   # eval harness, 6 checks
python repro.py                  # 3 untouched runs + diff  ->  runs/repro_report.md
python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
```

You can also run it without an API key: `python run.py --no-llm`. It uses the committed `notes_index.json` and template reasons. The verdicts and numbers are identical to an LLM run; `runs/no_llm/` holds the proof.

## How it works

```
shipment_records.csv ──► analytics.py ──► weekly cost/tonne-km, two baselines, anomaly rule
                                                  │ anomalous route-weeks
context_notes.csv ──► notes.py (1 LLM call) ──► structured, validated note facts
                                                  │
                         notes.match()  ◄─────────┘   pure route/date rules: the verdict and matched_note_id
                                                  │
                         explain.py (1 LLM call) ──► reason text, checked before use
```

**1. Numbers** (`analytics.py`, pandas only, no LLM)
- Each shipment goes into its Monday-to-Sunday week, and `week_of` is that week's Monday.
- `cost_per_tonne_km` = total weekly cost / total weekly tonne-km for the route. This pooled ratio reproduces all three sample rows exactly; an average of per-shipment ratios does not.
- **vs own history:** the mean of the route's weekly values over the 8 calendar weeks strictly before the current week.
- **vs similar routes:** the mean, in the same week, of the other routes with the same `route_type`. The route itself is excluded.
- **Fewer than 8 prior weeks:** this applies to the first 8 weeks of 2024 (2024-01-01 to 2024-02-19, 56 route-weeks). Those weeks use all the history available. The very first week has no baseline and can't be flagged. No flagged row falls in this period.

**2. Anomaly rule.** A week is anomalous when cost is above its own history (rising) **and** at least one of these holds:
- it is ≥ 10% above its own history, or
- it is ≥ 20% above similar routes.

The brief doesn't fix thresholds, so I picked them from the data:
- Week-to-week noise vs own history is about ±3–6% (max ≈ 8.5% outside known events), so 10% clears it.
- The peer check is needed for slow creep. Mumbai-Pune normally runs about 10% above its only Short peer. From Sep 2025 it drifts to +39%, while each week is only +3–9% vs its own trailing average. An 8-week baseline drifts up with the creep and never flags it alone.

| own ↓ / peer → | 15% | **20%** | 25% |
|---|---|---|---|
| 8%  | 28 | 21 | 16 |
| **10%** | 27 | **20** | 14 |
| 12% | 26 | 19 | 13 |

Known borderline: Mumbai-Pune 2025-06-23 (+6.2% own, +20.6% peer) is the only Mumbai-Pune week before the creep to cross 20%. Short routes have only one peer each, so their peer gap is noisy. The row is flagged for review, not justified.

**3. Notes.** The AI part is in `notes.py`. One LLM call reads all 10 notes. For each note it returns:
- which dataset routes the note affects,
- the dates it covers,
- whether it says costs **rose**,
- any stated magnitude,
- a verbatim evidence quote.

Code then validates every field:
- the routes exist in the dataset,
- the start date is near the note date,
- the quote really is in the note,
- any magnitude is stated in the note.

A note that fails any check is marked `unusable` and can never justify a flag.

This matters because 7 of the 10 notes are traps. They mention a route and a date but say costs were *not* affected, or that conditions were normal or improved, or that the affected routes aren't in the dataset. The LLM classified all 7 correctly as non-increase (see eval).

**4. Verdict** (`notes.match`, plain rules, no LLM). A flagged week is justified by a note only if all of these hold:
- the note covers the route ("All Routes" counts),
- its window overlaps the week,
- it reports a cost **increase**,
- its stated magnitude isn't far below the observed rise (more than 2× is rejected),
- if it applies to every route, the flag isn't about the gap vs peers. Peers got the same increase, so the note can't explain being out of line with them.

An open-ended note (diesel up from 2025-05-05) explains rises for 8 weeks, until the own-history baseline has absorbed the step. After that, a new rise needs a new reason. When several notes match, the route-specific one wins over All Routes, then the most recent.

**5. Reasons** (`explain.py`). One batched LLM call phrases a reason for every flagged row. It receives:
- the fixed verdict,
- the matched note,
- the nearby notes that were rejected, each with the rule that rejected it.

Each reason is then checked: it must start with the verdict, cite the matched note, and cite no other note. A failed reason is replaced by a deterministic template. In the submitted run, 0 reasons fell back.

### Why not a vector database?

There are 10 one-line notes, and the trap notes are *semantically* almost identical to the real ones ("Mumbai-Delhi maintenance, costs not affected"). Embedding similarity would retrieve them happily. What decides relevance here is route, dates and direction of effect. So the LLM turns each note into those fields once, and retrieval is an exact filter on them. With thousands of free-text notes I would add an embedding pre-filter in front of the same extraction step. The verdict logic would not change.

## How I convinced myself it's right

`python eval.py` runs 6 checks, and all 6 pass:

| check | result |
|---|---|
| Core math: every route-week recomputed in plain Python (csv + dicts, no pandas), independent of `analytics.py` | 728/728 match, max rel. error 2.5e-07 (CSV rounding) |
| The 3 rows in the sample file reproduce exactly (cost, both % strings, verdict, note) | pass |
| Output columns = contract | pass |
| LLM note facts vs my hand labels ([`eval/notes_gold.json`](eval/notes_gold.json)): routes, start, end, increase? | 10/10 notes, 0 mismatches |
| Hand-labelled weeks ([`eval/labeled_cases.csv`](eval/labeled_cases.csv)): flood weeks, post-flood weeks, every trap note, diesel, slow creep, weeks that must *not* appear | 15/15 |
| Grounding: every justified row re-checked against my **hand** labels, not the LLM's reading (route, window, note says costs rose); reasons cite the note | 4/4 |

The labelled cases worth calling out:
- **Chennai-Bangalore flood:** 02-24 and 03-03 are justified by N001. On 03-10 and 03-17 cost stayed high, but the flood ended on Mar 8 and N009 says the route was back to normal, so both weeks are **unexplained**.
- **Mumbai-Pune after 2025-10-27:** N010 says the compliance cost was absorbed with no rate change, so it isn't cited.

## Reproducibility

[`runs/repro_report.md`](runs/repro_report.md) covers three untouched `--fresh-index` runs. Each run re-reads the notes with the LLM, so the check covers the LLM step, not only a cache.
- Flags, numbers, verdicts and `matched_note_id` are identical across runs.
- Extracted note facts are identical across runs.
- Only reason wording varies, which the brief allows.

The LLM uses `temperature=0, seed=7, reasoning_effort=low`. The verdict never depends on LLM prose; it depends only on the validated note facts, and those matched across runs.

One wobble turned up while building: the model sometimes gave an end date to a "no cost increase" note (N009) and sometimes didn't. An end date only matters for notes that can justify a rise, so code now nulls it for all other notes.

## Cost of one full run

From [`output/cost_log.json`](output/cost_log.json), run with `--fresh-index`, over all 2,940 shipments:

| model | LLM calls | input tokens | output tokens (incl. 530 reasoning) | cost at Groq published rate |
|---|---|---|---|---|
| `openai/gpt-oss-120b` (open-weight, Groq free tier) | 2 | 4,841 | 2,397 | ≈ $0.0022 ($0.15 / $0.60 per M tokens) |

The call count doesn't grow with the shipment file. It is 1 call per notes file (cached by content hash) plus 1 batched call for the reasons. A run with a cached index makes 1 call. `ask.py` costs about 3k input tokens per question.

## Trade-offs and limits

- **Output scope:** the output lists only anomalous weeks (flagged or justified), as in the sample. All 728 route-weeks with baselines are in `output/weekly_costs.csv`.
- **Peer baselines:** the peer baseline for Short and Long routes is a single route, so it is noisy. A per-route "normal premium" would cut false peer flags. I kept the brief's exact definition because it's graded.
- **Magnitude guard:** the 2× limit and both thresholds are judgement calls, documented above and as constants at the top of `analytics.py` and `notes.py`.
- **Q&A:** `ask.py` answers from `flags.csv`, monthly averages and the notes, and is told never to re-judge verdicts. Its answers are LLM prose and can be incomplete. For example, it may summarise only some of a month's flagged weeks.
- **Sample file bug:** `sample_output_format_v2.csv` has an unquoted comma inside one `reason` (line 4), so strict CSV parsers reject it. `eval.py` reads it tolerantly. Our output quotes fields properly.

## Files

| file | purpose |
|---|---|
| `analytics.py` | weekly costs, baselines, anomaly rule |
| `notes.py` | LLM note extraction + validation + cache, rule-based matching |
| `explain.py` | reason text (LLM, checked; template fallback) |
| `llm.py` | Groq REST client, fixed sampling, token/cost accounting |
| `run.py` / `eval.py` / `repro.py` / `ask.py` | pipeline, eval harness, 3-run check, Q&A |
| `notes_index.json` | cached note facts (lets reviewers run without a key) |
