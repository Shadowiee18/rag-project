# Evaluation and experiment log

Every change to the system was measured here before it was kept. This file is the full record, in the order the
experiments ran, including the ideas the numbers rejected. The summary is in the [main README](../README.md).

```
uv run python evals/run_eval.py --name my_run                                       # page-style dev split
uv run python evals/run_eval.py --name my_run --golden evals/golden_student.jsonl   # student-style dev split
uv run python evals/rejudge.py my_run      # re-grade saved answers after a judge change (no new search/generation)
uv run python evals/score.py my_run        # agreement with the strong judge's grades, if present
uv run python evals/groups.py              # how many questions per question type x segment (empty cells included)
uv run python evals/groups.py --run my_run # a saved dev run's numbers per question type and segment
```

| File | What |
|---|---|
| `run_eval.py` | runs the full pipeline per question; writes `results/<run>.jsonl` and `results/<run>_summary.json` |
| `rejudge.py` | re-grades saved runs under the current judge rules |
| `groups.py` | coverage (questions per question type x segment) and a saved dev run's numbers per group |
| `score.py` | judge agreement (Cohen's kappa) and chunk-level evidence from saved runs |
| `judge_rubric.md` | rubric for the strong judge (correctness, faithfulness, citations) |
| `golden_set.example.jsonl` | the question-file format |
| `results/*_summary.json` | the numbers of every run (committed); per-question files stay local (course material) |

## Phase 1: baseline and first retrieval fixes

Golden set: 62 questions (54 answerable, 8 unanswerable). Local `qwen3:8b` + `qwen3-embedding:0.6b`, top 5 chunks.

| Run | What changed | hit@1 | hit@5 | MRR@10 | all pages in context | correct* | wrong* | false "I don't know" | unanswerable refused |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | naive 1500-char chunks | 59% | 87% | 0.71 | 80% | 33% | 7% | 9% | 7/8 |
| `exp1_headers` | + "Year 2 · Block B · title" header on every chunk | 63% | 87% | 0.73 | 80% | 39% | 4% | 7% | 7/8 |
| `exp2b_filter` | + year/block filter read from the question | 67% | 93% | 0.77 | 87% | 41% | 2% | 4% | 6/8 |
| `exp3b_user_year` | + student's own year when the question names none (simulated) | **70%** | **93%** | **0.80** | **87%** | **44%** | 2% | 4% | 7/8 |

\* judged by the same local model against the expected answer: treat as rough. Retrieval numbers are exact.
With 54 answerable questions, one question ≈ 2 points.

Lessons so far:
- Most baseline misses were year/block questions: every block has look-alike "DataLab"/deadline pages,
  and chunks didn't say which block they came from. Headers + filtering fixed most of them.
- The eval caught a bug: questions naming two blocks ("Block A and Block B") only kept the first.
- "Right page in top 5" is too generous: big block pages have 20+ chunks, and the retrieved one
  often isn't the chunk with the answer (b14, b27). Next: measure whether the answer *chunk* is retrieved.
- Baseline latency (10 s) was an eval bug (Ollama reloaded the model on every call); real answers take ~2 s.

## Phase 2 start: better metrics, a strong judge, and real student questions

Same system as `exp3b_user_year`. Two question sets:
- **page-style** (62): written from the pages; uses their vocabulary.
- **student-style** (31): written from a Year 2 Block B student's real questions ("what do i have to do for today",
  "is using llm allowed", "vpn?"), with context today = 2026-11-24, Year 2, Block B. Approved by that student in bulk.

| Set | right page in top 5 | **all answer evidence in top 5** | correct (local judge) | correct (Claude judge) |
|---|---|---|---|---|
| page-style | 93% | 61% | 44% | 63% |
| student-style | **50%** | **18%** | 14% | not graded yet |

Findings:
- **Page-level hit@5 was hiding a chunk-level problem.** With evidence phrases (short exact quotes that hold the answer),
  only 61% of page-style questions get all their answer chunks into the top 5, against 93% for "right page".
- **Real student questions are much harder.** Short, vague, date-dependent questions ("whats due this week") drop
  retrieval from 93% to 50%. The system has no idea what "today" or "this week" is.
- **A hub page.** One prompt-engineering page full of generic instructions ("your task is…") lands in the top 5 for
  20 of 31 student questions: it is vaguely similar to every short question.
- **Judges.** Claude graded all 62 page-style answers blind with [judge_rubric.md](judge_rubric.md) (v1.1).
  The local judge agrees on 76% (Cohen's kappa 0.57): reliable when it says "correct", too harsh with "partial",
  and it missed two wrong answers. Faithfulness (Claude): 84%. Citations pointing at the right chunk: 61%.
  No human checked the Claude grades.
- **The eval set had two stale/incomplete answers** (a07 EU AI Act dates, which two course pages disagree on; b10 missing
  "neutral"). Fixed.

## Query rewriting v1: rejected

An LLM (`qwen3:8b`) rewrote each question into a search query, given the student's date/year/block/week
(`student.py` finds today's week in the dated week rows on the student's block page).

| Set | hit@5 without → with rewrite | wrong answers | false "I don't know" |
|---|---|---|---|
| page-style | 93% → **61%** | 2% → 7% | 4% → 11% |
| student-style | 50% → 54% | 18% → 7% | 14% → 4% |

The 8B model copied "Block B" from its prompt examples into 34 of 62 queries (even when the student context had no
block) and dropped key terms ("Keras" disappeared from a Keras question). Guards in code fixed some cases but not this.
Kept: the "Student: today is…" line in the answer prompt (fewer wrong answers on student questions).
Rejected: letting the LLM freely edit the search query. Next design: the code adds date/block context, the LLM only
expands vague words, and both queries are searched and merged (multi-query + RRF).

## Method from here: dev / held-out test

Both question sets are split 50/50 per question type (seed 2026): page-style 32 dev / 30 test, student-style 17 / 14.
Per-question failures are only inspected on **dev**; **test** is reported as totals only.
Cleaning rules come from a corpus-only EDA ([analysis/corpus_eda.py](../analysis/corpus_eda.py)), not from eval failures.

## Cleaning and query rewriting v2: four-way dev comparison

Dev split only. Page-style: 28 answerable + 4 unanswerable (one question ≈ 3.6 points).
Student-style: 15 answerable + 2 unanswerable (one question ≈ 6.7 points). Correct/wrong are local-judge labels.

| Set | Config | hit@1 | hit@5 | MRR@10 | all evidence @5 | correct | wrong | false "I don't know" | unanswerable refused | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| page | `base` | 64% | 89% | 0.76 | 57% | 39% | 4% | 4% | 3/4 | 3.6 s |
| page | `clean` | 64% | 89% | 0.76 | 57% | 39% | 4% | 4% | 4/4 | 3.5 s |
| page | `v2` | 64% | 93% | 0.76 | 54% | 36% | 7% | 7% | 4/4 | 4.5 s |
| page | `clean_v2` | 64% | 93% | 0.76 | 54% | 39% | 7% | 7% | 4/4 | 3.9 s |
| student | `base` | 33% | 47% | 0.38 | 20% | 13% | 13% | 7% | 2/2 | 3.6 s |
| student | `clean` | 33% | 47% | 0.38 | 20% | 13% | 13% | 7% | 2/2 | 3.5 s |
| student | `v2` | 33% | 53% | 0.42 | 27% | 13% | 7% | 0% | 2/2 | 3.6 s |
| student | `clean_v2` | 33% | 53% | 0.43 | 27% | 13% | 7% | 0% | 2/2 | 4.0 s |

`clean` = `CLEAN_CORPUS=1` (454 of 506 pages; templates left out of search). `v2` = `QUERY_REWRITE=2`.

Findings:
- **Noise floor.** `v2` and `clean_v2` retrieve the same pages on page-style, yet "correct" differs by one question:
  answer generation and judging alone move a score by about one question. Every difference below is one or two questions.
- **Cleaning: neutral on retrieval.** No question's rank or evidence changed. It does tidy the context: template and
  near-empty pages held 6 of 160 top-5 slots (page) and 3 of 85 (student) before cleaning; none of those swaps decided
  whether the answer got in. One more unanswerable question was refused (within noise).
- **Rewrite v2: a trade, not a win.** It helps short, vague student questions ("vpn?", "presentation how long",
  "next week" questions): student hit@5 47% → 53%, all evidence 20% → 27%, wrong 13% → 7%, each one question.
  It slightly hurts long, precise page-style questions: the expansion adds synonyms that pull in loosely related chunks,
  and RRF pushes the exact answer chunk from rank 5 to 6 (all evidence 57% → 54%, wrong 4% → 7%). It adds an LLM call
  (+0.4 to 0.9 s median on page-style).
- **Bug found in v2:** in 1 of 49 queries the model copied the prompt's example format ("question -> query") into its
  output. Not fixed yet (v2 is off); strip that pattern when v2 is revisited.
- **Bug found in cleaning (fixed):** merging an exact duplicate kept only the surviving copy's year/block, so a page shared by
  Year 1 Block A and Block B became invisible to the Block B filter. The merged page now keeps every block its copies were in.
  No dev question searches that page with a block filter, so the numbers above are unaffected.

Defaults decided:
- `CLEAN_CORPUS=1` (and `COLLECTION` follows it: `adsai_chunks_clean`): no measurable cost, cleaner context, smaller index.
- `QUERY_REWRITE=0`: v2's gains and losses are each about one question, and it costs an extra LLM call.
  Its failure mode (answer chunk just outside the top 5) is what a reranker fixes, so v2 is re-tested together with reranking.

## Reranking: a cross-encoder re-sorts the top 30

Vector search fetches 30 chunks; a cross-encoder (fastembed, ONNX on CPU) reads each one together with the question
and the best 5 go to the LLM. Baseline is `clean` (the current default). Dev split only.

| Set | Config | hit@1 | hit@5 | MRR@10 | all evidence @5 | correct | wrong | unanswerable refused | median latency |
|---|---|---|---|---|---|---|---|---|---|
| page | `clean` | 64% | 89% | 0.76 | 57% | 39% | 4% | 4/4 | 3.5 s |
| page | `rerank_small` (MiniLM-L6, 0.08 GB) | 82% | 96% | 0.89 | 68% | 54% | 0% | 4/4 | 4.1 s |
| page | `rerank_base` (bge-reranker-base, 1.04 GB) | **96%** | **100%** | **0.97** | **82%** | **64%** | 0% | 4/4 | 9.8 s |
| page | `v2_rerank_base` (+ rewrite v2) | 96% | 96% | 0.96 | 82% | 61% | 0% | 3/4 | 10.4 s |
| student | `clean` | 33% | 47% | 0.38 | 20% | 13% | 13% | 2/2 | 3.5 s |
| student | `rerank_small` | 33% | 47% | 0.39 | 27% | 20% | 0% | 2/2 | 4.0 s |
| student | `rerank_base` | 27% | 53% | 0.37 | 33% | 20% | 7% | 2/2 | 10.2 s |
| student | `v2_rerank_base` (+ rewrite v2) | **40%** | **73%** | **0.55** | 33% | 20% | 0% | 2/2 | 10.6 s |

Findings:
- **First clear win of Phase 2.** On page-style, `rerank_base` puts the right page first for 9 more questions and loses none;
  all answer evidence reaches the LLM for 7 more, losing none. Far beyond the one-question noise floor.
- **Size matters.** The small model gets about half of that gain at +0.6 s; the base model all of it at +6.5 s per question
  (CPU, 30 chunks of ~400 tokens).
- **Rewrite v2 pays off only with a reranker.** Its extra queries add candidates, and the reranker sorts them. On student
  questions hit@5 goes 47% → 73% (+4 questions), mostly "this week / next week" questions found through the student's
  week that the code adds to the query. Without a reranker those candidates landed just outside the top 5.
  On page-style it costs one question of hit@5 and one unanswerable refusal.
- **Latency is the price:** about 10 s per answer with the base model, against 3.5 s now.
- Correct/wrong are local-judge labels; retrieval columns are exact. (All runs were re-graded after two judge fixes,
  see "Eval fixes" below.)

## Known gap: questions about dates ("tomorrow", "next")

Tried by hand with the default system (rerank + v2), as a Year 2 Block B student on a Wednesday:
- **"do i have any lectures tomorrow"** → "no", for the wrong reason: it missed the block schedule and cited an *undated*
  lecture from another course page, giving it today's date. A made-up fact with a citation.
- **"when is the next lecture we have"** → it found the block schedule, then picked an entry from two days earlier.

Two separate failures:
1. **Retrieval:** the schedule is one long page split into ~13 chunks; the right day's entry competes with look-alike pages.
2. **Date reasoning:** an 8B model compares dates unreliably, even with today's date in the prompt.

Prompting won't fix date arithmetic, so the plan is code: a schedule tool (Phase 3, agent layer) that reads the dated entries
of the student's block page and keeps those from today onwards; the LLM only phrases the answer.
Both questions are now in the student-style set (dev split, since they were seen failing first), so the fix can be measured.
From here on student-style dev has 19 questions instead of 17: earlier student-dev rows are not directly comparable with new runs.

## Speed-up round 1: fewer candidates, shorter reranker input

Target: at most 5 s per answer. The default (strong reranker over 30 full chunks + rewrite v2) takes ~10 s, of which
~7 s is reranking on the CPU; rewrite + search + answer take ~3 s, so the reranker's budget is ~2 s.
`RERANK_MAX_WORDS` = words of each chunk the reranker reads (chunks are ~250 words; the LLM always gets the full chunk).

Page-style dev (28 answerable):

| Config | candidates | words read | hit@1 | hit@5 | all evidence @5 | rerank step | median latency |
|---|---|---|---|---|---|---|---|
| default | 30 | all | **96%** | 96% | **82%** | 7.0 s | 10.4 s |
| `v2_cand15` | 15 | all | 96% | 96% | 75% | | 6.2 s |
| `mw150_c30` | 30 | 150 | 86% | 96% | 68% | 4.0 s | 6.7 s |
| `mw100_c30` | 30 | 100 | 86% | 93% | 46% | 2.7 s | 5.3 s |
| `mw150_c15` | 15 | 150 | 86% | 96% | 61% | 1.7 s | **4.8 s** |
| `mw100_c15` | 15 | 100 | 82% | 96% | 54% | 1.2 s | **4.1 s** |
| `week_rerank` (rewrite mode 3, no LLM expansion) | 30 | all | 96% | 100% | 82% | | 10.5 s |

Student-style dev, now 19 questions (17 answerable; two date questions added, so not comparable with earlier student rows):

| Config | candidates | words read | hit@1 | hit@5 | all evidence @5 | median latency |
|---|---|---|---|---|---|---|
| default | 30 | all | 41% | **71%** | 29% | 8.9 s |
| `mw150_c30` | 30 | 150 | 41% | 71% | 29% | 7.3 s |
| `mw100_c30` | 30 | 100 | 53% | 71% | 29% | 5.6 s |
| `mw150_c15` | 15 | 150 | 24% | 53% | 29% | 4.6 s |
| `mw100_c15` | 15 | 100 | 41% | 59% | 29% | 4.3 s |

Findings:
- **The reranker needs the whole chunk.** Reading only the first 100-150 words, it misses answers that sit further down:
  page-style all evidence drops from 82% to 46-68%. The configs under 5 s give back 4 to 8 page questions of the
  reranking gain (and 2-3 student questions).
- **Fewer candidates costs less than shorter input**: 15 full chunks lose 2 page questions of evidence at 6.2 s.
- **Rewrite mode 3 (week phrase only, no LLM)** saves no time (the expansion call is ~0.3 s) and, on the earlier 17-question
  student set, dropped hit@5 from 73% to 53%: rejected. The LLM expansion carries v2's gain, not the week phrase alone.
- **Conclusion:** no free setting reaches 5 s without losing much of the gain. Defaults stay as they are; next is making
  the full-length reranker itself faster (int8 quantization) or moving its work to indexing time (late interaction / ColBERT).

## Speed-up round 2: int8 cross-encoder and ColBERT

Two ways to keep the reranker reading the whole chunk and still be fast:
- **int8**: the same cross-encoder with its weights stored as 8-bit integers (dynamic quantization, 1.1 GB -> 0.28 GB).
  On this CPU (AVX-512 VNNI) it reranks 30 chunks in 2.9 s instead of 5.8 s, with the same top-1 on 9 of 10 checks.
- **ColBERT** (`answerai-colbert-small-v1`, late interaction): every chunk's word-piece vectors are computed once at
  indexing time (5 min, own Qdrant collection); at query time only the question is encoded and Qdrant scores the
  candidates with MaxSim. 0.25 s for 30 or 100 candidates.

| Set | Config | hit@1 | hit@5 | all evidence @5 | rerank step | median latency |
|---|---|---|---|---|---|---|
| page | default (cross-encoder, 30) | 96% | 96% | 82% | 7.0 s | 10.4 s |
| page | `int8_c30` | 89% | 96% | 82% | 2.9 s | 6.1 s |
| page | `int8_c15` | **93%** | 96% | **75%** | 1.3 s | **4.1 s** |
| page | `colbert_c30` | 86% | 96% | 71% | 0.2 s | **3.9 s** |
| page | `colbert_c100` | 86% | 100% | 71% | 0.2 s | 4.4 s |
| student | default | 41% | 71% | 29% | 5.7 s | 8.9 s |
| student | `int8_c30` | 47% | 65% | 18% | 2.7 s | 6.0 s |
| student | `int8_c15` | 41% | 59% | 29% | 1.2 s | **3.7 s** |
| student | `colbert_c30` | 29% | **71%** | 24% | 0.2 s | 4.3 s |
| student | `colbert_c100` | 35% | 65% | 24% | 0.2 s | 3.9 s |

Findings:
- Both get under 5 s while keeping most of the reranking gain (far better than shortening the input).
- int8 is slightly better on page-style, ColBERT slightly better on student hit@5: 1-2 questions either way, within noise.
- More ColBERT candidates (100 vs 30) do not help: what vector search misses in its top 30 it also misses lower down.
  That points at recall, which is what keyword search (hybrid, next) adds.

## Eval fixes: judge label parsing, half answers, list depth

A code review of the eval found three bugs; each now has a test.
- **Judge labels** were matched as substrings in the order CORRECT, PARTIAL, WRONG, so a reply "INCORRECT" or
  "Partially correct" counted as CORRECT. Labels are now read as whole words.
- **Half answers.** Every answer containing "I don't know" was graded WRONG without asking the judge, including answers
  that give part of the answer and say "I don't know" about the rest. Now only a bare "I don't know" skips the judge.
  "I don’t know" with a curly apostrophe is recognised too.
- **List depth.** RRF lets a chunk ranked 12th in two lists beat one ranked 1st in a single list, so how deep the merged
  lists are changes the top of the result. The app (5 chunks) merged shallower lists than the eval (10 chunks), so the
  eval's top 5 was not always what the app gave the LLM (rewrite v2 or hybrid search without a reranker). Every merged
  list is now 20 deep. Runs with a reranker already used deeper lists and are unaffected.

All 52 saved runs were re-graded from their saved answers (`evals/rejudge.py`; no new search or generation). Retrieval
numbers do not change. Out of ~1,100 graded answers 38 labels moved: 26 half answers from wrong to partial (the fix),
and 12 flips between correct and partial in both directions. No answer went from correct to wrong, so the "INCORRECT"
misreading never actually happened. The 12 flips show the local judge is **not fully repeatable** even at temperature 0:
about 1 in 100 labels changes when the same answer is graded again. The tables above use the re-graded numbers; only the
"wrong" column moved (by one or two questions), "correct" is unchanged in every row. Local vs Claude judge: 76% same
label, kappa 0.57 (was 0.58).

## Hybrid search: BM25 keyword search + vector search

Vector search matches meaning and blurs exact rare words ("keras", "task 8", "Block B"). BM25 scores chunks that contain
the question's words, weighting rare words most. Both lists are merged with RRF (`HYBRID=1`, own Qdrant collection with
sparse vectors, built in 2 s). In hindsight this should have been the baseline *before* reranking, so the first rows
measure it alone. Dev split; `v2` = query rewriting v2; rerankers get 30 candidates (int8: 15).

Page-style dev, 28 answerable + 4 unanswerable (one question ≈ 3.6 points):

| Reranker | Search | hit@1 | hit@5 | hit@10 | MRR@10 | all evidence @5 | @10 | correct | wrong | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| none | vector | 64% | 89% | 93% | 0.76 | 57% | 68% | 39% | 4% | 3.5 s |
| none | vector + v2 | 64% | 93% | 93% | 0.76 | 54% | 68% | 39% | 7% | 3.9 s |
| none | **hybrid** | **75%** | 93% | 96% | **0.84** | **71%** | 75% | 43% | 0% | 3.8 s |
| none | hybrid + v2 | 71% | 96% | 96% | 0.83 | 68% | 79% | 43% | 0% | 3.8 s |
| cross-encoder (+ v2) | vector | 96% | 96% | 96% | 0.96 | 82% | 82% | 61% | 0% | 10.4 s |
| cross-encoder (+ v2) | hybrid | **96%** | **100%** | **100%** | **0.97** | **86%** | **86%** | **64%** | 0% | 9.5 s |
| int8, 15 (+ v2) | vector | 93% | 96% | 96% | 0.95 | 75% | 75% | 54% | 0% | 4.1 s |
| int8, 15 (+ v2) | hybrid | 93% | 100% | 100% | 0.96 | 71% | 82% | 61% | 0% | 5.0 s |
| ColBERT (+ v2) | vector | 86% | 96% | 96% | 0.90 | 71% | 75% | 50% | 0% | 3.9 s |
| ColBERT (+ v2) | hybrid | 86% | 100% | 100% | 0.92 | 71% | 79% | 54% | 0% | **3.7 s** |

Student-style dev, on the 17 questions every run shares (15 answerable + 2 unanswerable; the two date questions were
added after the vector-only runs; one question ≈ 6.7 points):

| Reranker | Search | hit@1 | hit@5 | hit@10 | MRR@10 | all evidence @5 | @10 | correct | wrong | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| none | vector | 33% | 47% | 47% | 0.38 | 20% | 27% | 13% | 13% | 3.4 s |
| none | vector + v2 | 33% | 53% | 67% | 0.43 | 27% | 33% | 13% | 7% | 4.0 s |
| none | hybrid | 40% | 47% | 67% | 0.46 | 27% | 40% | 20% | 7% | 3.7 s |
| none | hybrid + v2 | 40% | 60% | 73% | 0.52 | 27% | 33% | 20% | 7% | 3.9 s |
| cross-encoder (+ v2) | vector | 40% | 73% | 80% | 0.55 | 33% | 40% | 13% | 0% | 9.0 s |
| cross-encoder (+ v2) | hybrid | **53%** | **80%** | **87%** | **0.63** | **40%** | **53%** | 27% | 7% | 9.5 s |
| int8, 15 (+ v2) | vector | 47% | 67% | 67% | 0.53 | 33% | 40% | 13% | 7% | 3.7 s |
| int8, 15 (+ v2) | hybrid | 47% | 67% | 80% | 0.59 | 27% | 40% | 20% | 7% | 5.1 s |
| ColBERT (+ v2) | vector | 33% | 73% | 80% | 0.48 | 27% | 40% | 20% | 0% | 4.0 s |
| ColBERT (+ v2) | hybrid | 47% | 67% | 80% | 0.56 | 27% | 33% | 20% | 7% | **3.6 s** |

Findings:
- **Hybrid alone is the cheapest step so far.** Without any reranker, page-style hit@1 goes 64% → 75% (+3 questions) and
  all evidence in the top 5 57% → 71% (+4), for +0.3 s. That is about a third of the cross-encoder's hit@1 gain at a
  twentieth of its cost.
- **On student questions hybrid mainly adds recall**: right page in the top 10 47% → 67% (+3), but the top 5 only
  improves together with rewrite v2 (47% → 60%). Short questions have few keywords for BM25 to match.
- **With a reranker on top, hybrid finds what vector search never does.** Page hit@10 reaches 100% under all three
  rerankers (the same question missed it under every 30-candidate vector config; only 100 ColBERT candidates had found it). With the cross-encoder, student hit@5
  goes 73% → 80% and all evidence @10 40% → 53%: the best retrieval measured so far, but at 9.5 s.
- **Under 5 s, the configs differ by 1-2 questions** (noise). Hybrid + ColBERT is the fastest (3.6-3.7 s) and its
  reranking work is done at indexing time, so it does not slow down as the candidate list grows.
- **The price of the 5 s target**, against hybrid + cross-encoder: page hit@1 96% → 86% (-3 questions), all evidence @5
  86% → 71% (-4), student hit@5 80% → 67% (-2). About 6 s per answer saved.

Defaults decided: `HYBRID=1`, `RERANK_KIND=colbert` (30 candidates), rewrite v2 kept. The cross-encoder stays one switch
away (`RERANK_KIND=cross`) for when quality matters more than speed.

## Held-out test split (end of Phase 2)

Run once with the final default (hybrid search + ColBERT, rewrite v2), totals only. "Start" is the Phase 2 starting
configuration on the same test questions, computed from the early run that covered all questions (on the dev split
that run and `clean` give identical retrieval: 64% / 89% page-style, 33% / 47% student-style).

| Set | Config | hit@1 | hit@5 | all evidence @5 | correct | wrong | unanswerable refused | median latency |
|---|---|---|---|---|---|---|---|---|
| page (26 + 4) | start | 77% | 96% | 65% | 50% | 0% | 3/4 | 4.0 s |
| page (26 + 4) | default | 77% | 96% | 73% | 54% | 4% | 3/4 | 4.1 s |
| student (13 + 1) | start | 23% | 54% | 15% | 8% | 15% | 1/1 | 2.9 s |
| student (13 + 1) | default | 46% | 69% | 31% | 15% | 0% | 1/1 | 4.8 s |

- **Page-style: no page-level gain on test.** The starting point was already at 77% / 96%, against 64% / 89% on dev.
  The gain that remains is chunk-level: all answer evidence reaches the LLM for 2 more questions.
- **Student-style: better on every retrieval measure**, by 2-3 questions out of 13.
- **Dev overstated the page-style gain.** The default was chosen among many configurations on 28 dev questions, so part
  of its dev advantage is selection on that split. This is what the held-out split is for.
- **Disclosure:** some test questions were read in the Phase 2 start run, before the split was made. No change was
  made from per-question test results.

## Phase 3, step 1: results per question type and segment

Every question now carries two labels:
- **question type**, labelled by an LLM from the question text only (no results looked at): schedule, deadline,
  overview, rules, how-to, concept, people, out of scope. A question that asks two things gets the riskier label
  (deadline > rules > the rest). Every unanswerable question is "out of scope".
  **Not reviewed by a person.** Checking every label by hand does not scale, so the check is automatic: the question
  router (step 9) labels every question independently, and only the questions where the two labels disagree are
  reviewed. That also gives an estimate of the label error rate.
- **segment**, computed from the year/block metadata of the answer's source pages: a block (e.g. "Year2 BlockB"),
  "shared" (pages of no year), "several", or "none" (unanswerable).

Development runs now report every group separately; test runs stay totals only (enforced in `run_eval.py`).

**Coverage** (dev/test question counts; `-` = no question):

| type | Year1 BlockA | Year1 BlockC | Year2 BlockA | Year2 BlockB | Year2 BlockD | Year3 Semester1 | shared | several | none |
|---|---|---|---|---|---|---|---|---|---|
| schedule | - | - | - | 4/1 | - | - | - | - | - |
| deadline | 1/0 | - | 0/1 | 1/3 | - | 0/1 | 0/1 | 0/1 | - |
| overview | - | 1/0 | - | 1/1 | - | 1/1 | 1/0 | 3/2 | - |
| rules | - | - | - | 5/3 | 1/0 | - | 6/4 | 1/1 | - |
| howto | - | - | - | 1/0 | - | 0/1 | 8/5 | 1/1 | - |
| concept | - | - | 0/1 | - | - | - | 9/11 | - | - |
| people | - | - | - | - | - | - | - | - | - |
| out of scope | - | - | - | - | - | - | - | - | 6/5 |

Columns with no question at all are left out: Year 1 Block B and D, Year 2 Block C, Year 3 Semester 2, and the
year-level pages. So 8 of the 14 parts of the site have no development question, all schedule questions are
about one block, there are 2 development deadline questions, and none asks for a person or contact. This is why the
week lookup looked fine while working for 1 block out of 10.

**Baseline per group**: the current default (hybrid + ColBERT), from the saved development runs, no new run needed:

| Question type | Page-style: n, right page in top 5, correct | Student-style: n, right page in top 5, correct |
|---|---|---|
| schedule | - | **4, 0%, 0%** |
| deadline | 1, 100%, 100% | 1, 100%, 0% |
| overview | 5, 100%, 40% | 2, 100%, 0% |
| rules | 8, 100%, 75% | **5, 40%, 20%** |
| howto | 6, 100%, 50% | 4, 100%, 25% |
| concept | 8, 100%, 38% | 1, 100%, 100% |
| out of scope (refused) | 3 of 4 | 2 of 2 |

| Segment | n | right page in top 5 | all evidence @5 | correct |
|---|---|---|---|---|
| shared (page / student) | 21 / 3 | 100% / 100% | 62% / 67% | 52% / 67% |
| Year 2 Block B (student) | 12 | **42%** | **17%** | **8%** |
| several | 3 / 2 | 100% / 100% | 100% / 0% | 67% / 0% |
| other blocks | 1 each | | | |

- **Schedule questions fail completely** (4 of 4): no right page in the top 5, no correct answer. This is the case for
  the calendar route.
- **Rules questions asked the student way fail too**: 2 of 5 find the right page, against 8 of 8 when phrased like the
  page. Short phrasing ("is using llm allowed") is the problem, not the topic.
- **Retrieval is solved for shared pages** (100% right page in the top 5); what is left there is the chunk and the
  answer (evidence 62%, correct about half, local judge).
- Most cells have 1-5 questions, so these numbers show where to look, not how big a difference is.

## Phase 3, step 4: search filter and student week from the calendar

Step 2 added the official 2026-27 year calendar and a resolver. Step 4 uses them in the pipeline:
- **Search filter**: "next / last block" is resolved from the student's block and the calendar (Year 1 Block D →
  Year 2 Block A), and block names map to the folders each year uses (Year 3 has semesters). Before, a "next block"
  question searched the student's own block.
- **Student context**: the week number and its dates come from the calendar for every year and block; holidays and
  closed days are named. Before, the week was read from one block page's timeline format (1 of 10 blocks).
- The query rewrite adds no current-week phrase to "next block" questions.

**Regression check** (dev, default config, against the same config before the change):

| Set | Same search queries | Same retrieved pages | hit@1 | hit@5 | all evidence @5 | correct |
|---|---|---|---|---|---|---|
| page (32) | 32 / 32 | 32 / 32 | 86% → 86% | 100% → 100% | 71% → 71% | 54% → 54% |
| student (19) | 19 / 19 | 19 / 19 | 41% → 41% | 59% → 59% | 24% → 24% | 18% → 18% |

- **Nothing moved, as expected**: no dev question asks about another block, and for the eval's Year 2 Block B dates the
  calendar gives the same week the page did. The check confirms the new wiring changes nothing it should not.
- **Median time rose 0.5-0.8 s, but not because of this change**: the new code takes under 2 ms per question, and the
  answer-generation step, whose prompts were identical, got 0.3 s slower as well. Run-to-run machine variation.
- **The gain is not measurable on this eval set.** "What do we have next block" and other blocks' weeks are covered by
  unit tests (including directional ones: changing the block must change the filter). Measuring it end to end needs
  the behavioural suite of step 3.

## Phase 3, step 3: behavioural suite, before and after step 4

The normal eval asks its student questions as Year 2 Block B on two November dates, so it could not show the step 4
gain. `evals/behavioral.py` asks template questions as a student of each of the 10 segments (Year 1-2 Blocks A-D,
Year 3 Semesters 1-2), on Tuesday of week 1, Tuesday of week 3, Thursday of the last teaching week and the Wednesday
of every break, in the CheckList style (Ribeiro et al., ACL 2020):
- **MFT** (minimum functionality): "which week of the block are we in?" (90), "what are the dates of week N?" (20),
  "when does the next block start?" (7, only where that block is in this year's calendar). Exact match: the first
  week number, or the first dates, in the answer; in a break the answer must say there is no teaching.
- **DIR** (directional): "what do we have next block?" / "what did we do last block?" (18): the top search result
  must be a page of that other block.
- **INV** (invariance): the two wordings of the week question (45 pairs) must give the same week.

Gold answers are computed from `calendar.yaml` by the suite's own code, not the app's resolver. Same default config
for both runs (hybrid + ColBERT, 30 candidates, rewrite v2); "before" is `main` with step 2 only.

| Family | Before step 4 | After step 4 |
|---|---|---|
| Which week: teaching days | 13 / 60 | **60 / 60** |
| Which week: breaks | 0 / 30 | 12 / 30 |
| Dates of week N | 1 / 20 | 6 / 20 |
| Next block starts | 0 / 7 | 1 / 7 |
| DIR next block (top result) | 0 / 9 | 2 / 9 |
| DIR last block (top result) | 0 / 9 | 0 / 9 |
| INV: both wordings agree | 14 / 45 | 35 / 45 |
| **All cases** | **14 / 135 (10%)** | **81 / 135 (60%)** |

Per segment (all families): Year 1 A-D 8 / 7 / 8 / 8% → 75 / 40 / 54 / 50%; Year 2 A-D 15 / **33** / 8 / 8% →
54 / 53 / 54 / 58%; Year 3 Semester 1-2 6 / 0% → 77 / 85%. Before, Year 2 Block B was the only segment above 15%:
it was the one block page whose timeline the old week lookup could read.

What the failures say (read on the answers, all segments):
- **The week number is solved by the student line**: with the calendar's week in the prompt, the model repeats it
  every time. Without it, it guessed (and gave different weeks for the two wordings).
- **Breaks**: the prompt says "This week is the Christmas holiday", yet in 18 of 30 cases the model still names a week
  number. An 8B model does not reliably follow a negative fact; a calendar route should answer this in code.
- **Dates of other weeks**: only the current week's dates are in the prompt. The model either says it does not know
  (the last week of every block) or subtracts days itself and is off by one (8-12 September for 7-11 September).
  Exactly the date arithmetic the design moves into code.
- **Next / last block**: the filter now lets the right block's pages through (some target-block page is in the top 5
  for 5 of 18, was 0), but it also keeps the shared Study Content and General pages, and for a vague question like
  "what do we have next block?" those rank first. The fix is general, not per question: an overview question about a
  resolved block should search that block's own pages (and its year/block words), not the shared ones.

## Phase 3: a larger answering model (BUas LLM service) on the behavioural suite

BUas offers students an LLM service with an OpenAI-compatible API (on its own servers). `LLM_BACKEND=buas` sends the
answer and query-rewrite calls there; embeddings, reranking and the search index stay local. Same suite, same day,
step 4 code; the only change is the answering model: local `qwen3:8b` (4-bit) against `qwen3.8-27b` on the service
(the only chat model the account offered that day).

Scored as before (exact match) and, because a wrong date costs a student more than "I don't know", also split into
right / wrong / abstained for the 117 cases with an answer to read (the CRAG score: +1 right, -1 wrong, 0 abstained):

| | Exact match (135) | Right | Wrong | Abstained | CRAG score |
|---|---|---|---|---|---|
| before step 4, `qwen3:8b` | 14 (10%) | 14 | 83 | 20 | -0.59 |
| step 4, `qwen3:8b` | 81 (60%) | 79 | 28 | 10 | +0.44 |
| step 4, `qwen3.8-27b` | 71 (53%) | 68 | **5** | 44 | **+0.54** |

| Family (right / wrong / abstained) | `qwen3:8b` | `qwen3.8-27b` |
|---|---|---|
| Which week, teaching days (60) | 60 / 0 / 0 | 60 / 0 / 0 |
| Which week, breaks (30) | 12 / 17 / 1 | 7 / 5 / 18 |
| Dates of week N (20) | 6 / 6 / 8 | 1 / 0 / 19 |
| Next block starts (7) | 1 / 5 / 1 | 0 / 0 / 7 |
| DIR next + last block, top result (18) | 2 | 3 |

- **The larger model is more honest, not more knowing.** It invents almost no week numbers or dates (5 wrong against
  28), but the facts it would need are still not in its prompt, so it abstains instead. On the questions where the
  answer IS in the prompt (the current week), both are perfect.
- **It ignores the holiday line.** The student line says "This week is the Christmas holiday", yet the 27B model
  answers "the context does not contain a calendar". The system prompt tells the model to use ONLY the numbered
  passages, and the larger model follows that rule more strictly; the student line is not a numbered passage. A
  general prompt fix (the student line counts as a source) is its own experiment, measured on both models.
- **Search changed too**: query rewriting also uses the answering LLM, so only 15 of 135 cases retrieved the same top 5
  pages. "Next / last block" is still a search problem (2 → 3 of 18), not a model problem.
- Model choice: not changed yet. The 27B model is the safer one (fewest wrong answers); the 8B scores higher on exact
  match because it guesses, and sometimes guesses right.
