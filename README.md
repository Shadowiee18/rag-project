# RAG Project

An evaluated, production-style RAG system, built one layer at a time.
The goal is a portfolio project you can explain line by line in an interview.

## Rules

1. **Write the first version yourself.** Use AI to review, explain, or give hints, not to write the code.
2. **Measure before you optimise.** Every retrieval change gets a before/after number in the results table.
3. **Commit after every exercise or feature.** Small commits with clear messages.
4. **LeetCode, 20–30 min a day:** easy/medium problems on arrays, hashmaps, two pointers, BFS/DFS and heaps.

## Phases

| Phase | Weeks | What you build | Done when |
|---|---|---|---|
| 0 Refresher | 1–2 | Exercises in `phase0/` | `uv run pytest` is all green and the Docker task works |
| 1 Baseline RAG | 3–4 | Load → chunk → embed → vector DB → answer with citations | 50+ questions in `evals/golden_set.jsonl` and a baseline score |
| 2 Advanced retrieval | 5–7 | Heading-aware chunking, BM25 + vector hybrid (RRF), reranker, query rewriting | Results table with a measured gain for each change |
| 3 Agent layer | 8–9 | Router (search or not), multi-step retrieval, "I don't know" | Eval includes unanswerable questions and they pass |
| 4 LLMOps | 10–12 | Tracing, evals in CI, cost/latency tracking, guardrails, Docker deploy; serving under load (below) | A PR that lowers quality fails CI; load-test table Ollama vs vLLM |

### Phase 4: serving many users

Almost all answer time is the LLM writing (~2 s), not search (~0.3 s), so with many students the
problem is throughput: one GPU queues requests. Plan, all free on the local GPU:

1. **Load test first** with Locust: 1, 10 and 50 simultaneous users. Record time to first token,
   median and p95 total latency, and answers per second.
2. **Streaming** answers from the API (same total time, but the answer starts almost at once).
3. **Semantic cache**: reuse answers for near-identical questions (question embeddings in Qdrant), with expiry.
4. **vLLM vs Ollama**: serve a 4-bit ~8B model with vLLM (Docker + WSL2 GPU, OpenAI-compatible API,
   continuous batching) behind a config switch in `generate.py`. Same load test, plus the 62-question
   eval to check the compressed model's quality. Result: one table of speed and quality for both.
| Ship | — | README diagram, results table, demo video, write-up | Someone else can run it from the README |

## Target layout (fill in as you go)

```
src/ragapp/
  ingest.py      # Phase 1: read files from data/raw into Document objects
  chunking.py    # Phase 1: reuse what you learned in phase0 ex02
  embed.py       # Phase 1: turn chunks into vectors
  store.py       # Phase 1: vector DB (pgvector or Qdrant)
  retrieve.py    # Phase 2: hybrid search + reranking
  generate.py    # Phase 1: prompt the LLM with retrieved chunks, return answer + citations
  api.py         # Phase 1: FastAPI app (you practise this in phase0 ex06)
evals/
  golden_set.jsonl   # your questions + expected answers (start from the .example file)
  run_eval.py        # Phase 1: scores the system against the golden set
data/raw/            # your documents (ignored by git)
```

## Commands

```
uv sync                 # install dependencies
uv run pytest           # run all tests
uv run pytest -k ex01   # run one exercise's tests
uv add <package>        # add a dependency when a phase needs it
```

Start here: [phase0/README.md](phase0/README.md)

## Phase 1: baseline RAG

Everything runs locally: answers never leave your PC.

| Part | What | Where |
|---|---|---|
| LLM | `qwen3:8b` | Ollama |
| Embeddings | `qwen3-embedding:0.6b` (1024 dims) | Ollama |
| Vector DB | Qdrant `v1.20.2` | Docker Compose |
| Corpus | 506 course pages, `data/raw/adsai_pages.jsonl` | not in git |

```
docker compose up -d                                   # start Qdrant (dashboard: http://localhost:6333/dashboard)
uv run python -m ragapp.ingest                         # indexing time: chunk + embed + store every page
uv run python -m ragapp.rag "How do I get VPN access?" # ask one question from the terminal
uv run uvicorn ragapp.api:app --reload                 # or as an API: http://127.0.0.1:8000/docs
uv run python evals/run_eval.py --name baseline        # score it against the golden set
```

Pipeline: `ingest.py` (chunking → embed → store) at indexing time; `rag.py` (retrieve → generate) at query time.
All settings live in `src/ragapp/config.py`.

The golden set and per-question results contain course material, so they stay out of git.
Only the `*_summary.json` numbers are committed.

## Results

Golden set: 62 questions (54 answerable, 8 unanswerable). Local `qwen3:8b` + `qwen3-embedding:0.6b`, top 5 chunks.

| Run | What changed | hit@1 | hit@5 | MRR@10 | all pages in context | correct* | wrong* | false "I don't know" | unanswerable refused |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | naive 1500-char chunks | 59% | 87% | 0.71 | 80% | 33% | 9% | 9% | 7/8 |
| `exp1_headers` | + "Year 2 · Block B · title" header on every chunk | 63% | 87% | 0.73 | 80% | 39% | 7% | 7% | 7/8 |
| `exp2b_filter` | + year/block filter read from the question | 67% | 93% | 0.77 | 87% | 41% | 6% | 4% | 6/8 |
| `exp3b_user_year` | + student's own year when the question names none (simulated) | **70%** | **93%** | **0.80** | **87%** | **44%** | 6% | 4% | 7/8 |

\* judged by the same local model against the expected answer: treat as rough. Retrieval numbers are exact.
With 54 answerable questions, one question ≈ 2 points.

Lessons so far:
- Most baseline misses were year/block questions: every block has look-alike "DataLab"/deadline pages,
  and chunks didn't say which block they came from. Headers + filtering fixed most of them.
- The eval caught a bug: questions naming two blocks ("Block A and Block B") only kept the first.
- "Right page in top 5" is too generous: big block pages have 20+ chunks, and the retrieved one
  often isn't the chunk with the answer (b14, b27). Next: measure whether the answer *chunk* is retrieved.
- Baseline latency (10 s) was an eval bug (Ollama reloaded the model on every call); real answers take ~2 s.

### Phase 2 start: better metrics, a strong judge, and real student questions

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
- **Judges.** Claude graded all 62 page-style answers blind with [judge_rubric.md](evals/judge_rubric.md) (v1.1).
  The local judge agrees on 76% (Cohen's kappa 0.58): reliable when it says "correct", too harsh with "partial",
  and it missed two wrong answers. Faithfulness (Claude): 84%. Citations pointing at the right chunk: 61%.
  No human checked the Claude grades.
- **The eval set had two stale/incomplete answers** (a07 EU AI Act dates, which two course pages disagree on; b10 missing
  "neutral"). Fixed.

### Query rewriting v1: rejected

An LLM (`qwen3:8b`) rewrote each question into a search query, given the student's date/year/block/week
(`student.py` reads the block timeline: 24 Nov 2026 = Year 2 Block B week 3).

| Set | hit@5 without → with rewrite | wrong answers | false "I don't know" |
|---|---|---|---|
| page-style | 93% → **61%** | 6% → 13% | 4% → 11% |
| student-style | 50% → 54% | 18% → 7% | 14% → 4% |

The 8B model copied "Block B" from its prompt examples into 34 of 62 queries (even when the student context had no
block) and dropped key terms ("Keras" disappeared from a Keras question). Guards in code fixed some cases but not this.
Kept: the "Student: today is…" line in the answer prompt (fewer wrong answers on student questions).
Rejected: letting the LLM freely edit the search query. Next design: the code adds date/block context, the LLM only
expands vague words, and both queries are searched and merged (multi-query + RRF).

### Method from here: dev / held-out test

Both question sets are split 50/50 per question type (seed 2026): page-style 32 dev / 30 test, student-style 17 / 14.
Per-question failures are only inspected on **dev**; **test** is reported as totals only.
Cleaning rules come from a corpus-only EDA ([analysis/corpus_eda.py](analysis/corpus_eda.py)), not from eval failures.

### Cleaning and query rewriting v2: four-way dev comparison

Dev split only. Page-style: 28 answerable + 4 unanswerable (one question ≈ 3.6 points).
Student-style: 15 answerable + 2 unanswerable (one question ≈ 6.7 points). Correct/wrong are local-judge labels.

| Set | Config | hit@1 | hit@5 | MRR@10 | all evidence @5 | correct | wrong | false "I don't know" | unanswerable refused | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| page | `base` | 64% | 89% | 0.76 | 57% | 39% | 7% | 4% | 3/4 | 3.6 s |
| page | `clean` | 64% | 89% | 0.76 | 57% | 39% | 7% | 4% | 4/4 | 3.5 s |
| page | `v2` | 64% | 93% | 0.76 | 54% | 36% | 11% | 7% | 4/4 | 4.5 s |
| page | `clean_v2` | 64% | 93% | 0.76 | 54% | 39% | 11% | 7% | 4/4 | 3.9 s |
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
  and RRF pushes the exact answer chunk from rank 5 to 6 (all evidence 57% → 54%, wrong 7% → 11%). It adds an LLM call
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
