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
| 2 Advanced retrieval | 5–7 | BM25 + vector hybrid (RRF), reranker, query rewriting | Results table with a measured gain for each change |
| 3 Agent layer | 8–9 | Router (search or not), multi-step retrieval, "I don't know" | Eval includes unanswerable questions and they pass |
| 4 LLMOps | 10–12 | Tracing, evals in CI, cost/latency tracking, guardrails, Docker deploy | A PR that lowers quality fails CI |
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
