"""Tune SCOPE_MODE=boost (retrieval only, no answers): does a bonus for the question's scope beat the filter?

Step 1, --dump (uses Ollama for embeddings, the LLM for query expansion, Qdrant and ColBERT): for every case, the
reranked candidates (~30) of both modes, saved once: "filter" (today's search) and "boost" (today's search + one over
the scope's own pages, before any bonus).
Step 2, --tune (no models): re-sort the boost candidates with every weight pair of the grid, score every set, and
choose the weights by the rule below. Seconds, so the grid costs nothing.

Case sets (--split dev to tune; --split test once, with the weights fixed from dev, totals only):
  page      dev page-style questions, no asker context: the scope comes only from the question itself.
            (Not the simulated asker year: it is the year of the answer page, so a year bonus would win by
            construction.)
  student   dev student questions with their own context (Year 2 Block B, November).
  dir       the behavioural suite's "what do we have next block?" / "what did we do last block?" (18):
            the top result must be a page of that block. (Templates: the same cases in both splits.)
  counter   counter-examples: every dev question whose answer is on a SHARED page, asked with a block in front
            ("Year 1 Block C: <question>"), rotating over the 10 segments. The shared page must stay in the top 5.
            A scope rule that hides or buries shared pages fails here.

Rule, fixed before the run: among the weight pairs whose hit@5 on page, student and counter is at least the filter
mode's, take the one with the most hits (hit@1 + hit@5 summed over all four sets); ties -> the smaller weights.
If no pair passes, boost is not adopted.

Run:  uv run python evals/scope_eval.py --dump scope2     (asks nothing: ask the student before running it)
      uv run python evals/scope_eval.py --tune scope2
      uv run python evals/scope_eval.py --dump scope2_test --split test
      uv run python evals/scope_eval.py --tune scope2_test --weights 0.05,0   (the dev choice; no grid)
Writes evals/results/<name>_scope_candidates.jsonl (git-ignored) and <name>_scope_summary.json.
"""

import argparse
import itertools
import json
from datetime import date
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
SETS = ("page", "student", "dir", "counter")
W_BLOCK = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0)
W_YEAR = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
COUNTER_SCOPES = [("Year1", "BlockA"), ("Year1", "BlockB"), ("Year1", "BlockC"), ("Year1", "BlockD"),
                  ("Year2", "BlockA"), ("Year2", "BlockB"), ("Year2", "BlockC"), ("Year2", "BlockD"),
                  ("Year3", "Semester1"), ("Year3", "Semester2")]


# ---------------------------------------------------------------- cases (no models)


def readable(year: str, block: str) -> str:
    """('Year1', 'BlockC') -> 'Year 1 Block C'; ('Year3', 'Semester1') -> 'Year 3 Semester 1'."""
    return f"Year {year[-1]} {block[:-1]} {block[-1]}"


def counter_cases(items: list[dict], page_meta: dict[str, tuple[str, str]]) -> list[dict]:
    """Answerable questions whose every answer page is shared (no year), with a block put in front."""
    shared = [i for i in items if i["source_ids"] and all(not page_meta.get(s, ("", ""))[0] for s in i["source_ids"])]
    cases = []
    for n, item in enumerate(shared):
        year, block = COUNTER_SCOPES[n % len(COUNTER_SCOPES)]
        cases.append({**item, "id": f"counter-{item['id']}", "set": "counter",
                      "question": f"{readable(year, block)}: {item['question']}"})
    return cases


# ---------------------------------------------------------------- scoring (no models)


def first_hit_rank(doc_ids: list[str], sources: list[str]) -> int | None:
    return next((rank for rank, d in enumerate(doc_ids, start=1) if d in sources), None)


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def in_segment(chunk: dict, target: list[str]) -> bool:
    return target[0] in _as_list(chunk.get("year", "")) and target[1] in _as_list(chunk.get("block", ""))


def hits(case: dict, ranked: list[dict]) -> tuple[bool, bool]:
    """(hit@1, hit@5) of one case for a ranked candidate list."""
    if case["set"] == "dir":
        return bool(ranked) and in_segment(ranked[0], case["target"]), any(in_segment(c, case["target"]) for c in ranked[:5])
    rank = first_hit_rank([c["doc_id"] for c in ranked], case["sources"])
    return rank == 1, rank is not None and rank <= 5


def score_sets(cases: list[dict], ranking) -> dict[str, dict[str, int]]:
    """{set: {"n", "hit@1", "hit@5"}} with ranking(case) -> ranked candidates."""
    out = {s: {"n": 0, "hit@1": 0, "hit@5": 0} for s in SETS}
    for case in cases:
        at1, at5 = hits(case, ranking(case))
        row = out[case["set"]]
        row["n"] += 1
        row["hit@1"] += at1
        row["hit@5"] += at5
    return out


def total(scores: dict) -> int:
    return sum(s["hit@1"] + s["hit@5"] for s in scores.values())


def choose(grid: dict[tuple[float, float], dict], baseline: dict) -> tuple[float, float] | None:
    """The rule in the module docstring. grid: {(w_block, w_year): scores}."""
    passing = [w for w, s in grid.items()
               if all(s[name]["hit@5"] >= baseline[name]["hit@5"] for name in ("page", "student", "counter"))]
    if not passing:
        return None
    return max(passing, key=lambda w: (total(grid[w]), -(w[0] + w[1]), -w[0]))


def tune(cases: list[dict], weights: tuple[float, float] | None = None) -> dict:
    """weights=None: the whole grid and the rule's choice. weights given: only those (the held-out check)."""
    from ragapp.retrieve import scope_boost  # pure function, no models

    baseline = score_sets(cases, lambda c: c["filter"])
    if weights:
        boosted = score_sets(cases, lambda c: scope_boost(c["boost"], c["scope"][0], c["scope"][1], *weights))
        passes = all(boosted[n]["hit@5"] >= baseline[n]["hit@5"] for n in ("page", "student", "counter"))
        return {"filter": baseline, "boost": {"w_block": weights[0], "w_year": weights[1], "scores": boosted},
                "passes": passes}
    grid = {}
    for w_block, w_year in itertools.product(W_BLOCK, W_YEAR):
        grid[(w_block, w_year)] = score_sets(
            cases, lambda c: scope_boost(c["boost"], c["scope"][0], c["scope"][1], w_block, w_year))
    chosen = choose(grid, baseline)
    return {
        "filter": baseline,
        "boost_no_bonus": grid[(0.0, 0.0)],
        "chosen": {"w_block": chosen[0], "w_year": chosen[1], "scores": grid[chosen]} if chosen else None,
        "grid": [{"w_block": wb, "w_year": wy, "total": total(s), **{k: f"{v['hit@1']}/{v['hit@5']}/{v['n']}" for k, v in s.items()}}
                 for (wb, wy), s in grid.items()],
    }


# ---------------------------------------------------------------- step 1: dump candidates (models)


def build_cases(golden_dir: Path, split: str = "dev") -> list[dict]:
    import behavioral
    import run_eval
    from ragapp import config
    from ragapp.cleaning import clean
    from ragapp.documents import load_documents

    corpus = load_documents(config.CORPUS_PATH)
    page_meta = {d.id: (d.metadata.get("year", ""), d.metadata.get("block", "")) for d in corpus}
    duplicate_of = clean(corpus).duplicate_of if config.CLEAN_CORPUS else {}

    def dev(name: str) -> list[dict]:
        lines = (golden_dir / name).read_text(encoding="utf-8").splitlines()
        return [i for i in (json.loads(line) for line in lines if line.strip()) if i.get("split") == split]

    page, student = dev("golden_set.jsonl"), dev("golden_student.jsonl")
    cases = [{**i, "set": "page"} for i in page if i["source_ids"]]
    cases += [{**i, "set": "student"} for i in student if i["source_ids"]]
    cases += counter_cases(page + student, page_meta)
    for case in cases:
        case["sources"] = list(dict.fromkeys(duplicate_of.get(s, s) for s in case["source_ids"]))
        ctx = run_eval.student_for(case, "none", {})
        case["context"] = [str(ctx.today), ctx.year, ctx.block] if ctx else None
    for c in behavioral.make_cases():
        if c["family"] in ("dir_next", "dir_last"):
            cases.append({"id": c["id"], "set": "dir", "question": c["question"], "target": c["gold"]["target"],
                          "context": [c["today"], c["year"], c["block"]]})
    return cases


def dump(name: str, golden_dir: Path, split: str = "dev") -> None:
    from ragapp import config
    from ragapp.colbert import load_colbert
    from ragapp.rerank import rerank
    from ragapp.retrieve import filter_for, retrieve_many
    from ragapp.rewrite import search_queries, week_queries
    from ragapp.student import StudentContext

    assert config.QUERY_REWRITE in ("2", "3") and config.RERANK_MODEL, "written for the default pipeline"
    load_colbert()
    cases = build_cases(golden_dir, split)
    keep = ("doc_id", "chunk_index", "year", "block", "score")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}_scope_candidates.jsonl", "w", encoding="utf-8") as f:
        for n, case in enumerate(cases, start=1):
            ctx = case["context"]
            student = StudentContext(date.fromisoformat(ctx[0]), ctx[1], ctx[2]) if ctx else None
            year, block, today = (student.year, student.block, student.today) if student else (None, None, None)
            queries = search_queries(case["question"], student) if config.QUERY_REWRITE == "2" else week_queries(case["question"], student)
            case["queries"] = queries  # the same queries for both modes, so only the mode differs
            for mode in ("filter", "boost"):
                config.SCOPE_MODE = mode
                found = rerank(case["question"], retrieve_many(queries, config.RERANK_CANDIDATES, case["question"], year, block, today))
                case[mode] = [{k: c.get(k) for k in keep} for c in found]
            case["scope"] = filter_for(case["question"], year, block, today)
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(f"  {n}/{len(cases)}  {case['id']}", end="\r")
    print(f"\n{len(cases)} cases -> {name}_scope_candidates.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", metavar="NAME", help="step 1: save the candidates of both modes (models)")
    parser.add_argument("--tune", metavar="NAME", help="step 2: grid over the weights (no models)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--weights", help="w_block,w_year: score only these (the held-out check), e.g. 0.05,0")
    parser.add_argument("--golden-dir", type=Path, default=EVALS_DIR, help="folder with golden_set.jsonl and golden_student.jsonl")
    args = parser.parse_args()
    if args.dump:
        dump(args.dump, args.golden_dir, args.split)
    if args.tune:
        lines = (RESULTS_DIR / f"{args.tune}_scope_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else None
        summary = tune([json.loads(line) for line in lines if line.strip()], weights)
        (RESULTS_DIR / f"{args.tune}_scope_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "grid"}, indent=2))
    if not (args.dump or args.tune):
        parser.error("--dump NAME and/or --tune NAME")


if __name__ == "__main__":
    main()
