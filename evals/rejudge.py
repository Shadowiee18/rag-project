"""Re-grade saved eval runs with the current judge rules, without searching or generating again.

Run:  uv run python evals/rejudge.py                     (every run that has a summary and older judge rules)
      uv run python evals/rejudge.py dev_base_page ...   (only these runs)

Uses the answers saved in evals/results/<run>.jsonl, so only the judge (the local LLM) runs.
Rewrites the judge labels and refusal flags in <run>.jsonl and the answer numbers in <run>_summary.json;
retrieval and latency numbers stay as they were. Prints how many labels changed per run.
"""

import json
import sys
from pathlib import Path

from run_eval import JUDGE_RULES, RESULTS_DIR, answers_summary, grade, is_refusal


def rejudge(name: str, results_dir: Path = RESULTS_DIR) -> int:
    """Re-grade one run in place. Returns how many judge labels changed."""
    run_path, summary_path = results_dir / f"{name}.jsonl", results_dir / f"{name}_summary.json"
    results = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for r in results:
        r["refused"] = is_refusal(r["answer"])
        if r["source_ids"]:
            label = grade(r)
            changed += label != r.get("judge")
            r["judge"] = label
    run_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(answers_summary(results))
    summary["config"]["judge_rules"] = JUDGE_RULES
    summary["config"]["rejudged_labels_changed"] = changed
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return changed


def outdated_runs(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Runs with a summary whose answers were graded under older judge rules."""
    names = []
    for summary_path in sorted(results_dir.glob("*_summary.json")):
        name = summary_path.name.removesuffix("_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (results_dir / f"{name}.jsonl").exists() and summary.get("config", {}).get("judge_rules", 1) < JUDGE_RULES:
            names.append(name)
    return names


def main() -> None:
    names = sys.argv[1:] or outdated_runs()
    for number, name in enumerate(names, start=1):
        print(f"{number}/{len(names)}  {name}: {rejudge(name)} labels changed", flush=True)


if __name__ == "__main__":
    main()
