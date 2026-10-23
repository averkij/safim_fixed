import argparse
import csv
import json
import os
from pathlib import Path
import sys
from collections import defaultdict

from data_utils import load_dataset, stream_jsonl


def metric_summary(pass_cnt, total):
    return {
        "metric": "empirical pass@1 (one completion per task)",
        "uncertainty": None,
        "uncertainty_note": "Sampling variability not measured; decimal digits are not confidence. Use independent generation seeds for repeated runs.",
        "groups": {
            lang: dict(passed=pass_cnt.get(lang, 0), total=total.get(lang, 0),
                       percent=100 * pass_cnt.get(lang, 0) / total[lang] if total.get(lang) else None,
                       one_task_percentage_points=100 / total[lang] if total.get(lang) else None)
            for lang in ["cpp", "java", "python", "csharp", "all"]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--allow-partial", action="store_true", help="Summarize an explicitly selected subset")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="JSON includes exact counts and uncertainty status")
    args = parser.parse_args()

    problems = load_dataset(args.completion_type)
    problem_to_lang = {problem["task_id"]: problem["lang"] for problem in problems}
    rows = list(stream_jsonl(args.result_path))
    if not rows:
        parser.error("Result file is empty")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or type(row.get("passed")) is not bool:
            parser.error("Every scored result requires a task_id and a boolean passed; incomplete infrastructure rows cannot be scored")
        if row["task_id"] in seen or row["task_id"] not in problem_to_lang:
            parser.error(f"Duplicate or unknown result task_id: {row['task_id']}")
        seen.add(row["task_id"])
    if seen != set(problem_to_lang) and not args.allow_partial:
        parser.error("Incomplete result coverage; use --allow-partial only for an intended subset")
    run_path = Path(args.result_path + ".run.json")
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get("evaluated") != len(rows) or run.get("expected") != len(rows):
            parser.error("Evaluation run is not complete or its counts do not match the result file")
    if seen != set(problem_to_lang):
        print(f"Partial summary: {len(rows)} / {len(problem_to_lang)} tasks", file=sys.stderr)

    pass_cnt, total = defaultdict(int), defaultdict(int)
    results_cnt = defaultdict(int)
    for r in rows:
        if 'passed' not in r:
            print(r)
        lang = problem_to_lang[r['task_id']]
        pass_cnt[lang] += int(r['passed'])
        total[lang] += 1
        pass_cnt["all"] += int(r['passed'])
        total["all"] += 1
        if not r['passed']:
            r_set = set()
            if isinstance(r['result'], str):
                if r['result'] != "EXACT_MATCH":
                    r_set.add(r['result'])
            else:
                for rr in r['result']:
                    if rr["exec_outcome"] not in ("PASSED", None):
                        r_set.add(rr["exec_outcome"])
            if not r_set:
                raise ValueError(f"Failed result has no executed failure: {r['task_id']}")
            if len(r_set) > 1:
                results_cnt["MIXED"] += 1
            else:
                results_cnt[list(r_set)[0]] += 1

    ALL_OUTCOMES = ['EMPTY', 'COMPILATION_ERROR', 'RUNTIME_ERROR', 'MEMORY_LIMIT_EXCEEDED', 'TIME_LIMIT_EXCEEDED',
                    'WRONG_ANSWER', 'MIXED']
    summary = metric_summary(pass_cnt, total)
    summary["errors"] = {outcome: results_cnt[outcome] for outcome in ALL_OUTCOMES}
    summary["dataset_variant"] = args.completion_type
    summary["partial"] = seen != set(problem_to_lang)
    print(f"Empirical pass@1: {pass_cnt['all']}/{total['all']} tasks; sampling variability not measured. "
          "Decimal digits are not confidence.", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(summary, indent=2))
        return
    writer = csv.writer(sys.stdout)
    # writer.writerow(['C++', 'Java', 'Python', 'C#', 'All'] + ALL_OUTCOMES)
    writer.writerow(
        [os.path.splitext(os.path.basename(args.result_path))[0]]
        + [f"{pass_cnt[lang] / total[lang] * 100:.6f}" if total[lang] else "NA" for lang in ['cpp', 'java', 'python', 'csharp', 'all']]
        + [str(results_cnt[outcome]) for outcome in ALL_OUTCOMES]
    )


if __name__ == '__main__':
    main()
