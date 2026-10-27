"""Item metrics, problem macro averages and family-clustered bootstrap intervals."""
from collections import defaultdict
import random


def summarize(tasks, results, seed=2026, bootstrap=1000):
    by_id = {}
    for result in results:
        if result["task_id"] in by_id:
            raise ValueError("Duplicate result ID")
        by_id[result["task_id"]] = result
    unknown = set(by_id) - {t["task_id"] for t in tasks}
    if unknown:
        raise ValueError("Results contain unknown task IDs")
    buckets, families, problems = defaultdict(list), defaultdict(list), defaultdict(list)
    infrastructure = []
    for task in tasks:
        result = by_id.get(task["task_id"], {"passed": False, "result": "EMPTY"})
        if result.get("result") == "INFRASTRUCTURE_ERROR":
            infrastructure.append(task["task_id"])
            continue
        success = int(result["passed"])
        families[task["family_id"]].append(success)
        problems[task["problem_id"]].append(success)
        rating = task.get("rating")
        dims = {"all": "all", "task": task["task_type"], "language": task["lang"],
                "month": (task.get("published_at") or "unknown")[:7], "difficulty": "unknown" if rating is None else str(rating // 400 * 400),
                "source_origin": task.get('source_origin', 'human'),
                "generator_model": task.get('generator_model') or 'not_applicable'}
        for dim, value in dims.items():
            buckets[f"{dim}/{value}"].append(success)
        for tag in task.get("tags") or []:
            buckets["tag/" + tag].append(success)
    rng, draws = random.Random(seed), []
    keys = sorted(families)
    if keys:
        for _ in range(bootstrap):
            values = [v for key in rng.choices(keys, k=len(keys)) for v in families[key]]
            draws.append(sum(values) / len(values))
    draws.sort()
    return {"complete": not infrastructure, "infrastructure_failures": infrastructure,
        "groups": {key: {"count": len(v), "passed": sum(v), "pass_at_1": sum(v) / len(v)} for key, v in sorted(buckets.items())},
        "problem_macro": sum(sum(v) / len(v) for v in problems.values()) / len(problems) if problems else None,
        "family_bootstrap_95_ci": [draws[int(.025 * len(draws))], draws[min(len(draws)-1, int(.975 * len(draws)))]] if draws else None,
        "seed": seed, "bootstrap_replicates": bootstrap,
        "note": "Scores exclude infrastructure failures and are provisional until those are rerun." if infrastructure else None}
