"""Leakage grouping, deterministic subsets, release checks, and HF export."""
from collections import Counter, defaultdict
import gzip
import json
import random
import re

from .storage import digest, read_rows, write_json, write_rows


def shingles(code):
    # Whitespace-insensitive lexical fingerprints, preserving literals/operators.
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\w+|[^\s\w]', code)
    return {tuple(tokens[i:i + 5]) for i in range(max(1, len(tokens) - 4))}


def similarity(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def deduplicate(tasks, reference_rows, threshold, decisions=None):
    decisions = decisions or {}
    if any(v not in ("duplicate", "independent") for v in decisions.values()):
        raise ValueError("Duplicate decisions must be duplicate or independent")
    groups = Groups()
    unique, rejected, reviews = [], [], []
    seen = {}
    fingerprints, postings = [], defaultdict(set)
    # Reference comparison uses complete code because masked spans differ between versions.
    references = [(r["task_id"], r["lang"], shingles(r["eval_prompt"].replace("{{completion}}", r["ground_truth"])))
                  for r in reference_rows]
    def index(record):
        position = len(fingerprints)
        fingerprints.append(record)
        for gram in record[2]:
            postings[(record[1], gram)].add(position)
    for rid, lang, fp in references:
        index((rid, lang, fp, None, None))
    statements = {}
    for row in sorted(tasks, key=lambda r: r["task_id"]):
        row = dict(row)
        cid = "contest:" + str(row["contest_id"])
        family = row.get("family_id")
        if family is None:
            if row.get("validation_scope") != "local_pilot":
                raise ValueError("Missing family review")
            # Keep every unresolved local-pilot family together conservatively.
            # Do not invent a reviewed family ID in the stored metadata.
            family = "local-unreviewed"
        groups.join(cid, "family:" + family)
        statement = " ".join(row["metadata"]["problem"]["statement"].split())
        if statement in statements:
            groups.join(cid, statements[statement])
        statements[statement] = cid
        code = row["eval_prompt"].replace("{{completion}}", row["ground_truth"])
        fp = shingles(code)
        exact = digest([row["lang"], row["task_type"], row["eval_prompt"], row["ground_truth"]])
        if exact in seen:
            groups.join(cid, seen[exact])
            rejected.append({"task_id": row["task_id"], "reason": "exact task duplicate"})
            continue
        seen[exact] = cid
        discard = False
        candidates = Counter(i for gram in fp for i in postings.get((row["lang"], gram), ()))
        for position, intersection in sorted(candidates.items()):
            other_id, other_lang, other_fp, other_cid, other_submission = fingerprints[position]
            if other_cid and other_submission is not None and other_submission == row.get('solution_id', row.get("submission_id")):
                continue
            score = intersection / (len(fp) + len(other_fp) - intersection)
            if score < threshold:
                continue
            pair = digest(sorted([row["task_id"], other_id]))
            decision = decisions.get(pair)
            if other_cid is None and fp == other_fp:
                decision = "duplicate"
            if other_cid:
                groups.join(cid, other_cid)
            reviews.append({"pair_id": pair, "task_id": row["task_id"], "other_id": other_id,
                "similarity": score, "historical": other_cid is None,
                "decision": decision or "pending"})
            if decision == "duplicate" or (other_cid is None and fp == other_fp):
                discard = True
        index((row["task_id"], row["lang"], fp, cid, row.get('solution_id', row.get("submission_id"))))
        if discard:
            rejected.append({"task_id": row["task_id"], "reason": "reviewed or historical duplicate"})
        else:
            unique.append(row)
    for row in unique:
        row["split_group"] = groups.find("contest:" + str(row["contest_id"]))
    return unique, rejected, reviews


def split_tasks(rows, config):
    rows = [dict(row) for row in rows]
    groups = sorted({r["split_group"] for r in rows}, key=lambda g: digest([config["seed"], g]))
    ndev = max(1, round(len(groups) * config["validation_fraction"])) if len(groups) > 1 else 0
    dev = set(groups[:ndev])
    buckets = defaultdict(list)
    for row in rows:
        row["split"] = "validation" if row["split_group"] in dev else "test"
        buckets[(row["task_type"], row["lang"])].append(row)
    selected = []
    for bucket in sorted(buckets):
        strata = defaultdict(list)
        for row in buckets[bucket]:
            rating = row["rating"]
            month = row["published_at"][:7] if row.get("published_at") else "unknown"
            strata[(month, -1 if rating is None else rating // 800)].append(row)
        for items in strata.values():
            items.sort(key=lambda r: (r["problem_id"], digest([config["seed"], r["task_id"]])))
        count, used_problems = 0, set()
        cap = config.get("max_items_per_task_language")
        while strata and (cap is None or count < cap):
            for key in sorted(list(strata)):
                if cap is not None and count >= cap:
                    break
                index = next((i for i, r in enumerate(strata[key]) if r["problem_id"] not in used_problems), 0)
                row = strata[key].pop(index)
                used_problems.add(row["problem_id"])
                selected.append(row)
                count += 1
                if not strata[key]:
                    del strata[key]
    return sorted(selected, key=lambda r: r["task_id"])


def select(rows, filters):
    from .collect import epoch
    allowed = {"date_from", "date_to", "rating_min", "rating_max", "tags", "test_origin",
               "contest_id", "problem_id", "split", "lang", "task_type", "runtime", "family_id",
               "source_origin", "generator_model"}
    if set(filters) - allowed:
        raise ValueError("Unknown filter fields: " + str(sorted(set(filters) - allowed)))
    def keep(row):
        for key, value in filters.items():
            if value is None:
                continue
            if key in ("date_from", "date_to"):
                date = row.get("published_at")
                if date is None or (key == "date_from" and epoch(date) < epoch(value)) or (key == "date_to" and epoch(date) >= epoch(value)):
                    return False
            elif key in ("rating_min", "rating_max"):
                rating = row.get("rating")
                if rating is None or (key == "rating_min" and rating < value) or (key == "rating_max" and rating > value):
                    return False
            elif key == "tags":
                if not set(value).issubset(row.get("tags") or []):
                    return False
            elif key == "test_origin":
                tests = row["unit_tests"]
                if isinstance(tests, str):
                    tests = json.loads(tests)
                if value not in {t.get("origin") for t in tests}:
                    return False
            elif row.get(key) != value:
                return False
        return True
    return [r for r in rows if keep(r)]


def audit(rows, config):
    errors, ids, groups = [], set(), {}
    from .collect import epoch
    for r in rows:
        tid = r["task_id"]
        if tid in ids:
            errors.append(f"Duplicate ID: {tid}")
        ids.add(tid)
        if r["eval_prompt"].count("{{completion}}") != 1:
            errors.append(f"Invalid marker: {tid}")
        code = r["eval_prompt"].replace("{{completion}}", r["ground_truth"])
        if digest(code.encode()) != r["source_sha256"]:
            errors.append(f"Reconstruction hash: {tid}")
        dates = [r['published_at']]
        if r.get('source_origin', 'human') == 'human':
            dates.append(r['submitted_at'])
        for date in dates:
            if date is None:
                errors.append(f"Missing reviewed date: {tid}")
            elif not epoch(config["start"]) <= epoch(date) < epoch(config["end"]):
                errors.append(f"Date outside freeze: {tid}")
        p = r["metadata"]["problem"]
        for key in ("problem_id", "contest_id", "rating", "tags", "published_at", "family_id"):
            if r[key] != p[key]:
                errors.append(f"Metadata mismatch {key}: {tid}")
        for key in ("split_group", "family_id", "contest_id", "problem_id", "source_sha256"):
            group = (key, r[key])
            if group in groups and groups[group] != r["split"]:
                errors.append(f"Split leakage {group}")
            groups[group] = r["split"]
        if not r["unit_tests"]:
            errors.append(f"No tests: {tid}")
    return errors


def export_release(rows, target, config, reviews, reference_manifest, manual_review):
    import pyarrow as pa
    import pyarrow.parquet as pq
    errors = audit(rows, config)
    if any(r.get("validation_scope") == "local_pilot" for r in rows):
        errors.append("Local pilot items cannot be released")
    if any(r.get('source_origin') == 'generated' for r in rows):
        errors.append('Generated-source release policy is pending')
    if not rows:
        errors.append("No validated items")
    if any(r["decision"] == "pending" for r in reviews):
        errors.append("Unresolved near-duplicate reviews")
    if reference_manifest.get("revision") != config["reference_revision"]:
        errors.append("Historical comparison revision missing or incorrect")
    if set(reference_manifest.get("configs", [])) != {"block", "control", "control_fixed", "block_v2"}:
        errors.append("Historical comparison must include all four Codeforces configurations")
    reviewed = set(manual_review.get("task_ids", []))
    if not manual_review.get("reviewer") or not manual_review.get("reviewed_at"):
        errors.append("Manual review provenance missing")
    for task in ("block", "control"):
        for lang in config["languages"]:
            subset = [r for r in rows if r["task_type"] == task and r["lang"] == lang]
            if not subset or len([r for r in subset if r["task_id"] in reviewed]) < min(20, len(subset)):
                errors.append(f"Manual review/coverage incomplete: {task}/{lang}")
    if len({r["problem_id"] for r in rows if r["task_id"] in reviewed}) < config["pilot_problems"]:
        errors.append(f"Reviewed pilot has fewer than {config['pilot_problems']} problems")
    pilot_ids = manual_review.get("pilot_problem_ids", [])
    if len(set(pilot_ids)) < config["pilot_problems"]:
        errors.append("Explicit pilot problem list is incomplete")
    if any(r["task_id"] not in reviewed for r in rows if r["problem_id"] in pilot_ids):
        errors.append("Every retained pilot item must be reviewed")
    if errors:
        raise ValueError("Release blocked:\n" + "\n".join(errors))
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise ValueError("Release directory must be empty; never overwrite a frozen release")
    configs = []
    for task in ("block", "control"):
        files = []
        for split in ("validation", "test"):
            subset = [r for r in rows if r["task_type"] == task and r["split"] == split]
            if not subset:
                continue
            # Complex metadata stays canonical JSON in HF rows; full tables are included below.
            exported = [{k: json.dumps(v, ensure_ascii=False, sort_keys=True) if k in
                         ("unit_tests", "metadata", "mask", "mutation_evidence", "limits") else v
                         for k, v in r.items()} for r in subset]
            stem = f"{task}/{split}"
            write_rows(target / (stem + ".jsonl.gz"), exported)
            pq.write_table(pa.Table.from_pylist(exported), target / (stem + ".parquet"), compression="zstd")
            files.append({"split": split, "path": stem + ".parquet"})
        configs.append({"config_name": task, "data_files": files})
    for entity, key, getter in (
        ("contests", "contest_id", lambda r: r["metadata"]["problem"]["contest_metadata"]),
        ("problems", "problem_id", lambda r: r["metadata"]["problem"]),
        ("submissions", "submission_id", lambda r: r["metadata"]["submission"]),
        ("test_suites", "test_suite_id", lambda r: {"test_suite_id": r["test_suite_id"], "tests": r["unit_tests"]}),
    ):
        table = {r[key]: getter(r) for r in rows}
        write_rows(target / "metadata" / (entity + ".jsonl.gz"), [table[k] for k in sorted(table, key=str)])
    header = "---\n" + "configs: " + json.dumps(configs) + "\n---\n"
    (target / "README.md").write_text(header + "\n# SAFIM v3\n\n" +
        f"{len(rows)} validated completion items, sourced in [{config['start']}, {config['end']}).\n\n" +
        "Block and control-flow completion in Python, C++, Java, and C#. " +
        "This dataset reduces temporal exposure; it does not establish absence of training contamination. " +
        "Difficulty and tags describe their recorded source snapshot. " +
        "See metadata tables, source URLs, review evidence, and manifest for provenance. " +
        "Rights are recorded per source; no blanket license is inferred.\n", encoding="utf-8")
    files = {p.relative_to(target).as_posix(): digest(p.read_bytes()) for p in sorted(target.rglob("*")) if p.is_file()}
    write_json(target / "manifest.json", {"schema_version": 1, "config": config,
        "counts": dict(Counter((r["task_type"] + "/" + r["lang"] + "/" + r["split"]) for r in rows)),
        "files": files, "reference": reference_manifest, "manual_review": manual_review})
    return target
