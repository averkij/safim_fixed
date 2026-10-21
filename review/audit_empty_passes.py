"""Sanity audit for a finished evaluation: does any task pass with an empty/whitespace completion?

usage: python review/audit_empty_passes.py OUTPUTS.jsonl RESULTS.jsonl

OUTPUTS.jsonl = file written by generate.py (task_id, completion)
RESULTS.jsonl = file written by evaluate.py (task_id, result, passed)
A healthy benchmark run prints ~0 passes for empty completions on block/control/api.
"""
import collections
import json
import sys

outs = {}
for line in open(sys.argv[1]):
    if line.strip():
        r = json.loads(line)
        outs[r["task_id"]] = r["completion"]
res = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]

empty = [r for r in res if r["task_id"] in outs and not outs[r["task_id"]].strip()]
missing = [r for r in res if r["task_id"] not in outs]
passed_empty = [r for r in empty if r["passed"]]
print(f"tasks scored: {len(res)}   with answer missing from outputs: {len(missing)}   with empty answer: {len(empty)}")
print(f"PASSED with an empty answer: {len(passed_empty)}")
if empty:
    labels = collections.Counter(
        r["result"] if isinstance(r["result"], str)
        else "/".join(sorted({o["exec_outcome"] for o in r["result"]}))
        for r in empty)
    print("outcomes of empty answers:", dict(labels))
print(f"overall pass rate: {100 * sum(r['passed'] for r in res) / max(1, len(res)):.2f}%")
print(f"pass rate excluding empty answers: "
      f"{100 * sum(r['passed'] for r in res if r not in empty) / max(1, len(res) - len(empty)):.2f}%")
for r in passed_empty[:10]:
    print("  suspicious:", r["task_id"])
