import argparse
import ast
import json
import re

import tqdm
from tqdm import tqdm

from ast_utils import ErrorCheckVisitor, get_parser
from data_utils import load_dataset, stream_jsonl
from exec_utils import build_execeval, run_test


def check_syntax(code):
    parser = get_parser("python")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0


def get_function_call_params(node):
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = [(kw.arg, ast.dump(kw.value)) for kw in node.keywords]
    # Keep every occurrence. Unpacking also makes keyword order significant.
    if all(name is not None for name, _ in keyword_args):
        keyword_args.sort()
    return positional_args, keyword_args


def function_calls_match(call1, call2):
    if ast.dump(call1.func, include_attributes=False) != ast.dump(call2.func, include_attributes=False):
        return False
    params1 = get_function_call_params(call1)
    params2 = get_function_call_params(call2)
    return params1 == params2


def syntax_match(code1, code2, lang):
    if lang == "python":
        code1 = code1.strip()
        code2 = code2.strip()
        try:
            tree1 = ast.parse(code1, mode='eval')
            tree2 = ast.parse(code2, mode='eval')

            if isinstance(tree1.body, ast.Call) and isinstance(tree2.body, ast.Call):
                return function_calls_match(tree1.body, tree2.body)
            return ast.dump(tree1, include_attributes=False) == ast.dump(tree2, include_attributes=False)
        except SyntaxError:
            # Do not erase literal contents or merge tokens in malformed code.
            return code1 == code2

    code1 = re.sub(r'\s+', '', code1).strip()
    code2 = re.sub(r'\s+', '', code2).strip()
    return code1 == code2


def validate_completions(rows, problems, allow_partial=False):
    completions = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not isinstance(row.get("completion"), str):
            raise ValueError("Each prediction requires string task_id and completion fields")
        if row["task_id"] in completions:
            raise ValueError(f"Duplicate prediction task_id: {row['task_id']}")
        completions[row["task_id"]] = row
    expected = {problem["task_id"] for problem in problems}
    unknown = set(completions) - expected
    if unknown:
        raise ValueError(f"Unknown prediction task IDs: {sorted(unknown)[:5]}")
    if not completions:
        raise ValueError("Prediction file is empty")
    missing = expected - set(completions)
    if missing and not allow_partial:
        raise ValueError(f"Missing predictions for {len(missing)} tasks; use --allow-partial to score only supplied IDs")
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("completion_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--allow-partial", action="store_true", help="Score only supplied known task IDs; report a subset score")
    args = parser.parse_args()

    problems = load_dataset(args.completion_type)
    try:
        completions = validate_completions(stream_jsonl(args.completion_path), problems, args.allow_partial)
    except ValueError as error:
        parser.error(str(error))
    if args.allow_partial:
        problems = [problem for problem in problems if problem["task_id"] in completions]
        print(f"Partial evaluation: {len(problems)} supplied tasks")
    build_execeval(args)

    pass_cnt, total = 0, 0
    results = []
    for problem in tqdm(problems):
        if problem["task_id"] not in completions:
            result = "EMPTY"
            passed = False
        else:
            completion = completions[problem["task_id"]]
            if "unit_tests" in problem and problem["unit_tests"]:
                if completion['completion'] == problem["ground_truth"]:
                    result = "PASSED"
                    passed = True
                else:
                    result, passed = run_test(problem, completion)
            else:
                if syntax_match(completion['completion'], problem["ground_truth"], problem["lang"]):
                    result = "EXACT_MATCH"
                    passed = True
                else:
                    result = "WRONG_ANSWER"
                    passed = False
        if not completion['completion'].strip() and not passed:
            result = "EMPTY"
        if problem["lang"] == "python" and not passed and completion['completion'].strip() and problem.get("unit_tests"):
            full_code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
            if not check_syntax(full_code):
                result = "COMPILATION_ERROR"
        pass_cnt += int(passed)
        total += 1
        results.append(
            {
                "task_id": problem["task_id"], "result": result, "passed": passed, "check_result": 0
            }
        )

    print(f"Pass {pass_cnt} / Total {total}")
    print(f"Pass@1: {pass_cnt / total * 100 :.04f}%")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == '__main__':
    main()
