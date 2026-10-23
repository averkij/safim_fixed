import argparse
import ast
import json
from pathlib import Path
import re

import tqdm
from tqdm import tqdm

from ast_utils import ErrorCheckVisitor, get_parser
from data_utils import load_dataset, stream_jsonl
from exec_utils import ExecEvalError, build_execeval, run_test


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

    tokens1, tokens2 = expression_tokens(code1, lang), expression_tokens(code2, lang)
    if tokens1 is None or tokens2 is None:
        return code1.strip() == code2.strip()
    return tokens1 == tokens2


def expression_tokens(code, lang):
    """Ignore formatting between tokens, never literal contents or boundaries."""
    prefix, suffix = {
        # C++ API references also contain object declarations/initializers.
        "cpp": ("void __safim_scope() { ", "; }"),
        "java": ("class __Safim { Object value = ", "; }"),
        "csharp": ("class __Safim { object value = ", "; }"),
    }[lang]
    tree = get_parser(lang).parse((prefix + code.strip() + suffix).encode("utf-8"))
    if tree.root_node.has_error:
        return None
    tokens, stack = [], [tree.root_node]
    while stack:
        node = stack.pop()
        if "comment" in node.type:
            continue
        if not node.children:
            tokens.append((node.type, node.text))
        else:
            stack.extend(reversed(node.children))
    return tokens


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


def validate_input_variant(path, completion_type):
    metadata = Path(str(path) + ".run.json")
    if metadata.exists():
        run = json.loads(metadata.read_text(encoding="utf-8"))
        declared = run.get("configuration", {}).get("completion_type")
        if declared is not None and declared != completion_type:
            raise ValueError(f"Predictions declare {declared}, but evaluation requests {completion_type}; dataset variants are distinct protocols")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("completion_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--exec-timeout", type=float, default=120, help="ExecEval read timeout in seconds")
    parser.add_argument("--allow-partial", action="store_true", help="Score only supplied known task IDs; report a subset score")
    args = parser.parse_args()

    problems = load_dataset(args.completion_type)
    try:
        validate_input_variant(args.completion_path, args.completion_type)
        completions = validate_completions(stream_jsonl(args.completion_path), problems, args.allow_partial)
    except ValueError as error:
        parser.error(str(error))
    if args.allow_partial:
        problems = [problem for problem in problems if problem["task_id"] in completions]
        print(f"Partial evaluation: {len(problems)} supplied tasks")
    build_execeval(args)

    pass_cnt, total = 0, 0
    manifest_path = Path(str(args.output_path) + ".run.json")
    manifest = dict(status="running", completion_type=args.completion_type, partial=args.allow_partial,
                    expected=len(problems), evaluated=0, passed=0, exec_timeout=args.exec_timeout)
    def save_manifest():
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    save_manifest()
    with open(args.output_path, "w", encoding="utf-8") as output:
        for problem in tqdm(problems):
            completion = completions[problem["task_id"]]
            try:
                if problem.get("unit_tests"):
                    if completion['completion'] == problem["ground_truth"]:
                        result, passed = "PASSED", True
                    else:
                        result, passed = run_test(problem, completion)
                else:
                    passed = syntax_match(completion['completion'], problem["ground_truth"], problem["lang"])
                    result = "EXACT_MATCH" if passed else "WRONG_ANSWER"
                if not completion['completion'].strip() and not passed:
                    result = "EMPTY"
                if problem["lang"] == "python" and not passed and completion['completion'].strip() and problem.get("unit_tests"):
                    full_code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
                    if not check_syntax(full_code):
                        result = "COMPILATION_ERROR"
            except ExecEvalError as error:
                output.write(json.dumps(dict(task_id=problem["task_id"], result="SERVER_ERROR", passed=None,
                                             check_result=0, error=str(error)[:1000])) + "\n")
                output.flush()
                manifest.update(status="incomplete", failed_task=problem["task_id"])
                save_manifest()
                parser.exit(1, f"Evaluation incomplete: {error}\n")
            output.write(json.dumps(dict(task_id=problem["task_id"], result=result, passed=passed, check_result=0)) + "\n")
            output.flush()
            pass_cnt += int(passed)
            total += 1
            manifest.update(evaluated=total, passed=pass_cnt)
            save_manifest()
    manifest["status"] = "complete"
    save_manifest()
    print(f"Pass {pass_cnt} / Total {total}")
    print(f"Pass@1: {pass_cnt / total * 100 :.04f}%")


if __name__ == '__main__':
    main()
