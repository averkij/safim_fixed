import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from source_support import source_at
from test_api_scoring import load_evaluator


class EvaluationLabelTests(unittest.TestCase):
    def score(self, completion, tests, execution_passed=False, lang="python"):
        module = load_evaluator(source_at("evaluate.py"))
        problem = dict(task_id="t", lang=lang, ground_truth="reference", unit_tests=tests, eval_prompt="{{completion}}")
        module.load_dataset = lambda _: [problem]
        module.stream_jsonl = lambda _: iter([dict(task_id="t", completion=completion)])
        module.tqdm = lambda rows: rows
        module.build_execeval = lambda _: None
        module.run_test = lambda *_: ("PASSED" if execution_passed else "WRONG_ANSWER", execution_passed)
        module.check_syntax = lambda _: False
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            with patch.object(sys, "argv", ["evaluate.py", "block_v2", "unused", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            return json.loads(output.read_text())

    def test_empty_python_keeps_empty_label(self):
        self.assertEqual(self.score("  ", [{}])["result"], "EMPTY")

    def test_empty_other_language_control(self):
        self.assertEqual(self.score("", [{}], lang="java")["result"], "EMPTY")

    def test_nonempty_python_syntax_failure(self):
        self.assertEqual(self.score("broken(", [{}])["result"], "COMPILATION_ERROR")

    def test_api_wrong_answer_not_relabelled_compilation(self):
        self.assertEqual(self.score("other()", [])["result"], "WRONG_ANSWER")

    def test_empty_passing_execution_preserves_verdict(self):
        row = self.score("", [{}], execution_passed=True)
        self.assertTrue(row["passed"])
        self.assertEqual(row["result"], "PASSED")

    def test_exact_reference_control(self):
        self.assertTrue(self.score("reference", [{}])["passed"])


if __name__ == "__main__":
    unittest.main()
