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


class EvaluationCoverageTests(unittest.TestCase):
    def run_cli(self, rows, partial=False):
        module = load_evaluator(source_at("evaluate.py"))
        problems = [dict(task_id=str(i), lang="python", ground_truth="f()", unit_tests=[],
                         eval_prompt="{{completion}}") for i in range(2)]
        module.load_dataset = lambda _: problems
        module.stream_jsonl = lambda _: iter(rows)
        module.tqdm = lambda items: items
        module.build_execeval = lambda _: None
        module.check_syntax = lambda _: True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            argv = ["evaluate.py", "api", "unused.jsonl", str(output)]
            if partial:
                argv.append("--allow-partial")
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module.main()
            return [json.loads(line) for line in output.read_text().splitlines()]

    def assert_rejected(self, rows):
        with self.assertRaises(SystemExit) as error:
            self.run_cli(rows)
        self.assertEqual(error.exception.code, 2)

    def test_missing_first_is_rejected(self):
        self.assert_rejected([dict(task_id="1", completion="f()")])

    def test_missing_later_is_rejected(self):
        self.assert_rejected([dict(task_id="0", completion="f()")])

    def test_duplicate_prediction_is_rejected(self):
        self.assert_rejected([dict(task_id="0", completion="f()"), dict(task_id="0", completion="g()"), dict(task_id="1", completion="f()")])

    def test_unknown_prediction_is_rejected(self):
        self.assert_rejected([dict(task_id=str(i), completion="f()") for i in range(3)])

    def test_empty_input_is_rejected(self):
        self.assert_rejected([])

    def test_invalid_completion_type_is_rejected(self):
        self.assert_rejected([dict(task_id="0", completion=None), dict(task_id="1", completion="f()")])

    def test_explicit_partial_scores_only_selected_ids(self):
        rows = self.run_cli([dict(task_id="1", completion="f()")], partial=True)
        self.assertEqual([(row["task_id"], row["passed"]) for row in rows], [("1", True)])

    def test_complete_input_preserves_verdicts(self):
        rows = self.run_cli([dict(task_id="0", completion="f()"), dict(task_id="1", completion="g()")])
        self.assertEqual([row["passed"] for row in rows], [True, False])


if __name__ == "__main__":
    unittest.main()
