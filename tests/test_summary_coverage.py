import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from source_support import source_at


class SummaryCoverageTests(unittest.TestCase):
    def summarize(self, rows, partial=False, run=None):
        data = types.ModuleType("data_utils")
        data.add_dataset_arguments = lambda parser: None
        data.load_dataset = lambda *args: [dict(task_id="p", lang="python"), dict(task_id="j", lang="java")]
        data.stream_jsonl = lambda _: rows
        module = types.ModuleType("summary_coverage")
        with patch.dict(sys.modules, {"data_utils": data}):
            exec(compile(source_at("show_detailed_results.py"), "show_detailed_results.py", "exec"), module.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "result.jsonl")
            if run:
                Path(path + ".run.json").write_text(json.dumps(run))
            argv = ["summary", "block_v2", path] + (["--allow-partial"] if partial else [])
            output = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                module.main()
            return next(csv.reader(io.StringIO(output.getvalue())))

    def test_absent_languages_are_na(self):
        row = self.summarize([dict(task_id=t, passed=True, result="PASSED") for t in ["p", "j"]])
        self.assertEqual(row[1:6], ["NA", "100.000000", "100.000000", "NA", "100.000000"])

    def test_partial_requires_explicit_option(self):
        rows = [dict(task_id="p", passed=True, result="PASSED")]
        with self.assertRaises(SystemExit):
            self.summarize(rows)
        self.assertEqual(self.summarize(rows, partial=True)[5], "100.000000")

    def test_unscored_infrastructure_row_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.summarize([dict(task_id="p", passed=None, result="SERVER_ERROR")], partial=True)

    def test_duplicate_unknown_and_empty_results_are_rejected(self):
        for ids in [["p", "p"], ["unknown"], []]:
            with self.subTest(ids=ids), self.assertRaises(SystemExit):
                self.summarize([dict(task_id=t, passed=True, result="PASSED") for t in ids], partial=True)

    def test_running_manifest_is_rejected_even_with_partial_option(self):
        with self.assertRaises(SystemExit):
            self.summarize([dict(task_id="p", passed=True, result="PASSED")], partial=True,
                           run=dict(status="running", evaluated=1, expected=1))


if __name__ == "__main__":
    unittest.main()
