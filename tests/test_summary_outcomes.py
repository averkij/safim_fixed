import contextlib
import csv
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from source_support import source_at


class SummaryOutcomeTests(unittest.TestCase):
    def summarize(self, result):
        data = types.ModuleType("data_utils")
        problems = [dict(task_id=lang, lang=lang) for lang in ["cpp", "java", "python", "csharp"]]
        rows = [dict(task_id=p["lang"], passed=p["lang"] != "python", result=result if p["lang"] == "python" else "PASSED") for p in problems]
        data.load_dataset = lambda _: problems
        data.stream_jsonl = lambda _: rows
        module = types.ModuleType("summary_test")
        with patch.dict(sys.modules, {"data_utils": data}):
            exec(compile(source_at("show_detailed_results.py"), "show_detailed_results.py", "exec"), module.__dict__)
        stream = io.StringIO()
        with patch.object(sys, "argv", ["summary", "block_v2", "fixture.jsonl"]), contextlib.redirect_stdout(stream):
            module.main()
        return list(csv.reader(io.StringIO(stream.getvalue())))[0]

    def test_skipped_tests_do_not_make_mixed(self):
        row = self.summarize([dict(exec_outcome="WRONG_ANSWER"), dict(exec_outcome=None)])
        self.assertEqual(row[-2:], ["1", "0"])

    def test_real_mixed_outcomes_remain_mixed(self):
        row = self.summarize([dict(exec_outcome="WRONG_ANSWER"), dict(exec_outcome="RUNTIME_ERROR"), dict(exec_outcome=None)])
        self.assertEqual(row[-1], "1")

    def test_no_executed_failure_is_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize([dict(exec_outcome=None)])

    def test_string_label_and_score_control(self):
        row = self.summarize("WRONG_ANSWER")
        self.assertEqual(row[5], "75.000000")
        self.assertEqual(row[-2:], ["1", "0"])


if __name__ == "__main__":
    unittest.main()
