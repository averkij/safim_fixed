import sys
import types
import unittest
from unittest.mock import patch
from source_support import source_at


def summary_module():
    dependency = types.ModuleType("data_utils")
    dependency.load_dataset = dependency.stream_jsonl = lambda _: []
    module = types.ModuleType("summary_statistics")
    with patch.dict(sys.modules, {"data_utils": dependency}):
        exec(compile(source_at("show_detailed_results.py"), "show_detailed_results.py", "exec"), module.__dict__)
    return module


class SummaryStatisticsTests(unittest.TestCase):
    def test_exact_counts_and_granularity(self):
        summary = summary_module().metric_summary({"all": 95}, {"all": 200})
        self.assertEqual(summary["groups"]["all"], dict(passed=95, total=200, percent=47.5, one_task_percentage_points=0.5))
        self.assertIsNone(summary["uncertainty"])

    def test_absent_group_is_undefined(self):
        group = summary_module().metric_summary({"all": 1}, {"all": 1})["groups"]["java"]
        self.assertEqual(group, dict(passed=0, total=0, percent=None, one_task_percentage_points=None))

    def test_json_cli_reports_counts_not_invented_uncertainty(self):
        import contextlib
        import io
        import json
        module = summary_module()
        module.load_dataset = lambda _: [dict(task_id="p", lang="python")]
        module.stream_jsonl = lambda _: [dict(task_id="p", passed=True, result="PASSED")]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["summary", "block_v2", "synthetic-statistics.jsonl", "--format", "json"]), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module.main()
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["groups"]["all"]["total"], 1)
        self.assertIsNone(summary["uncertainty"])
        self.assertIn("not measured", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
