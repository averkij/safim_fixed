import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
from source_support import source_at
from test_api_scoring import load_evaluator


class ReferenceExecutionTests(unittest.TestCase):
    def run_cli(self,force=False,engine_pass=False,tests=True):
        module=load_evaluator(source_at("evaluate.py"))
        module.load_dataset=lambda _:[dict(task_id="x",lang="cpp",ground_truth="f()",eval_prompt="{{completion}}",unit_tests=[{}] if tests else [])]
        module.stream_jsonl=lambda _:[dict(task_id="x",completion="f()")]
        module.tqdm=lambda rows:rows
        module.build_execeval=lambda _:None
        module.run_test=Mock(return_value=([dict(exec_outcome="PASSED" if engine_pass else "COMPILATION_ERROR")],engine_pass))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"results.jsonl"
            argv=["evaluate","block_v2","unused.jsonl",str(output)]+(["--force-execution"] if force else [])
            with patch.object(sys,"argv",argv),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):module.main()
            return json.loads(output.read_text()),json.loads(Path(str(output)+".run.json").read_text()),module.run_test

    def test_reference_shortcut_is_explicit(self):
        row,run,engine=self.run_cli()
        self.assertTrue(row["passed"])
        self.assertEqual(row["evaluation_method"],"reference_match")
        self.assertEqual(run["methods"]["reference_match"],1)
        engine.assert_not_called()

    def test_force_execution_can_disprove_reference_pass(self):
        row,run,engine=self.run_cli(force=True)
        self.assertFalse(row["passed"])
        self.assertEqual(row["evaluation_method"],"execution")
        self.assertEqual(run["methods"]["execution"],1)
        engine.assert_called_once()

    def test_healthy_reference_still_passes(self):
        row,_,engine=self.run_cli(force=True,engine_pass=True)
        self.assertTrue(row["passed"])
        engine.assert_called_once()

    def test_api_without_tests_keeps_syntax_protocol(self):
        row,_,engine=self.run_cli(force=True,tests=False)
        self.assertTrue(row["passed"])
        self.assertEqual(row["evaluation_method"],"syntax_match")
        engine.assert_not_called()


if __name__=="__main__":unittest.main()
