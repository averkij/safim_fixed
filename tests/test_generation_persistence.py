import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from source_support import source_at


class GenerationPersistenceTests(unittest.TestCase):
    def run_generation(self, error=None):
        dependencies = {}
        for name, attributes in {"data_utils": ["load_dataset", "add_dataset_arguments"], "model_utils": ["ModelWrapper", "build_model"],
                                 "prompt_utils": ["apply_postprocessors", "apply_prompt"], "tqdm": ["tqdm"]}.items():
            dependency = types.ModuleType(name)
            for attribute in attributes:
                setattr(dependency, attribute, lambda *args: None)
            dependencies[name] = dependency
        module = types.ModuleType("generate_under_test")
        with patch.dict(sys.modules, dependencies):
            exec(compile(source_at("generate.py"), "generate.py", "exec"), module.__dict__)
        module.load_dataset = lambda *args: [dict(task_id="0"), dict(task_id="1")]
        module.build_model = lambda _: object()
        module.tqdm = lambda rows: rows
        def handle(sample, task, mode, model, cache, processors):
            if sample["task_id"] == "1" and error is not None:
                raise error
            cache[sample["task_id"]] = "raw completion"
            return dict(task_id=sample["task_id"], completion="completion")
        module.handle_example = handle
        with tempfile.TemporaryDirectory() as directory:
            output, cache = Path(directory) / "output.jsonl", Path(directory) / "cache.json"
            exit_code = 0
            with patch.object(sys, "argv", ["generate", "fixture", "block_v2", str(cache), str(output), "infilling"]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    module.main()
                except SystemExit as stopped:
                    exit_code = stopped.code
            rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
            cached = json.loads(cache.read_text()) if cache.exists() else {}
            path = Path(str(output) + ".run.json")
            run = json.loads(path.read_text()) if path.exists() else {}
            return exit_code, rows, cached, run

    def test_exception_exits_nonzero_and_keeps_completed_rows(self):
        code, rows, cache, run = self.run_generation(RuntimeError("synthetic generation failure"))
        self.assertEqual(code, 1)
        self.assertEqual([r["task_id"] for r in rows], ["0"])
        self.assertEqual(cache, {"0": "raw completion"})
        self.assertEqual((run["status"], run["completed"], run["expected"]), ("incomplete", 1, 2))

    def test_interrupt_keeps_output_and_cache(self):
        code, rows, cache, run = self.run_generation(KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(len(rows), 1)
        self.assertEqual(cache, {"0": "raw completion"})
        self.assertEqual(run["status"], "interrupted")

    def test_complete_run_control(self):
        code, rows, cache, run = self.run_generation()
        self.assertEqual((code, len(rows), len(cache)), (0, 2, 2))
        self.assertEqual((run["status"], run["completed"]), ("complete", 2))


if __name__ == "__main__":
    unittest.main()
