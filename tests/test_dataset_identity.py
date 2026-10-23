import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from source_support import source_at
from test_api_scoring import load_evaluator


class DatasetIdentityTests(unittest.TestCase):
    def test_pinned_raw_input_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl.gz"
            row = dict(task_id="x", ground_truth="é", unit_tests="[]")
            path.write_bytes(gzip.compress((json.dumps(row, ensure_ascii=False)+"\n").encode()))
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = Mock(return_value=str(path))
            old = types.ModuleType("datasets")
            old.load_dataset = Mock(return_value=[dict(row)])
            module = types.ModuleType("data_under_test")
            with patch.dict(sys.modules, huggingface_hub=hub, datasets=old):
                exec(compile(source_at("data_utils.py"), "data_utils.py", "exec"), module.__dict__)
                rows = module.load_dataset("block_v2")
            hub.hf_hub_download.assert_called_once_with("gonglinyuan/safim", "block_completion_v2.jsonl.gz",
                repo_type="dataset", revision="be132cc15372e90b6f03a608e77f2d940e384edb")
            self.assertEqual(rows[0]["unit_tests"], [])
            self.assertEqual(rows[0]["ground_truth"], "é")
            self.assertEqual(rows.provenance["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(rows.provenance["rows"], 1)

    def test_mismatched_data_is_rejected_but_legacy_metadata_is_allowed(self):
        module = load_evaluator(source_at("evaluate.py"))
        identity = dict(repository="gonglinyuan/safim", revision="fixed", configuration="block_v2", sha256="a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            metadata = Path(str(path)+".run.json")
            for key in identity:
                with self.subTest(key=key):
                    metadata.write_text(json.dumps(dict(dataset={**identity, key:"different"})))
                    with self.assertRaises(ValueError):
                        module.validate_input_variant(path, "block_v2", identity)
            metadata.write_text(json.dumps(dict(dataset=identity)))
            module.validate_input_variant(path, "block_v2", identity)
            metadata.write_text("{}")
            module.validate_input_variant(path, "block_v2", identity)


if __name__ == "__main__":
    unittest.main()
