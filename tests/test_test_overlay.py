import json
from pathlib import Path
import tempfile
import unittest
from source_support import source_at
from test_api_scoring import load_evaluator


class TestOverlayTests(unittest.TestCase):
    def setUp(self):
        self.module=load_evaluator(source_at("evaluate.py"))
        self.data=dict(revision="fixed",sha256="hash")
        self.problems=[dict(task_id="x",unit_tests=[dict(input="1",output=["1"])])]
        self.overlay=dict(schema_version=1,dataset=self.data,tests=dict(x=[dict(input="2",output=["2"])]))

    def apply(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"tests.json";path.write_text(json.dumps(self.overlay))
            return self.module.apply_test_overlay(self.problems,path,self.data)

    def test_append_without_mutating_original(self):
        rows,metadata=self.apply()
        self.assertEqual(len(rows[0]["unit_tests"]),2)
        self.assertEqual(len(self.problems[0]["unit_tests"]),1)
        self.assertEqual(metadata["added_tests"],1)
        self.assertEqual(len(metadata["sha256"]),64)

    def test_reject_wrong_identity_and_unknown_ids(self):
        self.overlay["dataset"]={"revision":"other"}
        with self.assertRaises(ValueError):self.apply()
        self.overlay["dataset"]=self.data
        self.overlay["tests"]={"unknown":[dict(input="1",output=["1"])]}
        with self.assertRaises(ValueError):self.apply()

    def test_reject_malformed_tests_or_syntax_only_tasks(self):
        self.overlay["tests"]["x"][0]["output"]="not a list"
        with self.assertRaises(ValueError):self.apply()
        self.overlay["tests"]["x"][0]["output"]=["2"]
        self.problems[0]["unit_tests"]=[]
        with self.assertRaises(ValueError):self.apply()


if __name__=="__main__":unittest.main()
