from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from test_exec_infrastructure import client_module


def profile():
    limits=dict(core=0,data=-1,fsize=0,sigpending=0,rss=-1,nofile=4,msgqueue=0,rtprio=0,stack=-1,cpu=2,nproc=1,_as=2147483648,locks=0)
    return dict(schema_version=1,languages={lang:dict(runtime="fixture-"+lang,limits=dict(limits)) for lang in ["cpp","java","python","csharp"]})


class ExecutionProfileTests(unittest.TestCase):
    def test_explicit_profile_reaches_execution_and_metadata(self):
        module=client_module()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.json"
            value=profile()
            path.write_text(json.dumps(value))
            metadata=module.build_execeval(Namespace(port=5002,exec_profile=str(path),exec_timeout=30))
            self.assertEqual(metadata["profile"],value)
            self.assertEqual(len(metadata["profile_sha256"]),64)
            self.assertFalse(metadata["profile_verified_against_server"])
            module.execeval=Mock()
            module.execeval.execute_code.return_value=([dict(exec_outcome="PASSED")],None,"x")
            _,passed=module.run_test(dict(task_id="x",lang="java",eval_prompt="{{completion}}",unit_tests=[{}]),dict(task_id="x",completion="code"))
            self.assertTrue(passed)
            call=module.execeval.execute_code.call_args
            self.assertEqual(call.args[0],"fixture-java")
            self.assertEqual(call.kwargs["limits"],value["languages"]["java"]["limits"])

    def test_partial_limits_are_rejected(self):
        module=client_module()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.json"
            value=profile();value["languages"]["java"]["limits"]={"cpu":2}
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):module.load_execution_profile(path)

    def test_invalid_values_and_missing_languages_rejected(self):
        module=client_module()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.json"
            for invalid in [True,0,"2"]:
                value=profile();value["languages"]["cpp"]["limits"]["cpu"]=invalid
                path.write_text(json.dumps(value))
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):module.load_execution_profile(path)
            value=profile();del value["languages"]["java"]
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):module.load_execution_profile(path)

    def test_no_profile_explicitly_records_uncaptured_defaults(self):
        module=client_module()
        metadata=module.build_execeval(Namespace(port=5002))
        self.assertEqual(metadata["resource_limits"],"server defaults, not captured")
        self.assertIsNone(metadata["profile"])


if __name__=="__main__":unittest.main()
