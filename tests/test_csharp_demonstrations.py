import hashlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from parser_support import load_prompt
from source_support import source_at


@unittest.skipUnless(os.environ.get("SAFIM_TEST_EXEC_SERVER"), "Set SAFIM_TEST_EXEC_SERVER and SAFIM_TEST_EXEC_PROFILE for Docker integration")
class CSharpDemonstrationTests(unittest.TestCase):
    def test_completed_arithmetic_demonstrations_execute(self):
        prompt=load_prompt()
        module=types.ModuleType("demo_execution");sys.modules[module.__name__]=module
        exec(compile(source_at("exec_utils.py"),"exec_utils.py","exec"),module.__dict__)
        path=Path(os.environ["SAFIM_TEST_EXEC_PROFILE"])
        profile=module.load_execution_profile(path)
        with module.APICommunication(os.environ["SAFIM_TEST_EXEC_SERVER"]) as client:
            for task in ["statement","block_v2","control","control_fixed"]:
                with self.subTest(task=task):
                    code=prompt.FEW_SHOT_PROMPTS[(task,"csharp")].split("Code:\n\n",1)[1].replace(
                        prompt.COMPLETION_PLACEHOLDER["csharp"],prompt.FEW_SHOT_ANSWERS[(task,"csharp")].rstrip())
                    tests=[dict(input="4\n1 2\n-1000 1000\n0 0\n-5 -7\n",output=["3\n0\n0\n-12\n"])]
                    result=client.execute_code(profile["languages"]["csharp"]["runtime"],code,tests,
                        limits=profile["languages"]["csharp"]["limits"])[0]
                    print(json.dumps(dict(task=task,code=code,tests=tests,result=result,
                        profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
                    self.assertTrue(all(r["exec_outcome"]=="PASSED" for r in result),result)


if __name__=="__main__":unittest.main()
