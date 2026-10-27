import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import requests
from source_support import source_at
from test_api_scoring import load_evaluator


def client_module():
    module = types.ModuleType("exec_client_under_test")
    sys.modules[module.__name__] = module
    exec(compile(source_at("exec_utils.py"), "exec_utils.py", "exec"), module.__dict__)
    return module


class ExecInfrastructureTests(unittest.TestCase):
    def client(self, body, status=200):
        module = client_module()
        client = module.APICommunication()
        response = requests.Response()
        response.status_code = status
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        client._session = Mock()
        client._session.post.return_value = response
        return client

    def execute(self, client, count=1):
        return client.execute_code("PyPy 3", "print(1)", [{}] * count, task_id="fixture")[0]

    def test_http_error_cannot_be_a_pass(self):
        with self.assertRaises(RuntimeError):
            self.execute(self.client({"data": [{"exec_outcome": "PASSED"}]}, 500))

    def test_html_error_not_compilation_failure(self):
        with self.assertRaises(RuntimeError):
            self.execute(self.client(b"<html>Bad gateway</html>", 502))

    def test_missing_data_empty_and_malformed_results(self):
        for body in [{"error": "broken"}, {"data": []}, {"data": [{}]}, {"data": [{"exec_outcome": "UNKNOWN"}]}, {"data": [{"exec_outcome": []}]}]:
            with self.subTest(body=body), self.assertRaises(RuntimeError):
                self.execute(self.client(body))

    def test_timeout_is_infrastructure_failure(self):
        client = self.client({})
        client._session.post.side_effect = requests.Timeout("read timeout")
        with self.assertRaises(RuntimeError):
            self.execute(client)

    def test_request_has_bounded_timeout(self):
        client = self.client({"data": [{"exec_outcome": "PASSED"}]})
        self.execute(client)
        self.assertEqual(client._session.post.call_args.kwargs["timeout"], (5, 120))

    def test_incomplete_pass_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.execute(self.client({"data": [{"exec_outcome": "PASSED"}]}), count=2)

    def test_unexecuted_without_failure_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.execute(self.client({"data": [{"exec_outcome": None}]}))

    def test_legitimate_failure_and_skipped_tests_preserved(self):
        data = [{"exec_outcome": "WRONG_ANSWER"}, {"exec_outcome": None}]
        self.assertEqual(self.execute(self.client({"data": data}), count=2), data)

    def test_compilation_failure_preserved(self):
        data = [{"exec_outcome": "COMPILATION_ERROR", "result": "syntax error"}]
        self.assertEqual(self.execute(self.client({"data": data}), count=2), data)

    def test_failure_keeps_partial_results_and_no_score(self):
        module = load_evaluator(source_at("evaluate.py"))
        problems = [dict(task_id=str(i), lang="cpp", ground_truth="reference", unit_tests=[{}], eval_prompt="{{completion}}") for i in range(3)]
        module.load_dataset = lambda *args: problems
        module.stream_jsonl = lambda _: iter([dict(task_id=str(i), completion="reference" if i == 0 else "candidate") for i in range(3)])
        module.tqdm = lambda rows: rows
        module.build_execeval = lambda _: None
        module.run_test = Mock(side_effect=RuntimeError("synthetic server outage"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            stream = io.StringIO()
            with patch.object(sys, "argv", ["evaluate", "block_v2", "unused", str(output)]), contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
            self.assertEqual(error.exception.code, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["passed"] for row in rows], [True, None])
            self.assertEqual(rows[1]["result"], "SERVER_ERROR")
            manifest = json.loads(Path(str(output) + ".run.json").read_text())
            self.assertEqual((manifest["status"], manifest["evaluated"], manifest["expected"]), ("incomplete", 1, 3))
            self.assertNotIn("Pass@1:", stream.getvalue())
            module.run_test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
