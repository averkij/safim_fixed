from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import requests


class ExecEvalError(RuntimeError):
    """Infrastructure failure; never a model compilation or test failure."""


class ExecOutcome(Enum):
    PASSED = "PASSED"  # code executes and output matches expected output
    WRONG_ANSWER = "WRONG_ANSWER"  # code executes and output does NOT matches expected output
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"  # code executes and didn't exit in time, output is ignored in this case
    RUNTIME_ERROR = "RUNTIME_ERROR"  # code failed to execute (crashed)
    COMPILATION_ERROR = "COMPILATION_ERROR"  # code failed to compile
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"  # code exceeded memory limit during execution


@dataclass
class ExtendedUnittest:
    input: str
    output: List[str] = field(default_factory=list)
    result: Optional[str] = None
    exec_outcome: Optional[ExecOutcome] = None

    def json(self):
        _json = self.__dict__
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.name

        return _json

    @classmethod
    def from_json(cls, _json):
        return cls(
            input=_json.get("input", ""),
            output=_json.get("output", list()),
            result=_json.get("result", None),
            exec_outcome=_json.get("exec_outcome", None), )


class APICommunication:
    _session: requests.Session

    def __init__(self, server_url: str = "http://localhost:5000", timeout: float = 120):
        if timeout <= 0:
            raise ValueError("ExecEval timeout must be positive")
        self.timeout = (5, timeout)
        self._session = requests.Session()
        self.execute_code_url = f"{server_url}/api/execute_code"
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def get_runtimes(self):
        try:
            response = self._session.get(self.get_runtimes_url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise ExecEvalError(f"ExecEval runtime query failed: {error}") from error

    def execute_code(
        self,
        language: str,
        source_code: str,
        unittests: List[dict],
        limits: Optional[dict] = None,
        block_network: bool = True,
        stop_on_first_fail: bool = True,
        use_sanitizer: bool = False,
        compiler_program_name: Optional[str] = None,
        compiler_flags: Optional[str] = None,
        interpreter_cmd: Optional[str] = None,
        interpreter_flags: Optional[str] = None,
        sample_id: Optional[int] = None,
        task_id: Union[str, int, None] = None, ) -> Tuple[List[ExtendedUnittest], Optional[int], Union[str, int, None]]:
        if language is None:
            raise ValueError("EmptyLanguage")

        if source_code is None:
            raise ValueError("EmptySourceCode")

        if unittests is None or len(unittests) == 0:
            raise ValueError("EmptyUnittest")

        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer, )
        try:
            response = self._session.post(
                self.execute_code_url, json=request_body, headers={"Content-Type": "application/json"}, timeout=self.timeout)
            response.raise_for_status()
            json_response = response.json()
        except (requests.RequestException, ValueError) as error:
            raise ExecEvalError(f"ExecEval request failed for {task_id}: {error}") from error
        data = json_response.get("data") if isinstance(json_response, dict) else None
        if not isinstance(data, list) or not data or len(data) > len(unittests):
            raise ExecEvalError(f"Invalid ExecEval result list for {task_id}")
        allowed = {outcome.value for outcome in ExecOutcome}
        failed = False
        for test in data:
            if not isinstance(test, dict) or "exec_outcome" not in test:
                raise ExecEvalError(f"Missing ExecEval outcome for {task_id}")
            outcome = test["exec_outcome"]
            if outcome is None:
                if not failed:
                    raise ExecEvalError(f"Unexecuted test without an earlier failure for {task_id}")
            elif not isinstance(outcome, str) or outcome not in allowed:
                raise ExecEvalError(f"Unknown ExecEval outcome for {task_id}: {outcome}")
            elif outcome != "PASSED":
                failed = True
            if test.get("result") is not None and not isinstance(test["result"], str):
                raise ExecEvalError(f"Invalid ExecEval result text for {task_id}")
        if not failed and len(data) != len(unittests):
            raise ExecEvalError(f"Incomplete passing ExecEval response for {task_id}")
        return data, sample_id, task_id


LANG_TO_COMPILER = {
    "cpp": "GNU C++17", "csharp": "Mono C#", "java": "Java 17", "python": "PyPy 3"
}

execeval: APICommunication = None


def run_test(problem, completion):
    global execeval
    assert problem['task_id'] == completion['task_id']
    code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
    result = execeval.execute_code(
        LANG_TO_COMPILER[problem['lang']], code, problem['unit_tests'], task_id=problem['task_id']
    )[0]
    for o in result:
        if o.get('result') is not None and len(o['result']) > 1000:
            o['result'] = o['result'][:1000]
    return result, all(o['exec_outcome'] == 'PASSED' for o in result)


def build_execeval(args: Namespace):
    global execeval
    execeval = APICommunication(server_url=f"http://localhost:{args.port}", timeout=getattr(args, "exec_timeout", 120))
