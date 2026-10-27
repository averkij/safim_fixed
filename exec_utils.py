from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
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
execution_profile = None
docker_runner = None


def load_execution_profile(path):
    if path is None:
        return None
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Execution profile requires schema_version=1")
    languages = profile.get("languages")
    fields = {"core", "data", "fsize", "sigpending", "rss", "nofile", "msgqueue", "rtprio", "stack", "cpu", "nproc", "_as", "locks"}
    if not isinstance(languages, dict) or set(languages) != set(LANG_TO_COMPILER):
        raise ValueError("Execution profile requires all four SAFIM languages")
    for lang, entry in languages.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("runtime"), str) or not entry["runtime"]:
            raise ValueError(f"Missing runtime for {lang}")
        limits = entry.get("limits")
        if not isinstance(limits, dict) or set(limits) != fields or any(type(value) is not int or value < -1 for value in limits.values()) or limits["cpu"] <= 0:
            raise ValueError(f"Provide complete integer resource limits for {lang}; partial limits bypass engine language defaults")
    return profile


def run_test(problem, completion):
    global execeval
    assert problem['task_id'] == completion['task_id']
    code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
    if problem.get("schema_version") in (1, 2):
        from safim_v3.runner import InfrastructureError, passed
        if docker_runner is None:
            raise ExecEvalError("SAFIM v3 requires --runtimes with a pinned Docker registry")
        try:
            outcomes = docker_runner.run(code, problem['runtime'], problem['unit_tests'], problem.get('limits'))
        except (InfrastructureError, OSError) as error:
            raise ExecEvalError(f"SAFIM v3 execution failed: {error}") from error
        return [{"exec_outcome": r["outcome"], "result": r["stdout"][:1000]} for r in outcomes], passed(outcomes)
    result = execeval.execute_code(
        execution_profile["languages"][problem["lang"]]["runtime"] if execution_profile else LANG_TO_COMPILER[problem['lang']],
        code, problem['unit_tests'], task_id=problem['task_id'],
        limits=execution_profile["languages"][problem["lang"]]["limits"] if execution_profile else None,
    )[0]
    for o in result:
        if o.get('result') is not None and len(o['result']) > 1000:
            o['result'] = o['result'][:1000]
    return result, all(o['exec_outcome'] == 'PASSED' for o in result)


def build_execeval(args: Namespace):
    global execeval, execution_profile, docker_runner
    docker_runner = None
    is_v3 = getattr(args, "completion_type", None) in ("block_v3", "control_v3")
    if getattr(args, "runtimes", None) and not is_v3:
        raise ValueError("--runtimes requires block_v3/control_v3; legacy configs use ExecEval")
    if is_v3:
        if getattr(args, "exec_profile", None):
            raise ValueError("--exec-profile belongs to ExecEval; v3 uses --runtimes and per-item limits")
        if not getattr(args, "runtimes", None):
            raise ValueError("SAFIM v3 evaluation requires --runtimes")
        from safim_v3.runner import DockerRunner
        from safim_v3 import runner
        path = Path(args.runtimes)
        docker_runner = DockerRunner(path)
        execeval, execution_profile = None, None
        return dict(backend="safim_v3_docker", runtimes=docker_runner.runtimes,
                    runtimes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
                    worker_sha256=hashlib.sha256(Path(runner.__file__).with_name("worker.py").read_bytes()).hexdigest(),
                    resource_limits="per-item limits from pinned dataset metadata; DockerRunner defaults for absent values",
                    checker_policy="recorded checker SHA256; shared SAFIM v3 worker")
    path = getattr(args, "exec_profile", None)
    execution_profile = load_execution_profile(path)
    server_url = f"http://localhost:{args.port}"
    timeout = getattr(args, "exec_timeout", 120)
    execeval = APICommunication(server_url=server_url, timeout=timeout)
    return dict(server_url=server_url, request_timeout=dict(connect=5, read=timeout),
                runtime_mapping=LANG_TO_COMPILER if not execution_profile else {lang: entry["runtime"] for lang, entry in execution_profile["languages"].items()},
                resource_limits="server defaults, not captured" if not execution_profile else "explicit profile",
                profile=execution_profile, profile_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
                profile_verified_against_server=False)
