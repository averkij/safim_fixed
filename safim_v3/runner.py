"""Docker runner: never execute dataset code or generators on the host."""
import json
import subprocess
import uuid
from collections import Counter
from pathlib import Path

from .storage import read_json, write_json, digest


class InfrastructureError(RuntimeError):
    pass


def checker_path(tests):
    """Mount the exact recorded checker; never rewrite old test-suite hashes."""
    current = Path(__file__).with_name('checkers.py').resolve()
    hashes = {t.get('checker_sha256') for t in tests if t.get('checker')}
    if not hashes:
        return current
    if len(hashes) != 1:
        raise InfrastructureError('One execution requires one pinned checker version')
    expected = hashes.pop()
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise InfrastructureError('Invalid checker SHA256')
    path = current if digest(current.read_bytes()) == expected else current.parent / 'checker_versions' / (expected + '.py')
    if not path.is_file() or digest(path.read_bytes()) != expected:
        raise InfrastructureError('Recorded checker version unavailable: ' + expected)
    return path


class DockerRunner:
    def __init__(self, runtime_file, cache_dir=None):
        self.runtimes = read_json(runtime_file)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.occurrences = Counter()

    def run(self, source, runtime, tests, limits=None):
        if runtime not in self.runtimes:
            raise InfrastructureError("Unmapped runtime: " + str(runtime))
        if not tests:
            raise ValueError("Execution requires at least one test")
        spec = self.runtimes[runtime]
        image = spec["image"]
        if "@sha256:" not in image and not image.startswith("sha256:"):
            raise InfrastructureError("Execution image must be pinned by digest")
        limits = {"time_seconds": 3, "memory_mb": 512, "output_bytes": 1048576, **(limits or {})}
        request = {"source": source, "tests": tests, "limits": limits, **spec}
        name = "safim-v3-" + uuid.uuid4().hex
        worker = Path(__file__).with_name("worker.py").resolve()
        checker = checker_path(tests)
        identity = digest({"request": request, "worker_sha256": digest(worker.read_bytes()), "runner_sha256": digest(Path(__file__).read_bytes()),
                           'checkers_sha256':digest(checker.read_bytes())})
        ordinal = self.occurrences[identity]
        self.occurrences[identity] += 1
        cache_key = digest([identity, ordinal])
        cached = self.cache_dir / (cache_key + ".json") if self.cache_dir else None
        if cached and cached.exists():
            saved = read_json(cached)
            if saved["request_id"] != identity or saved["sha256"] != digest(saved["results"]):
                raise InfrastructureError("Execution cache checksum mismatch")
            return saved["results"]
        command = ["docker", "run", "--rm", "-i", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL", "--security-opt=no-new-privileges", "--pids-limit=64", "--cpus=1",
            "--memory", str(limits["memory_mb"]) + "m", "--memory-swap", str(limits["memory_mb"]) + "m",
            "--user=0:0", "--tmpfs", "/work:rw,exec,size=128m,mode=1777",
            "--tmpfs", "/tmp:rw,noexec,size=64m,mode=1777", "--workdir=/work",
            "--mount", f"type=bind,source={worker},target=/worker.py,readonly",
            "--mount", f"type=bind,source={checker},target=/checkers.py,readonly",
            image, "python3", "/worker.py"]
        try:
            result = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
                timeout=45 + len(tests) * (limits["time_seconds"] + 2))
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            raise InfrastructureError("Execution container did not complete") from exc
        if result.returncode:
            raise InfrastructureError(result.stderr[-2000:] or "Execution container failed")
        try:
            rows = json.loads(result.stdout)
        except ValueError as exc:
            raise InfrastructureError("Malformed worker response") from exc
        if not isinstance(rows, list) or len(rows) != len(tests):
            raise InfrastructureError("Incomplete worker results")
        if cached:
            write_json(cached, {"request_id": identity, "repetition": ordinal, "results": rows, "sha256": digest(rows)})
        return rows


def passed(results):
    return bool(results) and all(r["outcome"] == "PASSED" for r in results)
