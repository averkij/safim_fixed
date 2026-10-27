"""Executed INSIDE the isolated Linux container, never imported on the host."""
import json
import os
import resource
import signal
import subprocess
import time
import sys
from pathlib import Path
import hashlib
import checkers


def matches_output(stdout, expected, comparison='tokens'):
    if comparison not in ('tokens', 'ascii_case_insensitive_tokens'):
        raise ValueError('Unsupported output comparison: ' + str(comparison))
    def tokens(value):
        if comparison == 'ascii_case_insensitive_tokens':
            value = value.translate(str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'))
        return value.split()
    return expected is None or any(tokens(stdout) == tokens(value) for value in expected)


def invoke(command, stdin, seconds, output_bytes, file_bytes=None):
    def limits():
        cap = file_bytes or output_bytes
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    with open("stdin.txt", "wb") as f:
        f.write(stdin.encode("utf-8"))
    with open("stdin.txt", "rb") as inp, open("stdout.txt", "wb") as out, open("stderr.txt", "wb") as err:
        start = time.monotonic()
        proc = subprocess.Popen(command, stdin=inp, stdout=out, stderr=err, start_new_session=True, preexec_fn=limits)
        try:
            code = proc.wait(timeout=seconds)
            outcome = "OK" if code == 0 else "RUNTIME_ERROR"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            outcome = "TIME_LIMIT_EXCEEDED"
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    with open("stdout.txt", "rb") as out, open("stderr.txt", "rb") as err:
        stdout = out.read(output_bytes).decode("utf-8", errors="replace")
        stderr = err.read(2000).decode("utf-8", errors="replace")
    return {"outcome": outcome, "stdout": stdout,
            "stderr": stderr,
            "elapsed_seconds": time.monotonic() - start}


def main():
    # Expected outputs exist only in the privileged worker's memory, never in a
    # file visible to submitted programs. Children receive only the current input.
    job = json.load(sys.stdin)
    filename = job["filename"]
    if Path(filename).name != filename:
        raise ValueError("Runtime filename must be a basename")
    Path(filename).write_text(job["source"], encoding="utf-8")
    cap = job["limits"]["output_bytes"]
    if job.get("compile"):
        compiled = invoke(job["compile"], "", 30, cap,
                          job["limits"].get("compile_file_bytes", 64 * 1024 * 1024))
        if compiled["outcome"] != "OK":
            print(json.dumps([{**compiled, "outcome": "COMPILATION_ERROR"} for _ in job["tests"]]))
            return
    rows = []
    for test in job["tests"]:
        result = invoke(job["execute"], test["input"], job["limits"]["time_seconds"], cap)
        if result["outcome"] == "OK":
            expected = test.get("output")
            if test.get('checker'):
                if test.get('checker_sha256') != hashlib.sha256(Path(checkers.__file__).read_bytes()).hexdigest():
                    raise ValueError('Constructive checker implementation mismatch')
                correct = checkers.check_constructive(test['checker'],test['input'],result['stdout'],expected[0] if expected else None)
            else:
                correct = matches_output(result['stdout'],expected,test.get('comparison','tokens'))
            result['outcome'] = 'PASSED' if correct else 'WRONG_ANSWER'
        rows.append(result)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
