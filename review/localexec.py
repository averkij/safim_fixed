"""Local stand-in for ExecEval (no docker daemon in this sandbox).

Mirrors evaluate.py: eval_prompt.replace("{{completion}}", completion) is compiled and run on every
unit test; a task passes iff every test's stdout matches an expected output (token-wise).
Supports cpp / java / python (no mono/dotnet here, so csharp is skipped).

usage: localexec.py DATASET.jsonl.gz MODE OUT.jsonl [--lang L ...] [--limit N] [--seed S] [--jobs J]
MODE: empty | gt | ws   (completion = "" | ground_truth | " ")
"""
import argparse, glob, gzip, json, os, random, re, resource, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PCH = os.path.join(HERE, "pch")  # dir with bits/stdc++.h + precompiled .gch (see REVIEW.md)
CSC = "/usr/lib/dotnet/sdk/8.0.132/Roslyn/bincore/csc.dll"
CS_REF = "/usr/lib/dotnet/packs/Microsoft.NETCore.App.Ref/8.0.32/ref/net8.0"
TIMEOUT = 5  # seconds per test run


def limits(as_limit=True):
    if as_limit:
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


def sh(cmd, cwd, inp=None, timeout=TIMEOUT):
    as_limit = cmd[0] not in ("java", "javac", "mono", "dotnet")
    return subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True, timeout=timeout,
                          preexec_fn=lambda: limits(as_limit), errors="replace")


def run_cases(cmd, d, tests):
    outcomes = []
    for t in tests:
        try:
            p = sh(cmd, d, t["input"])
        except subprocess.TimeoutExpired:
            return "TIME_LIMIT_EXCEEDED"
        if p.returncode != 0:
            return "RUNTIME_ERROR"
        got = p.stdout.split()
        if not any(got == o.split() for o in t["output"]):
            return "WRONG_ANSWER"
    return "PASSED"


def evaluate(task):
    code, lang, tests = task["code"], task["lang"], task["tests"]
    d = tempfile.mkdtemp(prefix="se_")
    try:
        if lang == "cpp":
            open(f"{d}/a.cpp", "w", encoding="utf-8", newline="").write(code)
            try:
                p = sh(["g++", "-std=c++17", "-O2", "-I", PCH, "-w", "a.cpp", "-o", "a.out"], d, timeout=60)
            except subprocess.TimeoutExpired:
                return "COMPILATION_ERROR"
            if p.returncode != 0:
                return "COMPILATION_ERROR"
            return run_cases(["./a.out"], d, tests)
        if lang == "java":
            pub = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", code)
            name = pub.group(1) if pub else "Main"
            open(f"{d}/{name}.java", "w", encoding="utf-8", newline="").write(code)
            try:
                p = sh(["javac", "-nowarn", f"{name}.java"], d, timeout=60)
            except subprocess.TimeoutExpired:
                return "COMPILATION_ERROR"
            if p.returncode != 0:
                return "COMPILATION_ERROR"
            main = name
            if not pub:
                m = re.search(r"class\s+(\w+)[^{]*\{(?:(?!class\s).)*?static\s+void\s+main", code, re.S)
                main = m.group(1) if m else name
            return run_cases(["java", "-Xss64m", "-Xmx1g", "-cp", d, main], d, tests)
        if lang == "csharp":
            open(f"{d}/a.cs", "w", encoding="utf-8", newline="").write(code)
            refs = [f"-r:{x}" for x in sorted(glob.glob(CS_REF + "/*.dll"))]
            try:
                p = sh(["dotnet", CSC, "-nologo", "-nowarn:CS0168,CS0219,CS8632", "-target:exe", "-out:a.dll", "-langversion:latest", "a.cs"] + refs, d, timeout=90)
            except subprocess.TimeoutExpired:
                return "COMPILATION_ERROR"
            if p.returncode != 0:
                return "COMPILATION_ERROR"
            open(f"{d}/a.runtimeconfig.json", "w").write('{"runtimeOptions":{"tfm":"net8.0","framework":{"name":"Microsoft.NETCore.App","version":"8.0.0"}}}')
            return run_cases(["dotnet", "a.dll"], d, tests)
        if lang == "python":
            open(f"{d}/a.py", "w", encoding="utf-8", newline="").write(code)
            p = sh([sys.executable, "-I", "-c", "import sys;compile(open('a.py',encoding='utf-8').read(),'a.py','exec')"], d)
            if p.returncode != 0:
                return "COMPILATION_ERROR"
            return run_cases([sys.executable, "-I", "a.py"], d, tests)
        return "UNSUPPORTED"
    except Exception as e:  # noqa
        return "HARNESS_ERROR:" + type(e).__name__
    finally:
        shutil.rmtree(d, ignore_errors=True)


def work(args):
    tid, task = args
    return tid, evaluate(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("mode"); ap.add_argument("out")
    ap.add_argument("--lang", nargs="+", default=["cpp", "java", "python"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(l) for l in gzip.open(a.dataset, "rt")]
    rows = [r for r in rows if r["lang"] in a.lang]
    if a.limit:
        random.Random(a.seed).shuffle(rows)
        per = {}
        keep = []
        for r in rows:  # `limit` per language
            if per.get(r["lang"], 0) < a.limit:
                per[r["lang"]] = per.get(r["lang"], 0) + 1
                keep.append(r)
        rows = keep
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out)}
    jobs = []
    for r in rows:
        if r["task_id"] in done:
            continue
        comp = {"empty": "", "gt": r["ground_truth"], "ws": " ", "pass": "pass"}[a.mode]
        jobs.append((r["task_id"], {"code": r["eval_prompt"].replace("{{completion}}", comp), "lang": r["lang"],
                                    "tests": json.loads(r["unit_tests"])}))
    lang = {r["task_id"]: r["lang"] for r in rows}
    n = 0
    with open(a.out, "a") as f, ProcessPoolExecutor(a.jobs) as ex:
        for tid, res in ex.map(work, jobs, chunksize=4):
            f.write(json.dumps({"task_id": tid, "lang": lang[tid], "result": res}) + "\n"); f.flush()
            n += 1
            if n % 200 == 0:
                print(n, "/", len(jobs), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
