"""Content-addressed raw evidence and reproducible stage manifests."""
import gzip
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(canonical(row) + b"\n" for row in rows)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def environment():
    def commit(directory):
        p = subprocess.run(["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "parent_commit": commit("."), "submodule_commit": commit("safim_fixed"),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions()))}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def snapshot(self, data, url, status=200, retrieved_at=None, **metadata):
        raw_hash = digest(data)
        raw = self.root / "raw" / raw_hash
        raw.parent.mkdir(exist_ok=True)
        if not raw.exists():
            raw.write_bytes(data)
        record = {"url": url, "retrieved_at": retrieved_at or now(), "http_status": status,
                  "sha256": raw_hash, "raw_path": raw.relative_to(self.root).as_posix(), **metadata}
        record["snapshot_id"] = digest(record)
        write_json(self.root / "snapshots" / (record["snapshot_id"] + ".json"), record)
        return record

    def stage(self, name, inputs, config, tables, details=None):
        """New immutable stage directory; reruns from identical inputs use the same ID."""
        code = {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
        identity = {"stage": name, "inputs": inputs, "config": config, "implementation": code,
                    "tables": {key: digest(rows) for key, rows in tables.items()}}
        run_id = digest(identity)
        target = self.root / "stages" / name / run_id
        if target.exists():
            manifest = read_json(target / "manifest.json")
            for filename, expected in manifest["files"].items():
                if digest((target / filename).read_bytes()) != expected:
                    raise ValueError(f"Corrupt cached stage: {target / filename}")
            return target
        target.mkdir(parents=True)
        files = {}
        for key, rows in tables.items():
            write_rows(target / (key + ".jsonl"), rows)
            files[key + ".jsonl"] = digest((target / (key + ".jsonl")).read_bytes())
        write_json(target / "manifest.json", {**identity, "run_id": run_id, "created_at": now(),
                   "counts": {key: len(rows) for key, rows in tables.items()}, "files": files,
                   "environment": environment(), "details": details or {}})
        return target
