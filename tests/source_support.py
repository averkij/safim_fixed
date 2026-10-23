"""Load production source from a revision for paired regression audits."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def source_at(path):
    revision = os.environ.get("SAFIM_TEST_REVISION")
    if revision:
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT).decode("utf-8")
    return (ROOT / path).read_text(encoding="utf-8")
