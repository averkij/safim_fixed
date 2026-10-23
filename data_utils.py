import gzip
import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable

from huggingface_hub import hf_hub_download

DATASET_REVISION = "be132cc15372e90b6f03a608e77f2d940e384edb"
DATASET_FILES = {
    "block": "block_completion.jsonl.gz",
    "block_v2": "block_completion_v2.jsonl.gz",
    "control": "control_completion.jsonl.gz",
    "control_fixed": "control_completion_fixed.jsonl.gz",
    "api": "api_completion.jsonl.gz",
}


class DatasetRows(list):
    """List-compatible rows with the identity of the downloaded input."""
    def __init__(self, rows, provenance):
        super().__init__(rows)
        self.provenance = provenance


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt', encoding="utf-8") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def load_dataset(task):
    if task not in DATASET_FILES:
        raise ValueError(f"Unknown SAFIM configuration: {task}")
    filename = DATASET_FILES[task]
    path = Path(hf_hub_download("gonglinyuan/safim", filename,
                               repo_type="dataset", revision=DATASET_REVISION))
    lst = []
    for m in stream_jsonl(str(path)):
        m["unit_tests"] = json.loads(m["unit_tests"])
        lst.append(m)
    return DatasetRows(lst, dict(repository="gonglinyuan/safim", revision=DATASET_REVISION,
        configuration=task, split="test", filename=filename,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(lst)))
