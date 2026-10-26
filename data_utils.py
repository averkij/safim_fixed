import gzip
import json
from pathlib import Path
from typing import Dict, Iterable

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def add_dataset_arguments(parser):
    parser.add_argument("--dataset-source", default="gonglinyuan/safim")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-filters", help="JSON file with safim_v3 metadata filters")
    parser.add_argument("--runtimes", help="Pinned v3 Docker runtime registry")


def load_dataset(task, args=None):
    source = getattr(args, "dataset_source", "gonglinyuan/safim")
    split = getattr(args, "dataset_split", "test")
    if Path(source).is_file():
        ds = stream_jsonl(source)
    elif Path(source).is_dir():
        ds = stream_jsonl(str(Path(source) / task / (split + ".jsonl.gz")))
    else:
        import datasets
        ds = datasets.load_dataset(source, task, split=split, revision=getattr(args, "dataset_revision", None))
    lst = []
    for m in ds:
        m = dict(m)
        for key in ("unit_tests", "metadata", "limits", "mask"):
            if isinstance(m.get(key), str):
                m[key] = json.loads(m[key])
        if m.get("task_type", task) != task or m.get("split", split) != split:
            continue
        lst.append(m)
    if getattr(args, "dataset_filters", None):
        from safim_v3.release import select
        with open(args.dataset_filters, encoding="utf-8") as f:
            lst = select(lst, json.load(f))
    return lst
