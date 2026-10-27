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


V3_CONFIGS = {"block_v3": "block", "control_v3": "control"}
V3_REPOSITORY = "averoo/safim_v3"
V3_REVISION = "70edeef93658085e8ab905e5688996d098b8172e"


def add_dataset_arguments(parser):
    parser.add_argument("--dataset-source", help="HF repository, local snapshot directory, or JSONL file; defaults by config")
    parser.add_argument("--dataset-revision", help="HF revision; published original/v3 snapshots are pinned by default")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-filters", help="JSON file containing safim_v3 metadata filters")
    parser.add_argument("--runtimes", help="Pinned v3 Docker runtime registry (required for v3 evaluation)")


def dataset_sha256(samples):
    """Hash the canonical row list without materializing one enormous JSON string."""
    digest = hashlib.sha256(b"[")
    for index, sample in enumerate(samples):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(sample, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_configured_dataset(task, args):
    """Hydrate v3 metadata while preserving the fixed harness's input identity."""
    is_v3 = task in V3_CONFIGS
    source = getattr(args, "dataset_source", None) or (V3_REPOSITORY if is_v3 else "gonglinyuan/safim")
    split = getattr(args, "dataset_split", "test")
    revision = getattr(args, "dataset_revision", None)
    local = Path(source).is_file() or Path(source).is_dir()
    if split != "test":
        raise ValueError("SAFIM configurations only provide the test split")
    if not local and revision is None:
        revision = {V3_REPOSITORY: V3_REVISION, "gonglinyuan/safim": DATASET_REVISION}.get(source)
    if not local:
        # Resolve mutable user-supplied revisions once, before fetching any file.
        if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info(source, revision=revision).sha
    else:
        revision = None
    files = {}
    if Path(source).is_file():
        ds = stream_jsonl(source)
        files[Path(source).name] = file_sha256(source)
    elif is_v3:
        from safim_v3.hf_dataset import load_snapshot, resolve_file
        ds = load_snapshot(source, task, split, revision)
        for filename in (task + "/test.parquet", "metadata/items.jsonl.gz"):
            files[filename] = file_sha256(resolve_file(source, filename, revision))
    else:
        if task not in DATASET_FILES:
            raise ValueError(f"Unknown SAFIM configuration: {task}")
        filename = DATASET_FILES[task]
        path = Path(source) / filename if local else Path(hf_hub_download(source, filename, repo_type="dataset", revision=revision))
        ds = stream_jsonl(str(path))
        files[filename] = file_sha256(path)
    rows = []
    for row in ds:
        row = dict(row)
        for key in ("unit_tests", "metadata", "limits", "mask"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
        if row.get("task_type", V3_CONFIGS.get(task, task)) != V3_CONFIGS.get(task, task) or row.get("split", split) != split:
            continue
        if is_v3 and (row.get("schema_version") not in (1, 2) or not row.get("runtime") or not row.get("unit_tests")):
            raise ValueError("SAFIM v3 requires runtime metadata and nonempty executable tests")
        rows.append(row)
    filters = None
    if getattr(args, "dataset_filters", None):
        from safim_v3.release import select
        filters = json.loads(Path(args.dataset_filters).read_text(encoding="utf-8"))
        rows = select(rows, filters)
    identity = dict(files=files, filters=filters, task_ids=[row["task_id"] for row in rows])
    return DatasetRows(rows, dict(repository=str(Path(source).resolve()) if local else source,
        revision=revision, configuration=task, split=split, files=files, filters=filters,
        sha256=hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        hash_method="sha256-files-filters-ordered-task-ids-v1", rows=len(rows)))


def load_dataset(task, args=None):
    if task in V3_CONFIGS or (args is not None and (
        getattr(args, "dataset_source", None) or getattr(args, "dataset_revision", None)
        or getattr(args, "dataset_filters", None) or getattr(args, "dataset_split", "test") != "test")):
        return load_configured_dataset(task, args)
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
