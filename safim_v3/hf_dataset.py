"""Hydrate SAFIM-compatible HF rows with their pinned evaluation metadata."""
import gzip
import hashlib
import json
from pathlib import Path


CONFIGS = {'block_v3': 'block', 'control_v3': 'control'}


def resolve_file(source, filename, revision=None):
    if Path(source).is_dir():
        path = Path(source) / filename
        if not path.is_file():
            raise ValueError('Missing snapshot file: ' + str(path))
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(source, filename, repo_type='dataset', revision=revision))


def load_snapshot(source, config, split='test', revision=None):
    if config not in CONFIGS or split != 'test':
        raise ValueError('Expected block_v3/control_v3 with test split')
    if not Path(source).is_dir() and revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(source).sha
    with gzip.open(resolve_file(source, 'metadata/items.jsonl.gz', revision), 'rt', encoding='utf-8') as f:
        metadata = {}
        for line in f:
            item = json.loads(line)
            if item['task_type'] == CONFIGS[config]:
                if item['task_id'] in metadata:
                    raise ValueError('Duplicate metadata task id')
                metadata[item['task_id']] = item
    # Use Hub's cached whole-file transfer (including Xet acceleration), then
    # iterate bounded local batches. Repeated HTTP range reads were very slow in
    # the publication environment; this changes transport only, never row order.
    import pyarrow.parquet as pq
    def local_rows():
        file = pq.ParquetFile(resolve_file(source, config + '/test.parquet', revision))
        for batch in file.iter_batches(batch_size=1):
            yield from batch.to_pylist()
    rows = local_rows()
    suites, seen = {}, set()
    for row in rows:
        meta = metadata.get(row['task_id'])
        if meta is None or row['task_id'] in seen:
            raise ValueError('Missing or duplicate item metadata: ' + row['task_id'])
        seen.add(row['task_id'])
        if row['lang'] != meta['lang'] or meta['split'] != split or meta['schema_version'] not in (1, 2):
            raise ValueError('Inconsistent item metadata')
        expected = meta['unit_tests_sha256']
        if hashlib.sha256(row['unit_tests'].encode('utf-8')).hexdigest() != expected:
            raise ValueError('Unit test payload checksum mismatch')
        if expected not in suites:
            suites[expected] = json.loads(row['unit_tests'])
        yield {**meta, **row, 'unit_tests': suites[expected]}
    if seen != set(metadata):
        raise ValueError('Snapshot data and metadata contain different task ids')
