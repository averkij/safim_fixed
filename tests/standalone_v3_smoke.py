"""Small clone-local v3 contract test; run with python -I, optionally --docker."""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safim_v3
from safim_v3 import hf_dataset, metrics, release, runner, storage, syntax
import data_utils
import evaluate
import exec_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker', action='store_true')
    args = parser.parse_args()
    assert not syntax.parse_tree('print(1)\n', 'python').root_node.has_error
    manifest = json.loads((ROOT / 'safim_v3/VENDORED.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for path in (ROOT / 'safim_v3/checker_versions').glob('*.py'):
        assert runner.checker_path([dict(checker='fixture', checker_sha256=path.stem)]) == path

    import pyarrow as pa
    import pyarrow.parquet as pq
    configs = ['block_v3', 'control_v3']
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder / 'metadata').mkdir()
        metadata = []
        tests = json.dumps([dict(input='', output=['YES'], comparison='ascii_case_insensitive_tokens')])
        for config in configs:
            row = dict(task_id='fixture_' + config, lang='python', prompt='# TODO: Your code here',
                       eval_prompt='{{completion}}\n', ground_truth='print("yes")', unit_tests=tests)
            meta = dict(task_id=row['task_id'], lang='python', task_type=config.split('_')[0],
                        split='test', schema_version=2, runtime='python310', limits={'time_seconds': 2},
                        source_origin='human', family_id='fixture', problem_id='fixture',
                        unit_tests_sha256=hashlib.sha256(tests.encode()).hexdigest())
            metadata.append(meta)
            (folder / config).mkdir()
            pq.write_table(pa.Table.from_pylist([row]), folder / config / 'test.parquet')
        with gzip.open(folder / 'metadata/items.jsonl.gz', 'wt', encoding='utf-8') as output:
            for row in metadata:
                output.write(json.dumps(row) + '\n')
        for config in configs:
            rows = data_utils.load_dataset(config, argparse.Namespace(dataset_source=str(folder), dataset_split='test'))
            assert len(rows) == 1
            row = rows[0]
            assert row['runtime'] == 'python310'
            assert release.select(rows, {'lang': 'python'}) == rows
            assert release.select(rows, {'lang': 'java'}) == []
            assert metrics.summarize(rows, [dict(task_id=row['task_id'], passed=True)])['groups']['all/all']['passed'] == 1
            predictions, output = folder / 'predictions.jsonl', folder / 'results.jsonl'
            predictions.write_text(json.dumps(dict(task_id=row['task_id'], completion=row['ground_truth'])) + '\n')
            sys.argv = ['evaluate', config, str(predictions), str(output), '--dataset-source', str(folder),
                        '--runtimes', str(ROOT / 'configs/runtimes.v3.json')]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluate.main()
            assert json.loads(output.read_text())['passed'] is True
            if args.docker:
                # Deliberately non-exact answer: CLI must execute, not use reference matching.
                predictions.write_text(json.dumps(dict(task_id=row['task_id'], completion='print("YES")')) + '\n')
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    evaluate.main()
                executed = json.loads(output.read_text())
                assert executed['passed'] is True
                assert isinstance(executed['result'], list)
            results[config] = {'rows': len(rows), 'exact_reference': 'passed', 'docker_nonexact': args.docker}
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == 'safim_v3' or name.startswith('safim_v3.'):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(ROOT), (name, str(path))
            origins[name] = str(path.relative_to(ROOT))
    print(json.dumps(dict(status='passed', isolated_python=bool(sys.flags.isolated),
                         configs=results, verified_files=len(manifest['files_sha256']), imports=origins), indent=2))


if __name__ == '__main__':
    main()
