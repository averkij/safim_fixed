"""V3 integration: real metadata hydration, fixed prompts and execution contracts."""
import contextlib
import gzip
import hashlib
import io
import json
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from parser_support import load_prompt
from source_support import source_at
from test_api_scoring import load_evaluator


def module_from(name):
    module = types.ModuleType(name)
    exec(compile(source_at(name + '.py'), name + '.py', 'exec'), module.__dict__)
    return module


def snapshot(folder, config='block_v3'):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tests = json.dumps([dict(input='', output=['1'], checker='fixture',
                            checker_sha256='a' * 64, comparison='ascii_case_insensitive_tokens')])
    row = dict(task_id='v3_fixture', lang='python', prompt='# TODO: Your code here',
               eval_prompt='{{completion}}', ground_truth='print(1)', unit_tests=tests)
    meta = dict(task_id=row['task_id'], lang='python', task_type=config.split('_')[0],
                split='test', schema_version=2, runtime='python310', limits={'time_seconds': 2},
                source_origin='human', unit_tests_sha256=hashlib.sha256(tests.encode()).hexdigest())
    (folder / config).mkdir()
    (folder / 'metadata').mkdir()
    pq.write_table(pa.Table.from_pylist([row]), folder / config / 'test.parquet')
    with gzip.open(folder / 'metadata/items.jsonl.gz', 'wt', encoding='utf-8') as output:
        output.write(json.dumps(meta) + '\n')
    return row, meta


class V3LoadingTests(unittest.TestCase):
    def test_both_local_configs_restore_execution_semantics(self):
        data = module_from('data_utils')
        for config in ['block_v3', 'control_v3']:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as folder:
                snapshot(Path(folder), config)
                rows = data.load_dataset(config, Namespace(dataset_source=folder))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['runtime'], 'python310')
                self.assertEqual(rows[0]['limits'], {'time_seconds': 2})
                self.assertEqual(rows[0]['unit_tests'][0]['checker_sha256'], 'a' * 64)
                self.assertEqual(rows[0]['unit_tests'][0]['comparison'], 'ascii_case_insensitive_tokens')
                self.assertEqual(rows.provenance['configuration'], config)
                self.assertEqual(rows.provenance['sha256'], data.load_dataset(config, Namespace(dataset_source=folder)).provenance['sha256'])

    def test_remote_default_is_pinned_and_metadata_is_hydrated(self):
        data = module_from('data_utils')
        with tempfile.TemporaryDirectory() as folder:
            snapshot(Path(folder))
            def fetch(repo, filename, **kwargs):
                self.assertEqual(repo, 'averoo/safim_v3')
                self.assertEqual(kwargs, dict(repo_type='dataset', revision=data.V3_REVISION))
                return str(Path(folder) / filename)
            with patch('huggingface_hub.hf_hub_download', side_effect=fetch) as download:
                rows = data.load_dataset('block_v3')
            self.assertEqual(rows.provenance['revision'], data.V3_REVISION)
            self.assertGreaterEqual(download.call_count, 2)

    def test_modified_test_payload_is_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        data = module_from('data_utils')
        with tempfile.TemporaryDirectory() as folder:
            row, _ = snapshot(Path(folder))
            row['unit_tests'] = '[]'
            pq.write_table(pa.Table.from_pylist([row]), Path(folder) / 'block_v3/test.parquet')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                data.load_dataset('block_v3', Namespace(dataset_source=folder))

    def test_identity_captures_runtime_changes_and_is_checked(self):
        data = module_from('data_utils')
        evaluator = load_evaluator(source_at('evaluate.py'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, meta = snapshot(root)
            before = data.load_dataset('block_v3', Namespace(dataset_source=folder)).provenance
            meta['runtime'] = 'different-runtime'
            with gzip.open(root / 'metadata/items.jsonl.gz', 'wt', encoding='utf-8') as output:
                output.write(json.dumps(meta) + '\n')
            after = data.load_dataset('block_v3', Namespace(dataset_source=folder)).provenance
            self.assertNotEqual(before['sha256'], after['sha256'])
            prediction = root / 'prediction.jsonl'
            Path(str(prediction) + '.run.json').write_text(json.dumps(dict(dataset=before)))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                evaluator.validate_input_variant(prediction, 'block_v3', after)


class V3PromptTests(unittest.TestCase):
    def test_aliases_use_fixed_branch_templates_and_context_fitting(self):
        prompt = load_prompt()
        class Wrapper:
            def fit_infilling_prompt(self, prefix, suffix, reverse=False):
                return repr((prefix, suffix, reverse))
        for config, kind in [('block_v3', 'block'), ('control_v3', 'control_fixed')]:
            for lang in ['cpp', 'java', 'python']:
                sample = dict(lang=lang, prompt=prompt.FEW_SHOT_PROMPTS[(kind, lang)], eval_prompt="prefix{{completion}}suffix")
                for mode in ['infilling', 'reverse_infilling', 'left_to_right', 'instructed', 'fewshot', 'prefix_feeding']:
                    with self.subTest(config=config, lang=lang, mode=mode):
                        self.assertEqual(prompt.apply_prompt(sample, config, mode, Wrapper()),
                                         prompt.apply_prompt(sample, kind, mode, Wrapper()))

    def test_python_control_keeps_header_colon_in_context(self):
        prompt = load_prompt()
        sample = dict(lang='python', eval_prompt='if {{completion}}:\n    pass')
        self.assertEqual(prompt.apply_postprocessors('x:', sample, 'control_v3', ['truncate_control']), 'x')
        self.assertEqual(prompt.apply_postprocessors('x:', sample, 'control', ['truncate_control']), 'x:')


class V3ExecutionTests(unittest.TestCase):
    def test_v3_uses_docker_checker_and_limits_without_http(self):
        engine = module_from('exec_utils')
        engine.execeval = Mock()
        engine.docker_runner = Mock()
        engine.docker_runner.run.return_value = [dict(outcome='PASSED', stdout='yes')]
        problem = dict(task_id='v3_one', schema_version=2, runtime='python310',
                       eval_prompt='{{completion}}\n', unit_tests=[dict(checker='semantic')],
                       limits={'memory_mb': 256})
        result, passed = engine.run_test(problem, dict(task_id='v3_one', completion='print(1)'))
        self.assertTrue(passed)
        self.assertEqual(result[0]['exec_outcome'], 'PASSED')
        engine.docker_runner.run.assert_called_once_with('print(1)\n', 'python310', problem['unit_tests'], problem['limits'])
        engine.execeval.execute_code.assert_not_called()

    def test_docker_infrastructure_errors_keep_fixed_failure_contract(self):
        from safim_v3.runner import InfrastructureError
        engine = module_from('exec_utils')
        problem = dict(task_id='v3_one', schema_version=2, runtime='python310', eval_prompt='{{completion}}', unit_tests=[{}])
        engine.docker_runner = Mock()
        for error in [InfrastructureError('unavailable'), FileNotFoundError('docker missing')]:
            engine.docker_runner.run.side_effect = error
            with self.assertRaises(engine.ExecEvalError):
                engine.run_test(problem, dict(task_id='v3_one', completion='pass'))

    def test_runtime_manifest_is_recorded_and_conflicting_profile_rejected(self):
        engine = module_from('exec_utils')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtimes.json'
            raw = json.dumps({'python310': {'image': 'sha256:' + 'a' * 64}}).encode()
            path.write_bytes(raw)
            args = Namespace(completion_type='block_v3', runtimes=str(path))
            result = engine.build_execeval(args)
            self.assertEqual(result['backend'], 'safim_v3_docker')
            self.assertEqual(result['runtimes_sha256'], hashlib.sha256(raw).hexdigest())
            args.exec_profile = 'ignored.json'
            with self.assertRaisesRegex(ValueError, 'exec-profile'):
                engine.build_execeval(args)
            with self.assertRaisesRegex(ValueError, 'requires --runtimes'):
                engine.build_execeval(Namespace(completion_type='control_v3'))

    def test_v3_cli_shortcut_force_execution_and_outage(self):
        for mode in ['shortcut', 'force', 'outage']:
            module = load_evaluator(source_at('evaluate.py'))
            module.load_dataset = lambda *args: [dict(task_id='v3_one', schema_version=2, runtime='python310',
                lang='python', ground_truth='print(1)', eval_prompt='{{completion}}', unit_tests=[{}])]
            module.stream_jsonl = lambda _: [dict(task_id='v3_one', completion='print(1)')]
            module.tqdm = lambda rows: rows
            module.build_execeval = lambda _: {'backend': 'fixture'}
            module.run_test = Mock(return_value=([dict(exec_outcome='WRONG_ANSWER')], False))
            module.check_syntax = Mock(side_effect=AssertionError('v3 must keep worker syntax verdict'))
            if mode == 'outage':
                module.run_test.side_effect = module.ExecEvalError('synthetic outage')
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'results.jsonl'
                argv = ['evaluate', 'block_v3', 'fixture', str(output)]
                if mode != 'shortcut':
                    argv += ['--force-execution']
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if mode == 'outage':
                        with self.assertRaises(SystemExit):
                            module.main()
                    else:
                        module.main()
                row = json.loads(output.read_text())
                manifest = json.loads(Path(str(output) + '.run.json').read_text())
                self.assertEqual(row['passed'], {'shortcut': True, 'force': False, 'outage': None}[mode])
                self.assertEqual(module.run_test.call_count, 0 if mode == 'shortcut' else 1)
                self.assertEqual(manifest['status'], 'incomplete' if mode == 'outage' else 'complete')


if __name__ == '__main__':
    unittest.main()
