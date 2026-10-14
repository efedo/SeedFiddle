"""Single-entry CLI routing and bounded adapters; GPU producers tested separately."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import seed_vision as launch
from seedvision.benchmarking import benchmark_cli as cli
from seedvision.bootstrap import BootstrapError
from tests import test_grouped_comparison as fixtures


class BenchmarkCLITests(unittest.TestCase):
    def comparison(self, root, *, classifier=False):
        study = fixtures.manifest()
        plan = fixtures.plan(study)
        if classifier:
            plan = replace(plan, candidate=replace(plan.candidate,
                model_sha256=fixtures.digest('model'), training_evidence_sha256=fixtures.digest('training')))
        rows = fixtures.records(study, plan)
        path = root / 'input.json'
        path.write_text(json.dumps({'manifest': study.to_dict(), 'plan': plan.to_dict(),
                                   'records': [r.to_dict() for r in rows]}), encoding='utf-8')
        return path, plan

    def evidence(self, root):
        path = root / 'evidence.json'
        path.write_text(json.dumps({'bindings': [{'method_id': 'candidate', 'family': 'naive_bayes',
            'model_path': 'model.txt', 'training_path': 'training.json'}]}), encoding='utf-8')
        return path

    def test_benchmark_actions_are_exclusive_with_each_other_and_learning(self):
        for args in (['--benchmark-comparison', 'x', '--benchmark-resources', 'y'],
                     ['--benchmark-resources', 'x', '--learning-audit', 'y']):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                launch.build_parser().parse_args(args)

    def test_output_and_evidence_arguments_fail_before_runtime_bootstrap(self):
        for args in (['--benchmark-comparison', 'x'], ['--benchmark-output', 'out'],
                     ['--benchmark-resources', 'x', '--benchmark-output', 'out', '--classifier-evidence', 'e']):
            with (self.subTest(args=args), patch.object(launch, 'load_configuration') as configuration,
                    redirect_stdout(StringIO()), redirect_stderr(StringIO())):
                self.assertEqual(launch.main(args), 2)
                configuration.assert_not_called()

    def test_bootstrap_forwarding_preserves_paths_and_benchmark_arguments(self):
        original = ['--benchmark-comparison', 'inputs with space.json', '--benchmark-output', 'out with space.json',
                    '--classifier-evidence', 'evidence.json']
        args = launch.build_parser().parse_args(original)
        self.assertEqual(launch._learning_arguments(args), original)
        with patch.object(launch.subprocess, 'call', return_value=0) as call:
            self.assertEqual(launch._run_with_interpreter('python.exe', offline=True, bootstrap='never',
                forwarded_arguments=launch._learning_arguments(args)), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[-len(original):], original)
        self.assertEqual(command.count(str(launch.ROOT / 'seed_vision.py')), 1)

    def test_ready_runtime_dispatches_real_comparison_without_launching_desktop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, plan = self.comparison(root)
            output = root / 'report.json'
            with (patch.object(launch, 'load_configuration', return_value=object()),
                    patch.object(launch, 'inspect_runtime', return_value=SimpleNamespace(ready=True)),
                    patch.object(launch, '_launch_application') as desktop, redirect_stdout(StringIO())):
                self.assertEqual(launch.main(['--benchmark-comparison', str(source), '--benchmark-output', str(output)]), 0)
            desktop.assert_not_called()
            report = json.loads(output.read_bytes())
            self.assertEqual(report['state'], 'pass')
            self.assertEqual(report['plan_sha256'], plan.plan_sha256)
            self.assertFalse(report['scientifically_validated'])

    def test_adapter_errors_preserve_bootstrap_error_cause(self):
        args = launch.build_parser().parse_args(['--benchmark-resources', 'bad.json', '--benchmark-output', 'out.json'])
        original = ValueError('invalid recipe')
        with patch.object(cli, 'run_resources', side_effect=original), self.assertRaises(BootstrapError) as caught:
            launch._run_learning_action(args)
        self.assertIs(caught.exception.__cause__, original)

    def test_classifier_claims_require_typed_evidence_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.comparison(root, classifier=True)
            destination = root / 'report.json'
            destination.write_bytes(b'previous report')
            with self.assertRaisesRegex(ValueError, 'classifier-evidence'):
                cli.run_comparison(source, destination)
            self.assertEqual(destination.read_bytes(), b'previous report')

    def test_exact_eligible_binding_attaches_receipt_without_rewriting_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, plan = self.comparison(root, classifier=True)
            evidence = self.evidence(root)
            binder = Mock(return_value=(plan.candidate, json.dumps({'evaluation_eligible': True, 'scientifically_validated': False})))
            with patch.dict(sys.modules, {'seedvision.benchmarking.classifier_evaluation': SimpleNamespace(bind_classifier_evaluation=binder)}):
                result = cli.run_comparison(source, root / 'report.json', classifier_evidence=evidence)
            self.assertEqual(result['plan'], plan.to_dict())
            self.assertEqual(result['plan_sha256'], plan.plan_sha256)
            self.assertTrue(result['classifier_evidence']['receipts'][0]['receipt']['evaluation_eligible'])
            self.assertFalse(result['scientifically_validated'])
            self.assertEqual(binder.call_args.kwargs['model_path'], root / 'model.txt')
            self.assertEqual(binder.call_args.kwargs['training_path'], root / 'training.json')
            self.assertEqual(binder.call_args.kwargs['training_access_ids'], plan.candidate.training_access_ids)

    def test_unknown_training_or_binding_mismatch_preserves_prior_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, plan = self.comparison(root, classifier=True)
            evidence = self.evidence(root)
            destination = root / 'report.json'
            destination.write_bytes(b'old')
            for bound, receipt in ((plan.candidate, {'evaluation_eligible': False, 'blockers': ['unknown_membership']}),
                                   (replace(plan.candidate, model_sha256=fixtures.digest('different')), {'evaluation_eligible': True, 'scientifically_validated': False})):
                binder = Mock(return_value=(bound, json.dumps(receipt)))
                with self.subTest(receipt=receipt), patch.dict(sys.modules, {
                    'seedvision.benchmarking.classifier_evaluation': SimpleNamespace(bind_classifier_evaluation=binder)}), self.assertRaises(ValueError):
                    cli.run_comparison(source, destination, classifier_evidence=evidence)
                self.assertEqual(destination.read_bytes(), b'old')

    def test_evidence_missing_second_classifier_or_duplicate_method_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, plan = self.comparison(root, classifier=True)
            payload = json.loads(source.read_bytes())
            both = replace(plan, reference=replace(plan.reference, model_sha256=fixtures.digest('reference model')))
            payload['plan'] = both.to_dict()
            payload['records'] = [r.to_dict() for r in fixtures.records(fixtures.manifest(), both)]
            source.write_text(json.dumps(payload), encoding='utf-8')
            evidence = self.evidence(root)
            binder = Mock(return_value=(both.candidate, json.dumps({'evaluation_eligible': True, 'scientifically_validated': False})))
            with patch.dict(sys.modules, {'seedvision.benchmarking.classifier_evaluation': SimpleNamespace(bind_classifier_evaluation=binder)}), self.assertRaisesRegex(ValueError, 'omits'):
                cli.run_comparison(source, root / 'out.json', classifier_evidence=evidence)
            supplied = json.loads(evidence.read_bytes())
            supplied['bindings'].append(supplied['bindings'][0])
            evidence.write_text(json.dumps(supplied), encoding='utf-8')
            with patch.dict(sys.modules, {'seedvision.benchmarking.classifier_evaluation': SimpleNamespace(bind_classifier_evaluation=binder)}), self.assertRaisesRegex(ValueError, 'duplicate'):
                cli.run_comparison(source, root / 'out.json', classifier_evidence=evidence)

    def test_resource_controls_and_relative_assets_reach_existing_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'resource.json'
            source.write_text(json.dumps({'recipe': {'selected': 'native'}, 'source': 'image.png', 'repeats': 2,
                'warmups': 1, 'modes': ['cold', 'warm'], 'device': 'cuda:0', 'maximum_elapsed_seconds': 5.,
                'maximum_peak_bytes': 1024, 'model_asset_paths': {'asset': 'model.dat'}, 'correction_records': []}), encoding='utf-8')
            recipe = object()
            benchmark = Mock(return_value={'status': 'unavailable', 'scientifically_validated': False})
            exporter = Mock()
            with patch.dict(sys.modules, {
                'seedvision.persistence.native_recipes': SimpleNamespace(native_recipes_from_payload=Mock(return_value=(recipe,))),
                'seedvision.benchmarking.performance': SimpleNamespace(benchmark_native_recipe=benchmark, export_performance_report=exporter)}):
                result = cli.run_resources(source, root / 'out.json')
            self.assertEqual(benchmark.call_args.args, (recipe, root / 'image.png'))
            self.assertEqual(benchmark.call_args.kwargs, {'repeats': 2, 'warmups': 1, 'modes': ('cold', 'warm'),
                'device': 'cuda:0', 'maximum_elapsed_seconds': 5., 'maximum_peak_bytes': 1024,
                'model_asset_paths': {'asset': root / 'model.dat'}, 'correction_records': ()})
            exporter.assert_called_once_with(result, root / 'out.json')

    def test_original_reference_resource_recipe_has_no_native_substitution(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        recipe = next(i for i in build_recipe_catalogue().recipes if i.recipe_id == 'plantcv_seed_binary')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'resource.json'
            source.write_text(json.dumps({'recipe': recipe.to_dict(), 'source': 'image.png'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                cli.run_resources(source, root / 'out.json')
            self.assertFalse((root / 'out.json').exists())

    def test_resource_malformed_controls_and_duplicate_nonfinite_config_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'resource.json'
            cases = [dict(recipe={}, source='x', cancel=True), dict(recipe={}, source='x', device='cpu'),
                     dict(recipe={}, source='x', repeats=True), dict(recipe={}, source='x', modes=['cold', 'cold'])]
            raw = [json.dumps(i).encode() for i in cases] + [b'{"recipe":{},"source":"x","source":"y"}',
                b'{"recipe":{},"source":"x","maximum_elapsed_seconds":1e999}']
            for data in raw:
                source.write_bytes(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    cli.run_resources(source, root / 'out.json')
                self.assertFalse((root / 'out.json').exists())
            with patch.object(cli, 'MAX_CONFIG_BYTES', 1), self.assertRaisesRegex(ValueError, 'byte limit'):
                cli.run_resources(source, root / 'out.json')

    def test_comparison_receipt_publication_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, plan = self.comparison(root, classifier=True)
            evidence = self.evidence(root)
            destination = root / 'report.json'
            destination.write_bytes(b'old')
            binder = Mock(return_value=(plan.candidate, json.dumps({'evaluation_eligible': True, 'scientifically_validated': False})))
            with (patch.dict(sys.modules, {'seedvision.benchmarking.classifier_evaluation': SimpleNamespace(bind_classifier_evaluation=binder)}),
                    patch.object(cli.os, 'replace', side_effect=OSError('disk failure')), self.assertRaises(OSError)):
                cli.run_comparison(source, destination, classifier_evidence=evidence)
            self.assertEqual(destination.read_bytes(), b'old')
            self.assertEqual(list(root.glob('*.tmp')), [])
