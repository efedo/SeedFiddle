"""Single-entry CLI routing and bounded adapters; GPU producers tested separately."""

from tests.functional_units import functional_units
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import seed_vision as launch
from seedvision.benchmarking import benchmark_cli as cli
from seedvision.bootstrap import BootstrapError
from tests import test_grouped_comparison as fixtures


@functional_units
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

    def check_test_benchmark_actions_are_exclusive_with_each_other_and_learning(self):
        for args in (['--benchmark-comparison', 'x', '--learning-audit', 'y'],
                     ['--benchmark-comparison', 'x', '--create-synthetic-learning-data', 'y']):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                launch.build_parser().parse_args(args)

    def check_test_output_and_evidence_arguments_fail_before_runtime_bootstrap(self):
        for args in (['--benchmark-comparison', 'x'], ['--benchmark-output', 'out']):
            with (self.subTest(args=args), patch.object(launch, 'load_configuration') as configuration,
                    redirect_stdout(StringIO()), redirect_stderr(StringIO())):
                self.assertEqual(launch.main(args), 2)
                configuration.assert_not_called()

    def check_test_bootstrap_forwarding_preserves_paths_and_benchmark_arguments(self):
        original = ['--benchmark-comparison', 'inputs with space.json', '--benchmark-output', 'out with space.json']
        args = launch.build_parser().parse_args(original)
        self.assertEqual(launch._learning_arguments(args), original)
        with patch.object(launch.subprocess, 'call', return_value=0) as call:
            self.assertEqual(launch._run_with_interpreter('python.exe', offline=True, bootstrap='never',
                forwarded_arguments=launch._learning_arguments(args)), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[-len(original):], original)
        self.assertEqual(command.count(str(launch.ROOT / 'seed_vision.py')), 1)

    def check_test_ready_runtime_dispatches_real_comparison_without_launching_desktop(self):
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

    def check_test_adapter_errors_preserve_bootstrap_error_cause(self):
        args = launch.build_parser().parse_args(['--benchmark-comparison', 'bad.json', '--benchmark-output', 'out.json'])
        original = ValueError('invalid plan')
        with patch.object(cli, 'run_comparison', side_effect=original), self.assertRaises(BootstrapError) as caught:
            launch._run_learning_action(args)
        self.assertIs(caught.exception.__cause__, original)

    def check_test_classifier_bearing_plans_are_refused_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.comparison(root, classifier=True)
            destination = root / 'report.json'
            destination.write_bytes(b'previous report')
            with self.assertRaisesRegex(ValueError, 'not supported'):
                cli.run_comparison(source, destination)
            self.assertEqual(destination.read_bytes(), b'previous report')

    def test_workflow_lifecycle(self):
        """5 related BenchmarkCLITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_benchmark_actions_are_exclusive_with_each_other_and_learning)
        self.scenario(self.check_test_bootstrap_forwarding_preserves_paths_and_benchmark_arguments)
        self.scenario(self.check_test_ready_runtime_dispatches_real_comparison_without_launching_desktop)
        self.scenario(self.check_test_adapter_errors_preserve_bootstrap_error_cause)
        self.scenario(self.check_test_classifier_bearing_plans_are_refused_before_publication)

    def test_validation_and_recovery(self):
        """1 related BenchmarkCLITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_output_and_evidence_arguments_fail_before_runtime_bootstrap)
