"""Runner failure reporting, exhaustive tier partition and fixture lifecycle."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import selection
from tests.__main__ import TimedSuite
from tests.prerequisites import check_prerequisites


ROOT = Path(__file__).resolve().parents[1]


class TestRunnerTests(unittest.TestCase):
    def test_parallel_assignment_partitions_once_and_result_merge_cannot_hide_missing_or_failed_units(self):
        from tests.parallel import assignments,merge
        class Item:
            def __init__(self,name):self.name=name
            def id(self):return self.name
        groups={'cpu-qt':[Item('tests.desktop.Case.test_flow')],
            'cuda':[Item('tests.large.Case.test_a'),Item('tests.large.Other.test_b'),
                    Item('tests.large.Case.test_c'),Item('tests.small.Case.test_math'),
                    Item('tests.graph.Case.test_flow')]}
        expected={item.id() for group in groups.values() for item in group}
        large={'tests.large.Case.test_a','tests.large.Other.test_b','tests.large.Case.test_c'}
        for jobs in (2,3):
            buckets=assignments(groups,jobs)
            self.assertEqual(len(buckets),jobs)
            self.assertEqual({name for bucket in buckets for name in bucket},expected)
            self.assertEqual(sum(map(len,buckets)),len(expected))
            self.assertEqual(buckets[0],['tests.desktop.Case.test_flow'])
            # Each module, including all of its classes, stays whole on one worker.
            self.assertTrue(any(large<=set(bucket) for bucket in buckets[1:]))
            reports=[dict(tests=len(ids),passed=True,selection={'selected_ids':ids}) for ids in buckets]
            result=merge(buckets,reports,[0]*jobs)
            self.assertTrue(result['passed']);self.assertTrue(result['all_discovered_units_accounted_for'])
            self.assertEqual(result['tests'],len(expected))
            broken=[*reports];broken[-1]=None
            self.assertFalse(merge(buckets,broken,[0]*jobs)['passed'])
            broken[-1]={**reports[-1],'errors':['numerical regression']}
            self.assertFalse(merge(buckets,broken,[0]*jobs)['passed'])
            broken[-1]={**reports[-1],'tests':len(buckets[-1])-1}
            self.assertFalse(merge(buckets,broken,[0]*jobs)['passed'])
        # The large module goes to one worker, the rest to the other.
        self.assertEqual(sorted(map(len,assignments(groups,3)[1:])),[2,3])
        with self.assertRaisesRegex(ValueError,'exactly once'):
            assignments({**groups,'cpu-qt':groups['cpu-qt']*2},3)

    def test_stall_watchdog_names_the_unit_a_worker_journal_left_open(self):
        from tests.parallel import last_started_unit
        with tempfile.TemporaryDirectory() as directory:
            journal=Path(directory)/'worker.journal.jsonl'
            self.assertIsNone(last_started_unit(journal))
            rows=[dict(id='tests.a.Case.test_one',event='start'),
                  dict(id='tests.a.Case.test_one',event='stop',seconds=1.0,failures=0,errors=0,skips=0)]
            journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertIsNone(last_started_unit(journal))
            journal.write_text(journal.read_text()+json.dumps(dict(id='tests.a.Case.test_hang',event='start'))+'\n')
            self.assertEqual(last_started_unit(journal),'tests.a.Case.test_hang')

    def test_functional_units_own_scenario_state_and_report_failure_skip_and_cleanup(self):
        from tests.functional_units import functional_units
        events = []
        @functional_units
        class Witness(unittest.TestCase):
            def setUp(self):
                self.state = []
                events.append(('setup', self))
                self.addCleanup(events.append, ('cleanup', self))
            def tearDown(self):
                events.append(('teardown', self))
            def check_pass(self):
                self.assertEqual(self.state, [])
                self.state.append(1)
            def check_failure(self):
                self.assertEqual(self.state, [])
                self.fail('numerical witness')
            def check_skip(self):
                self.skipTest('explicit platform witness')
            def check_cleanup_error(self):
                def fail_cleanup(): raise RuntimeError('cleanup witness')
                self.addCleanup(fail_cleanup)
            def test_unit(self):
                self.scenario(self.check_pass)
                self.scenario(self.check_failure)
                self.scenario(self.check_skip)
                self.scenario(self.check_cleanup_error)
                self.scenario(self.check_pass)
        result = unittest.TextTestRunner(stream=io.StringIO()).run(Witness('test_unit'))
        self.assertEqual((result.testsRun,len(result.failures),len(result.errors),len(result.skipped)), (1,1,1,1))
        self.assertFalse(result.wasSuccessful())
        self.assertIn('numerical witness',result.failures[0][1])
        self.assertIn('cleanup witness',result.errors[0][1])
        self.assertEqual([kind for kind,_ in events], ['setup','teardown','cleanup']*5)
        self.assertEqual(len({id(case) for kind,case in events if kind=='setup'}),5)
        # Other test modules also use TestCase objects as explicit fixtures.
        fixture=Witness('test_unit');fixture.setUp()
        self.assertEqual(fixture.state,[])
        fixture.tearDown();fixture.doCleanups()
        self.assertEqual([kind for kind,_ in events[-3:]],['setup','teardown','cleanup'])

    def test_compilation_fixture_keys_exact_bytes_excludes_mutable_ast_and_restores(self):
        import ast, builtins
        from tests.compilation_fixture import compilation_fixture
        original = builtins.compile
        with self.assertRaisesRegex(RuntimeError, 'body witness'):
            with compilation_fixture() as cache:
                first=compile('value=1','fixture.py','exec',dont_inherit=True)
                self.assertIs(first,compile('value=1','fixture.py','exec',dont_inherit=True))
                changed=compile('value=2','fixture.py','exec',dont_inherit=True)
                scope={};exec(changed,scope);self.assertEqual(scope['value'],2)
                tree=ast.parse('value=3');scope={};exec(compile(tree,'fixture.py','exec'),scope)
                self.assertEqual(scope['value'],3)
                tree.body[0].value.value=4
                scope={};exec(compile(tree,'fixture.py','exec'),scope)
                self.assertEqual(scope['value'],4)
                self.assertEqual(cache.cache_info().hits,1)
                raise RuntimeError('body witness')
        self.assertIs(builtins.compile,original)

    def test_local_observer_keeps_real_frames_returns_c_calls_and_restores_on_error(self):
        from tests.observation import observe_functions
        seen = []
        def unrelated():
            return 99
        def target(value):
            unrelated()
            return len(value)
        def observe(frame, event, arg):
            seen.append((frame.f_code, event, arg, frame.f_locals.get('value')))
        prior = sys.getprofile()
        tools = [sys.monitoring.get_tool(i) for i in range(6)] if hasattr(sys, 'monitoring') else None
        with self.assertRaisesRegex(ValueError, 'body failure'):
            with observe_functions(observe, target, c_calls=(target,)):
                self.assertEqual(target([1, 2]), 2)
                from threading import Thread
                worker = Thread(target=target, args=([3],))
                worker.start()
                worker.join()
                raise ValueError('body failure')
        self.assertIs(sys.getprofile(), prior)
        if tools is not None:
            self.assertEqual([sys.monitoring.get_tool(i) for i in range(6)], tools)
        self.assertTrue(all(row[0] is target.__code__ for row in seen))
        self.assertEqual([row[2] for row in seen if row[1] == 'return'], [2])
        self.assertTrue(any(row[1] == 'c_call' and row[2] is len for row in seen))
        count = len(seen)
        target([3])
        self.assertEqual(len(seen), count)

    def test_invalid_module_exits_before_execution_and_reports_discovery_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as directory:
            report = Path(directory) / 'invalid.json'
            process = subprocess.run([sys.executable, '-m', 'tests',
                'tests.intentionally_missing_runner_smoke', '--report', str(report)],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 2, process.stderr)
            record = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(record['tests'], 0)
            self.assertFalse(record['passed'])
            self.assertTrue(record['discovery_errors'])
            self.assertEqual(record['test_durations'], [])
            self.assertIn('no tests executed', process.stderr)
            process = subprocess.run([sys.executable, '-m', 'tests', '--list', '--preflight'],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn('cannot be combined', process.stderr)
            self.assertNotIn('prerequisites passed', process.stdout)
            manifest=Path(directory)/'worker-manifest.json'
            manifest.write_text(json.dumps(['tests.intentionally_missing_runner_smoke']))
            worker_report=Path(directory)/'worker.json';journal=Path(directory)/'worker.jsonl'
            process=subprocess.run([sys.executable,'-m','tests.worker',str(manifest),str(worker_report),str(journal)],
                cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(process.returncode,2,process.stderr)
            from tests.parallel import merge
            child=json.loads(worker_report.read_text())
            merged=merge([['tests.intentionally_missing_runner_smoke']],[child],[process.returncode])
            self.assertFalse(merged['passed'])
            self.assertFalse(merged['all_discovered_units_accounted_for'])

    def test_worker_ends_with_its_code_and_flushed_output_but_no_interpreter_shutdown(self):
        script=("import atexit\nfrom tests.worker import end_without_shutdown\n"
            "atexit.register(print,'interpreter shutdown ran')\n"
            "print('flushed',end='')\nend_without_shutdown(7)\n")
        process=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,7,process.stderr)
        self.assertEqual(process.stdout,'flushed')

    @unittest.skipUnless(sys.platform=='win32','The per-test memory limit is enforced on Windows only.')
    def test_memory_limit_fails_an_oversized_allocation_inside_its_own_test(self):
        # A child process, so the cap this runner already set does not interfere.
        script=("import numpy as np\nfrom tests.memory_limit import install\n"
            "limit=install(0.25)\n"
            "kept=np.ones(200*2**20,np.uint8)\n"
            "try:\n    np.ones(200*2**20,np.uint8)\nexcept MemoryError:\n    print('refused')\n"
            "limit.start_test()\n"
            "more=np.ones(200*2**20,np.uint8)\n"
            "print('next test',limit.peak()>=400*2**20)\n")
        process=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,0,process.stderr)
        # 200 MiB fits a 0.25 GiB budget; 200 more do not. The next test's
        # budget counts from what the process holds when it starts.
        self.assertEqual(process.stdout.split('\n')[:2],['refused','next test True'])

    def test_partition_is_complete_and_fixture_timings_preserve_lifecycle(self):
        class Item:
            def __init__(self, identifier): self.identifier = identifier
            def id(self): return self.identifier
        items = [Item('sample.Cpu.test_a'), Item('sample.Cuda.test_b'), Item('sample.Other.test_c')]
        with patch.object(selection, 'CPU_QT_TARGETS', ('sample.Cpu', 'sample.Cuda')), patch.object(selection, 'CUDA_TARGETS', ('sample.Cuda',)):
            groups = selection.partition(items)
            self.assertEqual({k:len(v) for k,v in groups.items()}, {'cpu-qt':1, 'cuda':2})
            self.assertEqual({t.id() for v in groups.values() for t in v}, {t.id() for t in items})
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                selection.partition(items + items[:1])
            with self.assertRaisesRegex(ValueError, 'resolves to no'):
                selection.partition(items[1:])
        events = []
        class Witness(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                events.append('setup')
                cls.addClassCleanup(events.append, 'cleanup')
            def test_a(self): events.append('a')
            def test_b(self): events.append('b')
            @classmethod
            def tearDownClass(cls): events.append('teardown')
        suite = TimedSuite([TimedSuite([Witness('test_a')]), TimedSuite([Witness('test_b')])])
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(events, ['setup','a','b','teardown','cleanup'])
        self.assertEqual([row['fixture'] for row in result.fixture_durations], ['setUpClass','tearDownClass/cleanups'])
        self.assertTrue(all(row['seconds'] >= 0 for row in result.fixture_durations))

    def test_windows_left_by_a_test_are_deleted_but_class_fixture_windows_last_the_class(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication, QWidget
        from shiboken6 import isValid
        from tests.functional_units import functional_units
        application = QApplication.instance() or QApplication([])
        seen = {}
        def self_retaining_window(name):
            # The timer's connection keeps a closed window alive with no Python
            # reference left, which is what leaks a real MainWindow.
            window = seen[name] = QWidget()
            seen[name + '-child'] = QWidget(window)
            QTimer(window).timeout.connect(lambda: window)
            window.close()
        class Witness(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                self_retaining_window('class')
            def test_a(self):
                self_retaining_window('a')
            def test_b(self):
                self.assertTrue(isValid(seen['class']))
                self.assertFalse(isValid(seen['a']))
                self.assertFalse(isValid(seen['a-child']))
                self_retaining_window('b')
        @functional_units
        class Later(unittest.TestCase):
            def check_first(self):
                self.assertFalse(isValid(seen['class']))
                self.assertFalse(isValid(seen['b']))
                self_retaining_window('first')
            def check_second(self):
                self.assertFalse(isValid(seen['first']))
                pending = seen['pending'] = QWidget()
                pending.deleteLater()
                application.processEvents()
                self.assertTrue(isValid(pending))
            def check_third(self):
                self.assertFalse(isValid(seen['pending']))
            def test_unit(self):
                self.scenario(self.check_first)
                self.scenario(self.check_second)
                self.scenario(self.check_third)
        suite = TimedSuite([Witness('test_a'), Witness('test_b'), Later('test_unit')])
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)

    def test_prerequisites_require_cuda_only_for_full_and_cuda_runs(self):
        with patch('torch.cuda.is_available', return_value=False):
            for mode in ('core', 'cpu-qt', 'focused'):
                self.assertTrue(check_prerequisites(mode)['passed'], mode)
            for mode in ('full', 'cuda'):
                failed = check_prerequisites(mode)
                self.assertFalse(failed['passed'])
                self.assertIn('CUDA is required', failed['errors'][0])
