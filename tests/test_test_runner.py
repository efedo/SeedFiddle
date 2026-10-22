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
        from tests.parallel import assignments,merge,BALANCED_NATIVE_MODULES
        class Item:
            def __init__(self,name):self.name=name
            def id(self):return self.name
        groups={'cpu-qt':[Item('tests.desktop.Case.test_flow')],
            'cuda':[Item('tests.test_plantcv_size.Case.test_math'),Item('tests.graph.Case.test_flow')],
            'acceptance':[Item('tests.test_recipe_acceptance.Case.test_matrix')]}
        moved={module+'.'+cls+'.test_flow' for module in BALANCED_NATIVE_MODULES
               for cls in ('ContractTests','CalculationTests')}
        groups['cuda'].extend(Item(name) for name in sorted(moved))
        expected={item.id() for group in groups.values() for item in group}
        for jobs in (2,3):
            buckets=assignments(groups,jobs)
            self.assertEqual(len(buckets),jobs)
            self.assertEqual({name for bucket in buckets for name in bucket},expected)
            self.assertEqual(sum(map(len,buckets)),len(expected))
            # Each measured module, including all of its classes, stays whole.
            self.assertTrue(moved.issubset(buckets[0 if jobs == 3 else 1]))
            self.assertTrue(all(name in moved or name == 'tests.desktop.Case.test_flow'
                                for name in buckets[0]) if jobs == 3 else True)
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

    def test_catalogue_fixture_reuses_only_templates_and_restores_lazy_import_aliases(self):
        from dataclasses import replace
        from types import ModuleType
        from tests.catalogue_fixtures import _catalogue, catalogue_fixture
        from seedvision.benchmarking import native_recipe as runner, recipe_catalogue
        original = recipe_catalogue.build_recipe_catalogue
        _catalogue.cache_clear()
        lazy = ModuleType('tests.fixture_lazy_import_witness')
        with patch.dict(sys.modules, {lazy.__name__: lazy}):
            with catalogue_fixture() as first:
                lazy.build_recipe_catalogue = recipe_catalogue.build_recipe_catalogue
                self.assertIs(runner.build_recipe_catalogue(), first)
                selected = first.recipe('plantcv_seed_binary')
                forged = replace(selected, stages=(replace(selected.stages[0],
                    operation_id='external.unexpected'), *selected.stages[1:]))
                with self.assertRaises(runner.NativeRecipeError):
                    runner._validate_recipe(forged)
            self.assertIs(lazy.build_recipe_catalogue, original)
            self.assertIs(runner.build_recipe_catalogue, original)
            with catalogue_fixture() as second:
                self.assertIs(first, second)
            self.assertEqual(_catalogue.cache_info().misses, 1)
            self.assertEqual(_catalogue.cache_info().hits, 1)
        _catalogue.cache_clear()

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

    def test_partition_is_complete_and_fixture_timings_preserve_lifecycle(self):
        class Item:
            def __init__(self, identifier): self.identifier = identifier
            def id(self): return self.identifier
        items = [Item('sample.Cpu.test_a'), Item('sample.Cuda.test_b'), Item('sample.Acceptance.test_c')]
        with patch.object(selection, 'CPU_QT_TARGETS', ('sample.Cpu', 'sample.Cuda')), patch.object(selection, 'ACCEPTANCE_TARGETS', ('sample.Acceptance',)), patch.object(selection, 'CUDA_TARGETS', ('sample.Cuda',)):
            groups = selection.partition(items)
            self.assertEqual({k:len(v) for k,v in groups.items()}, {'cpu-qt':1, 'cuda':1, 'acceptance':1})
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

    def test_fixture_bundle_checks_hashes_and_never_overwrites_reviewed_files(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base/'project'; bundle = base/'bundle'
            (root/'config').mkdir(parents=True)
            (bundle/'artifacts').mkdir(parents=True)
            data = b'immutable expected record'
            (bundle/'artifacts/expected.json').write_bytes(data)
            (root/'config/test_fixtures.json').write_text(json.dumps({'files':[
                {'path':'artifacts/expected.json','group':'acceptance','sha256':sha256(data).hexdigest()}]}))
            missing = check_prerequisites('history', root=root)
            self.assertFalse(missing['passed'])
            restored = check_prerequisites('history', root=root, fixture_source=bundle)
            self.assertTrue(restored['passed'], restored)
            self.assertEqual(restored['copied_files'], ['artifacts/expected.json'])
            target = root/'artifacts/expected.json'; target.write_bytes(b'changed')
            changed = check_prerequisites('history', root=root, fixture_source=bundle)
            self.assertFalse(changed['passed'])
            self.assertIn('SHA256 mismatch', changed['errors'][0])
            self.assertEqual(target.read_bytes(), b'changed')
