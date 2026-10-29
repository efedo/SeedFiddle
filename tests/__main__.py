"""Run core/focused checks, an explicit tier, or unfiltered --full acceptance."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
import unittest

from tests.prerequisites import check_prerequisites
from tests.selection import (CORE_TARGETS, CPU_QT_TARGETS, ACCEPTANCE_TARGETS,
                             HISTORY_TARGETS, flatten, matches, partition)

ROOT = Path(__file__).resolve().parents[1]


class TimedSuite(unittest.TestSuite):
    """Attribute class fixture boundaries without replacing fixture methods."""

    def _time_fixture(self, result, cls, name, invoke):
        started = time.perf_counter()
        try:
            return invoke()
        finally:
            if not hasattr(result, 'fixture_durations'):
                result.fixture_durations = []
            result.fixture_durations.append({'id': cls.__module__ + '.' + cls.__qualname__,
                'fixture': name, 'seconds': time.perf_counter() - started})

    def _handleClassSetUp(self, test, result):
        cls = test.__class__
        original = super()._handleClassSetUp
        if cls is getattr(result, '_previousTestClass', None) or getattr(cls, '__unittest_skip__', False):
            return original(test, result)
        fixture = getattr(cls, 'setUpClass', None)
        if getattr(fixture, '__func__', fixture) is unittest.TestCase.setUpClass.__func__:
            return original(test, result)
        return self._time_fixture(result, cls, 'setUpClass', lambda: original(test, result))

    def _tearDownPreviousClass(self, test, result):
        cls = getattr(result, '_previousTestClass', None)
        original = super()._tearDownPreviousClass
        if cls is None or cls is getattr(test, '__class__', None) or getattr(cls, '__unittest_skip__', False) or getattr(cls, '_classSetupFailed', False):
            return original(test, result)
        fixture = getattr(cls, 'tearDownClass', None)
        if (getattr(fixture, '__func__', fixture) is unittest.TestCase.tearDownClass.__func__
                and not getattr(cls, '_class_cleanups', ())):
            return original(test, result)
        return self._time_fixture(result, cls, 'tearDownClass/cleanups', lambda: original(test, result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Discover every test; no filtering or tier skips.')
    parser.add_argument('--jobs',type=int,choices=(1,2,3),default=3,
        help='Isolated processes for full runs (default: 3); focused/tier runs use one.')
    parser.add_argument('--tier', choices=('core', 'cpu-qt', 'cuda', 'acceptance', 'history'), help='Explicit subset; never a full-suite claim.')
    parser.add_argument('--list', action='store_true', help='List selected IDs and partition counts without running fixtures/tests.')
    parser.add_argument('--preflight', action='store_true', help='Check selected prerequisites without running tests.')
    parser.add_argument('--fixtures-from', type=Path, help='Copy missing reviewed fixtures from a local bundle; never overwrite expectations.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--failfast', action='store_true', help='Stop a focused diagnostic at the first failure.')
    parser.add_argument('--durations', type=int, default=10, help='Show this many slow tests (0 shows all).')
    parser.add_argument('--report', type=Path, help='Write a new timing/result JSON under artifacts/.')
    parser.add_argument('--stall-seconds', type=float, default=900.0,
        help='Full runs: stop a worker whose journal shows no progress for this long (default 900).')
    parser.add_argument('targets', nargs='*', help='Optional unittest module, class or method names.')
    args = parser.parse_args(argv)
    if sum((args.full, bool(args.tier), bool(args.targets))) > 1:
        parser.error('--full, --tier and selected targets are mutually exclusive')
    if args.list and (args.preflight or args.fixtures_from):
        parser.error('--list cannot be combined with --preflight or --fixtures-from')
    if args.durations < 0:
        parser.error('--durations must be nonnegative')
    if args.report:
        args.report = args.report.resolve()
        if (ROOT / 'artifacts').resolve() not in args.report.parents or args.report.exists():
            parser.error('--report must name a new file under artifacts/')

    def write_report(record):
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x', encoding='utf-8') as output:
                json.dump(record, output, indent=2)
                output.write('\n')

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    # Keep native Qt/image fixtures inside the writable workspace, including
    # when the host's default Windows temporary directory is restricted.
    from uuid import uuid4
    temp_root = ROOT / 'artifacts' / 'test-temp' / uuid4().hex
    temp_root.mkdir(parents=True)
    os.environ.update(TEMP=str(temp_root), TMP=str(temp_root))
    import tempfile
    tempfile.tempdir = str(temp_root)
    started = time.perf_counter()
    mode = 'full' if args.full else args.tier or ('focused' if args.targets else 'core')
    targets = args.targets or (CORE_TARGETS if mode == 'core' else CPU_QT_TARGETS if mode == 'cpu-qt'
                              else HISTORY_TARGETS if mode == 'history' else ACCEPTANCE_TARGETS if mode == 'acceptance' else ())
    # --list is import-only and works even on a machine without full prerequisites.
    # Selected acceptance modules cannot evade prerequisite checks via a focused name.
    prerequisite_mode = ('acceptance' if mode == 'focused' and any(matches(t, ACCEPTANCE_TARGETS) for t in targets) else mode)
    preflight = {'passed': True, 'errors': [], 'checked_files': [], 'copied_files': []}
    if not args.list:
        preflight = check_prerequisites(prerequisite_mode, fixture_source=args.fixtures_from)
    if not preflight['passed']:
        for error in preflight['errors']:
            print(error, file=sys.stderr)
        print('Prerequisites failed; no tests executed. Restore the reviewed fixture bundle and pinned reference environment.', file=sys.stderr)
        write_report({'mode': mode, 'tests': 0, 'passed': False, 'preflight': preflight,
                      'failures': [], 'errors': [], 'skips': [], 'discovery_errors': [],
                      'wall_seconds': time.perf_counter() - started, 'test_durations': [], 'fixture_durations': []})
        return 2
    if args.preflight:
        write_report({'mode': mode, 'tests': 0, 'passed': True, 'preflight_only': True,
                      'preflight': preflight, 'wall_seconds': time.perf_counter() - started})
        print(f'{mode} prerequisites passed ({len(preflight["checked_files"])} reviewed fixture files); no tests executed.')
        return 0
    loader = unittest.TestLoader()
    loader.suiteClass = TimedSuite
    # A single package namespace also shares fixture caches across imports.
    suite = (loader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT)) if args.full or args.tier or args.list
             else loader.loadTestsFromNames(targets))
    if loader.errors:
        print('Test discovery failed; no tests executed.', file=sys.stderr)
        for error in loader.errors:
            print(error, file=sys.stderr)
        write_report({'mode': mode, 'targets': None if args.full else list(targets),
                      'tests': 0, 'passed': False, 'failures': [], 'errors': [], 'skips': [],
                      'discovery_errors': list(loader.errors),
                      'wall_seconds': time.perf_counter() - started,
                      'temporary_directory': str(temp_root), 'test_durations': []})
        return 2
    selection = {'selected_ids': [], 'omitted_ids': [], 'partition_counts': None}
    if args.full or args.tier or args.list:
        all_tests = list(flatten(suite))
        groups = partition(all_tests)
        selection['partition_counts'] = {name: len(items) for name, items in groups.items()}
        selected = (all_tests if args.full else groups[mode] if mode in groups
                    else [test for test in all_tests if matches(test.id(), targets)])
        selected_ids = {test.id() for test in selected}
        selection['omitted_ids'] = [test.id() for test in all_tests if test.id() not in selected_ids]
        suite = TimedSuite(selected)
    selection['selected_ids'] = [test.id() for test in flatten(suite)]
    discovery_seconds = time.perf_counter() - started
    if args.list:
        print('\n'.join(selection['selected_ids']))
        print(f'{mode}: {len(selection["selected_ids"])} selected, {len(selection["omitted_ids"])} omitted; partitions {selection["partition_counts"]}', file=sys.stderr)
        write_report({'mode': mode, 'tests': 0, 'passed': True, 'collection_only': True,
                      'selection': selection, 'wall_seconds': discovery_seconds})
        return 0
    print(f'SeedFiddle {mode} suite: {suite.countTestCases()} tests. '
          + ('All discovered tests included.' if args.full else 'Use --full for complete acceptance.'),
          file=sys.stderr)
    if args.full and args.jobs>1:
        from tests.parallel import run as parallel_run
        execution_started=time.perf_counter()
        record=parallel_run(groups,root=ROOT,jobs=args.jobs,stall_seconds=args.stall_seconds)
        wall=time.perf_counter()-started
        # The budget is advisory: recorded and printed, never a verdict. Hangs
        # are caught per worker by the journal stall watchdog instead.
        budget={'maximum_units':999,'maximum_wall_seconds':900,'applicable':True,'advisory':True,
                'passed':len(selection['selected_ids'])<1000 and wall<=900}
        if not budget['passed']:
            print(f'Full-suite budget exceeded (advisory): {len(selection["selected_ids"])} units, {wall:.3f}s.',file=sys.stderr)
        record.update(mode=mode,targets=None,selection=selection,preflight=preflight,
            wall_seconds=wall,test_seconds=time.perf_counter()-execution_started,
            discovery_and_preflight_seconds=discovery_seconds,performance_budget=budget,
            temporary_directory=str(temp_root))
        write_report(record)
        print(f'Full suite: {record["tests"]}/{len(selection["selected_ids"])} units; '
              f'{wall:.3f}s; '+('PASS' if record['passed'] else 'FAIL'),file=sys.stderr)
        return 0 if record['passed'] else 1
    options = {'verbosity': 2 if args.verbose else 1, 'failfast': args.failfast}
    if 'durations' in inspect.signature(unittest.TextTestRunner).parameters:
        options['durations'] = args.durations
    test_started = time.perf_counter()
    from tests.compilation_fixture import compilation_fixture
    from tests.functional_units import SCENARIO_DURATIONS
    from tests.reference_session import reference_session
    SCENARIO_DURATIONS.clear()
    with compilation_fixture() as compiled, reference_session() as reference:
        result = unittest.TextTestRunner(**options).run(suite)
        compilation_stats = compiled.cache_info()._asdict()
    test_seconds = time.perf_counter() - test_started
    wall_seconds = time.perf_counter() - started
    budget = {'maximum_units': 999, 'maximum_wall_seconds': 900, 'advisory': True,
              'applicable': args.full, 'passed': not args.full or (
                  len(selection['selected_ids']) < 1000 and wall_seconds <= 900)}
    if not budget['passed']:
        print(f'Full-suite budget exceeded (advisory): {len(selection["selected_ids"])} units, {wall_seconds:.3f}s.', file=sys.stderr)
    fixtures = sorted(getattr(result, 'fixture_durations', ()), key=lambda row: -row['seconds'])
    for fixture in fixtures[:args.durations or len(fixtures)]:
        print(f'{fixture["seconds"]:.3f}s {fixture["fixture"]} ({fixture["id"]})', file=sys.stderr)
    if args.report:
        record = {'mode': mode, 'targets': None if args.full else list(targets),
                  'tests': result.testsRun, 'passed': result.wasSuccessful(),
                  'failures': [test.id() for test, _ in result.failures],
                  'errors': [test.id() for test, _ in result.errors],
                  'discovery_errors': [],
                  'preflight': preflight, 'selection': selection,
                  'skips': [{'id': test.id(), 'reason': reason} for test, reason in result.skipped],
                  'wall_seconds': wall_seconds, 'performance_budget': budget,
                  'discovery_and_preflight_seconds': discovery_seconds, 'test_seconds': test_seconds,
                  'temporary_directory': str(temp_root),
                  'fixture_durations': fixtures,
                  'scenario_durations': list(SCENARIO_DURATIONS),
                  'reference_fixture': reference.stats,
                  'compilation_fixture': compilation_stats,
                  'test_durations': list(getattr(result, 'collectedDurations', ())) }
        from tests.acceptance_workflows import LAST_SELECTION
        record['recipe_acceptance_scope'] = dict(LAST_SELECTION)
        write_report(record)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
