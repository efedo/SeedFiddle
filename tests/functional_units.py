"""Functional units with independently owned scenario fixtures.

Scenario assertions still run as subtests, without creating nested unittest
runs or reporting old method executions as new discovered tests. Mutable
windows, directories, tensors and cleanup stacks remain scenario-local.
"""
from contextlib import nullcontext
import time


SCENARIO_DURATIONS = []


def functional_units(cls):
    setup, teardown = cls.setUp, cls.tearDown
    original_run = cls.run
    # The containing unit has no scenario state of its own. Each scenario
    # explicitly receives the same original fixture lifecycle on a fresh case.
    def run(self, *args, **kwargs):
        self._functional_container = True
        try:
            return original_run(self, *args, **kwargs)
        finally:
            self._functional_container = False

    def manual_setup(self):
        if not getattr(self, '_functional_container', False):
            return setup(self)

    def manual_teardown(self):
        if not getattr(self, '_functional_container', False):
            return teardown(self)

    cls.run, cls.setUp, cls.tearDown = run, manual_setup, manual_teardown

    def scenario(self, method, *, labelled=True):
        case = type(self)(self._testMethodName)
        case._outcome = self._outcome
        case._subtest = self._subtest
        started = time.perf_counter()
        label = method.__name__.removeprefix('check_')
        # Some archive fixtures include the scenario name in private paths.
        # Preserve that identity instead of making siblings collide on the
        # containing functional unit's name.
        case._testMethodName = label
        with self.subTest(scenario=label) if labelled else nullcontext():
            case._subtest = self._subtest
            try:
                setup(case)
                try:
                    getattr(case, method.__name__)()
                finally:
                    teardown(case)
            finally:
                case.doCleanups()
                case._outcome = None
                SCENARIO_DURATIONS.append(dict(unit=self.id(),scenario=label,
                    seconds=time.perf_counter()-started))
    cls.scenario = scenario
    return cls
