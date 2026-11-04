"""Functional units with independently owned scenario fixtures.

Scenario assertions still run as subtests, without creating nested unittest
runs or reporting old method executions as new discovered tests. Mutable
windows, directories, tensors and cleanup stacks remain scenario-local.
"""
from contextlib import nullcontext
import sys
import time


SCENARIO_DURATIONS = []


def live_windows():
    """Return the parentless widgets now alive; no other widget will delete them."""
    widgets = sys.modules.get('PySide6.QtWidgets')
    if widgets is None:
        return frozenset()
    # A widget embedded in a graphics item has no parent, but that item owns it.
    return frozenset(widget for widget in widgets.QApplication.topLevelWidgets()
                     if widget.parentWidget() is None and widget.graphicsProxyWidget() is None)


def discard_windows(kept=frozenset()):
    """Delete the windows a finished test left behind, except those in ``kept``.

    Closing only hides a window, and connections to its own methods keep it
    alive with no Python reference. Without this, a worker holds every window
    of its run until interpreter shutdown deletes them, in time quadratic in
    the widgets alive. Deferred deletion also finishes the deleteLater() calls
    of tests, which processEvents() leaves queued outside an event loop.
    """
    core = sys.modules.get('PySide6.QtCore')
    if core is None or core.QCoreApplication.instance() is None:
        return
    for window in live_windows() - kept:
        window.deleteLater()
    core.QCoreApplication.sendPostedEvents(None, core.QEvent.Type.DeferredDelete)


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
        earlier_windows = live_windows()
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
                discard_windows(earlier_windows)
                case._outcome = None
                SCENARIO_DURATIONS.append(dict(unit=self.id(),scenario=label,
                    seconds=time.perf_counter()-started))
    cls.scenario = scenario
    return cls
