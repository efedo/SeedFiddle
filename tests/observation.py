"""Observe selected Python boundaries without profiling unrelated calculations.

The callbacks retain the existing ``(frame, event, arg)`` convention. Python
3.12's local monitoring events leave the actual functions, aliases and code
identities intact, unlike wrapping the functions with mocks.
"""
from contextlib import contextmanager
import sys
from threading import get_ident
from types import CodeType


@contextmanager
def observe_functions(callback, *functions, c_calls=()):
    codes = frozenset(value if isinstance(value, CodeType) else value.__code__
                      for value in functions if value is not None)
    call_codes = frozenset(value if isinstance(value, CodeType) else value.__code__
                          for value in c_calls)
    owner = get_ident()
    monitoring = getattr(sys, 'monitoring', None)
    tool = (next((i for i in range(5, -1, -1) if monitoring.get_tool(i) is None), None)
            if monitoring is not None else None)
    if tool is None:
        previous = sys.getprofile()

        def filtered(frame, event, arg):
            if frame.f_code in codes or frame.f_code in call_codes and event == 'c_call':
                callback(frame, event, arg)
            if previous is not None:
                previous(frame, event, arg)

        sys.setprofile(filtered)
        try:
            yield
        finally:
            sys.setprofile(previous)
        return

    def entered(code, offset):
        if get_ident() == owner:
            callback(sys._getframe(1), 'call', None)

    def returned(code, offset, value):
        if get_ident() == owner:
            callback(sys._getframe(1), 'return', value)

    def called(code, offset, function, arg0):
        if get_ident() == owner:
            callback(sys._getframe(1), 'c_call', function)

    events = monitoring.events
    monitoring.use_tool_id(tool, 'seedfiddle-test-observer')
    try:
        monitoring.register_callback(tool, events.PY_START, entered)
        monitoring.register_callback(tool, events.PY_RETURN, returned)
        monitoring.register_callback(tool, events.CALL, called)
        for code in codes | call_codes:
            mask = (events.PY_START | events.PY_RETURN) if code in codes else 0
            monitoring.set_local_events(tool, code, mask | (events.CALL if code in call_codes else 0))
        yield
    finally:
        for code in codes | call_codes:
            monitoring.set_local_events(tool, code, 0)
        for event in (events.PY_START, events.PY_RETURN, events.CALL):
            monitoring.register_callback(tool, event, None)
        monitoring.free_tool_id(tool)
