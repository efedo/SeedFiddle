"""Cap the memory one test may commit, so a runaway test fails on its own.

Each test may commit ``budget`` bytes beyond what its process held when the
test started. Going over makes the allocation fail (``MemoryError`` from NumPy
or Python, an allocation error from Torch) inside that test, which then reports
the traceback, instead of the process taking the machine's memory.

Windows only: a job object caps the process's committed private memory, and
the cap is moved at the start of every test. Windows charges CUDA allocations
to that commit as well, so the budget covers GPU and host memory together.
Elsewhere nothing is enforced and ``install`` says so.
"""
from __future__ import annotations

import os
import sys

# The GPU's own size (an 8 GiB card): beyond it the NVIDIA driver silently
# spills CUDA allocations into system memory, which is how one analysis once
# reached 28 GB. The heaviest test (a 27-megapixel dense dish) needs ~6.7 GiB.
DEFAULT_GB = 8.0
ENVIRONMENT = 'SEEDFIDDLE_TEST_MEMORY_GB'
_GIB = 1024 ** 3


def configured_gb() -> float:
    """The per-test budget in GiB: the environment's, else the default."""

    value = os.environ.get(ENVIRONMENT)
    return DEFAULT_GB if value in (None, '') else float(value)


class _WindowsProcessMemoryCap:
    """Committed-memory cap on this process through its own job object."""

    _EXTENDED_LIMIT_INFORMATION = 9
    _LIMIT_PROCESS_MEMORY = 0x00000100

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', BasicLimits),
                ('IoInfo', IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]

        self._ctypes = ctypes
        self._limits_type = ExtendedLimits
        self._counters_type = MemoryCounters
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p)
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.K32GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._process = kernel32.GetCurrentProcess()
        self._job = kernel32.CreateJobObjectW(None, None)
        if not self._job:
            raise ctypes.WinError(ctypes.get_last_error())
        # Windows 8 and later nest jobs, so this works under a launcher's job.
        if not kernel32.AssignProcessToJobObject(self._job, self._process):
            raise ctypes.WinError(ctypes.get_last_error())

    def committed(self) -> int:
        counters = self._counters_type()
        counters.cb = self._ctypes.sizeof(counters)
        if not self._kernel32.K32GetProcessMemoryInfo(
                self._process, self._ctypes.byref(counters), counters.cb):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(counters.PrivateUsage)

    def _limits(self):
        limits = self._limits_type()
        if not self._kernel32.QueryInformationJobObject(
                self._job, self._EXTENDED_LIMIT_INFORMATION, self._ctypes.byref(limits),
                self._ctypes.sizeof(limits), None):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return limits

    def peak(self) -> int:
        return int(self._limits().PeakProcessMemoryUsed)

    def cap(self, limit: int) -> None:
        limits = self._limits()
        limits.BasicLimitInformation.LimitFlags |= self._LIMIT_PROCESS_MEMORY
        limits.ProcessMemoryLimit = int(limit)
        if not self._kernel32.SetInformationJobObject(
                self._job, self._EXTENDED_LIMIT_INFORMATION, self._ctypes.byref(limits),
                self._ctypes.sizeof(limits)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())


def _prepare_cuda() -> None:
    """Start Torch's CUDA runtime before the first test's budget is set.

    Importing Torch and creating its CUDA context commits about 1.5 GiB that
    no test asks for; without this the first test in a process would pay it.
    """

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.zeros(1, device='cuda')
        torch.cuda.synchronize()


def _release_cached_gpu_memory() -> None:
    # Under Windows' display driver model GPU allocations count as committed
    # process memory, so blocks Torch keeps cached would count against the
    # next test's budget and the process total.
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class TestMemoryLimit:
    """Moves the process cap to (committed now + budget) at each test start.

    GPU allocations count too: Windows charges them to the process's commit.
    """

    def __init__(self, budget_gb: float, process_cap) -> None:
        self.budget = int(budget_gb * _GIB)
        self._process = process_cap
        self.start_test()

    def start_test(self) -> None:
        _release_cached_gpu_memory()
        self._process.cap(self._process.committed() + self.budget)

    def peak(self) -> int:
        return self._process.peak()


def install(budget_gb: float) -> TestMemoryLimit | None:
    """Cap this process; ``None`` (with a note) when the budget is 0 or unsupported."""

    if budget_gb <= 0:
        return None
    if os.name != 'nt':
        print(f'Test memory limit ({budget_gb:g} GiB) is only enforced on Windows; '
              'none is set here.', file=sys.stderr)
        return None
    process_cap = _WindowsProcessMemoryCap()
    _prepare_cuda()
    return TestMemoryLimit(budget_gb, process_cap)
