"""Low-overhead per-node CPU/CUDA execution timing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


class AnalysisCancelled(RuntimeError):
    """A background analysis was superseded at a safe node boundary."""


@dataclass(slots=True)
class _TimingSpan:
    node_id: str
    cpu_started: float
    report_progress: bool = True
    cpu_elapsed: float = 0.0
    cuda_start: object | None = None
    cuda_end: object | None = None


class NodeTimingRecorder:
    """Record CUDA events without synchronizing between pipeline nodes.

    All CUDA spans are resolved together in :meth:`finalize`. The reported
    duration is the larger of submission wall time and device-event time, so
    compact CPU work and kernels are both represented without serializing the
    normal pipeline.
    """

    def __init__(
        self,
        device=None,
        progress_callback=None,
        cancellation_requested=None,
    ) -> None:
        self._device = device
        self._progress_callback = progress_callback
        self._cancellation_requested = cancellation_requested
        self._spans: list[_TimingSpan] = []
        self._explicit: dict[str, float] = {}
        self._cuda = False
        try:
            import torch

            self._cuda = bool(
                device is not None
                and getattr(device, "type", str(device)) == "cuda"
                and torch.cuda.is_available()
            )
        except ImportError:
            self._cuda = False

    def start(self, node_id: str, *, report_progress: bool = True) -> _TimingSpan:
        self.raise_if_cancelled()
        node_id = str(node_id)
        if report_progress:
            self._report(node_id, "started")
        cuda_start = None
        if self._cuda:
            import torch

            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        return _TimingSpan(
            node_id=node_id,
            cpu_started=perf_counter(),
            report_progress=report_progress,
            cuda_start=cuda_start,
        )

    def stop(self, span: _TimingSpan, *, completed: bool = True) -> None:
        if self._cuda:
            import torch

            span.cuda_end = torch.cuda.Event(enable_timing=True)
            span.cuda_end.record()
        span.cpu_elapsed = perf_counter() - span.cpu_started
        self._spans.append(span)
        if span.report_progress and completed:
            self._report(span.node_id, "completed")
        if completed:
            self.raise_if_cancelled()

    @contextmanager
    def measure(self, node_id: str):
        span = self.start(node_id)
        try:
            yield
        except BaseException:
            self.stop(span, completed=False)
            raise
        else:
            self.stop(span)

    def _report(self, node_id: str, state: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(node_id, state)
        except Exception:
            # UI telemetry must never be able to fail an analysis.
            pass

    def raise_if_cancelled(self) -> None:
        """Stop before another node starts when a newer run supersedes this one."""

        if (
            self._cancellation_requested is not None
            and self._cancellation_requested()
        ):
            raise AnalysisCancelled("Analysis superseded by a newer request.")

    def add_seconds(self, node_id: str, seconds: float) -> None:
        self._explicit[str(node_id)] = self._explicit.get(str(node_id), 0.0) + max(
            0.0, float(seconds)
        )

    def finalize(self) -> dict[str, float]:
        self.raise_if_cancelled()
        if self._cuda:
            final_event = next(
                (
                    span.cuda_end
                    for span in reversed(self._spans)
                    if span.cuda_end is not None
                ),
                None,
            )
            if final_event is not None:
                final_event.synchronize()
        result = dict(self._explicit)
        for span in self._spans:
            gpu_elapsed = 0.0
            if span.cuda_start is not None and span.cuda_end is not None:
                gpu_elapsed = (
                    float(span.cuda_start.elapsed_time(span.cuda_end)) / 1000.0
                )
            elapsed = max(span.cpu_elapsed, gpu_elapsed)
            result[span.node_id] = result.get(span.node_id, 0.0) + elapsed
        return result
