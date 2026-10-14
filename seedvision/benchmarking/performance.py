"""Explicit CUDA benchmark runs, separate from ordinary analysis/cache ownership."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from seedvision.benchmarking.native_recipe import run_native_recipe_path, NativeRecipeCancelled, _validate_recipe
from seedvision.benchmarking.checkpoints import _canonical_json
from seedvision.benchmarking.native_image_inputs import source_identity
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.timing import NodeTimingRecorder

MAX_RUNS = 64
MAX_REPORT_BYTES = 16 * 1024 * 1024


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}].')
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return float(value)


class TransferRecorder(TorchDispatchMode):
    """Count logical PyTorch copy payloads and explicit CUDA scalar reads.

    These are observed dispatcher operations, not PCIe traffic measurements or
    a claim to trace hidden transfers inside third-party kernels.
    """
    def __init__(self):
        super().__init__()
        self.upload_bytes = self.download_bytes = self.scalar_reads = self.scalar_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten._to_copy.default and args and torch.is_tensor(args[0]) and torch.is_tensor(result):
            source, target = args[0].device.type, result.device.type
            size = result.numel() * result.element_size()
            if source == 'cpu' and target == 'cuda':
                self.upload_bytes += size
            elif source == 'cuda' and target == 'cpu':
                self.download_bytes += size
        elif func == torch.ops.aten._local_scalar_dense.default and args and args[0].device.type == 'cuda':
            self.scalar_reads += 1
            self.scalar_bytes += args[0].element_size()
        return result

    def to_dict(self):
        return {'method': 'PyTorch dispatcher logical tensor copy payloads and explicit scalar reads',
            'cpu_to_cuda_tensor_bytes': self.upload_bytes, 'cuda_to_cpu_tensor_bytes': self.download_bytes,
            'cuda_scalar_reads': self.scalar_reads, 'cuda_scalar_payload_bytes': self.scalar_bytes,
            'physical_bus_bytes': None, 'unobserved_external_transfers': 'unknown'}


@contextmanager
def measure_operator_correction(*, operator, acceptance_standard, source_sha256, recipe_sha256):
    """Explicitly time work against a supplied fixed acceptance standard."""
    for value in (operator, acceptance_standard):
        if type(value) is not str or not value.strip() or len(value) > 4096:
            raise ValueError('Correction timing needs an operator and acceptance standard.')
    for value in (source_sha256, recipe_sha256):
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Correction timing requires exact source/recipe digests.')
    record = {'operator': operator, 'acceptance_standard': acceptance_standard,
        'source_sha256': source_sha256, 'recipe_sha256': recipe_sha256,
        'method': 'explicit perf_counter start/stop; supplied acceptance standard',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'in_progress', 'seconds': None}
    started = time.perf_counter()
    try:
        yield record
    except BaseException:
        record['status'] = 'interrupted'
        raise
    else:
        record['status'] = 'completed'
    finally:
        record['seconds'] = time.perf_counter() - started
        record['ended_at_utc'] = datetime.now(timezone.utc).isoformat()


def _timings(recorder):
    resolved = recorder.finalize()
    values = {}
    for span in recorder._spans:
        gpu = (float(span.cuda_start.elapsed_time(span.cuda_end)) / 1000
               if span.cuda_start is not None and span.cuda_end is not None else None)
        values[span.node_id] = {'submission_wall_seconds': span.cpu_elapsed,
            'cuda_event_seconds': gpu, 'resolved_seconds': resolved[span.node_id]}
    for name in recorder._explicit:
        values.setdefault(name, {'submission_wall_seconds': 0.0, 'cuda_event_seconds': 0.0,
                                 'resolved_seconds': resolved[name]})
    return values


def benchmark_native_recipe(recipe, source, *, repeats=3, warmups=1, modes=('cold', 'warm'),
                            device='cuda', maximum_elapsed_seconds=None, maximum_peak_bytes=None,
                            cancel=None, model_asset_paths=None, correction_records=()):
    """Measure the existing path producer using an exclusively owned cache.

    Thresholds stop subsequent runs after a synchronized measurement. They are
    not hard kernel-time or allocation limits. Full-resolution products stay
    on CUDA and are released when this benchmark returns.
    """
    _validate_recipe(recipe)
    if recipe.provenance.mode.value == 'original_reference':
        raise ValueError('Select an explicitly native or paper-port working recipe for native performance measurement.')
    _integer(repeats, 1, MAX_RUNS, 'repeats')
    _integer(warmups, 0, MAX_RUNS, 'warmups')
    if type(modes) is not tuple or not modes or len(set(modes)) != len(modes) or set(modes) - {'cold', 'warm'}:
        raise ValueError('Cache modes must be distinct cold/warm values.')
    if len(modes) * repeats + warmups > MAX_RUNS:
        raise ValueError('Benchmark exceeds its run budget.')
    if 'warm' in modes and warmups == 0:
        raise ValueError('Warm-cache measurements require at least one actual warmup.')
    if maximum_elapsed_seconds is not None:
        _positive(maximum_elapsed_seconds, 'maximum elapsed seconds')
    if maximum_peak_bytes is not None:
        _integer(maximum_peak_bytes, 1, 2**63-1, 'maximum peak bytes')
    if cancel is not None and not callable(cancel):
        raise TypeError('Cancellation must be a callable.')
    source = Path(source)
    source_digest = source_identity(source).sha256
    if not isinstance(correction_records, (tuple, list)) or len(correction_records) > 1024:
        raise ValueError('Too many correction observations.')
    corrections = []
    for row in correction_records:
        if (type(row) is not dict or row.get('source_sha256') != source_digest
                or row.get('recipe_sha256') != recipe.identity.recipe_sha256
                or row.get('status') not in {'completed', 'interrupted'}
                or not row.get('acceptance_standard') or not row.get('operator') or not row.get('method')):
            raise ValueError('Correction observation differs from the evaluated workload.')
        _positive(row.get('seconds'), 'correction seconds')
        corrections.append(json.loads(_canonical_json(row, name='correction observation', maximum=16384)))
    report = {'format': 'seedfiddle.native-performance', 'version': 1,
        'execution_backend': 'seedvision.benchmarking.native_recipe.run_native_recipe_path',
        'recipe': recipe.to_dict(), 'source_sha256': source_digest,
        'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                     'python': platform.python_version(), 'torch': str(torch.__version__),
                     'torch_cuda': torch.version.cuda},
        'configuration': {'repeats': repeats, 'warmups': warmups, 'modes': list(modes),
            'maximum_elapsed_seconds': maximum_elapsed_seconds, 'maximum_peak_bytes': maximum_peak_bytes,
            'threshold_enforcement': 'stop between synchronized runs; not hard allocation/kernel limits'},
        'measurement_method': 'CUDA events and device synchronization before/after each measured run; process-wide allocator peaks reset per run; exclusive owned node cache; dispatcher instrumentation overhead included',
        'warmup_outcomes': [], 'runs': [], 'status': 'unavailable',
        'requested_measured_runs': len(modes) * repeats,
        'attempted_measured_runs': 0, 'completed_measured_runs': 0,
        'operator_correction': {'records': corrections, 'method': 'explicit recorded start/stop against supplied fixed acceptance standard',
            'availability': 'recorded' if corrections else 'unavailable',
            'scientific_acceptance_standard_verified': False},
        'scientifically_validated': False}
    selected = torch.device(device)
    if selected.type != 'cuda':
        raise ValueError('Native recipe benchmarks require a CUDA device.')
    if not torch.cuda.is_available():
        report['reason'] = 'CUDA is unavailable; no CPU substitute executed.'
        report['report_sha256'] = sha256(_canonical_json(report, name='performance report', maximum=MAX_REPORT_BYTES)).hexdigest()
        return report
    stop = False
    warm_cache = PipelineAnalysisCache()
    with torch.cuda.device(selected):
        props = torch.cuda.get_device_properties(selected)
        report['hardware'].update(device_index=torch.cuda.current_device(), device_name=props.name,
            total_device_bytes=props.total_memory, compute_capability=[props.major, props.minor])
        for index in range(warmups):
            if cancel and cancel():
                report['status'] = 'cancelled'
                stop = True
                break
            try:
                result = run_native_recipe_path(recipe, source, device=selected, cache=warm_cache,
                    cancel=cancel, model_asset_paths=model_asset_paths)
                if result.source_sha256 != source_digest:
                    raise ValueError('Benchmark source changed during warmup.')
                report['warmup_outcomes'].append({'index': index, 'status': result.status,
                    'source_sha256': result.source_sha256, 'runtime': dict(result.runtime)})
                del result
            except Exception as error:
                report['warmup_outcomes'].append({'index': index, 'status': 'cancelled' if isinstance(error, NativeRecipeCancelled) else 'failed',
                    'reason': f'{type(error).__name__}: {error}'[:4096]})
                report['status'] = report['warmup_outcomes'][-1]['status']
                stop = True
                break
        for mode in modes:
            for index in range(repeats):
                if stop:
                    break
                if cancel and cancel():
                    report['status'] = 'cancelled'
                    stop = True
                    break
                cache = PipelineAnalysisCache() if mode == 'cold' else warm_cache
                tracer, recorder = TransferRecorder(), NodeTimingRecorder(selected)
                torch.cuda.synchronize(selected)
                allocated, reserved = torch.cuda.memory_allocated(selected), torch.cuda.memory_reserved(selected)
                torch.cuda.reset_peak_memory_stats(selected)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                row = {'mode': mode, 'index': index, 'cache_entries_before': len(cache.values),
                    'baseline_allocated_bytes': allocated, 'baseline_reserved_bytes': reserved,
                    'runtime': None, 'nodes': None, 'coordinate_frame': None}
                begin.record()
                started = time.perf_counter()
                try:
                    with tracer:
                        result = run_native_recipe_path(recipe, source, device=selected, cache=cache,
                            cancel=cancel, model_asset_paths=model_asset_paths, timing_recorder=recorder)
                    if result.source_sha256 != source_digest:
                        raise ValueError('Benchmark source changed between selected runs.')
                    row.update(status=result.status, runtime=dict(result.runtime),
                        coordinate_frame=result.coordinate_frame.to_dict(),
                        nodes=[{'stage_id': node.stage_id, 'operation': node.operation_id,
                                'status': node.status, 'reused': node.reused, 'reason': node.reason} for node in result.nodes])
                    del result
                except Exception as error:
                    row.update(status='cancelled' if isinstance(error, NativeRecipeCancelled) else 'failed',
                        reason=f'{type(error).__name__}: {error}'[:4096])
                    stop = True
                row.update(cuda_event_seconds=None, peak_allocated_bytes=None,
                    peak_reserved_bytes=None, peak_additional_allocated_bytes=None, node_timings=None,
                    transfers=tracer.to_dict(), synchronized=False)
                try:
                    end.record()
                    torch.cuda.synchronize(selected)
                    row.update(cuda_event_seconds=float(begin.elapsed_time(end))/1000,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(selected),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(selected),
                        node_timings=_timings(recorder), synchronized=True)
                    row['peak_additional_allocated_bytes'] = max(0, row['peak_allocated_bytes']-allocated)
                except Exception as error:
                    row.update(status='failed', measurement_error=f'{type(error).__name__}: {error}'[:4096])
                    stop = True
                row['wall_seconds'] = time.perf_counter()-started
                row['threshold_exceeded'] = bool(
                    maximum_elapsed_seconds is not None and row['wall_seconds'] > maximum_elapsed_seconds or
                    maximum_peak_bytes is not None and row['peak_allocated_bytes'] is not None
                    and row['peak_allocated_bytes'] > maximum_peak_bytes)
                report['runs'].append(row)
                report['status'] = (row['status'] if row['status'] in ('failed', 'cancelled')
                                    else 'threshold_exceeded' if row['threshold_exceeded'] else row['status'])
                stop = stop or row['threshold_exceeded']
                if mode == 'cold':
                    cache.values.clear()
        warm_cache.values.clear()
    report['attempted_measured_runs'] = len(report['runs'])
    report['completed_measured_runs'] = sum(row['status'] in ('executed', 'partial') for row in report['runs'])
    report['report_sha256'] = sha256(_canonical_json(report, name='performance report', maximum=MAX_REPORT_BYTES)).hexdigest()
    return report


def export_performance_report(report, path):
    raw = _canonical_json(report, name='performance report', maximum=MAX_REPORT_BYTES)
    with Path(path).open('xb') as stream:
        stream.write(raw + b'\n')
    return Path(path)
