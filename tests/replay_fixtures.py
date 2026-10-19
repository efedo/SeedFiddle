"""Private archive and instrumentation helpers for saved-stage replay tests.

Only scaffolding is shared: callers own their runs, CUDA buffers, directories and
mutation callbacks. No mutable archive or numerical result is cached globally.
"""
from dataclasses import replace
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import sys
from uuid import uuid4

from tests.observation import observe_functions

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.export import native_results as archives


@contextmanager
def hidden_origins(*paths):
    """Combine full product comparisons with the source-absent replay witness."""
    moved = []
    try:
        for path in dict.fromkeys(map(Path, paths)):
            hidden = path.with_name(path.name + '.hidden-' + uuid4().hex)
            path.rename(hidden)
            moved.append((path, hidden))
            if path.exists():
                raise AssertionError('Replay origin is still available')
        yield
    finally:
        for path, hidden in reversed(moved):
            hidden.rename(path)


def archive_hashes(directory):
    """Fingerprint every archive file, including unselected array payloads."""
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def export_private_archive(root, recipe, saved, *, prefix='changed'):
    """Allocate a caller-owned archive directory without sibling-count names."""
    directory = Path(root) / (prefix + '-' + uuid4().hex)
    archives.export_native_recipe_result(directory, recipe, saved)
    return directory


def changed_archive(root, recipe, saved, replacements, *, partial=False):
    """Export a private run with replaced nodes; retain all untouched products."""
    changed = replace(saved, nodes=tuple(replacements.get(node.stage_id, node)
                                        for node in saved.nodes))
    if partial and any(node.status == 'partial' for node in changed.nodes):
        changed = replace(changed, status='partial')
    return export_private_archive(root, recipe, changed)


def replay_attempt(directory, stage, *, cancel=None, inventory_change=None,
                   observer=None, observer_codes=(), **kwargs):
    """Observe actual dispatch/load calls without replacing replay execution.

    The previous profiler is restored even when callbacks raise. Optional hooks
    preserve the adapter tests' exact inventory and private-buffer fault sites.
    """
    calls, selected = [], []
    dispatch_code = runner._dispatch.__code__
    # Admission tests can replace the loader with a Mock and require no calls.
    load_code = getattr(archives.load_native_recipe_result, '__code__', None)
    inventory_code = getattr(archives.read_native_recipe_inventory, '__code__', None)

    def profile(frame, event, arg):
        if event == 'call':
            code = frame.f_code
            if code is dispatch_code:
                calls.append(frame.f_locals['operation'])
            elif code is load_code:
                selected.append(frame.f_locals['array_names'])
        elif (event == 'return' and inventory_change is not None
              and frame.f_code is inventory_code and arg is not None):
            inventory_change(*arg)
        if observer is not None:
            observer(frame, event, arg)

    with observe_functions(profile, dispatch_code, load_code, inventory_code, *observer_codes):
        result = replay.replay_native_stage(directory, stage, cancel=cancel, **kwargs)
    return result, calls, selected


def assert_stage_inputs_equal(testcase, recipe, expected, stage_ids):
    """Prove selected stage contracts and every incoming edge are unchanged.

    Outgoing edges may add consumers, and the composed external source adds a
    source-frame output. Neither changes the prefix's consumed image contract.
    """
    stage_ids = frozenset(stage_ids)
    for stage_id in stage_ids:
        testcase.assertEqual(recipe.stage(stage_id), expected.stage(stage_id), stage_id)
    testcase.assertEqual(
        tuple(edge for edge in recipe.dependencies if edge.target_stage in stage_ids),
        tuple(edge for edge in expected.dependencies if edge.target_stage in stage_ids))
