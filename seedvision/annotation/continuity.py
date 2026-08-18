"""Efficient continuity summaries for categorical seed-instance rasters.

Instance IDs are categorical: touching pixels with different positive values
must never become one component.  Calling a binary connected-component
routine once per ID preserves that rule, but scales as ``pixels * IDs``.  The
scan below instead run-length encodes each row and joins only equal-ID runs in
adjacent rows.  Components which no longer touch the active row are finalized,
so union-find storage is bounded by the active run frontier rather than the
number of pixels in the image.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np


_MAX_INSTANCE_ID = int(np.iinfo(np.uint16).max)


@dataclass(frozen=True, slots=True)
class InstanceContinuitySummary:
    """Compact 8-connected component statistics for an instance label map.

    ``component_counts`` is aligned with ``identifiers``.  A value greater
    than one means that the corresponding seed ID occupies disconnected areas.
    """

    identifiers: tuple[int, ...]
    pixel_count: int
    component_counts: tuple[int, ...]

    def component_count(self, identifier: int) -> int:
        """Return the component count for one ID, or zero when it is absent."""

        identifier = int(identifier)
        index = bisect_left(self.identifiers, identifier)
        if index == len(self.identifiers) or self.identifiers[index] != identifier:
            return 0
        return self.component_counts[index]

    @property
    def disconnected(self) -> tuple[tuple[int, int], ...]:
        """Return ``(ID, component count)`` pairs needing review."""

        return tuple(
            (identifier, count)
            for identifier, count in zip(
                self.identifiers, self.component_counts, strict=True
            )
            if count > 1
        )

    @property
    def disconnected_identifiers(self) -> tuple[int, ...]:
        """Return every ID represented by more than one connected area."""

        return tuple(identifier for identifier, _count in self.disconnected)


def summarize_instance_continuity(
    labels: np.ndarray,
) -> InstanceContinuitySummary:
    """Summarize positive seed IDs using exact 8-neighbour connectivity.

    Runtime is linear in raster pixels plus the number of row runs.  Different
    IDs remain separate even when their pixels touch, and diagonally adjacent
    pixels of the same ID are connected.  Internal Seed Fiddle label rasters
    use unsigned 16-bit IDs; accepting the same range here keeps the fixed-size
    counters small and catches malformed imported labels early.
    """

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("Instance labels must be a two-dimensional raster.")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("Instance labels must use an integer dtype.")
    if values.size:
        minimum = int(np.min(values))
        maximum = int(np.max(values))
        if minimum < 0 or maximum > _MAX_INSTANCE_ID:
            raise ValueError("Instance IDs must be between 0 and 65,535.")

    pixel_counts = np.zeros(_MAX_INSTANCE_ID + 1, dtype=np.int64)
    component_counts = np.zeros(_MAX_INSTANCE_ID + 1, dtype=np.int32)

    # A run is (inclusive start, exclusive end, ID, active component token).
    previous_runs: list[tuple[int, int, int, int]] = []
    parents: dict[int, int] = {}
    component_ids: dict[int, int] = {}
    next_component = 1

    def find(component: int) -> int:
        root = component
        while parents[root] != root:
            root = parents[root]
        while component != root:
            parent = parents[component]
            parents[component] = root
            component = parent
        return root

    def union(first: int, second: int) -> int:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return first_root
        # Stable roots make the result deterministic and simplify tests/debugging.
        if second_root < first_root:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        return first_root

    width = values.shape[1]
    for row in values:
        if width:
            boundaries = np.flatnonzero(row[1:] != row[:-1]) + 1
            starts = np.empty(boundaries.size + 1, dtype=np.intp)
            ends = np.empty(boundaries.size + 1, dtype=np.intp)
            starts[0] = 0
            starts[1:] = boundaries
            ends[:-1] = boundaries
            ends[-1] = width
            run_ids = row[starts]
            positive = run_ids > 0
            starts = starts[positive]
            ends = ends[positive]
            run_ids = run_ids[positive].astype(np.intp, copy=False)
            np.add.at(pixel_counts, run_ids, ends - starts)
        else:
            starts = np.empty(0, dtype=np.intp)
            ends = np.empty(0, dtype=np.intp)
            run_ids = np.empty(0, dtype=np.intp)

        current_runs: list[tuple[int, int, int, int]] = []
        previous_cursor = 0
        for start_value, end_value, identifier_value in zip(
            starts, ends, run_ids, strict=True
        ):
            start = int(start_value)
            end = int(end_value)
            identifier = int(identifier_value)

            # With exclusive ends, end == start still represents a one-column
            # diagonal contact between adjacent rows and therefore qualifies
            # under 8-neighbour connectivity.
            while (
                previous_cursor < len(previous_runs)
                and previous_runs[previous_cursor][1] < start
            ):
                previous_cursor += 1
            matching_components: list[int] = []
            candidate = previous_cursor
            while (
                candidate < len(previous_runs)
                and previous_runs[candidate][0] <= end
            ):
                _old_start, _old_end, old_identifier, old_component = (
                    previous_runs[candidate]
                )
                if old_identifier == identifier:
                    matching_components.append(find(old_component))
                candidate += 1

            if matching_components:
                component = matching_components[0]
                for match in matching_components[1:]:
                    component = union(component, match)
            else:
                component = next_component
                next_component += 1
                parents[component] = component
                component_ids[component] = identifier
            current_runs.append((start, end, identifier, component))

        # Union operations above can have changed roots belonging to both rows.
        previous_roots = {find(run[3]) for run in previous_runs}
        current_roots = {find(run[3]) for run in current_runs}
        for completed_root in previous_roots - current_roots:
            component_counts[component_ids[completed_root]] += 1

        # Only components represented on the current row can ever receive more
        # pixels.  Canonicalize them and discard finalized union-find state.
        current_runs = [
            (start, end, identifier, find(component))
            for start, end, identifier, component in current_runs
        ]
        active_roots = {run[3] for run in current_runs}
        active_ids = {root: component_ids[root] for root in active_roots}
        parents = {root: root for root in active_roots}
        component_ids = active_ids
        previous_runs = current_runs

    for completed_root in {find(run[3]) for run in previous_runs}:
        component_counts[component_ids[completed_root]] += 1

    identifiers_array = np.flatnonzero(pixel_counts[1:]) + 1
    identifiers = tuple(int(identifier) for identifier in identifiers_array)
    counts = tuple(int(component_counts[identifier]) for identifier in identifiers)
    return InstanceContinuitySummary(
        identifiers=identifiers,
        pixel_count=int(pixel_counts[1:].sum()),
        component_counts=counts,
    )


__all__ = ("InstanceContinuitySummary", "summarize_instance_continuity")
