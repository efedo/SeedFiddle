# Current architecture and analysis contracts

**Reviewed through revision `f244df8` on 29 September 2026.** See
[Current project status](CURRENT_STATUS.md) for the capability and validation
summary.

The only launcher is `seed_vision.py`; the interactive application is native PySide6.
The pipeline graph owns typed dependencies, active/unused membership and analytical
parameters. `pipeline/settings.py` translates that graph once for desktop, batch and
optimization. Display-only edits do not invalidate analytical caches.

## Data and computation ownership

- `segmentation/baseline.py`: source decoding/content identity, calibrated coordinates,
  node-local recomputation and production orchestration. Source changes invalidate
  image results; ordinary parameter edits invalidate only their dependents.
- `cuda/`, `calibration/`, `measurement/`: device-resident evidence and geometry.
  CPU transfer is limited to display, compact metadata and documented bounded topology.
- `learning/features.py`: shared live/export feature channels. Versioned feature specs
  identify raw foreground evidence; unsupported ambiguous recipes are rejected.
- `annotation/eligibility.py` and `annotation/coordinates.py`: consumer-specific shape
  eligibility and exact recorded-frame transport. Drawing progress is not scientific
  approval; complete contours clipped during transport lose shape eligibility.
- `reference_library/`: immutable library artifacts, source/group-balanced products
  and coverage audits. Coverage tiers never prove performance.
- `optimization/`: shared capability registry, fixed reference objectives, dependency
  order search and production evaluator. Optimization proposals have provenance and
  explicit review/application; they are in-sample adaptation.

## Workflow and persistence

`ui/main_window.py` remains the native coordinator. Bounded responsibilities are
extracted into `ui/results.py`, `reference_frames.py`, `work_control.py`,
`persistence_jobs.py` and `optimization.py`. Workers compute/save while Qt remains
responsive. Shutdown polls worker completion; cancellation occurs at safe boundaries
and between learned-model tiles.

`persistence/project_analysis.py` stores fingerprinted project masters that reference
mutable working sidecars. `persistence/snapshot.py` separately packages verified
content copies and restores a relocated working master. Supplementary audit evidence
is retained with its original identity; restoration does not silently reassign
review decisions or optimization targets to changed source/result identities.

`export/results.py` defines authoritative method selection, unavailable-vs-empty
state, pixel-space object rows, exact result revisions and JSON/CSV/image export.
Physical units and unvalidated traits abstain. `export/project_runner.py` loads the
same saved production recipe and source-bound references as the desktop. The legacy
`BaselineAnalysis.count` property remains a proposal-count compatibility API and
must not be used for new result reporting.

## Annotation and display coordinates

Reference and seed annotations are stored against the corrected, deskewed image
frame. The application suppresses annotation rasters, centres, brush outlines,
assisted previews, and hilum editing whenever the source image, gamut slice,
prototype collage, or another non-deskewed view is displayed. This avoids a
visually plausible but geometrically incorrect overlay.

The application window owns the movable annotation palette. The image view owns
only scene content and zoom/pan behavior. Toolbar zoom preserves the viewport
centre; wheel zoom preserves the scene point beneath the pointer. Hilum placement
is a one-shot interaction tied to the current seed ID and derives its saved
direction from the painted seed centroid.

## Learning and independent evaluation

`learning/data.py` publishes immutable component revisions before atomically changing
a manifest. Audits detect changed components, duplicate source/feature content,
group leakage and conflicting species labels. Training caches are byte bounded;
evaluation predictions use a budgeted temporary spool with deterministic cleanup.
Checkpoint loading is restricted, and cached learned models use content hashes.

`learning/evaluation_protocol.py` records immutable component/source membership and
checks checkpoint/decoder selection overlap. Development, frozen and reference-
assisted protocols have distinct target-access provenance. Recorded provenance is
necessary but cannot prove study conduct; reviewed labels do not automatically
certify scientific validity. `learning/metrics.py` uses global object assignment,
explicit empty/unavailable policies and micro/image/group reporting. Empirical
calibration and biological validation follow the separate laboratory protocol.

## Extending the application

Keep controls tied to actual calculations. Add the node's optimization capability or
an explicit nonoptimizable reason, then regenerate the catalogue. Preserve native
Qt, one launcher and device-resident reusable rasters. Run focused regressions and
the full unittest suite. Use reviewed real data only for scientific claims; fixture
and synthetic tests establish engineering properties, not laboratory accuracy.
