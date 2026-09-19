# Current operator guide

This guide describes the September 2026 review remediation. Dated review reports
and experiment logs document the state at their original date; their original
findings are retained for audit. See `CRITICAL_REVIEW_REMEDIATION.md` for disposition,
`NODE_CATALOGUE.md` for generated controls and `SCIENTIFIC_VALIDATION_PROTOCOL.md`
for the laboratory work still required.

1. Launch `python seed_vision.py`. The routine workspace opens on the image.
   Species starts **Unknown / unassigned**. Assign verified project metadata;
   this selection is not an automatic species identification.
2. Load images and inspect acquisition/calibration diagnostics. Supply Foreground
   references or an applicable pinned library. **Unavailable** means mandatory
   evidence or an instance result is missing; it is not an empty-dish count.
3. Draw reference regions or seed instances. Seed selection and **Apply + save**
   precede optional trait/shape metadata. Saving progress does not approve an
   incomplete contour as a physical seed edge. Only connected, complete,
   shape-reviewed, non-excluded outlines supervise physical contours.
4. Run the pipeline. **Stop current work** (`Ctrl+.`) requests cancellation at a
   cooperative boundary. Closing waits asynchronously for running work, then
   resolves drafts with Save/Discard/Cancel. Failed saves retain in-memory edits.
5. Open **Review / export results** (`Ctrl+R`). Inspect proposed, accepted,
   excluded and unreviewed counts. Review decisions are tied to the exact source,
   recipe, calibration, annotation and prediction revision. Double-click a row
   to inspect it. Inspect unselected procedural candidates for missed atypical
   seeds; an alternative hypothesis is not necessarily a separate rejected seed.
6. Export CSV, structured JSON and an annotated PNG as a new result revision.
   Visible 2-D pixel extents and areas are separate from inferred full shapes.
   Physical units and unvalidated trait assignments are withheld. Uncalibrated
   scores are not probabilities of correctness. The annotated PNG is a full-image
   ID/decision overview; **Save current image viewport** (`Ctrl+Shift+E`) instead
   captures the current zoom and overlays for display documentation.

## References and recovery

New reference archives include the source-to-corrected transform. Calibration
changes reproject labels with nearest-neighbor sampling and transform landmarks;
shape changes are not repaired by arbitrary resizing. Old archives with unknown
transforms are withheld. **Analysis → Review legacy reference alignment** offers
an explicit overlay inspection when canvases match. Different legacy canvases
require the original transform or new annotations. Keep the original archive.

Replacing an image in place invalidates its cached calculations. Existing in-memory
annotations cannot silently bind to new bytes: restore the original file or start
a fresh workspace and revalidate its references.

Working project masters still refer to shared mutable sidecars. Use **File → Create
immutable portable snapshot** to freeze verified copies of images, annotations,
manual centers, enabled model checkpoints and the pinned species library. A
snapshot publishes its completion index last. **Restore portable snapshot** creates
a separate verified working copy and reports the new master path. Open that path
with **File → Open project**; the original snapshot remains unchanged. Missing or
modified snapshot components prevent restoration.

## Keyboard and expert workflow

- `Ctrl+1` / `Ctrl+2`: image/pipeline visibility; an empty workspace recovers the image.
- `Ctrl+K`: search active nodes and frame the selected node with its neighbors.
- `Ctrl+Z` / `Ctrl+Shift+Z`: annotation undo/redo. New edits clear redo; applying or
  reprojecting annotations is a history boundary. History has count and byte limits.
- `Alt+Left` / `Alt+Right`: previous/next existing annotated seed.
- `Ctrl+Alt+B`, `Ctrl+Alt+T`, `Ctrl+Alt+F`: brush, edge trace, smart fill in annotation mode.

Inspector labels wrap above their editors. Splitter sizes and palette position are
remembered. The full typed dependency graph, intermediate diagnostics and node-local
cache invalidation remain available. Use the generated catalogue and
`NODE_OPTIMIZATION.md` for node and sequential project optimization.

## Training and evaluation

Learning export requires complete reviewed contours for physical-boundary targets,
verified species for species-conditioned features, and an explicit biological/capture
group. A filename is not a grouping policy. Related captures must share one split.
Exports create immutable component revisions before switching the audited manifest;
old features/labels remain paired if writing fails.

Feature specification v2 consistently means raw Foreground colour evidence in both
live inference and export. V1 foreground recipes are ambiguous and must be re-exported
and retrained; do not simply change the version number. Colour-only v1 specifications
remain compatible. Checkpoints load only tensor/primitive state dictionaries through
PyTorch's restricted loader; unrestricted legacy pickle import is not offered.

GUI exports record image-local adaptation and cannot create a locked test split.
The evaluation CLI supports `--evaluation-protocol development`, `frozen` or
`reference_assisted`; independent protocols require reviewed membership and upstream
provenance. Scientific validity is never inferred from a checkbox, source count or
split name. Dataset audits verify component hashes, duplicate content and grouping.
Reports include dataset/checkpoint/decoder identities, micro totals and group results;
unavailable metrics use null. Training has a bounded LRU sample cache; evaluation
spools predictions within a disk budget and retains only bounded thumbnails.

Batch execution uses the same saved recipe and reference archives:

```powershell
python scripts/analyze_pilot.py --project projects/analysis.seedfiddle-project.json --output-dir artifacts/project-results
```

The batch command rejects unresolved fingerprints and unknown reference frames.
`BaselineAnalysis.count` remains the legacy proposal-count compatibility property;
new consumers must use `seedvision.export.results.result_report` for authoritative
instance results and availability. It never silently substitutes a legacy zero.


Snapshot supplementary evidence includes configuration/trait vocabulary, review
history and project optimization records/targets. The snapshot index maps their
original identities to verified content copies. Restoration preserves that index;
supplementary decisions are not automatically attached to relocated result IDs.
Review decisions record UTC time and the local account and retain append-only
history. This is an audit trail, not an identity-authentication system.

Parameter-section arrow buttons collapse controls without changing calculations.
Learned inference accepts Stop between tiles; already running GPU kernels and
filesystem commits complete safely. Legacy coverage tiers displayed as validated
are explicitly identified as unvalidated performance in the library manager.


The result table filters by decision as well as seed ID. Choose Unreviewed or
Reviewed and use **Next matching seed / Alt+N** to move through that subset.
**Compare completed instance methods** reports available branch proposal counts;
use their overlays to inspect boundaries. An unavailable method must be enabled
and run before comparison, and count agreement alone is not validation.
Image-list icons distinguish calculation, computed results and unapplied/unsaved
edits; hover or accessible descriptions give the textual status. Saved result
review counts are shown for the current session.

Viewport export preserves the displayed pixels and adds a caption containing the
overlay name, interpretation key and pixel-space scale bar. It does not assert a
validated physical length. Use result export for structured measurements and
revision provenance.
