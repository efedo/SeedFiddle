# Seed Fiddle implementation and validation roadmap

**Direction updated:** 3 October 2026
**Engineering baseline:** `094aa7d`; prior implementation review `f244df8`
**Current summary:** [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)

## Objective

Build a reproducible, on-premises desktop workflow for reviewing, counting,
measuring, and broadly classifying soybean and lupin seeds in calibrated
laboratory photographs. A photograph contains one verified species and may contain
isolated, touching, overlapping, damaged, patterned, or partially visible seeds.

The existing software engineering workflow is implemented. The newly accepted
PlantCV–SMPTS peer-functionality extension is pending; completion of the baseline
does not imply completion of this expanded scope. Laboratory readiness is still
blocked by representative reviewed data, prespecified tolerances, independent
physical and colour measurements, trait definitions, and operator validation.
Software tests and synthetic experiments do not close those gates.

Initial species remain soybean, *Lupinus mutabilis*, *Lupinus polyphyllus*, and
*Lupinus mexicanus*. Species is operator supplied metadata; the application does
not identify it automatically.

## Deployment contract

- `seed_vision.py` is the only launch point.
- The application remains unbundled Python source with a native PySide6 interface.
  It has no server, installer, listening port, telemetry, or required cloud service.
- Python 3.12 x64 is the baseline. Python 3.13 and 3.14 are allowed only with the
  complete pinned dependency set validated on the target host.
- The optional bootstrapper may create `.venv` or use an explicit offline
  wheelhouse. It does not silently fall back to user-site installation.
- Full-resolution evidence and reusable intermediates stay on the PyTorch CUDA
  device unless Qt display, compact metadata, or bounded topology requires CPU data.
- The typed graph owns calculation parameters and dependency invalidation. Changing
  one node recomputes that node and its dependents; display-only changes do not
  invalidate analytical caches.
- Every exposed analytical control must change a calculation. Every computational
  card has an optimization route; factual, reporting, or noncomputational cards
  have an explicit exemption.
- Images under `images/` are committed fixtures. Generated logs, screenshots,
  exports, and experiments belong under ignored `artifacts/`.

## Product workflow

1. Create or open a project and load original images.
2. Assign verified species, lot/capture group, acquisition facts, and notes.
3. Detect and inspect the ruler, colour card, vessel, deskew, scale, and neutral
   balance evidence.
4. Paint or restore Background, Foreground, and Other material references.
5. Run the pipeline and inspect material, edge, lighting, quality, procedural, and
   optional learned outputs.
6. Paint and review complete seed IDs. Record condition, outline, pose, exclusion,
   and hilum location where appropriate.
7. Optionally adapt eligible nodes against project references. Review a shared
   before/after proposal before applying it. This is in-sample adaptation.
8. Review the selected instance method, accepted/excluded/unreviewed IDs, missed
   alternatives, and available pixel-space measurements.
9. Export a versioned CSV, JSON, annotated image, or viewport record. Save the
   working project or create an immutable portable snapshot.
10. Export only eligible reviewed annotations for learning. Audit grouping and
    provenance before training or evaluation.

## Implemented engineering phases

### 1. Runtime and application shell — complete

The launcher, diagnostics, optional environment bootstrap, native image and graph
workspaces, background workers, cooperative stop, responsive shutdown, log paths,
layout persistence, and Windows CPU/Qt plus optional CUDA workflow definitions are
implemented. The image is the default routine workspace; the typed graph remains an
expert view.

### 2. Calibration and analytical evidence — engineering complete

Implemented capabilities include ruler/card detection, corrected-frame transforms,
neutral balance, dual-edge Petri-dish layout, physical scale evidence, material
colour/noise models, wavelets, gradients, ridges, traces, lighting decomposition,
image-quality diagnostics, reference-edge probabilities, and shape/trait diagnostic
branches. The current default vessel type is Glass Petri-dish.

The application deliberately withholds validated millimetres and colorimetric claims.
Traceable card/camera geometry, independent dimensions, acquisition constraints, and
held-out colour patches remain required. Alternative vessel detectors for weigh
boats, watch glasses, and unbounded paper surfaces are deferred extension points.

### 3. References and annotation — complete

Material and seed references have independent draft/apply/revert state, bounded
undo/redo, source identities, corrected-coordinate transforms, and explicit legacy
alignment review. The application-owned palette supports brush, trace edge, shape
fill, smart fill, eraser, seed visibility, clear actions, Next empty, and Next
unannotated.

Condition and shape metadata are explicit. No defects excludes defect labels and
vice versa. Outline, full-length visibility, and pose define size/shape eligibility;
Exclude from modelling defaults clear. Hilum editing is location-only and one-shot,
with direction derived from the painted centroid. Annotations appear and accept
input only on the corrected deskewed image.

Load matching reference prefers saved applied labels, then the bundled source-bound
pre-annotation, then an explicit corrected-coordinate file. Loaded labels stay an
undoable draft. Bundled masks remain `reviewed: false` until a person verifies every
instance at full resolution.

### 4. Instance proposal methods — engineering complete

Procedural seed separation combines material evidence, scale-aware morphology,
boundary costs, centre evidence, marker-controlled watershed, and explicit
confidence. It is a transparent fallback and annotation bootstrap, not a validated
counter.

Native PyTorch U-Net/watershed and StarDist branches, tiled inference, training,
checkpoint metadata, dataset export/audit, global matching metrics, and evaluation
protocols are implemented. Learned nodes are disabled by default and require an
explicit compatible checkpoint. Existing controlled results are synthetic
engineering evidence; no real-image accuracy claim is made.

### 5. Species libraries and reviewed geometry — complete

Immutable species-library versions can publish source-balanced material, edge,
trait, shape, and dimensions/shape products. Projects pin exact ID/version/content
hashes and exclude the current source during resolution. Build, grouped validation,
publish, fork, import/export, retirement, and pinning are implemented. Coverage
reports inventory evidence and never certify accuracy.

Reviewed 2-D geometry includes Feret diameters, area, perimeter, equivalent diameter,
aspect ratio, circularity, roundness, solidity/convexity, pose-conditioned shape,
uncertainty sensitivity, concavity, turning, and curvature summaries. Intrinsic 3-D
shape remains unknowable from one silhouette without paired views or thickness data.

### 6. Reference-driven optimization — complete

The shared registry covers all 48 cards: 38 computational cards have optimization
capabilities and 10 have explicit exemptions. Project optimization runs eligible
nodes in dependency order with fixed reference objectives, production calculations,
isolated caches, cancellation, stale-input checks, per-image losses, before/after
overlays, proposal review, history, rollback, and project persistence.

Optimization is always labelled image-local in-sample adaptation. It does not train
model weights and cannot be presented as independent validation.

### 7. Results, export, recovery, and provenance — complete

The permanent result workflow distinguishes unavailable from a valid empty result,
selects an authoritative method, supports per-ID accepted/excluded/unreviewed review,
and creates immutable result revisions. CSV, structured JSON, annotated image,
viewport export, and saved-project batch execution share the production recipe.

Source and checkpoint hashes govern reuse. Project masters use shared mutable
sidecars; immutable portable snapshots copy and verify analytical dependencies and
restore to a new working master. Unknown legacy coordinate frames are withheld until
explicit alignment review. Physical units and unvalidated per-object traits abstain.

## Current compatibility versions

- Project analysis document: version 2
- Reference-region archive: version 6
- Analysis-settings profile: version 20
- Bundled instance-reference manifest: version 1
- Foreground feature recipe: version 2 for unambiguous raw Foreground evidence

Legacy data migrates only through the implemented readers. Never relabel an unknown
coordinate frame or ambiguous v1 feature recipe by editing its version number.

## Accepted direction: PlantCV–SMPTS peer functionality

The [comparison report](docs/PLANTCV_SMPTS_COMPARISON.md) supplies the rationale;
[the implementation and benchmark plan](docs/PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md)
is the canonical extension specification. Achieve functional peer coverage for
selected seed-relevant preparation, segmentation, extraction, geometry, colour,
QC and batch capabilities. Exact numerical compatibility, independent scientific
quality and resource/operator costs have separate acceptance evidence. No claim
of all-PlantCV coverage or demonstrated peer performance is made.

Keep the application and its current production methods. Reuse existing geometry,
learning, annotation, results and evaluation code wherever the feature audit
confirms its contract. Do not introduce a second graph, required PlantCV desktop
dependency, fixed splitter cascade or replacement results schema. Reference runs
use isolated environments; production retains CUDA residency and node-local caches.

Accurate display of the reference implementation graphs in the existing node editor
is an explicit objective (F22). Show the audited operations, typed dependencies,
branches, effective parameters and intermediate outputs with pinned recipe identity.
Distinguish original, compatible, reconstructed and enhanced graphs and mark unresolved
stages. Acceptance includes topology/parameter checks, actual Qt visual review,
persistence and cache behavior; a generic diagram or opaque backend card is insufficient.

All work packages below are pending, including the operation-level feature register.

| Work package | Deliverable and gate |
|---|---|
| WP0: scope and baseline | Pin the current recipe and external references; expand F01–F22 into exact operations/outputs, existing-code mappings, gaps, exclusions and acceptance tests. Resolve original SMPTS access or label reconstruction explicitly. |
| WP1: comparison infrastructure | Versioned reference-to-node graph mappings, reference adapters, immutable stage checkpoints/replay, matched-ID audits and conformance/resource reports; detect deliberate coordinate, mask and metric errors. |
| WP2: PlantCV capabilities | Faithful PlantCV node-editor graphs, selected compatible operations and same-input conformance evidence; reuse existing native capabilities where verified. |
| WP3: SMPTS capabilities | Display the SMPTS reference graph and recover/compare threshold, contact, erosion and extraction stages; distinguish original code from a paper port. |
| WP4: native enhancements | CUDA implementations, scale-aware alternatives and contour recovery supported by development ablations; preserve compatible modes. |
| WP5: measurements and workflow | Definition-specific measurements, eligibility, comparison overlays and provenance across desktop/batch/snapshots; retain physical and trait gates. |
| WP6: comparative validation | Locked independent study, paired grouped uncertainty, absolute/comparative gates, correction costs and documented promotion decision. |

Start WP0/WP1 while obtaining the laboratory inputs below. Methods can be implemented
and tested for engineering behavior before those inputs exist; scientific release
and default promotion require the prespecified evidence. Scope changes belong in
this roadmap and the feature register, not in retroactive benchmark exclusions.

## Remaining validation phases

### A. Intended use and acceptance criteria — required first

The laboratory must define the decisions supported by the tool, representative
hardware and workloads, acceptable count/boundary/dimension/trait errors, rejection
rules, coverage expectations, and acceptable correction time before the final test
set is opened.

### B. Independent reviewed corpus — required

Collect original images across species, lots, sessions, density, contact, overlap,
pattern, glare, damage, size, and empty/negative cases. Retain source hashes,
corrected transforms, complete evaluation regions, reviewer identity and revisions,
biological/capture groups, and rejected candidates. A second expert reviews a
prespecified subset and disagreements are adjudicated.

### C. Real-data model development and frozen evaluation — required

Use development groups for annotation, library construction, model training, decoder
selection, and parameter adaptation. Freeze preprocessing, features, checkpoint,
decoder, library, and result policy before independent testing. Report global object
matching, precision/recall, PQ/IoU, boundary quality, count error, empty-dish false
positives, calibration, micro totals, and group/stratum results. Bootstrap biological
or acquisition groups, never pixels or repeated views of the same seed.

### D. Physical, colour, and trait validation — required

Provide ruler/card dimensions or camera calibration, coplanarity and seed-height
constraints, and independent horizontal/vertical measurements across the image.
Provide traceable colour patch values and held-out patch measurements. Experts must
define operational trait labels, ambiguous cases, abstention, and adjudication, then
measure agreement and object-level error. Until these gates pass, pixel units and
trait abstention remain the authoritative output.

### E. Operator and release validation — required

Exercise the full acquisition-to-export workflow with representative operators,
project sizes, target GPUs, laptop/desktop layouts, scaling, keyboard-only use,
screen readers, colour-vision needs, interrupted saves, cancellation, snapshots, and
offline recovery. Freeze and hash the final wheelhouse and model/library artifacts
only after the workflow and resource envelope pass.

## Current verification baseline

Full local discovery during the 3 October 2026 reference-graph objective follow-up
against baseline `dfda336` ran 670 tests: 668 passed and 2 were skipped in 347.608
seconds. The suite includes corrected-frame-only
annotations, compact palette behavior, saved-reference precedence, stable zoom, and
one-shot hilum placement. The generated node catalogue and whitespace checks pass.
See [docs/CRITICAL_REVIEW_REMEDIATION.md](docs/CRITICAL_REVIEW_REMEDIATION.md) for
the finding register and [docs/SCIENTIFIC_VALIDATION_PROTOCOL.md](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md)
for the detailed empirical protocol.

## Next concrete work

1. Execute WP0: freeze the baseline and reference recipes, audit F01–F22 against
   actual code, and create the operation-level feature register with scoped gates.
2. Execute WP1: build reproducible reference adapters and stage comparisons before
   tuning or replacing production algorithms.
3. In parallel, obtain intended-use tolerances, reviewed grouped images, acquisition
   geometry, trait definitions and operator requirements; complete annotation review.
4. Deliver WP2–WP5 on development data, reusing existing libraries and learned
   branches. Keep original, compatible, reconstructed and enhanced methods distinct.
5. Freeze the complete recipe and final split for WP6 and the scientific protocol.
   Promote only capabilities that meet their functional, absolute and comparative
   gates; report blocked or inconclusive capabilities explicitly.
6. Address failures on development data and issue a new frozen revision without
   reusing the final test set for tuning.
