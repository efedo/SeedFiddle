# Seed Fiddle

Size and units saved-stage replay checkpoint (9 October). `analysis.size` now replays original RGB/labels and all seventeen observation records in source or verified seed-ROI frames, comparing the optional CUDA diagnostic separately. Exact source descriptors, nominal datatypes and finite local coordinates are preserved. `calibrated_seed.size_units` replays the compact physical conversion with zero selected arrays and binds the saved size/frame and card-scale producers. Its conversion allocates no CUDA tensors; the shared runtime audit has a separately admitted 64 KiB reserve. Upstream label maxima and card measurements are not recomputed by the units stage. There are now 22 declared replay operations.

The shared 15-module gate passes 185 tests in 1155.051s before the final audit-budget refinement. The final affected size/unit and existing planner/contract gate passes 28 tests in 86.761s after that refinement (zero skips; overlapping gates). Nine final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled direct-input 1027x1025 graph uses 128 actually computed labels with independently prepared, hashed gray/RGB arrays; it makes no single-image decoding claim and changes no upstream outputs. All seventeen size variables agree with pinned public PlantCV under the declared 1e-6 numerical tolerances and ten-decimal longest-path check; the diagnostic agrees byte-for-byte. The public explicit-one empty-mask behavior agrees too. Source replay selects three arrays (10,526,750 bytes), compares 3,171,593 values and adds 21,663,232 CUDA bytes within 268,960,128 planned; 13,312 preexisting bytes are separate. Whole source replay traces 63,354,720 additional peak host bytes. Physical/pixel units select no arrays and add 15,872/15,872 CUDA bytes within the separate 65,536-byte shared-audit reserve; an isolated cold audit measured 29,184 bytes. Encoded metadata caps do not guarantee Python RSS. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence.

Freeze: artifacts/native-size-replay-frozen-hashes.json; final gates: artifacts/native-size-replay-final-shared-gate.log and native-size-replay-final-gate.log; source/oracle/recipe/archive/output/runtime hashes and actual Qt views: artifacts/native-size-replay-probe-final-v2/evidence.json. Actual async size/unit summaries and the size image were inspected at 1440x900 and 1050x650 with Windows fonts; the active run and every archive were preserved. Review uses supplied projections and root execution/visual evidence, with no independent file reads/tests/visual inspection claimed.

The F01-F22/WP0-WP6 goal remains active and partial_native. Next: archived-model classifier replay, then remaining card, recovery and paper helper contracts and the complete method/input/domain audit. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required before completed-goal handoff.

Earlier verified checkpoints follow.

Colour saved-stage replay checkpoint (9 October). `analysis.color` now supports typed saved-input replay in source and verified calibrated seed-ROI frames. All RGB/LAB/HSV/ALL histograms, bins, statistics, label records and returned hue-chart fields are compared; explicit nonfinite tags preserve NaN/inf outcomes. The native Qt view shows the returned chart and concise observations while preserving the active run. There are now 20 declared replay operations.

The shared 15-module gate passes 179 tests in 979.683s before the final character-length precheck; the final affected colour gate passes 14 tests in 125.179s after that refinement (zero skips; overlapping gates). Six final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

See [all requested feature statuses](docs/FEATURE_IMPLEMENTATION_STATUS.md) and the checkpoint evidence recorded in HANDOFF.md.

F01-F22/WP0-WP6 remains active and partial_native. Next: typed size/unit observation replay, then remaining model, card, recovery and paper helper contracts and the full method/input/domain audit. Original SMPTS source/license/dependencies and laboratory inputs remain separate open gates. Fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

Calibrated range workflow checkpoint (9 October). The additive `plantcv_calibrated_seed_range` graph corrects the colour card, crops an explicit seed ROI, applies channel-range selection, smooths its binary mask, and executes fill/morphology/watershed plus size/colour observations. Traits use the original corrected cropped RGB; the range masked RGB is a pre-median diagnostic. This is a user-selected composition of pinned operations, not a recovered historical fixed-range workflow or scientific validation.

The final 11-module gate passes 143 tests in 647.331s, including 11 new integration tests, with zero skips. All nine prior calibrated/diagnostic recipe serializations remain exact. Tests cover five spaces plus partial ranges, complete pinned corrected-ROI outputs/physical observations, range/median/ROI/units cache boundaries, empty-result recovery, fresh/cached/debug admission, excess input backing, final callback/runtime mutation, actual Qt replay/Keep/reopen and saved-project/batch/archive round trips.

The Reference graphs calibrated seed selector now includes range. Choose the seed ROI, channel and exact-length inclusive byte bounds, then run the same native/project/batch workflow. See [all requested feature statuses](docs/FEATURE_IMPLEMENTATION_STATUS.md).

F01-F22/WP0-WP6 remains active and partial_native. Remaining model/helper/paper/observation replay and the full method/input/domain freeze remain open, alongside original SMPTS source and laboratory gates. Fresh full unittest discovery is required before completed-goal handoff. Earlier source freezes retain historical scope when shared modules change.

Earlier checkpoint history follows.

Custom-range graph and saved-stage replay checkpoint (9 October). The separate `plantcv_threshold_custom_range` recipe now executes the existing pinned CUDA primitive with real RGB/HSV/LAB/CMYK/GRAY and inclusive byte-bound controls. Both the binary mask and masked original RGB are retained, exported and replayed. RGB bounds use pinned B,G,R storage-slot semantics; CMYK retains the source C/Y swap. The four existing extended recipe identities and historical reference catalogue remain unchanged.

The final shared gate passes 133 tests in 220.734s (11 new range tests; zero skipped). Actual 887x887 LAB/CMYK probes select 353,079/214,138 pixels and match both pinned public outputs and isolated saved-input replay. Replay adds 29,599,232/52,102,656 CUDA bytes within 234,924,160 planned; 13,312 preexisting bytes are separate. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence. There are now 19 declared saved-stage replay operations.

See [all requested feature statuses](docs/FEATURE_IMPLEMENTATION_STATUS.md). In **Reference graphs**, choose **Run extended threshold**, then **PlantCV custom-range threshold**. Select an image; set channel and lower/upper JSON byte lists. RGB/HSV/LAB require three bounds, CMYK four, GRAY one. Run, Keep recipe in project or Export run. Saved-stage replay previews the masked original RGB without replacing the active run.

Earlier checkpoint history follows.

SMPTS executable paper reconstruction checkpoint (8 October). The additive `smpts_paper_fixed_mean` graph now executes the paper-supported branch/filter/intersection/contact/erosion/extraction sequence. Defaults25/5/13 come from the paper; offset/polarity, fixed cutoff, connectivity, axis-aligned boxes, box-area cutoff, border, erosion shape and crop rules remain declared assumptions. The historical unresolved graph is unchanged.

The 111-test relevant gate passes in 117.973s (19 new tests;0 skipped). The 2080x3120 committed fixture matches nine pinned OpenCV substep rasters and all 1118 packed crops. This verifies the declared reconstruction, not original SMPTS conformance or scientific accuracy. The full F01-F22/WP0-WP6 goal remains active; full discovery is still required before completed handoff.

Open Reference graphs, choose **SMPTS paper port…**, choose an image and run. Inspect the ten calculation nodes; the extraction node offers full-frame final labels and individual original RGB crops. Keep the selected recipe in the project for exact later/batch execution, or export its run. Assumed controls are editable and affect their own calculations/dependents.

Earlier checkpoint history follows.

Extended threshold saved-stage replay checkpoint (8 October): eighteen declared
operations now support saved-input replay. Gaussian, triangle, texture and
dual-channel each replay their existing standalone source-frame graph. The
reader selects exactly the input raster and expected mask; calculation and
every binary-mask comparison remain on CUDA. Producer-port/frame/runtime
binding and active-run preservation use the existing guarded replay path.
Malformed controls/headers are invalid archives; valid wrong masks report
mismatch. Source precision/preamble/resource limits report unavailable without
products. Cancellation and mutation withhold all incomplete outputs.

Primitive execution and replay share allocation-free tile/scratch admission.
Gaussian 11/13/15 retains its 1B window-sample cap; texture retains 4M pixels
and bounded pair/sort scratch; triangle retains the 2**24 histogram source
ceiling beneath the stricter replay CUDA ceiling. Dual counts its RGB-to-BGR
copy and at most two distinct tiled conversions. Selected arrays remain capped
at 48MiB and CUDA at 1GiB. The fourteen earlier plans are unchanged. Baseline
replay admission units and primitive visits are recorded separately; neither
is an instruction count or elapsed-time guarantee. Texture coverage concerns
binary decisions, not byte parity for its internal texture map.

Verification: 141 relevant tests pass in 619.534s before the final CMYK repair;
then 58 affected tests pass in 212.202s after the repair (0 skipped;
gates overlap). A new CUDA regression compares all four CMYK channel bytes
against the pinned helper on a 196,625-pixel pattern including all 256 neutral values.
The shared conversion now uses a CUDA float64 tensor divisor, preserving true
double division before byte truncation; the earlier scalar reciprocal path
caused 30 public dual-mask disagreements in the larger probe. Seven actual archives match
pinned public PlantCV masks, including guarded Gaussian and LAB/CMYK dual
conversion. Actual asynchronous Qt replay works for all four operations.
Three controlled probes cross three tiles plus a tail and match public/native
masks and isolated replay: gaussian 125x241: 59,357,696 additional / 138,073,728 planned CUDA bytes, 13,312 preexisting; texture 113x101: 24,600,064 additional / 138,481,343 planned CUDA bytes, 13,312 preexisting; dual_channels 275x715: 12,585,472 additional / 159,385,728 planned CUDA bytes, 13,312 preexisting.
Primitive scratch stays within the shared allowance; archive hashes remain
unchanged. These probes do not establish maximum-domain or corpus coverage.

Bounded review approves the five files recorded in
artifacts/extended-threshold-replay-frozen-hashes.json using supplied projections
and root execution evidence only; no independent reads/tests are claimed.
Earlier freezes and fixture evidence retain historical scope. F01-F22/WP0-WP6
remains active, partial_native and scientifically unvalidated. Complete method
matrices, remaining model/helper replay, broader domains, original/SMPTS
execution and laboratory gates remain open. Fresh full unittest discovery is
required before the completed-goal hand-off.

Earlier watershed debug saved-stage replay checkpoint (8 October): fourteen declared
operations now support saved-input replay. The new debug contract works in all
six source/calibrated binary, mean and Otsu graphs. It binds both incoming
branches and each raster frame, selects exactly RGB input, int32 labels and
expected RGB, then compares every fresh image value on CUDA. Its frameless
compact output has an exact 31-field schema and immutable palette/provenance.
Primitive decoding has a conservative 256 KiB encoded budget, 30,000 visits and
depth 4. Every leaf is compared, including each missing palette channel.
Structurally valid wrong counters, palettes, populations and pixels report
mismatch. Malformed contracts, incompatible runtimes, cancellation and native
resource unavailability remain distinct; incomplete products are withheld.

Selected arrays admit up to 128 MiB. This operation uses CUDA admission
max(128P+128MiB,13P+256MiB+64min(P,1M)+3MiB)<=1GiB, where P is frame pixels.
The previous thirteen plans are unchanged. Native scan/topology/geometry/host
and painter limits remain effective; bounds do not guarantee runtime or RSS.

Verification: 113 tests pass in 659.099s (0 skipped); then 15 tests pass in 156.141s (0 skipped); gates overlap.
Fresh 2080x3120 verification matches pinned public PlantCV labels, palette and
debug pixels (1294 labels). Final isolated replay adds 118,130,688 CUDA bytes
within 964,886,528 planned bytes and takes 178.878s; upstream tensors
are released and 0 preexisting bytes are separate. Only 64,896,000
array bytes are selected. All 19,468,800 RGB values and 5,208 metadata leaves
match, and archive hashes remain unchanged. The controlled grayscale/gray-repeat
RGB, cutoff 100, fill 1 and peak-distance 8 fixture remains diagnostic coverage,
outside the separate 128-label trait domain. Geometry remains near its 64M cap.
Actual replay-specific Qt views were inspected at 1440x900 and 1050x650.

Bounded review approves the four final source/test files recorded in
artifacts/native-watershed-diagnostic-replay-final-hashes.json, using supplied
projections and root execution/visual evidence only. Earlier freezes retain
historical scope. PlantCV returns labels only; the image remains a debug side
effect, and count is separate. F01-F22/WP0-WP6 stays active, partial_native and
scientifically unvalidated. Remaining operation/helper replay, broader domains,
original/SMPTS execution, laboratory gates and fresh full unittest before
completed hand-off stay open. Next: saved-stage Gaussian, triangle, texture and
dual-channel thresholds with source-derived precision/resource admission.

Earlier watershed debug overlay checkpoint (8 October): six additive binary/mean/Otsu
graphs support the source frame or calibrated seed ROI. PlantCV returns labels
only; the new contour image implements its separate debug side effect. The
count remains a separate observation. Colormap, explicit reproducible uint32
shuffle seed and line thickness affect the image and reuse segmentation.

The private CUDA RGB renderer preserves pinned contour order, holes, islands,
overdraw, closure and LINE_8 pixels. Adjacent original edges are drawn in bounded
batches, without contour approximation. Strict callback/storage checks withhold
partial products. Shared scan/topology/stroke/paint limits and a 256MiB modeled
host geometry allowance remain explicit. Composed CUDA admission is
max(base,64P+16min(P,1M)+256MiB)+3P <=1GiB, including the private canvas;
larger input backing stores are counted. Bounds do not guarantee process RSS or runtime.

Verification: 30 tests pass in 80.641s (0 skipped); then 183 tests pass in 645.147s (0 skipped); gates overlap. The 2080x3120 committed fixture matches
actual public labels, palette and diagnostic pixels (1294 labels).
Additional CUDA peak is 517,786,624 bytes against 719,238,656
planned bytes; caller source backing is excluded from the measured additional peak.
Native execution took 276.814s. Geometry used 63,164,962/64M
work units; peak modeled host geometry was 12,369,424 bytes.
This geometry result is near the cap and does not establish broader input coverage.
The full-fixture oracle uses decoded grayscale and explicit gray-repeated RGB,
binary cutoff 100, fill size 1 and peak distance 8. Its 1,294 labels exceed the
separate 128-label downstream measurement domain. This is diagnostic coverage.
Actual async calibrated Qt views were inspected at 1440x900 and 1050x650.
Bounded review approves eleven frozen files from supplied projections and root
execution/visual evidence only: artifacts/native-watershed-diagnostic-frozen-hashes.json.
This supersedes current hashes of files changed since earlier checkpoints;
their recorded freezes retain historical scope. Earlier wording about a
public diagnostic return is corrected here to a debug-only side effect.

F01-F22/WP0-WP6 remains active; affected features stay partial_native and science
is unvalidated. The new debug stage itself still lacks saved-input replay.
Remaining replay/domain/original/SMPTS/laboratory gates and fresh full unittest
discovery before completed hand-off remain open.

After reconciliation, all six feature-register checks pass in 0.637s; every
existing test reference resolves. The canonical peer manifest and all eleven
frozen source/test hashes are unchanged. No task test process remains live.

Earlier typed watershed saved-stage replay checkpoint (8 October): five typed watershed stages now replay
exact saved inputs in source and canonical calibrated seed-ROI frames, bringing
the supported total to thirteen operation contracts. Every output is compared;
coordinates, boolean peaks, marker/instance IDs, exact completion and count
records have explicit contracts. All branches, frame ancestry and archive bytes
are verified. Selected data and calculation stay on CUDA. The five new stages
admit 128 MiB selected data within the unchanged replay CUDA envelope; earlier
operations retain 48 MiB. Peak suppression uses equivalent separable maxima and a
separate ceiling of 64 billion estimated NMS sample visits. Full-resolution typed replay and pinned chains
pass. 93 tests pass in 563.810s (0 skipped), then 5 tests pass in 19.608s (0 skipped) after the final count-display repair (overlapping gates).
Actual maxima/count views are checked at both window sizes. Full implementation,
remaining replay/domains/source and scientific gates stay open; functionality
remains partial. See [current status](docs/CURRENT_STATUS.md).

Earlier Calibrated local-stage replay checkpoint (8 October): eight saved-input raster operations now
also support the explicit seed ROI in calibrated binary/mean/Otsu graphs. Parent
and local frames, transform, producer-port ancestry and recomputed card overlap
are checked before selected pixels load. Only saved local inputs/expected output
are uploaded; local dimensions drive resource admission. Full-bounds crops retain
their distinct local frame. 35 focused tests pass in 281.940s (0 skipped); actual Qt replay views are checked at both
window sizes. Active runs and archive bytes are preserved.
Full feature implementation remains active; affected features stay partial and
scientific validation is false. Other stage contracts, broader domains and
source/laboratory gates remain open. See [current status](docs/CURRENT_STATUS.md).

Earlier Saved-stage replay checkpoint (8 October): Replay saved stage now reruns eight supported
full-source-frame raster operations from verified archived inputs, compares
fresh CUDA output and preserves active runs/history. All archive payloads are
verified with bounded selected loading; runtime/settings and input lineage bind
the attempt. Qt execution, cancellation and both window sizes are checked.
Integration gate: 171 tests pass in481.283s (0 skipped); archive-origin/watershed gate: 12 tests pass in35.600s (0 skipped); final API/Qt gate after the cropped-frame outcome repair: 17 tests pass in122.279s (0 skipped). These gates overlap and are not a count of distinct tests.
Local/cropped, UHD, other stage contracts and broader replay remain open.
Computational agreement does not establish scientific validation or full parity.
See [current status](docs/CURRENT_STATUS.md). Full feature implementation remains active.

Earlier F13 native recovery graph checkpoint (8 October): Experimental core
recovery now connects threshold/erosion, four-connected core IDs, explicit
negative-distance elevation, full-frame validity, partition, uncertainty and
permitted pixel measurements. Radius/policy edits reuse labels. Qt execution,
preview, Save/Keep/reopen, project move, companion batch and archive integration
are tested. Known validity-clipped IDs and unknown unseeded foreground remain
explicit, and scientific validation stays false. The committed full-resolution
markers/labels match the pinned source chain within the joint CUDA plan.
Final recovery/preview gate: 20 tests in 141.423s (20 pass; 0 skipped). Shared executor/Qt/project/export/classifier compatibility gate on the final frozen source: 198 tests in 370.864s (198 pass; 0 skipped).
Bounded supplied-projection/root-evidence review covers seven frozen files. F13
remains partial_native and WP4 is in progress, with corpus ablations, complete
replay and laboratory gates open. See [current status](docs/CURRENT_STATUS.md).
Full feature implementation remains active.

Earlier f13 recovery primitive checkpoint (8 October): the experimental CUDA API
now partitions valid original foreground from supplied eroded-core IDs and
explicit elevation, preserving core IDs and the original union. Unseeded
foreground stays unresolved; shared boundaries and invalid/canvas support are
flagged. Separate radius/policy annotations reuse labels and do not establish
biological identity, hidden outlines or scientific validation. The21-test
recovery/watershed gate passes49.845s, including actual pinned-source equality
on the committed full-resolution fixture and bounded allocator checks. Bounded
review covers three frozen source/test files using supplied projections/root
execution. F13 remains partial: native recipe/UI/project/batch/archive integration
and contact/laboratory controls are open. See [current status](docs/CURRENT_STATUS.md)
for evidence. Full feature implementation remains active.

After documentation reconciliation, all six feature-register checks pass in0.579s; every existing-test reference resolves. No test process remains live.

Earlier Bayes mask-coordinate checkpoint (8 October): Images and masks training
now accepts different dimensions using source row/column coordinates. Every
0/255 mask coordinate must lie within the image; other values are ignored,
including outside it. The implementation preserves source failure/RNG order,
CUDA intermediates and separate image/mask resource accounting. New directory
records use parser-v2; old parser-v1 bytes, saved projects and archives remain
readable. The focused gate passes59 tests and the shared20-module gate passes258
tests in358.626s. Bounded review covers seven unchanged files using supplied
projections/root execution. Other formats, full parity and scientific validation
remain open. See [current status](docs/CURRENT_STATUS.md) for evidence.

After the eight-document/register reconciliation, all six register checks pass
in0.573s and every existing-test reference resolves. No test process remains live.

Earlier Bayes constant-channel checkpoint (8 October): native training now retains
the pinned source's successful constant-channel roundoff fits and genuine
singular failures, without a variance floor. Separate typed constant records
preserve legacy records and unchanged TSV models. Actual source PDF comparisons,
both desktop training modes, project move/source removal, companion archives and
optional SVG plots are covered. The20-module gate passes248 tests in330.530s;
bounded review covers nine unchanged files using supplied projections/root
execution. Numerical normalization allowances remain empirical and runtime
limited. Different mask dimensions, broader domains and full implementation/
validation gates remain open. See [current status](docs/CURRENT_STATUS.md) for evidence.

After reconciling the eight documents/register, all six register checks pass
in0.582s and every existing-test reference resolves. No test process remains live.

Earlier Bayes buffered-reduction checkpoint (8 October): weighted CUDA statistics now retain
the pinned NumPy 2.2.6 default 8,192-element buffer order. Actual moments match
231 fixtures / 693 channels, and all 256 actual KDE PDF bins match on tested sizes
through 1,000,000 samples within declared empirical and separate tighter bounds.
New training records identify the fixed reference reduction profile; legacy
records remain readable and profile-only metadata edits reuse predictions.
The 20-module compatibility gate passes 240 tests in 308.330 s. Constant-channel
training, broader source domains and full implementation/validation gates remain
open. See [current status](docs/CURRENT_STATUS.md) for checkpoint evidence.

Naive Bayes training is available in the native desktop through **Train model…**:
choose **Sampled RGB table** or **Images and masks**, train asynchronously, then
save the exact PlantCV TSV. Directory jobs preserve source pairing/order and
the existing global random stream; Stop and closing drain the worker. Only a
committed, reread and verified model is selected. **Keep in project** and **Save
Project** retain the verified model and distinct typed training record for
reopen/move, companion batch and whole-run archives without original inputs.
The previous plotting compatibility gate passes236 tests, including15 new plotting
source/UI cases and the existing directory integration checks. PNG/JPEG contents use source row/column mask indexing. Every0/255 mask
coordinate must be in bounds; other values are ignored. Singular or uncertain fits remain unavailable. Optional **Save probability density plots (SVG) beside the model** writes
ordered hue/saturation/value PDFs after verifying the TSV. Numerical conformance
is empirical; broader domains and full implementation/validation gates remain open. See `docs/CURRENT_STATUS.md` for checkpoint evidence.

Earlier calibrated seed workflow checkpoint (8 October): three additive
24-node native recipes connect automatic colour-card affine correction to a
required half-open seed ROI, corrected RGB-to-gray conversion, median/threshold/
fill/morphology/watershed and size/colour observations. Binary, mean and Otsu
variants retain the existing source operations and image-specific controls.
Labels and measurements use a declared local pixel frame; `labels_coordinate_frame`
and compact crop metadata preserve the exact translation back to the source.
Physical chip dimensions update only calibration metadata and derived size values;
pixel measurements and segmentation remain reusable. CUDA products, combined
preflight, verified helper closure and changed-node/descendant caching are retained.
Qt launcher/previews/Keep/reopen, project persistence, actual companion batch and
whole-run archives are implemented. The seven-module compatibility gate passes
100 tests in 207.409s; the six register checks pass separately. Actual1440x900
launcher, seed crop/labels and calibrated-size views were inspected. Bounded review
covers supplied source projections and root execution evidence only.
The selected public fixtures match all seed intermediates/labels/diagnostics and
colour histograms; physical size observations use source metadata and declared
numeric tolerance. Whole-image correction retains the one-byte tolerance. The mean
fixture's unstable ellipse size remains unavailable; no fallback is substituted.
Full F01-F22/WP0-WP6 remains active: broader computational/input domains, helper
graphs/training gaps, complete per-stage replay, original/SMPTS execution, fresh
full-suite verification before completed hand-off and laboratory gates remain open.

Automatic colour-card correction is available in the reference graph
overview through **Automatic colour card…**. Choose the image, select a node to
edit its controls, and run the selected recipe. ROI bounds are half-open
`[x0,y0,x1,y1]`; use `null` for the whole image. Chip radius controls sampling
circles. Physical chip size accepts a named card or `[width,height]` in mm and
reports calibration metadata. The standard orientation defaults to3, matching
automatic correction. **Keep recipe in project** keeps the selected controls and
layout; **Save Project** writes the project. Reopen through **Project native
recipes**. **Export run…** saves the available products and execution record.
Unsupported or uncertain calculations display failed/unavailable descendants.
This workflow has bounded public computational comparisons, not scientific
colour-accuracy validation.

Current desktop checkpoint (8 October): correction Keep/reopen retains verified
four-file selections for the exact project image. Keep merges compact session
metadata after worker capture and stale/cancel checks; Save Project saves the
master. Reviewed evidence and remaining full-scope gates are recorded below.

Earlier bounded checkpoints (8 October): reviewed size tracing supports holes,
nested islands and disconnected regions, pinned contour order/hierarchy and all
17 public observations within supported geometry. Numerically ambiguous ellipse
orientations remain unavailable without angle normalization or relaxed tolerance.
Explicit line thickness now reaches size measurements and CUDA RGB diagnostics;
the source's hard-coded-five caliper fallback remains separate from overlay width.
One/five-pixel public observations and complete images pass, including the actual
isolated worker's one-pixel protocol on the tested fixture. The size/raster gate
passes 48 tests in 90.267s. Additive native working seed copies expose effective
thickness editing, node-local invalidation, async preview, reset and save; their
24-test API/Qt gate passes in 70.210s. Historical catalogue graphs are unchanged.
Gaussian kernels 11/13/15 use pinned float32 coefficients and withhold masks at
uncertain byte-rounding boundaries; work and actual multi-tile scratch stay bounded.
The extended-threshold gate passes 22 tests in 71.401s; the larger-kernel recipe/
desktop/cache gate passes 14 tests in 90.053s. These additions have bounded source
review. Actual 1440x1000 size-thickness and classifier views were inspected.
Frozen-model Naive Bayes has reviewed recipe/API/desktop binding and class previews.
Frozen-model K-means prediction is now implemented and independently reviewed:
37 focused tests pass in26.683s against isolated pinned PlantCV/sklearn calls.
This includes sigma5 preprocessing, ordered gray/RGB patch tiles, bounded numeric
model identity, exact integer ties and guarded fractional nearest-centre decisions.
Prepared CUDA centres are reused across tiles; actual allocation, combined work,
cancellation and input preservation are covered. The separate model/predict/masks
working recipe and desktop workflow are also reviewed;27 API/Qt/compatibility tests
pass in22.282s, including effective category edits and image/prediction cache reuse.
Canonical numeric-model export is also reviewed;24 focused model/prediction/workflow
tests pass in13.257s after the incremental8MiB serializer addition.
Reviewed training components now provide unpadded sigma5 full-patch extraction
and source-ordered unit-weight MiniBatch centre/count updates on CUDA. Eight
training-data tests and ten update tests pass against pinned public/Cython calls,
including exact binary64 values, measured storage, cancellation and mutation checks.
The prior reviewed training/watershed/register gate passes38 tests in47.640s.
Sampled sigma5 patches, exact estimator RandomState calls and byte-valued CUDA
k-means++ initialization are also reviewed, with9/7/7 focused tests passing.
Public PlantCV/sklearn calls verify row order, selected centres and exact RNG state.
The assembled training/register gate passes47 tests in18.869s.
Compact MiniBatch reassignment timing, scalar convergence and low-count planning
are now independently reviewed:10 focused tests pass in1.685s against actual
pinned estimator methods, including byte-exact updates/replacements/count resets,
RNG state and all-selected failure order. Scalar control requires certified
inertia inputs; the complete bounded byte-feature fitter is recorded below.
The combined control/RNG/update/register gate passes33 tests in5.636s.
Mutable-centre CUDA scoring/inertia and compact interval convergence are also
reviewed, with7/8 focused tests passing. Integral totals are exact; fractional
source reductions have certified bounds, and undecidable decisions withhold the
result. The scorer/interval/control/classifier/register gate passes37 tests
in7.315s. The bounded byte-feature MiniBatch fitter and public-tolerance producers
are now independently reviewed. Their strengthened combined gate passes12 tests
in5.469s against traced actual public fits, including every RNG draw, initialization,
update/reassignment/convergence decision and final outputs. A4097x243-row fixture
covers complete custom-stream allocation/cancellation; variance/shift intervals
retain source rounding and withhold undecidable decisions. Total job work and live
storage are checked before RNG/output allocation. Ordered CUDA image training
and compact fitted-model export are also implemented and independently reviewed:
8 image-training/interoperability and7 export tests pass. Source image order/BGR,
per-image patch seed1, complete fit parameters, reusable CUDA rows and combined
preflight are preserved. Exported numeric models match actual public joblib
predictions/masks; only compact centres transfer to CPU, with measured metadata
peak and incremental8MiB serialization. The metadata-only fit preflight now has9
passing fitter tests. Bounded directory selection/materialization is independently
reviewed:14 directory tests and9 ordered-image tests pass together in8.434s.
Selection preserves sorted names, independent Python random choices, duplicate
occurrences and the source's index-zero failure. Qt header metadata bounds OpenCV
decoding; all matching regular files, including decoder skips, receive byte identity.
Unique images upload once; repeated choices retain repeated CUDA feature rows.
Explicit all-reduction provenance is also reviewed:26 interval/scorer/fitter tests
pass in7.584s, including public fits at1/2/4 threads. No host count is inferred.
Native directory training, reusable selection/row caches, async controls, worker
draining and atomic model saving are independently reviewed:21 new/existing
API/Qt workflow tests pass in13.657s. Fit edits reuse rows; extraction edits retain
the file selection. Saved centres enter the existing picker and invalidate only
model descendants. Persistent training provenance is now implemented and reviewed:
one bounded artifact retains the ordered byte-identified selection, effective
settings, fixed patch seed, exact estimator RNG identity, fit plan/scalar intervals,
loaded producer/runtime identities and pinned sources. Legacy centres-only files
remain compatible; recipe selection binds the entire artifact's raw bytes.
Production PlantCV model conversion is now implemented and reviewed. Shared
standard-library codecs retain legacy import aliases; the explicit converter runs
in the source training environment and writes the same atomic numeric artifact.
Eleven conversion tests pass in6.181s against actual pinned public training,
prediction and masks;38 legacy/model/native tests pass in15.916s after the move.
Native whole-run output persistence and saved-project companion execution are now
implemented. The shared exporter records exact recipe/source/frame/runtime, ordered
node states and effective controls, compact observations, every available CUDA
product and verified raw classifier-model bytes. Desktop **Export run...** drains
its worker; project batch executes exact saved native-compatible attachments and
records failures or unavailable original/reference executors explicitly. Historical
graph records and baseline reports retain their own identities. Array groups support
256 ordered class masks; metadata supports the full 128-label colour domain.
These output archives are not per-stage replay certificates or scientific validation.
The reviewed native-output/project/register integration gate runs142 tests in109.556s:
141 pass and one existing platform-specific test skips.
Project schema6 retains editable native working recipes separately from historical
graph attachments, with safe migration from versions1-4. Seed, extended-threshold,
Bayes and K-means dialogs expose Keep recipe in project; Save Project captures the
selected controls, actual canvas layout and available verified model copies.
Project native recipes reopens or removes working selections. Historical recipes
and StageRunRecords retain their exact identities even when a working selection
shares an ID but selects a different model. Snapshot closure covers both collections.
Batch uses the same runner and records historical_graph or working_recipe rows;
working recipes have no fabricated historical stage records. Native output schema3
reads v1/v2 and additionally supports four-file correction outputs. The expanded18-module gate runs279 tests in233.278s:278 pass and
one existing platform skip. All16 new working-recipe tests pass in10.003s; the two
tests added after the expanded run are included in that final focused gate.
Bounded independent approval covers supplied source/test projections and root
execution evidence. Actual1440x900 overview/working views and menus were inspected.
Complete per-stage replay, original/SMPTS execution, remaining feature domains
and fresh full-suite/scientific gates stay open; correction project integration is implemented below.

Correction now has a shared four-file runner and whole-run output schema3,
with v1/v2 archive compatibility. All four images/masks retain raw byte identities,
decoded frames and decoder facts. Verified preparation/core/runtime identities
govern reuse and archive binding. Qt Export run saves corrected CUDA RGB and
compact immutable fit matrices, locks all input controls and drains on close.
The final eight-module gate passes 102 tests in 144.574s; bounded independent
review covers supplied source/test projections and root execution evidence.
Correction project image-to-mask bindings, assets/snapshots/batch and desktop
integration are now implemented in the newer checkpoints below. Full F01-F22/WP0-WP6 and fresh full/scientific verification stay active.

Saved per-image correction bindings now execute through the shared CUDA adapter.
Exact project image ID/locator and analytical recipe identity select the four pins;
identical RGB bytes do not merge distinct image-to-mask selections. Missing selections
produce unavailable rows; selected invalid inputs produce failed rows without origin
fallback. Other project images and baseline reports continue. Source and transport
byte reads remain bounded, including final callback-free publication audits.
Output schema4 reads versions1-3 and preserves original selection records separately
from the actual opened paths and current decoder facts. Compact origin metadata is
bound independently in checkpoint identity; local schema3 identities remain unchanged.
Batch ledger3 retains exact bindings and project image IDs, adjacent correction runs
reuse prepared CUDA inputs, and those inputs are released before generic graphs.
Project schema6 persistence/snapshots and desktop Save As/degraded recovery remain.
Bounded review covers supplied source/test projections and root execution evidence:
123 tests in411.355s (122 pass;1 platform skip), then 8 focused passes in 44.869s. The project orchestration test uses a fixture
baseline and actual CUDA correction companions; it does not validate production
baseline numerics or scientific accuracy. The desktop correction checkpoint follows.

Correction dialog Keep/reopen is implemented for the exact selected project image.
A successful current CUDA run enables Keep; changed inputs, failed/incomplete outputs
and stale worker results disable it. A frozen request carries recipe/layout and four
input records, never CUDA results. The existing persistence worker verifies bounded
source/input bytes and stages immutable copies. GUI completion checks cancellation
and exact project/image/recipe/binding state before merging compact session metadata.
Keep marks the project modified; Save Project separately writes its master.
Reopened runs use project pins and preserve original selection records separately
from actual reads. Editing a reopened mask/target starts a fresh selection using the
exact selected project RGB path; an explicit different RGB selection is preserved
and cannot be kept for that image. Closing, layout or parameter changes during Keep
discard the pending merge after the worker drains. Historical attachments remain.
Bounded independent review covers supplied source/test projections and root evidence:
87 tests in289.194s (86 pass;1 skip) before the final two reviewed UI repairs; then 31 tests pass in198.483s. The final guard reuses bounded output contracts;
it does not transfer the corrected image to CPU or rerun analysis. This checkpoint
adds desktop functionality, not scientific colour validation or full feature parity.
Full F01-F22/WP0-WP6, automatic card/helper and remaining computational domains,
complete per-stage replay, original/SMPTS execution, fresh full discovery and
independent scientific/laboratory gates remain open.
The reviewed conversion/dependency/register gate passes181 tests in81.100s.
Fresh full-suite verification stays open.
The assembled fitter/tolerance/scoring/convergence/control/RNG/initializer/update/
register gate passes67 tests in15.344s. The expanded image-training/export/full-
dependency gate historically passed121 tests in54.982s. The expanded directory/
native-workflow/dependency/register gate passes161 tests in73.954s; fresh full-suite
verification remains open.
The last full discovery ran 1149 tests in 1544.843s: 1144 passed, two UI tests failed
and three skipped. Both UI failures are repaired/reviewed with focused passes.
That full run predates current additions; fresh full verification remains required
before hand-off. Remaining numerical/drawing/threshold domains, Bayes training and other K-means domains,
automatic card/helper/SMPTS execution, complete saved-project/batch reports and
model/checkpoint bindings, and laboratory gates remain open. All F01–F22/WP0–WP6
stay in scope. Earlier full-suite evidence is historical.

Seed Fiddle is a local PySide6 application for reference-assisted seed-image
analysis, annotation, instance review and reproducible result export. It uses
PyTorch CUDA for full-resolution image evidence and retains node-local reusable
intermediates on the device. There is no application server or bundled executable.

**Project status (8 October 2026):** the existing engineering workflow is
implemented through reviewed result export and portable snapshots. The latest
full local verification and in-progress checks are recorded in the
[current project status](docs/CURRENT_STATUS.md) for the capability matrix,
annotation behavior, schema versions, and remaining validation work.

**New project direction:** achieve peer functionality for selected PlantCV and
SMPTS seed-analysis capabilities within SeedFiddle. The
[revised comparison](docs/PLANTCV_SMPTS_COMPARISON.md) and
[implementation/benchmark plan](docs/PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) define
the accepted extension. Capability auditing and comparison infrastructure come
first. Reviewed checkpoint, recipe, conformance and isolated original-operation
APIs, grouped benchmark manifests and **View → Reference graphs…** are implemented.
Reviewed bounded implementations add CUDA thresholds, median/morphology, small-object
removal, stage comparisons and verified original-operation checkpoint inspection.
Original/native five-stage benchmark preprocessing is reviewed and passes focused
comparisons. Reviewed CUDA distance/marker stages and per-seed image/mask crops
add Qt and saved-project batch artifact export. The latest frozen full suite passes
989 tests with two existing skips.
The next batch adds reviewed bounded peak/range primitives and native inspection.
An exact LAB candidate passes focused tests and exhaustive byte-colour comparison;
its bounded review is approved; fresh full verification remains pending. Per-label colour and default-mode
CUDA watershed candidates now pass focused reference comparisons. The full-resolution
distance/peak/marker/partition chain matches every stage. A new **Run native
calculation…** working view executes the three canonical seed graphs with editable
controls, real CUDA previews, node-local cache reuse, cancellation and standalone
selected-recipe save. All three threshold arms pass full-chain reference comparisons.
Returned hue-chart data and native sample display are integrated and tested;
size observations and nonverbose diagnostics are available for supported geometry.
Native seed working copies expose line thickness, with exact one/five-pixel reference
checks; historical graphs remain unchanged. Gaussian working controls now cover odd
kernels 3..15, withholding uncertain byte decisions. Numerical/drawing limits remain explicit. The four-input
correction desktop workflow is reviewed and tested.
Remaining size/card work, SMPTS execution, complete project/batch reports and operation compatibility remain
pending; peer performance is unvalidated.
Full-suite evidence and the current integration gate are recorded in the status document.

**Current limits:** outputs are proposals for review. Real-image counting accuracy,
physical measurement accuracy and biological trait validity remain unestablished.
Physical result units and unvalidated per-seed traits are withheld. Neutral colour
balance is not validated colorimetry. Read the
[scientific validation protocol](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md) before
planning quantitative laboratory use.

Most project code was authored by AI and has not had comprehensive human review.
Independently review and validate it before relying on results.

## Start

Python 3.12 x64 is the baseline runtime. With compatible dependencies available:

```powershell
python .\seed_vision.py
python .\seed_vision.py --diagnostics
```

The single launcher can create a project-local virtual environment or install
missing packages after confirmation. `--no-bootstrap` suppresses bootstrap;
`--offline` requires compatible wheels in `wheels/`. Runtime requirements are in
[requirements-runtime.txt](requirements-runtime.txt). Full-raster analysis is
GPU-first; the UI reports the active device and fallback.

## Work through an image

1. Open images or a saved project. Verify acquisition, species and biological
   context; species starts as Unknown/unassigned.
2. Supply appropriate Foreground/Background/Other references. Missing required
   evidence means unavailable, not a zero seed count.
3. Run the pipeline. Inspect intermediate overlays or use the optional expert
   graph. Reference-driven node and project optimization propose settings for
   review; fitting to the same references is adaptation, not validation.
4. Use **Review / export results** (`Ctrl+R`) to inspect seed IDs and mark accepted,
   excluded or unreviewed instances. Filter decisions and compare completed methods.
5. Export JSON, CSV, an annotated image or image/mask crops for visible seed rows,
   or save the displayed viewport with its
   pixel-scale key. Save a project for continued editing; create an immutable
   portable snapshot to freeze verified analytical dependencies and audit evidence.

Stop uses `Ctrl+.`. Annotation Undo/Redo use `Ctrl+Z` / `Ctrl+Shift+Z`; node search
uses `Ctrl+K`. See the guide for drawing and review shortcuts.

## Current documentation

- [Current project status](docs/CURRENT_STATUS.md): implemented capabilities,
  latest verification, current annotation behavior, and remaining work.
- [Documentation index](docs/README.md): current guides, implementation records,
  historical audits, and research ideas.
- [Implementation and validation roadmap](PLAN.md): completed baseline phases,
  the accepted peer-functionality extension and remaining laboratory milestones.
- [Operator guide](docs/OPERATOR_GUIDE.md): workflow, shortcuts, projects, snapshots,
  results, learning and migration.
- [Architecture and contracts](docs/ARCHITECTURE.md): ownership, caching and provenance.
- [Generated node/control catalogue](docs/NODE_CATALOGUE.md).
- [Node optimization](docs/NODE_OPTIMIZATION.md): objectives, eligibility and exemptions.
- [Critical-review remediation](docs/CRITICAL_REVIEW_REMEDIATION.md): all 30 findings,
  corrections, test evidence, limitations and remaining laboratory inputs.
- [Scientific validation protocol](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md).

The [historical README](README_HISTORY.md), dated audits and experiment reports
preserve their original findings. They are not current run instructions or validated
performance claims. [HANDOFF.md](HANDOFF.md) contains agent constraints and dated
implementation history.

## Compatibility and batch work

Reference schema 6 binds rasters to calibration coordinates. Unknown legacy frames
require explicit alignment review; never migrate them by resizing or changing a
schema number. Raw-foreground feature recipe v2 replaces ambiguous v1 foreground
recipes, which require re-export/retraining. Checkpoint loading is restricted to
supported tensor/primitive payloads.

Batch execution consumes the saved desktop recipe and fingerprinted references:

```powershell
python scripts/analyze_pilot.py --project projects/analysis.seedfiddle-project.json --output-dir artifacts/project-results
```

Add `--seed-crop-padding-px 4` for per-seed image/mask PNGs and their provenance
manifest. Partial extraction is reported explicitly.

Working masters retain shared mutable sidecars. Use an explicit snapshot when an
immutable revision is required. Source/checkpoint content changes invalidate cached
results and require recomputation or reference review.

## Development and checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/generate_node_catalogue.py --check
```

`images/` holds committed fixtures. Generated logs, screenshots and exports belong
under ignored `artifacts/`. Windows CPU/Qt CI and an optional configured CUDA runner
are defined in [.github/workflows/tests.yml](.github/workflows/tests.yml).

Desktop diagnostic logs rotate under `%LOCALAPPDATA%\Seed Fiddle\logs\` on Windows.
**Help → Runtime summary** shows the actual log and crash-log paths. Software tests
and synthetic examples do not substitute for independent laboratory validation.

## License

[MIT License](LICENSE).

Sample-table Naive Bayes training is implemented as a bounded native CUDA API
and an asynchronous desktop action. Use **Train model…** in the frozen
Naive Bayes dialog, save the exact PlantCV TSV, run the selected model, then
**Keep in project** and **Save Project** to retain its training record and model
copy. Saved projects and whole-run archives preserve the record without requiring
the original sample table. Legacy TSV selections remain compatible. Numerical
conformance is empirical, with explicit constant-channel exclusions. Directory training and optional SVG plots are integrated; broader training
domains remain open.
The lower-level CUDA image/mask sampler now preserves the public channel/RNG
sequence and feeds the shared fitter. See the
[operator guide](docs/OPERATOR_GUIDE.md) and current
[feature register](docs/plantcv_smpts_features.json).
