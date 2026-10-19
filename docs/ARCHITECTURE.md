# Current architecture and analysis contracts

Size and units saved-stage replay checkpoint (9 October). `analysis.size` now replays original RGB/labels and all seventeen observation records in source or verified seed-ROI frames, comparing the optional CUDA diagnostic separately. Exact source descriptors, nominal datatypes and finite local coordinates are preserved. `calibrated_seed.size_units` replays the compact physical conversion with zero selected arrays and binds the saved size/frame and card-scale producers. Its conversion allocates no CUDA tensors; the shared runtime audit has a separately admitted 64 KiB reserve. Upstream label maxima and card measurements are not recomputed by the units stage. There are now 22 declared replay operations.

The shared 15-module gate passes 185 tests in 1155.051s before the final audit-budget refinement. The final affected size/unit and existing planner/contract gate passes 28 tests in 86.761s after that refinement (zero skips; overlapping gates). Nine final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled direct-input 1027x1025 graph uses 128 actually computed labels with independently prepared, hashed gray/RGB arrays; it makes no single-image decoding claim and changes no upstream outputs. All seventeen size variables agree with pinned public PlantCV under the declared 1e-6 numerical tolerances and ten-decimal longest-path check; the diagnostic agrees byte-for-byte. The public explicit-one empty-mask behavior agrees too. Source replay selects three arrays (10,526,750 bytes), compares 3,171,593 values and adds 21,663,232 CUDA bytes within 268,960,128 planned; 13,312 preexisting bytes are separate. Whole source replay traces 63,354,720 additional peak host bytes. Physical/pixel units select no arrays and add 15,872/15,872 CUDA bytes within the separate 65,536-byte shared-audit reserve; an isolated cold audit measured 29,184 bytes. Encoded metadata caps do not guarantee Python RSS. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence.

Freeze: artifacts/native-size-replay-frozen-hashes.json; final gates: artifacts/native-size-replay-final-shared-gate.log and native-size-replay-final-gate.log; source/oracle/recipe/archive/output/runtime hashes and actual Qt views: artifacts/native-size-replay-probe-final-v2/evidence.json. Actual async size/unit summaries and the size image were inspected at 1440x900 and 1050x650 with Windows fonts; the active run and every archive were preserved. Review uses supplied projections and root execution/visual evidence, with no independent file reads/tests/visual inspection claimed.

The F01-F22/WP0-WP6 goal remains active and partial_native. Next: archived-model classifier replay, then remaining card, recovery and paper helper contracts and the complete method/input/domain audit. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required before completed-goal handoff.

Earlier verified checkpoints follow.

Colour saved-stage replay checkpoint (9 October). `analysis.color` now supports typed saved-input replay in source and verified calibrated seed-ROI frames. All RGB/LAB/HSV/ALL histograms, bins, statistics, label records and returned hue-chart fields are compared; explicit nonfinite tags preserve NaN/inf outcomes. The native Qt view shows the returned chart and concise observations while preserving the active run. There are now 20 declared replay operations.

The shared 15-module gate passes 179 tests in 979.683s before the final character-length precheck; the final affected colour gate passes 14 tests in 125.179s after that refinement (zero skips; overlapping gates). Six final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

Replay accepts only six whitelisted exact dataclass schemas, explicit tuple/nonfinite tags and immutable compact snapshots. Per-product metadata admission is 8 MiB encoded, 1 million visits, depth 10, 2048-byte strings and 1024-byte sample bases, with character checks before UTF-8 conversion. Malformed records fail before array loading; valid wrong numerical products report mismatch. Automatic counts bind labels.max(); explicit counts retain source remapping/ignored-ID behavior. Empty RGB/LAB returns valid zero histograms and NaN statistics; missing HSV/ALL samples are unavailable. RGB/InstanceLabels remain CUDA, including a private guarded 3P BGR adapter. Source/storage/version mutation, callback errors/non-bool decisions, cancellation, changed code/constants and final frozen-record mutation withhold incomplete outputs. Colour admission is max(128P+128MiB,10P+512min(P,65536)+32768(n+1)+4MiB);7P selected arrays <= 48 MiB and owned CUDA <= 1 GiB. Work 128P+32ceil(P/65536)*9(n+1)*256<=8B names scan/channel/table visits, not instructions or elapsed time. The preceding 19 plans remain unchanged.

F01-F22/WP0-WP6 remains active and partial_native. Next: typed size/unit observation replay, then remaining model, card, recovery and paper helper contracts and the full method/input/domain audit. Original SMPTS source/license/dependencies and laboratory inputs remain separate open gates. Fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

Calibrated range workflow checkpoint (9 October). The additive `plantcv_calibrated_seed_range` graph corrects the colour card, crops an explicit seed ROI, applies channel-range selection, smooths its binary mask, and executes fill/morphology/watershed plus size/colour observations. Traits use the original corrected cropped RGB; the range masked RGB is a pre-median diagnostic. This is a user-selected composition of pinned operations, not a recovered historical fixed-range workflow or scientific validation.

The final 11-module gate passes 143 tests in 647.331s, including 11 new integration tests, with zero skips. All nine prior calibrated/diagnostic recipe serializations remain exact. Tests cover five spaces plus partial ranges, complete pinned corrected-ROI outputs/physical observations, range/median/ROI/units cache boundaries, empty-result recovery, fresh/cached/debug admission, excess input backing, final callback/runtime mutation, actual Qt replay/Keep/reopen and saved-project/batch/archive round trips.

The new arm is a 24-node composition with typed BinaryMask median ports. Original corrected ROI RGB feeds flood and measurements; masked RGB has no analytical dependent. Range/median/ROI/scale changes preserve independent cached branches. Calibrated admission adds16N to the prior64N+16min(N,1M)+256MiB envelope, using full-source N; debug adds3N. Unique source backing above4N is charged before dispatch. Fresh/cached preparation and final callback-free loaded-closure auditing use the exact calibrated range base identity. Local replay uses P only after verified crop/producer/frame ancestry. The nineteen replay operation identities are unchanged.

F01-F22/WP0-WP6 remains active and partial_native. Remaining model/helper/paper/observation replay and the full method/input/domain freeze remain open, alongside original SMPTS source and laboratory gates. Fresh full unittest discovery is required before completed-goal handoff. Earlier source freezes retain historical scope when shared modules change.

Earlier checkpoint history follows.

Custom-range graph and saved-stage replay checkpoint (9 October). The separate `plantcv_threshold_custom_range` recipe now executes the existing pinned CUDA primitive with real RGB/HSV/LAB/CMYK/GRAY and inclusive byte-bound controls. Both the binary mask and masked original RGB are retained, exported and replayed. RGB bounds use pinned B,G,R storage-slot semantics; CMYK retains the source C/Y swap. The four existing extended recipe identities and historical reference catalogue remain unchanged.

The final shared gate passes 133 tests in 220.734s (11 new range tests; zero skipped). Actual 887x887 LAB/CMYK probes select 353,079/214,138 pixels and match both pinned public outputs and isolated saved-input replay. Replay adds 29,599,232/52,102,656 CUDA bytes within 234,924,160 planned; 13,312 preexisting bytes are separate. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence. There are now 19 declared saved-stage replay operations.

The additive factory is `seedvision/benchmarking/native_range_recipe.py`; the primitive and graph/replay share allocation-free control/resource admission. RGB-to-BGR and masked-BGR-to-RGB use one allocation each with direct plane copies. Primitive callbacks guard input and owned-output identity/version/storage before and after callbacks; trusted output writes refresh baselines between callbacks. Ordinary PyTorch mutation is covered; raw-pointer writes are outside that contract. Native execution uses existing descendant-only cache eviction and final loaded-closure auditing. Joint CUDA admission is max(128P+128MiB,17P+512min(P,262144)+4MiB)<=1GiB plus excess graph input backing. Replay selects exactly RGB input, expected mask and expected masked RGB (7P<=48MiB), compares all 4P output bytes on CUDA, and validates the fresh masking relationship. Well-typed historical wrong products report mismatch; malformed controls/headers report invalid archive. Resource, runtime and cancellation outcomes remain distinct.

Earlier checkpoint history follows.

SMPTS executable paper reconstruction checkpoint (8 October). The additive `smpts_paper_fixed_mean` graph now executes the paper-supported branch/filter/intersection/contact/erosion/extraction sequence. Defaults25/5/13 come from the paper; offset/polarity, fixed cutoff, connectivity, axis-aligned boxes, box-area cutoff, border, erosion shape and crop rules remain declared assumptions. The historical unresolved graph is unchanged.

The 111-test relevant gate passes in 117.973s (19 new tests;0 skipped). The 2080x3120 committed fixture matches nine pinned OpenCV substep rasters and all 1118 packed crops. This verifies the declared reconstruction, not original SMPTS conformance or scientific accuracy. The full F01-F22/WP0-WP6 goal remains active; full discovery is still required before completed handoff.

Implementation: `seedvision/benchmarking/native_smpts_recipe.py`, `seedvision/cuda/smpts_paper.py` and `_smpts_paper.cu`. The existing native runner audits loaded Python/CUDA source and caches each node; project acceptance permits only this canonical executable PAPER_RECONSTRUCTION graph. Historical paper contracts stay unavailable. Packed RGB/masks are flat CUDA byte arrays; immutable crop rows bind half-open source boxes, offset_pixels, crop_shape, final ID and parent ID through the outer source/frame identity. At most4096 components and 128MiB of crop bytes; graph admission64P+192MiB<=1GiB plus excess input backing. Compact metadata stays bounded. The separate experimental full-foreground recovery feature is not silently substituted.

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
remains partial. See [current status](CURRENT_STATUS.md).

Earlier Calibrated local-stage replay checkpoint (8 October): eight saved-input raster operations now
also support the explicit seed ROI in calibrated binary/mean/Otsu graphs. Parent
and local frames, transform, producer-port ancestry and recomputed card overlap
are checked before selected pixels load. Only saved local inputs/expected output
are uploaded; local dimensions drive resource admission. Full-bounds crops retain
their distinct local frame. 35 focused tests pass in 281.940s (0 skipped); actual Qt replay views are checked at both
window sizes. Active runs and archive bytes are preserved.
Full feature implementation remains active; affected features stay partial and
scientific validation is false. Other stage contracts, broader domains and
source/laboratory gates remain open. See [current status](CURRENT_STATUS.md).

Earlier Saved-stage replay checkpoint (8 October): Replay saved stage now reruns eight supported
full-source-frame raster operations from verified archived inputs, compares
fresh CUDA output and preserves active runs/history. All archive payloads are
verified with bounded selected loading; runtime/settings and input lineage bind
the attempt. Qt execution, cancellation and both window sizes are checked.
Integration gate: 171 tests pass in481.283s (0 skipped); archive-origin/watershed gate: 12 tests pass in35.600s (0 skipped); final API/Qt gate after the cropped-frame outcome repair: 17 tests pass in122.279s (0 skipped). These gates overlap and are not a count of distinct tests.
Local/cropped, UHD, other stage contracts and broader replay remain open.
Computational agreement does not establish scientific validation or full parity.
See [current status](CURRENT_STATUS.md). Full feature implementation remains active.

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
replay and laboratory gates open. See [current status](CURRENT_STATUS.md).
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
and contact/laboratory controls are open. See [current status](CURRENT_STATUS.md)
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
remain open. See [current status](CURRENT_STATUS.md) for evidence.

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
validation gates remain open. See [current status](CURRENT_STATUS.md) for evidence.

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
open. See [current status](CURRENT_STATUS.md) for checkpoint evidence.

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

Project schema6 retains immutable reference graph metadata, separate editable native
working recipes and bounded model-asset descriptors, with versions1-5 migration.
Recipe/run/layout attachments and working selections survive desktop saves and snapshots;
historical checkpoint digests explicitly do not imply restored or verified arrays.
Native analysis-settings schema remains 20. Reference checkpoint artifacts require
separate binding and inspection.

`export/results.py` defines authoritative method selection, unavailable-vs-empty
state, pixel-space object rows, exact result revisions and JSON/CSV/image export.
Physical units and unvalidated traits abstain. `export/project_runner.py` loads the
same saved production recipe and source-bound references as the desktop. The legacy
`BaselineAnalysis.count` property remains a proposal-count compatibility API and
must not be used for new result reporting.

`export/native_results.py` writes a separate v4 native whole-run projection and reads v1/v2/v3. Tagged
ordered metadata preserves tuple/list/mapping/record distinctions and explicit
nonfinite observations; named output bindings reference bounded checkpoint groups.
Only an explicit save transfers dense CUDA products to CPU. All descriptors, total
bytes, group inventories and report size are checked first. The loader checks
semantic relationships and aggregate inventory before restoring any arrays, then
verifies the existing checkpoint hashes and raw model-byte identity. It returns
data without executing models or instantiating recorded Python classes. Partial
nodes retain only actual products; output persistence does not establish conformance.

`export/project_native_graphs.py` executes exact saved native attachments and working recipes
through the same path as the desktop. It atomically publishes a companion ledger
and output archives under the baseline report. Original/reference graphs receive
explicit unavailable rows. Schema2 ledger rows identify historical versus working recipes;
working recipes contain no invented run history. Each result releases ownership after export, so
normal node-cache eviction can free its tensors; baseline tensors release before
native execution and the image cache clears before the next source. Separate
`persistence/reference_models.py` descriptors bind each exact historical/working model
selection to verified workspace-relative copies. Transport paths are excluded
from analytical identity and every run rereads selected bytes before cache reuse.
Snapshots require complete model closure; model inventory and copy reads are
capped by descriptor length even after preflight. This does not add per-stage
replay certificates or scientific eligibility.

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

## Accepted peer-functionality extension (4 October 2026)
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

The 7 October native seed-calculation candidate in `benchmarking/native_recipe.py`
evaluates the three canonical `RecipeSpec` graphs using their typed fan-in and
the existing `PipelineAnalysisCache`. It uses no separate application launcher.
Node keys bind actual upstream products, selected parameters, source/frame,
preparation, runtime and loaded code; tensor mutation counters reject stale products.
Display/layout changes preserve calculation reuse. Source changes clear only the
reference-recipe cache scope. A single prepared-image slot and conservative 1 GiB
retention/preflight limits bound this candidate's persistent CUDA products; each
primitive separately bounds scratch allocations. Actual OpenCV decoding and
NVRTC/driver versions are recorded without claiming decoder reference conformance.

`ui/native_recipe_dialog.py` reuses the canvas/inspector for an explicitly separate
native working view. Its QRunnable computes on CUDA, samples previews on the device
before a bounded Qt transfer, and drains cancellation before close. Parameter edits
and reset change the selected recipe; atomic UTF-8 JSON save preserves that selection.
The parent historical attachment is unchanged. `run_native_recipe_path` provides
the shared saved-recipe Python path. `benchmarking/color_histogram.py` builds the
returned hue-chart rows from compact observations, and `ui/color_histogram.py`
renders them in Qt. Selecting a displayed sample does not change calculation or
cache identity. RGB/LAB return a valid empty hue chart. Size returns immutable per-sample
observations for supported empty, external and nested contours, including holes and
disconnected regions. Sparse RETR_TREE hierarchy/composition has bounded review.
The reviewed deterministic4.11 ellipse compatibility path is restricted to runtime5.0.0;
nearzero cross coefficients at the atan2 cut/isotropy withhold the whole observation
bundle. Unsupported outputs remain explicit. Reviewed explicit thickness is stored
with measurements and must match the renderer. The source hard-coded-five fallback
applies only to caliper measurement; requested width reaches every overlay stroke.
The nonverbose overlay clones the caller RGB image on CUDA and paints bounded compact source-derived
spans. Shared point/boundary/contour limits apply while collecting labels; crossing
releases geometry and retains all accurate observations as a partial node.
Diagnostic-only resource limits preserve observations; cancellation publishes no
partial image. Four additional threshold RecipeSpecs are created separately by
`native_extended_threshold_recipes.py` and opened through the reference dialog's
extended-threshold selector. Their typed CUDA dispatch uses the same node-local cache,
loaded-source/runtime identity, cancellation and selected-recipe save contracts.
Gaussian 11/13/15 use pinned float32 coefficient bits, a CUDA binary64 estimate and
a conservative source-arithmetic error interval; uncertain byte rounding withholds
the complete mask. Work and actual multi-tile scratch are bounded. Native seed copies
have distinct `*_working` identities and an effective size-thickness contract;
the historical catalogue and attachments are unchanged.
`cuda/plantcv_naive_bayes.py` parses bounded immutable TSV likelihood tables on CPU
and returns all per-class CUDA masks using byte HSV and left-associated binary64
products with strict first-class ties. This reviewed primitive has separate model,
output and scratch caps. `native_naive_bayes_recipe.py` adds a separate frozen-model
input stage and immutable raw/table/class-order identity. The shared executor rereads
the bounded asset before preparation and execution/cache reuse; model changes retain
the image node. Recursive mask signatures and alias-aware storage protect cached
class products. `NativeNaiveBayesDialog` uses the native async worker and selected
class preview. K-means has separate reviewed CUDA mask and sigma5 preprocessing
primitives. Its Gaussian uses virtual edge-padded halos, explicit symmetric pairs,
literal binary64 coefficients, whole-raster rounding guards and halo-aware work
preflight; actual tile scratch stays below64MiB. Tiled patch scoring, compact model
identity and full frozen-model prediction are reviewed. Bounded sigma5 uint8
training composes source-ordered directory selection, Qt header metadata/OpenCV
decoding, unique CUDA input uploads, reusable uint16 patch rows, the complete
MiniBatch fitter and compact numeric export. The native adapter retains an upstream
byte-verified file manifest across fit/extraction edits; row keys bind loaded
producer/runtime identity, patch/sampling identity and CUDA device. Input edits
invalidate selection; extraction edits discard rows; fit edits retain rows.
Native fitting explicitly certifies all OpenMP reductions without inventing a
source thread count. Only compact centres transfer for atomic JSON model saving.
The async dialog drains before close and publishes no cancelled/stale model.
The separate bounded artifact codec wraps the unchanged numeric-v1 model with
compact typed training provenance. It never traverses CUDA rows/counts/labels.
Single-file QSaveFile publication prevents centres/metadata half-saves. The recipe
loader accepts legacy numeric files and artifacts, preserving the centre digest
and binding source SHA/size to the actual outer file. Artifact parsing is included
in the executor's loaded calculation identity. Historical training directories
need not remain present to use a saved model; live training still verifies assets.
The numeric/artifact implementation lives in the inert standard-library model_io
package. Historical CUDA/benchmarking imports alias those exact module objects;
patched globals and loaded-source verification still cover the real implementation.
Stable persisted producer labels retain historical artifact compatibility.
Explicit source-environment conversion imports sklearn/joblib only for conversion,
extracts compact centres, records source identity/environment/derived geometry and
atomically publishes the same JSON artifact. No source estimator enters the desktop.
Other training domains and complete saved-project/batch binding remain pending.
Original checkpoint publication and full
project/batch result integration remain separate pending work.

The [PlantCV–SMPTS plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) extends these contracts.
Checkpoint, recipe/conformance and isolated original-operation APIs are implemented;
native threshold primitives and source-inspection reference views are also implemented.
Tested local candidates add CUDA median/morphology, original mask-stage binding,
paired threshold reports and explicit checkpoint-store inspection. Complete backend
compatibility and runnable recipes remain in progress. Keep
reference adapters and their dependency environments separate from
the production runtime. Adapters cannot own the graph, project persistence or canonical
results. Reuse existing metric/geometry definitions with explicit compatibility names
where reference formulas differ. Normal production must not acquire benchmark-only
CPU transfers or a mandatory external framework dependency.

The existing node editor must accurately display each pinned reference recipe's
operations, typed connections, branches, effective parameters and intermediate products.
A shared versioned recipe description binds displayed topology to reference execution
and checkpoints; unavailable or reconstructed stages are explicit. Reuse the existing
graph model and renderer, not a parallel graph engine. Read-only reference inspection
is supported by the design; editable controls must affect actual computation. Require
F22 topology/parameter tests, actual Qt visual inspection, persistence round-trips
and cache independence for display-only actions. The separate Reference window is removed. Factory RecipeSpec objects remain immutable; fresh factory loads create working PipelineWorkflow values and apply the edit lock by default. Node parameters, wiring, bypass and model/input mutation routes check that lock; layout and execution/backend selection remain available. PipelineDocumentsController supplies standalone pipeline/layout commands; PipelineDocumentStore validates bounded JSON and saves atomically with distinct factory-derived names. PipelineBackendsController resolves global/per-node execution choices; BackendExecution dispatches only admitted equivalent public calls, uploads returned raster products to CUDA, and binds their checked provenance to node-local cache keys. Watershed locals share one atomic public capture. Existing ReferenceGraphAttachment/checkpoint/QC data and verification services remain compatible; no reference records are generated merely by opening a factory.

Replay checkpoints must retain source/frame/settings identities and bounded storage.
The original-threshold `run_reference_stage` bridge loads and validates exact
input artifacts, invokes the isolated runner itself and publishes a bound run/output
record. It supports three public threshold calls. The reviewed bounded mask-stage
bridge adds original median/fill/erode/dilate, and paired threshold execution retains
both backends and their comparison. The reviewed original preprocessing coordinator
wires median→threshold→fill→erode→dilate with exact checkpoint provenance. Its
native counterpart is a reviewed bounded benchmark adapter using explicit per-stage bridge
projections, not a second production graph engine. For mean/Otsu recipes, native
mask checkpoints use the mask bridge's binary-recipe stage IDs; the outcome retains
the selected parent identity and stage mapping. These records cannot be treated as
one unified desktop recipe execution. Full recipe orchestration remains open.
Comparison outputs preserve available/empty/failed/unsupported states, instance-ID
mapping, method identity and measurement eligibility. Snapshot and desktop/batch
contracts remain authoritative. Eroded cores, original foreground and measurement
masks are distinct products. New controls still require computational effect,
optimization coverage or an explicit exemption, and generated catalogue updates.

The reviewed bounded native watershed-stage bridge binds distance and markers to
explicit typed checkpoints. The distance primitive reproduces the pinned labeled
OpenCV fixed-point sweeps; it does not substitute the production procedural distance
field. Original public watershed execution still exposes final outputs only.

`export/crops.py` keeps extracted image/mask tensors on CUDA. Crop geometry derives
from the selected label frame, actual per-axis procedural scaling and corrected
image offsets. CPU and CUDA corrected frames must agree; half-open rectangles,
padding, clipping facts and source-to-crop transforms are explicit. The artifact
writer transfers only selected crops for PNG encoding, binds current source/result
identity and publishes a new manifest directory after final revalidation. Qt workers
export currently visible result rows and support Stop/worker-draining close. Optional
saved-project crop export stages the report and crops together before publication.
Independent Sol reviews and frozen full discovery verify these bounded paths.
Full recipe execution and original SMPTS extraction conformance remain separate work.

The follow-on CUDA peak primitive preserves pinned ordered coordinates, plateau,
border and priority-spacing rules within explicit resource limits. Supported range
conversion intersects inclusive BGR-slot/HSV/CMYK/gray/LAB byte bounds. The new LAB
candidate uses compact CPU-generated, CUDA-cached OpenCV tables and tiled CUDA pixel
arithmetic; exhaustive pinned byte-colour comparison passes, with independent review
pending. Upstream notices are retained in `OPENCV_MATH_NOTICES.md`.
Raw size premeasurements keep dense masks on CUDA and transfer bounded
compact boundary topology; the PlantCV observation bundle remains withheld.

Native checkpoint-inspection candidates read actual archived inputs/outputs for nine
operations, requiring exact selected recipe/source/frame/settings/roles and current
loaded code/CUDA environment. Shared support declarations are verified. Sampled
distance normalization and categorical marker colours affect Qt display only; no
operation is rerun and no active desktop image is independently matched. Independent
review approved the bounded inspection repair; fresh full verification of this
follow-on batch remains pending.

Portable checkpoint bundles copy existing immutable manifests/NPZ archives and the
complete selected upstream closure. Per-checkpoint decode limits are 64 MiB total
arrays, 80 MiB encoded NPZ, 1 MiB metadata and eight arrays; bundle limits are 256
checkpoints and 512 MiB encoded files. Only one decoded checkpoint is retained at a
time. Raw copies and source stability are verified before atomic directory publication.
Snapshots opt in with `reference_checkpoint_store`; ordinary snapshots keep their
existing metadata-only reference behavior. The optional Qt action exposes that choice.
Restoration verifies the inventory and closure, retaining a separate local checkpoint
store without changing recipe history, eligibility, active analysis or node caches.
This candidate passes focused tests and bounded independent review at matching frozen
hashes. Fresh full verification remains pending.

Native PlantCV colour-correction candidates reduce labelled uint8 chips on CUDA and
transfer only their compact RGB mean tables for pinned-formula NumPy fits. Affine
correction uses the source's augmented pseudoinverse; polynomial correction uses
its nine-term normal equations and sequential image summation. Image operations are
float64 CUDA tiles with source clipping/truncation and BGR output. Actual NumPy
version is recorded. Focused public-call comparisons declare matrix tolerances and
one-byte pixel tolerance; this is not byte-exact or scientific colour validation.
The four-input correction recipe and Qt view share
`benchmarking/native_correction_inputs.py`. Bounded selected byte snapshots feed
Qt header checks and OpenCV decoding; decoded geometry and CPU/GPU limits precede
CUDA transfer. Reuse also requires verified loaded preparation code and decoder
versions. Final publication guards cover four files, input tensors and corrected
outputs, including NumPy matrix metadata. `export/native_results.py` schema4 binds
all four source records and composed preparation/core implementation identity; v1/v2/v3
archives remain readable. Matrix ports use explicit compact roles and corrected RGB
uses its source-frame HWC image role. API/Qt export retains atomic staging and
worker-draining close. `persistence/correction_inputs.py` supplies exact per-image
origin/transport bindings and bounded byte capture; project schema6 and snapshots
preserve them, including duplicate-image identities. The project adapter executes pinned correction inputs and schema4 retains
separate origins. `persistence/correction_session.py` carries frozen exact-image
Keep requests and stages bounded immutable inputs in the existing persistence
worker. GUI completion checks its passive cancellation token and exact session
snapshot before merging records; it performs no full-image transfer or hash.
Native correction UI validates declared outputs before preview/Keep, drains
workers on close and reopens pins through the shared adapter. Save Project
separately publishes the master. Automatic-card primitives are implemented below; their working-recipe/Qt/project/batch integration remains pending.

`cuda/plantcv_color.py` computes per-label RGB/HSV/LAB byte histograms on CUDA using
bounded tiles, reusing the exact range converters. Only compact counts and observation
metadata transfer to CPU. Circular hue and positive-only saturation/value summaries,
binary-mask remapping, sample/bin/trait metadata and explicit empty/missing-label
failures follow the pinned public call. Focused comparisons pass; degenerate circular
results and complete project/batch integration remain open; bounded primitive and
chart/Qt source reviews are approved.

`cuda/plantcv_watershed.py` uses the fixed `_plantcv_watershed.cu` priority queue and
in-memory bundled NVRTC runtime in `_nvrtc.py`. Kernel batches preserve the sequential
reference heap/age/tie and label-at-push semantics on the current PyTorch CUDA stream.
Dense rasters and queue storage remain on CUDA, with scalar progress/cancellation
between batches. Limits are 64M pixels and 512 MiB output/queue workspace; finite
contiguous float32 elevation, int32 markers and uint8/bool masks are required. Only
the pinned default four-connectivity/compactness=0/watershed_line=False is implemented.
Every stage of the full-resolution reference chain matches; the Windows bundled
compiler path is tested. Native checkpoint identity must include the CUDA source and
runtime helper when bindings are added. Review/full integration remain pending.

## Extending the application

Keep controls tied to actual calculations. Add the node's optimization capability or
an explicit nonoptimizable reason, then regenerate the catalogue. Preserve native
Qt, one launcher and device-resident reusable rasters. Run focused regressions and
the full unittest suite. Use reviewed real data only for scientific claims; fixture
and synthetic tests establish engineering properties, not laboratory accuracy.

## Automatic-card preprocessing checkpoint

CUDA automatic-card preprocessing is implemented for BGR grayscale, exact 11-tap byte Gaussian blur and inverse adaptive mean/Gaussian kernels3..255. Only SHA-verified compact Gaussian taps use CPU OpenCV; all raster filtering stays on CUDA. Non-dyadic Gaussian binary decisions require outward exact-rational error certificates. Shared adaptive mean now checks scalar-double, SIMD-float32 and SHIFT23 arithmetic, including full-frame singleton-axis reduction. Backend-sensitive binary decisions explicitly withhold output. The actual pinned public detector is executed only through its preprocessing boundary in this gate; complete contour/card geometry, automatic correction recipe/API/Qt integration and scientific colour validation remain open. Focused final gate: Ran 57 tests in 191.942s; bounded independent review covers supplied computational/guard projections and root execution evidence only. Fresh full-suite verification remains required at completed handoff.

## Automatic-card detector and affine primitives

Automatic colour-card CUDA primitives are implemented in bounded supported
domains in `cuda/plantcv_card_detection.py`. RETR_TREE/SIMPLE contours come from
the new contour-only CUDA boundary collector; no dense CPU image or mask is used.
The collector preserves cyclic direction changes, repeated visits, source order
and hierarchy, with distinct empty, unsupported, resource and cancellation cases.
Sparse topology retains the existing point/row/component/trace/work limits; its
Python state is not claimed to fit the packed boundary byte count. CUDA extraction
scratch is preflighted at 256 MiB and tested on an actual custom stream.

The detector preserves strict contour area/aspect/solidity and median-area filters,
white-corner ordering, source int32 transform WITHOUT a perspective divide,
strict transformed-center membership, inclusive clipped bounding rectangles and
ordered overlapping circle labels10..240. It retains the source case where23
detected contours still generate24 grid masks. ROI bounds are explicitly half-open
xyxy. Interior bounds are compared with actual `pcv.roi.rectangle` effective crops;
frame-edge bounds are compared with rectangular source Objects directly. Named
or custom physical chip scaling is returned as metadata without changing global
application units. Reusable filters and full float64 labels/uint8 bounding masks
stay on CUDA; only sparse geometry, four sampled pixels and compact fit matrices
cross to CPU. Input/filter/output versions and final callback mutation checks
discard incomplete results; compact arrays are immutable bytes-backed views.

`cuda/plantcv_minrect.py` ports pinned4.11 rotating-caliper float32 operation
boundaries and `RotatedRect::points`, retaining last-equal-area selection and
source corner order without angle normalization. The loaded OpenCV5 min-area
rectangle sign convention and far-corner arithmetic caused actual label and ULP
failures; these were fixed rather than relaxing geometry comparisons. The fixed
4x6 grid box is pinned to its exact32 bytes of float32 metadata and SHA256
39c9018bf2b16570e4c2871891f031b76f5cdfb52131b1673b26e0672cec0ac6.
Actual OpenCV/NumPy runtime and reference-source identities are returned.
Loaded hull generation remains a runtime dependency with bounded reference tests.
Aggregate geometry preflight charges all incoming SIMPLE points and contour count
before candidate hulls, retained-centers geometry and24-center membership tests.
Circle/retained-storage/filter work limits precede full output allocation.

`auto_correct_color_cuda` uses the detected mask for chip means, standard matrix
pos=3 and the source affine fit, then corrects the whole image. It is separate
from the existing four-file polynomial correction workflow. Compact matrices are
immutable and corrected full-resolution BGR remains on CUDA. Pinned public
auto-correction tests require source means within2e-15, exact standard matrices
and at most one byte/channel corrected-image difference.

Focused final six-module gate: 49 tests PASS in 147.705s,
`artifacts/automatic-card-reviewed-gate.log`. This includes all512 binary3x3
masks, nested/seam/thin/random contour arrays and hierarchy,109 compact geometry
fixtures with exact hull/rectangle/corner comparisons, actual full public
detect/mask/auto calls, both adaptive methods, four right-angle rotations plus13
degrees, equal white-corner ties, named/custom scales, interior/edge/clipped ROI,
overlap and radius-zero labels,23-contour source behavior, numerical uncertainty,
resource preflight, cancellation/mutation including reusable filters/final masks,
changed grid metadata and actual custom-stream allocation.

Bounded independent approval uses supplied computational/guard projections,
frozen hashes and root execution evidence. The reviewer did not independently
read files or execute tests. This establishes bounded primitive functionality and
tested reference conformance, not scientific colour accuracy, arbitrary runtime
conformance, a loaded working-recipe certificate or complete feature parity.
Next work is additive automatic-card working-recipe/Qt binding, loaded callable
closure covering contour/minrect/grid/preprocessing/correction helpers, and shared
saved-project/batch/export integration. Existing four-file correction bindings,
historical catalogue identities and the canonical peer source manifest remain.
Full F01-F22/WP0-WP6 and fresh full unittest/scientific gates remain open.

## 8 October — automatic-card working recipe and desktop/project integration

`plantcv_automatic_color_card` v1.0.0 is an additive editable native graph.
RGB input converts once to reusable BGR; ROI cropping precedes grayscale,
Gaussian11 and inverse adaptive filtering. Separate geometry, circle/bounding
mask, physical scale, chip means, standard matrix and affine nodes preserve
source channel order. Standard position3 matches public auto-correction; other
positions explicitly compose the public standard matrix and affine functions.
Physical dimensions produce mm-per-pixel metadata without changing image pixels
or global application units. Radius changes reuse geometry; layout edits reuse
all analytical products. Compact geometry/matrices are deeply immutable finite
JSON tuples, while full images/masks/reusable filters remain CUDA tensors.

Loaded callable identity includes the card factory, detector, contour/minrect,
preprocessing/Gaussian pins, colour-card and correction helpers alongside the
shared runner closure. The exact immutable grid byte hash is verified separately.
Input versions are captured before the first callback; accepted cached/new
products are checked after callbacks, including the final callback. Mutations
discard the result/cache; failed or uncertain stages leave empty outputs and
unavailable descendants. Dirty descendants are evicted before replacement
allocation. The graph-owned32N+256MiB CUDA bound includes25N accepted products,
temporary corrected BGR, tile scratch and bounded contour extraction; old result
objects held independently by callers are outside cache ownership. Checks occur
before CUDA upload and direct-run allocation/runtime loading.

The reference overview offers Automatic colour card. The shared async canvas
supports effective controls, ROI-local/BGR previews, float64 chip-label colours,
compact scale facts, Keep/reopen and export. Project schema6 and the existing
batch/whole-run archive preserve exact working selections separately from
historical attachments, without inventing historical stage records. The canonical
peer source manifest and historical catalogue remain unchanged. Four-file
polynomial correction remains a separate workflow.

Post-review display verification: all18 card/shared Qt tests pass in38.226s
(`artifacts/automatic-card-working-preview-gate.log`). Actual1440x900 overview,
ROI-labelled-mask and scale views were rendered with Windows fonts and inspected.
The preview now fits its current label dimensions, preserving aspect ratio and
showing all24 chip circles. This display-only repair changes the shared UI hash;
the other six current source/test hashes remain unchanged. The103-test gate
predates the preview repair; the18-test gate includes the12 strengthened card
tests and all shared Qt regressions. Six register tests pass after reconciliation.
Full F01-F22/WP0-WP6 and fresh full-suite/scientific gates remain active.

## Composed calibrated seed graph

`native_calibrated_seed_recipe` adds three24-node graphs without changing either
standalone automatic-card or historical seed identities. A source-frame metadata
port binds the actual caller frame/SHA. The crop validates integer half-open bounds
and top-left/x-right,y-down pixel coordinates before allocating; its contiguous
CUDA RGB crop feeds every seed RGB consumer and its tiled pinned grayscale feeds
the gray branch. Crop-local semantics mark every seed port. Frozen local-frame,
parent-frame and translation records preserve coordinate meaning independently of
the source input frame. Full-frame colour correction remains available.

The compact size conversion consumes pixel observations and card calibration as
separate inputs. A physical edit invalidates only calibration and converted size;
an ROI edit invalidates the crop and its descendants. The joint graph reserves
64N +16min(N,peak_candidate_cap)+256MiB under the1GiB cap before transfer and
direct execution. Existing primitive scratch/work limits remain. The executor
counts unique backing storage, evicts dirty descendants before replacement, and
guards all accepted CUDA tensors through the final callback. Caller-retained old
runs remain outside cache ownership. Whole-run export retains mixed local/full
products with exact recipe/crop metadata; differently shaped local arrays never
receive a source-frame raster role. Per-stage replay remains separate work.

## Sample-table Naive Bayes training

`benchmarking/native_naive_bayes_training.py` separates bounded text decoding
from CUDA computation. Raw UTF-8 input is capped at16MiB, rows64KiB, headers32
columns and labels255UTF-8 bytes; these are input limits, not a16MiB Python
working-memory claim. Decoded text, StringIO, row fields and parsed input buffers
may coexist. Duplicate columns combine in source row/column order; missing/empty
cells are skipped and only row quotes are removed. RGB input transport is the
CPU boundary. Conversion tiles and retained HSV samples stay CUDA thereafter.

`cuda/plantcv_naive_bayes_training.py` accepts1..32 classes and at least two
samples per class, within one million total samples and one billion declared
work visits. A single3x256x256 lookup plus source-ordered density scans avoids
sample-by-query storage. Unique complete input backing storage plus scratch must
fit128MiB; fitting scratch is capped at16MiB. Conversion/fit combined preflight
runs before upload. NumPy pairwise weighted moments retain source arithmetic;
covariance uses a positive ordered reduction whose BLAS differences are bounded
separately. Compact24-scalar fit summaries and final PDFs alone cross to CPU.
Scott bandwidth, original sample order and source normalization are retained.
Numerical provenance distinguishes reduction estimates from empirical libm
allowances; this is neither a formal PDF guarantee nor scientific validation.

Every callback verifies complete input/returned tensor signatures, including
pointer/backing storage, version, shape/stride, offset, dtype/device/layout, in a
finally block. The file API's final bounded selected-byte check follows its last
callback. Actual loaded Python/helper closure is verified before and after the
fit. Returned canonical JSON binds source bytes, parsed samples, exact TSV/table
identity, fit/resource facts and actual runtime; origin paths remain provenance,
separate from numeric identity. The desktop/project integration below retains
this record separately from the interoperable TSV.

## Bayes training desktop, project and cache integration

`ui/native_naive_bayes_training_dialog.py` fits on a QRunnable, snapshots the
resolved sample path and selection revision, disables conflicting controls and
drains cancelled/closing workers. It validates compact TSV/model/provenance only;
display never traverses CUDA samples or fit products. QSaveFile commits the exact
TSV, then the recipe factory rereads and checks actual saved bytes before returning
the model selection. Commit, short-write and postcommit reload failures cannot
publish a selection. Parent close waits for child drain; modal exit releases its
completed CUDA fit and schedules child deletion.

`model_io/naive_bayes_provenance.py` is a standard-library codec for optional
`training_provenance` metadata alongside the existing five model-source fields.
The canonical JSON record is capped at512KiB,50,000 visited items,depth20 and
32,768-character strings; duplicate keys, nonfinite numbers and non-int64 integers
are rejected. Typed validation binds the model/table SHA, ordered classes, sample
counts, fit/resource facts, source pins and empirical profile. Runtime and origin
paths describe the producer; they grant no source access and establish no
authenticity, numeric certificate or scientific validity. The standalone TSV
contains no record, and no sidecar or alternate model envelope is introduced.

The shared runner verifies the codec's actual loaded body. Its Bayes model-node
key uses the original five fields after full metadata/asset validation. History
changes alone reuse numerical outputs; reused nodes freeze current effective
parameters so reports/archive metadata retain the new record. Project selection
digests include the entire record, while existing model assets deduplicate exact
TSV bytes. Invalid new records fail before asset reads even when unavailable
historical assets are permitted. Existing atomic master publication binds the
selection and model copy. Batch and output archives validate the full record and
compare raw model identity using the five numerical fields. Original training
inputs are unnecessary for project reopen/inference. Legacy selections and the
existing K-means training workflow retain their behavior.

## Binary Bayes CUDA image/mask sampling

`cuda/plantcv_naive_bayes_sampling.py` accepts matching contiguous CUDA uint8
BGR HWC and mask HW. Explicit kernels count exact255/0 memberships per4096-pixel
tile, then compact int32 positions using bounded exclusive tile offsets. Only
the compact count table crosses to CPU; no full mask/population/HSV transfer or
opaque nonzero/scan workspace is used. HSV conversion uses existing pinned byte
arithmetic in bounded tiles. Selected values and all dense intermediates remain
CUDA. Inputs record the caller's current CUDA stream.

Sampling uses the original global NumPy stream or a supplied legacy RandomState;
the API has no seed control. It calls foreground then background per H/S/V,
with replacement and exclusive high=population-1. Each class takes floor(F/10)
values. Independent columns can form an Nx3 fitter input without pretending the
rows are colocated RGB samples. Zero-row datasets are valid sampling results;
the separate KDE fitter requires at least two varying samples per class/channel.
Invalid population calls occur in public sequence order after prior draws.
Cancellation/source errors return no dataset and do not roll back consumed RNG.

Immutable MT19937 records retain624 keys, position and exact Gaussian-cache bits.
An independently advanced shadow checks each draw's values and post-state;
callbacks check the full expected RNG/tensor signatures before and finally after
execution, including raised callbacks and final output-storage changes. Raw CUDA
writes do not advance tensor versions, so allocated partial outputs remain watched.
One bounded source C-long vector is processed at a time and ordinal tiles widen
to CUDA int64; all six draw identities are hashed incrementally.

The16M pixel,1M combined sampled-row,16MiB compact CPU metadata,2B declared work
and256MiB CUDA live caps are separate constraints. Sparse foreground may fit a
large frame while dense foreground exceeds the row cap. A count-only first phase
is preflighted separately; exact counts then determine sample/population/output
storage before those allocations or RNG calls. Full unique input backing storage
is charged. Repeated conversion-table/setup and gather launches contribute to
work, so tiny tiles cannot bypass that bound. A1M-pixel custom-stream test measures
actual peak; source observations/RNG states and tile repartition are tested.
The directory adapter composes these stages with ordered decoding, cumulative
retained-sample admission, final source audits and the typed project/Qt workflow.


## Directory Bayes provenance and native worker integration

The existing standard-library codec dispatches exact sampled-RGB and directory
schemas. Directory JSON preserves Unicode with ensure_ascii=False; bounded walk
records reconstruct source candidate/pair/skip order without sorting. Unique
source bytes, matching shapes, per-occurrence plans, aggregate work/sample/live
facts, MT19937 state digests and before/after chains are checked against model
and table identity. Zero-row occurrences retain zero gather and unchanged RNG;
impossible completed background populations and mixed index dtypes are rejected.
These are descriptive consistency checks, not draw authenticity certificates.

The asynchronous training dialog freezes mode, roots, revision and callable.
A process-local lock serializes whole directory GUI jobs sharing NumPy state;
queued cancellation returns before dispatch. The adapter retains its tensor/RNG
guards for external changes. UI validation reads compact TSV/model/record facts
without traversing CUDA samples, fits or selection objects. Parent/child close
drains workers and releases the completion. The existing exact QSaveFile/reload,
five-field numeric model identity, annotation-only cache reuse, project assets,
companion batch and archive paths apply to both record variants. Original input
paths grant no filesystem access during record validation or inference.


## Compact Naive Bayes SVG rendering

`export/naive_bayes_plots.py` takes the existing immutable CPU PDF model, never
dense samples or images. Separate SVG Figures reproduce the pinned helper's
ordered plot/legend calls under Matplotlib factory defaults. A lock covers the
complete rc_context/figure/render/cleanup lifetime for this module's jobs; it
does not coordinate arbitrary other-library Matplotlib use. Streaming per-file
and aggregate caps, bounded labels, exact model facts and cancellation checks
precede publication of a frozen three-payload bundle. Payload SHA, SVG root and
model/table/class bindings are checked before UI publication. Descriptive
renderer/default metadata remains separate from numerical training provenance.

The dialog freezes the save revision/model/path, renders in a worker and
revalidates completion after the file chooser and after rendering. Existing
atomic TSV reload/selection precedes individually atomic sibling SVG writes.
Written paths and failures describe actual publication; rendered metadata
describes all rendered payloads. A publication guard also defers parent/child
close during modal failure reporting. Model-only saving remains the default.

## Saved native stage replay

`benchmarking/native_stage_replay.py` consumes the ordinary whole-run archive;
it does not execute a historical record or replace a current cache entry.
`read_native_recipe_inventory` preflights report/manifests only. The subsequent
selected loader binds that exact revision and verifies all payload bytes.
`load_checkpoint(array_names=...)` retains only named immutable arrays while
streaming every unselected member. `()` is verify-only; finite correction fits
are small auxiliary loads. Default all-array callers retain their prior contract.
Replay delegates calculation to the shared native dispatcher and returns fresh
CUDA rasters plus separate immutable facts. Runtime records now persist the
calculation-module closure digest. Missing/changed old bindings are unavailable;
old archives remain readable. Replay now has thirteen typed operation contracts:
the earlier eight raster operations plus maxima coordinates/peaks, connected
markers, negative distance, flood labels/completion and compact marker count.
Each selected input/output binds its declared type and exact incoming producer;
recursive frame verification visits all branches, including RGB and completion.
`_crop_frame` derives canonical local transforms; crop inputs, source/card
metadata and recomputed overlap remain verified. Full-bounds crops retain a
distinct local identity. Selected arrays are uploaded once, watched for mutation,
and every output is validated/compared before final loaded-code audits publish
results. Qt selects the declared preview port and displays compact observations
before replay details. Remaining model/geometry/measurement/helper replay and
other cropped graphs remain open.
