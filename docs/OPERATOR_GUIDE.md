# Current operator guide

Shared project inputs (20 October 2026): every Pipeline workspace contains **Project → Image selector → Species and metadata**, plus **Species reference library**. Select **Image selector** and use its inspector dropdown, or activate an image in the left Images list; both choose the same current raw image. Image selection remains available while the calculation is frozen. These four nodes and their context connections cannot be removed or disabled. Their positions and the selected node are preserved in pipeline/layout files. Older Default settings acquire the selector automatically without changing calculation parameters.

**Project** supplies the image inventory, image-local annotations and library pin. **Species and metadata** describes the selected image; edit its species with the existing **Image metadata → Species** control. Select **Species reference library** to inspect its pin/resolution and open the existing library manager. Default's connected consumers use the resolved banks. The PlantCV and SMPTS recipes expose the same project context, but do not implicitly consume species-library banks or change their factory algorithms.

The two colour-correction choices serve different purposes. Built-in **Deskew and colour balance** rectifies image geometry and balances RGB channels using neutral patches. The separate PlantCV colour-card workflows fit source patch colours to target patch colours: the automatic card route uses an affine RGB transform with offsets, while four-input **Correct colour** uses a nine-term polynomial RGB fit. This can represent colour mixing and nonlinear response that neutral balance cannot; it does not replace geometric rectification. Keep it as optional advanced calibration when target patch data are available. Applying it alone does not establish better measurement accuracy.

Current Pipeline workflow (19 October 2026): choose a factory in the compact Workflow selector or **… → Other pipelines**, or use **… → Load pipeline** for a saved custom pipeline. All loads start with **Lock** checked. Clear Freeze to edit node parameters, supported connections, bypass state and model/input selections. The original factory stays immutable; **Save** requires a distinct custom name and filename. Re-selecting a factory loads its original values.

Use **Run pipeline** and **Stop** on the application toolbar. The global **Native/PlantCV** selector and selected-node **Backend** selector choose actual execution; **Inherit** follows the global choice. They remain available while frozen. Stages without equivalent public calls show a native-only reason. The six watershed stages share one backend choice because the original is one public call. Explicit PlantCV execution requires the pinned local runtime; unavailable execution is reported without silently switching to native.

The Pipeline **…** menu provides pipeline and layout load/save. On a narrow pane, Save/Export remain available in this menu when their buttons are hidden. A pipeline file preserves calculation settings, independent node IDs, connections, enabled state, backend choices, layout, selection and referenced assets. It does not embed image pixels or run outputs. Missing asset references remain explicit. A layout file changes positions and presentation by stable node ID; applying it preserves calculation caches. Project saving and snapshots retain their existing historical evidence. Native saved-stage replay explicitly reports unavailable for a stage produced by PlantCV; it does not substitute the native implementation.

The shared image toolbar follows the active pipeline. **Overlay** offers **Source image**, **Calibrated image and annotations**, and the available outputs grouped by node. **Opacity** blends a mask or label output only when a matching RGB image has the same dimensions and coordinate frame. HSV controls appear only where the selected display supports their meaning. **Materials** or **Seeds** opens the calibrated annotation image established by Default seed analysis for that image; source and crop previews are never treated as the annotation canvas. Run Default seed analysis first if that frame is unavailable. Returning to Default restores its image controls.

For working binary, adaptive-mean and Otsu threshold nodes, and the SMPTS intersection node, clear **Lock** and use **Select foreground region…** or **Settings → Foreground region (input pixels)**. Enter `null` for the whole frame, or nonempty half-open integer bounds `[left, top, right, bottom]`: left/top are included, right/bottom are excluded. Bounds use that node's input pixels. Drawing is available for source-frame inputs; transformed inputs require bounds entered in their own input frame. The adapter runs after the threshold or intersection, sets foreground outside the rectangle to zero, and preserves the full raster dimensions and coordinates. It does not crop or resize the image. With PlantCV selected, the public threshold runs first and the explicit native mask adapter follows it; saved provenance identifies that addition. The region belongs to the working pipeline, survives save/reopen, and invalidates its node and dependents.

Full-resolution analysis and reusable rasters remain on CUDA; bounded previews can be smaller for display. Admission still reports explicit unsupported precision, memory or work limits instead of silently changing image resolution. Size/colour count controls admit up to 1,024 resolved labels; independent precision, encoded-byte, metadata-traversal and work budgets still apply to calculation, archives and saved-stage replay. SMPTS inventory allows 7,936 components under the unchanged 32 MiB modeled metadata allowance; its packed-crop allowance is 144 MiB within the unchanged 1 GiB composed per-operation CUDA admission. These operation bounds do not promise total application GPU memory or process RSS. Successful execution, an admitted photograph, or a displayed component count does not establish correct seed counts. SMPTS remains a declared paper reconstruction with assumptions; eroded components and extracted crops are not automatically complete biological seed outlines or validated original-SMPTS results.

During a run, the activity panel reports **Preparing image…**, the current node and its execution/reuse state, and elapsed time. A failed, partial or unavailable result shows its reason and selects the affected node for inspection. **Stop** requests safe cancellation; the display reports that it is waiting for the current calculation and discards obsolete outputs after cancellation. Read the node reason before changing controls or interpreting a partial result.

The separate Reference window is removed. Factory workflows are inspected directly in Pipeline. The dated engineering checkpoints below retain their historical scope; their former Reference-window launcher instructions are superseded by the controls above.

Size and units saved-stage replay checkpoint (9 October). `analysis.size` now replays original RGB/labels and all seventeen observation records in source or verified seed-ROI frames, comparing the optional CUDA diagnostic separately. Exact source descriptors, nominal datatypes and finite local coordinates are preserved. `calibrated_seed.size_units` replays the compact physical conversion with zero selected arrays and binds the saved size/frame and card-scale producers. Its conversion allocates no CUDA tensors; the shared runtime audit has a separately admitted 64 KiB reserve. Upstream label maxima and card measurements are not recomputed by the units stage. There are now 22 declared replay operations.

The shared 15-module gate passes 185 tests in 1155.051s before the final audit-budget refinement. The final affected size/unit and existing planner/contract gate passes 28 tests in 86.761s after that refinement (zero skips; overlapping gates). Nine final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled direct-input 1027x1025 graph uses 128 actually computed labels with independently prepared, hashed gray/RGB arrays; it makes no single-image decoding claim and changes no upstream outputs. All seventeen size variables agree with pinned public PlantCV under the declared 1e-6 numerical tolerances and ten-decimal longest-path check; the diagnostic agrees byte-for-byte. The public explicit-one empty-mask behavior agrees too. Source replay selects three arrays (10,526,750 bytes), compares 3,171,593 values and adds 21,663,232 CUDA bytes within 268,960,128 planned; 13,312 preexisting bytes are separate. Whole source replay traces 63,354,720 additional peak host bytes. Physical/pixel units select no arrays and add 15,872/15,872 CUDA bytes within the separate 65,536-byte shared-audit reserve; an isolated cold audit measured 29,184 bytes. Encoded metadata caps do not guarantee Python RSS. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence.

Freeze: artifacts/native-size-replay-frozen-hashes.json; final gates: artifacts/native-size-replay-final-shared-gate.log and native-size-replay-final-gate.log; source/oracle/recipe/archive/output/runtime hashes and actual Qt views: artifacts/native-size-replay-probe-final-v2/evidence.json. Actual async size/unit summaries and the size image were inspected at 1440x900 and 1050x650 with Windows fonts; the active run and every archive were preserved. Review uses supplied projections and root execution/visual evidence, with no independent file reads/tests/visual inspection claimed.

The F01-F22/WP0-WP6 goal remains active and partial_native. Next: archived-model classifier replay, then remaining card, recovery and paper helper contracts and the complete method/input/domain audit. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required before completed-goal handoff.

Earlier verified checkpoints follow.

Colour saved-stage replay checkpoint (9 October). `analysis.color` now supports typed saved-input replay in source and verified calibrated seed-ROI frames. All RGB/LAB/HSV/ALL histograms, bins, statistics, label records and returned hue-chart fields are compared; explicit nonfinite tags preserve NaN/inf outcomes. The native Qt view shows the returned chart and concise observations while preserving the active run. There are now 20 declared replay operations.

The shared 15-module gate passes 179 tests in 979.683s before the final character-length precheck; the final affected colour gate passes 14 tests in 125.179s after that refinement (zero skips; overlapping gates). Six final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

In Native recipe execution select PlantCV per-label colour observations, then Replay saved stage and choose the exported run. Inspect match/mismatch/unavailable status, per-label counts and the returned hue chart. RGB/LAB charts are valid empty charts. Replay uses saved RGB and labels; it preserves the current run and archive. Agreement establishes repeat computation of that saved stage; public-source and scientific evidence remain separate.

F01-F22/WP0-WP6 remains active and partial_native. Next: typed size/unit observation replay, then remaining model, card, recovery and paper helper contracts and the full method/input/domain audit. Original SMPTS source/license/dependencies and laboratory inputs remain separate open gates. Fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

Calibrated range workflow checkpoint (9 October). The additive `plantcv_calibrated_seed_range` graph corrects the colour card, crops an explicit seed ROI, applies channel-range selection, smooths its binary mask, and executes fill/morphology/watershed plus size/colour observations. Traits use the original corrected cropped RGB; the range masked RGB is a pre-median diagnostic. This is a user-selected composition of pinned operations, not a recovered historical fixed-range workflow or scientific validation.

The final 11-module gate passes 143 tests in 647.331s, including 11 new integration tests, with zero skips. All nine prior calibrated/diagnostic recipe serializations remain exact. Tests cover five spaces plus partial ranges, complete pinned corrected-ROI outputs/physical observations, range/median/ROI/units cache boundaries, empty-result recovery, fresh/cached/debug admission, excess input backing, final callback/runtime mutation, actual Qt replay/Keep/reopen and saved-project/batch/archive round trips.

In **Pipeline → … → Other pipelines**, choose the calibrated range workflow. Select card controls, an explicit seed ROI, channel and lower/upper lists (RGB/HSV/LAB three bytes, CMYK four, GRAY one). Endpoints are inclusive; RGB uses pinned B,G,R bound order and CMYK retains the source C/Y swap. The range preview is before median; Smooth range mask changes the segmentation mask. Size and colour use the unmasked corrected crop and resulting working labels. Keep/save/reopen preserves controls; export/batch preserves products and frames; range or mask-median replay verifies the saved local stage. Add watershed debug image for the optional independent diagnostic.

F01-F22/WP0-WP6 remains active and partial_native. Remaining model/helper/paper/observation replay and the full method/input/domain freeze remain open, alongside original SMPTS source and laboratory gates. Fresh full unittest discovery is required before completed-goal handoff. Earlier source freezes retain historical scope when shared modules change.

Earlier checkpoint history follows.

Custom-range graph and saved-stage replay checkpoint (9 October). The separate `plantcv_threshold_custom_range` recipe now executes the existing pinned CUDA primitive with real RGB/HSV/LAB/CMYK/GRAY and inclusive byte-bound controls. Both the binary mask and masked original RGB are retained, exported and replayed. RGB bounds use pinned B,G,R storage-slot semantics; CMYK retains the source C/Y swap. The four existing extended recipe identities and historical reference catalogue remain unchanged.

The final shared gate passes 133 tests in 220.734s (11 new range tests; zero skipped). Actual 887x887 LAB/CMYK probes select 353,079/214,138 pixels and match both pinned public outputs and isolated saved-input replay. Replay adds 29,599,232/52,102,656 CUDA bytes within 234,924,160 planned; 13,312 preexisting bytes are separate. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence. There are now 19 declared saved-stage replay operations.

In **Pipeline → … → Other pipelines**, choose **PlantCV custom-range threshold**. The channel determines the bound list length: RGB/HSV/LAB three, CMYK four, GRAY one. Use integer bytes 0..255; bounds include both endpoints. RGB bounds follow B,G,R order. Reversed bounds successfully select no pixels. The masked image retains original RGB colours. Changing channel or bounds recomputes the threshold only. Keep the recipe for reopen/batch; Export run for actual products; Replay saved stage compares both outputs and previews masked RGB while preserving the current run.

Earlier checkpoint history follows.

SMPTS executable paper reconstruction checkpoint (8 October). The additive `smpts_paper_fixed_mean` graph now executes the paper-supported branch/filter/intersection/contact/erosion/extraction sequence. Defaults25/5/13 come from the paper; offset/polarity, fixed cutoff, connectivity, axis-aligned boxes, box-area cutoff, border, erosion shape and crop rules remain declared assumptions. The historical unresolved graph is unchanged.

The 111-test relevant gate passes in 117.973s (19 new tests;0 skipped). The 2080x3120 committed fixture matches nine pinned OpenCV substep rasters and all 1118 packed crops. This verifies the declared reconstruction, not original SMPTS conformance or scientific accuracy. The full F01-F22/WP0-WP6 goal remains active; full discovery is still required before completed handoff.

In Pipeline choose the SMPTS paper workflow from Workflow or **… → Other pipelines**. The fixed-global variant uses mean 25/offset2/light, median 5, fixed 100/light, strict box-area cutoff 10000 and erosion 13 by default. Offset/polarity/cutoff/contact rule and borders are assumptions, not recovered settings. Run, inspect contact IDs and selected erosion, then choose an individual crop in the extraction node. **Keep recipe in project** preserves settings/layout for reopen and batch; **Export run…** preserves actual products/provenance. Shrunk components/crops are not recovered complete seed outlines. Unsupported work/precision/resource domains are explicit.

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
outside the then-supported 128-label trait domain. That dated check does not define the current 1,024-label limit. Geometry remains near its recorded 64M cap.
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

After exporting a successful native run, select **Watershed debug image** and
choose **Replay saved stage...**. Select that run directory to compare a fresh
RGB diagnostic and palette facts with the saved products. The active run and
archive stay unchanged. The separate view reports matched, mismatch or explicit
unavailability; Stop and closing drain the worker. Agreement is computational.

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
then-supported 128-label downstream measurement domain. This is historical diagnostic coverage; the current measurement limit is 1,024 resolved labels.
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

Use **Add watershed debug image** in a source/calibrated binary, mean or Otsu
native recipe. The branch is added in the same window and preserves dragged
node positions. Run the recipe, then select the debug node to inspect its
image and palette facts. Palette or stroke edits reuse analytical labels.
Project recipe and whole-run archive round trips include the branch.
Saved-input replay for this new node is still unavailable.

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
256 ordered class masks; current size/colour metadata admits up to 1,024 resolved labels, subject to the separate resource budgets. The older controlled 128-label checks retain their recorded scope.
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

This guide describes the workflow current on 8 October 2026. Dated review reports
and experiment logs document the state at their original date; their original
findings are retained for audit. See `CRITICAL_REVIEW_REMEDIATION.md` for disposition,
`NODE_CATALOGUE.md` for generated controls and `SCIENTIFIC_VALIDATION_PROTOCOL.md`
for the laboratory work still required. The concise capability matrix is in
`CURRENT_STATUS.md`.

1. Launch `python seed_vision.py`. The routine workspace opens on the image.
   Species starts **Unknown / unassigned**. Assign verified project metadata;
   this selection is not an automatic species identification.
2. Load images and inspect acquisition/calibration diagnostics. Supply Foreground
   references or an applicable pinned library. **Unavailable** means mandatory
   evidence or an instance result is missing; it is not an empty-dish count.
3. Choose **Annotate: Materials** or **Seeds** and draw only while the corrected,
   deskewed image is displayed. Seed selection and **Apply + save** precede
   trait/shape review. Saving progress does not approve an
   incomplete contour as a physical seed edge. Only connected, complete,
   shape-reviewed, non-excluded outlines supervise physical contours.
4. Run the pipeline. **Stop current work** (`Ctrl+.`) requests cancellation at a
   cooperative boundary. Closing waits asynchronously for running work, then
   resolves drafts with Save/Discard/Cancel. Failed saves retain in-memory edits.
5. Open **Results** (`Ctrl+R`). Inspect proposed, accepted,
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

## Frozen-model Naive Bayes

Open the reference graph view and choose **Run frozen Naive Bayes…**. Select the
image and a PlantCV probability-table TSV through **Choose PDF model…**, then run
the native calculation. Select a class to inspect its actual mask. Save recipe
records the selected model identity; the model file must remain available and
unchanged. Reselect a changed model to calculate its masks while reusing the image.
Stop or close requests cancellation and drains the worker. This workflow performs
frozen-model inference. Native training and complete saved-project/batch model
binding remain pending.

## Frozen-model K-means

Open the reference graph view and choose **Run frozen K-means…**. Select an image
and use **Choose centres model…** for a numeric-centres JSON file. The model fixes
patch size, channel count and ordered centres; its path, raw-file hash, numeric
digest and geometry are derived facts. They are not separate editable controls.
The saved recipe references the selected local file, which must remain available
and unchanged. Reselecting a changed file reuses the prepared image while updating
prediction and masks. This picker accepts the numeric JSON format below and can
train a new model through **Train centres…**. Conversion of existing PlantCV joblib
estimators remains pending.

```json
{"schema":"seedfiddle.plantcv-kmeans-centres.v1",
 "preprocessing":"plantcv-4.11.3-predict-sigma5-uint16-v1",
 "dtype":"float64","patch_size":1,"channels":1,
 "centres":[[30.0],[220.0]]}
```

This illustrates the file format, not a validated seed model. Each centre must
contain `patch_size * patch_size * channels` finite values in0..255, in patch-row,
patch-column, channel order. One-channel models consume the declared prepared
grayscale raster; three-channel models consume BGR feature order after CUDA channel
reversal of the prepared RGB image. Prediction uses the pinned sigma5 preprocessing.
The native `encode_kmeans_model` API exports validated canonical JSON bytes with an
incremental8MiB limit. Re-encoding preserves the ordered numeric identity while
format changes acquire a new raw-file identity. Model/export/prediction/workflow
verification passes24 focused tests after this addition.

Run and select the **PlantCV K-means category masks** node, then choose a cluster
in the top selector to inspect its actual mask. The node's **cat list** setting is
effective: `null` returns every decimal cluster ID in order; `[0,2]` returns one
union mask named `selected`. Repeated or absent IDs follow PlantCV behavior;
an empty list is invalid. Editing this setting recomputes only masks and reuses
the prediction. Reset restores the node's initial value. Stop or close drains the
worker; a failed or cancelled run clears the previous preview. Numerically ambiguous
Gaussian rounding or fractional-centre decisions withhold the complete prediction.

The reviewed engineering gates pass37 backend and27 workflow/compatibility tests;
the shared68-test runner/desktop/reference/register gate also passes. Scientific
model validation and complete project/batch/report/checkpoint integration remain open.

### Train K-means centres

Choose **Train centres…**, select a folder of uint8 grayscale or BGR images, then
set clusters, patch size, seed and initializations. Defaults match PlantCV patch
size10/seed1/n_init10 with sigma5 preprocessing. **Filename contains** applies a
substring filter. Image count0 selects the sorted directory; a positive count
selects with replacement before filtering. PlantCV requires the first selected
entry to be a matching decodable image; the native workflow reports that source
failure explicitly. Alpha and higher-depth images remain outside this domain.

Use all patches, a positive patch count, or a fraction strictly between0and1.
Counts clip to available patch positions while retaining source sampling with
replacement. For large images choose a smaller count/fraction to fit the bounded
CUDA dataset; unsupported resource requests fail before analysis allocation.
The estimator seed affects fitting; per-image patch sampling remains source seed1.
File choices use independent Python randomness. Changing fit controls reuses CUDA
rows and the selected files. Changing patch controls rebuilds rows while keeping
that selection. Choose the folder again to request a fresh file selection.

**Stop** cancels between work stages. Closing waits for the worker to finish.
**Save model…** writes the model and its training provenance together in one
atomic JSON artifact and selects it in the existing K-means view. The artifact
records the chosen image names/order, file identities, effective settings and
fit/RNG/calculation metadata. Existing centres-only JSON models remain loadable.
Keep recipe in project and Save Project bind the selected training artifact for
reopen/snapshots/batch. Held-out model evaluation remains open.
The runtime certifies numerical decisions across all valid source
thread reductions, withholding undecidable results.

To use an existing PlantCV `.fit` model, run this from the project folder with
the Python environment that trained the model:

```powershell
python -m seedvision.model_io.convert_plantcv_kmeans model.fit model.json
```

Then select the resulting JSON in the K-means view. The converter preserves the
ordered float64 centres and records the source file/environment and inferred
patch/channel geometry. It supports the declared gray/BGR patch domain and
rejects unsupported models before replacing an existing destination. The desktop
runtime does not require sklearn or joblib; the conversion command uses the
source environment's packages. Conversion does not retrain or validate a model.

## Individual seed crops

The reviewed crop-export workflow adds **Export crops for visible rows…** to
**Results**. Use the review/seed filters to choose rows, then supply
pixel padding and a destination. Each available seed produces a corrected image
PNG, an isolated binary-mask PNG and coordinate/provenance metadata in
`seed-crops.json`. The padded image may contain neighbouring seeds; their IDs and
pixel counts are recorded separately from the isolated target mask.

Stop cancels at an extraction or file boundary. Incomplete exports remain in an
explicitly incomplete working directory; only a completed manifest is published.
Changed sources, results or corrected frames require reopening/recomputing the
result. Export does not approve an outline or establish scientific validity.

For a saved project, `scripts/analyze_pilot.py --seed-crop-padding-px 4` adds the same
crop artifacts to each result report. The option requires nonnegative integer
padding. Full peer recipes and original SMPTS extraction compatibility remain open.

## Planned peer capabilities

The [PlantCV–SMPTS direction](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) is accepted
work in progress. Developer-level checkpoint, recipe, conformance and isolated
original-operation APIs and reference-graph inspection are implemented. A
focused-tested native calculation view now runs the three canonical PlantCV seed
graphs with editable controls and real CUDA products. Complete peer outputs and
the project/batch comparison workflow remain pending. Scientific validation and
physical/trait availability limits remain unchanged.

Project saves now use schema6 and read versions1-5. They preserve attached
reference graph metadata, including historical run hashes, through save/load and
snapshots. Those hashes do not mean checkpoint arrays are present or restored.

Save and Save As copy available models selected by historical and working native recipes
into `native-models/` beneath the workspace root. Keep that folder when moving
the workspace. Moving only the project JSON does not copy its dependencies.
Legacy or unavailable models can still be saved as metadata and are reported
as unavailable; snapshots require complete verified copies. Opening a saved
native graph launches its exact selected recipe with the project model copy.
Choosing or training a replacement model clears the old copy binding; Reset
restores the saved selection and binding.

To retain an edited seed, extended-threshold, Bayes or K-means recipe, choose
**Keep recipe in project**, then **Save Project**. This saves selected controls,
node positions and available model copies separately from historical graph records.
Use **Project native recipes** in the reference-graphs window to reopen a working
selection or remove it from the project. The corresponding recipe launch button
also reopens its saved working choice. **New Project** clears those draft choices.
Save As changes the JSON destination while dependencies remain under the workspace
root. The colour-correction workflow below retains its exact per-image four-file selection.

## Inspect factory pipelines and retained evidence

Load a fresh factory in Pipeline and keep Freeze checked to inspect its original topology and node controls. Runs show actual outputs in the same workspace. Factory templates cannot be overwritten. Existing project reference attachments, checkpoint files and QC records remain compatible; opening a factory does not manufacture execution history. The retired Reference-window checkpoint-store browser is no longer a user-facing route. Existing checkpoint verification/export services and portable snapshot data remain available.

## Run a native seed calculation

In Pipeline choose the PlantCV binary, adaptive-mean or Otsu seed workflow. Open the current image, clear Freeze to supply required controls, and select Native or PlantCV execution. Required unselected fields remain explicit. An unselected analysis label count is resolved from the actual final labels and shown with the result.

Click **Run pipeline** to calculate. Select a computed node to see its bounded preview,
resolved controls and available compact observations. The colour node shows its
returned hue distribution with a sample selector. RGB/LAB return an empty hue chart;
the numerical observations for their selected channels remain available. Selecting
a displayed chart sample preserves cached calculations. Changing an analysis control recomputes
that node and its dependents on the next run; moving cards preserves cached work.
**Stop** requests cancellation, and closing waits for the worker to drain.
**Save selected recipe…** atomically saves the selected `RecipeSpec` as UTF-8 JSON.
The same selection can be loaded with `RecipeSpec.from_json` and passed to
`seedvision.benchmarking.native_recipe.run_native_recipe_path` in Python.

After a completed run, **Export run...** lets you choose a parent folder and writes
a new run folder containing `native-results.json` and its output checkpoints.
This also works for a partial run: unavailable outputs keep their reason and only
available products are saved. The archive includes ordered observations, diagnostic
images/masks and the exact selected Bayes/K-means model bytes when present. Export
runs in a worker; **Stop** cancels and closing waits for cleanup. Changing an image
or calculation control disables export until the next result is calculated.

Saved-project batch now adds `native_graphs/native-graphs.json` and per-graph output
folders to each baseline report when native graph attachments or working recipes exist.
It uses their exact saved selections, records model/source failures, and reports
original/reference executors as unavailable. The archive loader
`seedvision.export.native_results.load_native_recipe_result` verifies the report and
product bytes. Ledger schema2 identifies historical and working selections;
working rows do not invent historical run records. Model relocation and verified
snapshot closure cover both collections. Per-stage replay remains pending.

This focused-tested candidate records the actual decoded image frame and runtime.
It does not apply the production calibrated/deskewed frame. Select the size node to
inspect all 17 observations for supported empty, external and nested contours,
including holes and disconnected regions. Label and object-count controls affect
returned samples. Unsupported numerical ellipse fits withhold the whole bundle;
the adaptive-mean fixture exercises an angle-boundary limit. The reviewed deterministic
ellipse path supports its guarded runtime5.0.0 domain. The size node displays its
actual nonverbose diagnostic overlay. **Run native calculation…** now creates an
additive working copy with a **line thickness** field, initially 5. Set it to 1
for the isolated reference worker's stroke width; both measurements and drawing use
the selection. The reference's fallback caliper calculation still uses five pixels.
Changes recompute size and its dependents; reset/save preserve the working recipe's
selection without editing the historical graph. Drawing-resource limits preserve
available measurements and mark the node partial. Other numerical/drawing domains
remain pending; widths one/five have focused public-reference coverage.
Whole-run native output publication and saved-project companion runs are available.
Correction-session project integration, per-stage replay and fresh full verification
remain open. Computational agreement does not establish scientific accuracy.

To run Gaussian, dual-channel, texture or triangle thresholding, click
**Run extended threshold…** in the reference-graphs dialog and choose the operation.
Each opens a separate working recipe. Choose an image, select the threshold node,
edit its controls and run; reset and selected-recipe save use the same controls as
the seed view. Gaussian accepts odd kernel sizes from 3 to 15; larger supported
kernels can report an unavailable mask when byte rounding is uncertain. These
authored defaults demonstrate calculations and are not validated acquisition
settings. Texture and triangle can report explicit unsupported numerical
or input domains. Triangle exposes polarity; its source's xstep seed does not affect
the final Otsu mask and is not a working control.

For the PlantCV colour-card correction workflow, choose its graph and click
**Run native calculation…**. Select target image/mask and source image/mask using
the four file controls, then run. Masks must be single-channel uint8 and match their
image dimensions; the pinned correction path also requires equal image heights.
The working view shows corrected RGB pixels and reports the available compact
matrix outputs. Repeat runs reuse unchanged preparation/calculations; changed
file bytes, verified preparation/runtime identity or tensor versions invalidate
them. Inputs require supported eight-bit PNG/JPEG/BMP/TIFF headers; image/mask sizes
must match without an implicit transform. **Export run…** saves corrected RGB and
compact fit matrices with the byte/frame/decode identity of all four inputs. The
selectors and recipe controls lock while export runs; Stop and close drain workers.
Selected-recipe save stores the native graph and layout, not four input files.
Saved per-image correction bindings now survive Save As and portable snapshots.
Saved exact per-image bindings run in batch with explicit unavailable/failed rows.
Select the project image before opening correction. After a successful run, choose
**Keep recipe in project**, then **Save Project**. Keep verifies copies in a worker
and retains the current canvas layout for this exact image. A different RGB path
cannot be kept for it, even with identical bytes. Use **Project native recipes**
to reopen the selection; Run verifies project copies without reading original
selections. Editing a reopened target/mask uses the selected project RGB image
for the next fresh run. Missing/corrupt pinned inputs fail without fallback.
Keep leaves the master unchanged until Save Project; close or a project change
during copying discards the pending session merge after the worker drains.
Automatic card detection and independent colour validation remain open.

## Material and seed annotation

The annotation palette belongs to the application window and can be dragged beside
the image without being clipped by the image view. Material and seed annotations
use corrected-image coordinates. They are deliberately hidden, and drawing is
disabled, on source, gamut, prototype-collage, or any other non-deskewed display.

For material references, choose Background, Foreground, or Other and paint the
reviewed region. Use the independent exclusion controls only when the corresponding
class model should ignore an area. **Apply + save** applies the draft to analysis and
attempts to save the per-image sidecar. A failed save leaves the applied state in
memory and is reported explicitly; use **Save Project** to retry. Undo/Redo remain
local to the current annotation history.

Click **Seeds** to edit the current seed labels; opening the editor does not import
another reference. For seed instances:

1. Select an ID, **Next empty**, or **Next unannotated**. The latter cycles painted
   seeds whose condition or shape review is incomplete. Set **Show** to all or
   selected as needed.
2. Choose Brush, Trace edge, Shape fill, Smart fill, or Eraser. Tool instructions
   are in each button's tooltip. Clear seed and Clear all are beside the visibility
   control.
3. **Import bundled seed outlines…** imports only the source-bound bundled labels.
   If none match, it explains how to use **Import seed outlines from file…** instead.
   File import accepts lossless categorical PNG, TIFF or NPZ labels whose dimensions
   exactly match the corrected image; the chooser shows the expected dimensions.
   It does not transform a source-coordinate mask or silently select another source.
   Both commands create an undoable draft. Review it before applying; imported
   outlines are not automatically accepted as ground truth.
4. Review **Seed condition**. **No defects** and the defect choices are mutually
   exclusive. Choose explicit Outline and Pose values. A Complete outline checks
   and locks **Full length visible**; incomplete outlines can set it explicitly for
   size-only use.
5. Leave **Exclude from modelling** clear unless the seed must be omitted from
   model/library contributions. It sits beside **Apply + save**.
6. **Pick location** arms the hilum selector for one placement on the current seed.
   It turns off after the pick, when the selected seed changes, or when Escape is
   pressed. **Clear location** removes it. Direction is calculated internally from
   the painted seed centroid and is not a separate operator input.

The pixel and ID summary appears in the relevant Project node details rather than
occupying permanent palette space.

Keep these states separate:

| State/action | What it means |
| --- | --- |
| Draft | Editable/imported labels, separate from the labels currently used by analysis. |
| **Apply + save** | Uses the labels in analysis and attempts a sidecar disk save. A save failure leaves applied labels in memory; the confirmation text reports it. **Save Project** also records the sidecar digest and project settings. |
| **Export seed outlines…** | Writes the applied full-resolution labels as a 16-bit PNG in corrected-image coordinates. Apply a draft first to include its changes. |
| **Add image and seed labels to training dataset…** | Creates a normalized image crop, model input features, seed labels, derived boundary targets and a manifest entry. This prepares data; training is a separate Learning action. |

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

## Optimize from applied project references

Save a project, apply the intended references, and clear **Lock**. Use
**Analysis → Optimize eligible nodes from project references…** or a supported
node's **Optimize from project references…** action. Current-image and mixed
library/current-image reference modes work from local applied references without
a library pin. **Species library only** requires a compatible pinned library;
a configured missing, broken or incompatible pin is reported rather than replaced
with another version.

The plan shows eligible and excluded seed counts, excluded image/draft reasons,
and each node's image readiness. Only complete, connected, shape-reviewed outlines
without an explicit shape exclusion are instance targets. Apply or discard drafts
before fitting them; hover excluded counts for their seed IDs. Each image retains
its own biological context and local references. Choose images and a bounded budget,
then inspect the proposed changes before applying. Cancellation or changed inputs
withhold the proposal, and accepted changes apply together. Optimization is
in-sample project adaptation; it does not establish independent validation.

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

Image zoom buttons retain the current viewport centre even when the pointer is over
the toolbar. Mouse-wheel zoom retains the scene point under the pointer.

## Training and evaluation

Use **Learning → Seed outlines → Add image to training dataset…** after analysis
and label review. The readiness message reports complete reviewed outlines,
excluded IDs and unapplied drafts before writing. Fix incomplete, disconnected,
unreviewed or explicitly excluded outlines and apply/discard drafts; export refuses
to silently omit those seeds. The ready dialog explains the normalized crop,
features, seed labels, derived targets and manifest it will create. The suggested
sample revision avoids identifiers already in the manifest or immutable revision
folders.

Export also requires verified species for species-conditioned features and an
explicit biological/capture group. A filename is not a grouping policy. Related
captures must share one split. Exports create immutable component revisions before
switching the audited manifest; old features/labels remain paired if writing fails.
After export, use **Train or refine U-Net / StarDist…** in Learning to train a model.
Saving a label PNG or adding a dataset sample does not start training.

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

Post-review display verification: all18 card/shared Qt tests pass in38.226s
(`artifacts/automatic-card-working-preview-gate.log`). Actual1440x900 overview,
ROI-labelled-mask and scale views were rendered with Windows fonts and inspected.
The preview now fits its current label dimensions, preserving aspect ratio and
showing all24 chip circles. This display-only repair changes the shared UI hash;
the other six current source/test hashes remain unchanged. The103-test gate
predates the preview repair; the18-test gate includes the12 strengthened card
tests and all shared Qt regressions. Six register tests pass after reconciliation.
Full F01-F22/WP0-WP6 and fresh full-suite/scientific gates remain active.

## Calibrated seed workflow

Open Reference graphs, choose **Calibrated seeds…**, and select binary, mean or
Otsu thresholding. Select the source image and, when needed, set the card
preprocessing `roi_bounds_xyxy`. On **Select seed region**, enter
`seed_roi_xyxy` as `[x0,y0,x1,y1]` in the corrected full image; right/bottom
coordinates are exclusive. This selection is required. Select the full bounds
explicitly to analyze the whole image. Card overlap is reported; pixels are never
silently removed. Choose the required image-specific median, threshold, fill and
morphology settings, then run. Failed card detection blocks dependent seed nodes.

The crop precedes seed filtering and adaptive thresholding. Labels, diagnostic
images, centroids and ellipse coordinates belong to the local crop. The facts
panel shows its frame and translation: parent xy = local xy + (x0,y0). API users
can read `result.labels_coordinate_frame`; `result.coordinate_frame` continues to
describe the source inputs. Whole-run archives retain exact crop metadata and
port semantics, but are not per-stage replay certificates.

Set **Chip size calibration** to a named card size or `[width,height]` in mm for
physical size values. The **Size in calibrated units** node preserves original
pixel measurements separately. As in pinned PlantCV, every linear trait uses
the width factor and areas use width times height; coordinates stay in local
pixels. The UI shows squared area units. Serialized source `scale`/`label` fields
use `mm` for both linear and area traits, matching PlantCV; conversion metadata
records the area factor and variable lists. With no chip size, this node uses
pixel units. Unsupported size geometry remains unavailable.

Use Keep in project and Save Project to retain the selected working recipe.
Project native recipes reopens it; saved-project companion batch uses the same
executor and selected settings. Changing the crop reuses card correction;
changing only chip dimensions reuses image processing and pixel measurements.

## Sample-table Naive Bayes training API

Sampled RGB table training is available through
`seedvision.benchmarking.native_naive_bayes_training.train_bayes_sample_file(path)`.
The returned completion contains `tsv_bytes`, the parsed `model`, CUDA `samples`
and `fit`, and immutable `provenance_json`. The TSV can be selected by the existing
**Run frozen Naive Bayes…** workflow. The API does not save files; the desktop
action below provides atomic saving and project-held training provenance.

Use tab-delimited class headers and comma-delimited integer RGB cells. Class
order, row order and duplicate-header combinations affect the fit. Missing/empty
cells are skipped; one class is allowed. Each resulting class needs at least two
samples with variation in all three HSV channels. Constant channels are rejected
as an unsupported native domain, even when SciPy's roundoff produces positive
covariance. Inputs are limited to16MiB,32 header columns/unique classes and one
million combined samples. This numerical fit has empirical source conformance;
it does not establish held-out accuracy or scientific validity.

## Train a Naive Bayes model in the desktop

1. Open **Run frozen Naive Bayes…**, then choose **Train model…**.
2. Select **Sampled RGB table** and choose a table, or select **Images and masks**
   and choose both directories. Images are traversed recursively in recorded
   filesystem order; masks use the image basename directly under the mask root.
   The source's case-sensitive last-three-character rule accepts `png`/`jpg`;
   ordinary `.jpeg` names are excluded. Missing masks are skipped. Decoded image
   and mask shapes must match. Mask0 is background and mask255 is foreground;
   other values are ignored. PNG/JPEG contents are supported.
3. Press **Train**. Both modes use the source default Gaussian bandwidth.
   Directory training consumes the current global NumPy random stream and has
   no seed control. Independent HSV columns take floor(foreground/10) samples
   for each class, retaining the public last-population-pixel exclusion.
   **Stop** cancels; closing waits for workers to drain. Failed, cancelled or
   changed selections require a new fit. Earlier random draws stay consumed.
4. Review the compact class counts and, for directories, ordered pair count and
   post-training RNG identity. Choose **Save model…**. The exact PlantCV TSV is
   saved atomically and verified before selection. A failed save preserves the
   previous destination; a failed post-save reload selects no model.
5. Run the selected classifier. Choose **Keep in project**, then **Save Project**
   to retain the editable recipe, typed training record and verified model copy.
   Reopened/moved projects and companion batch runs can use that copy after both
   training roots, the original samples and original model are removed.
   Whole-run archives retain the current record and actual TSV bytes.

The standalone TSV is interoperable with PlantCV and contains probability tables
only. Selecting that TSV afresh through **Choose PDF model…** creates a plain
selection without a training record. Keep/save the trained recipe to preserve
its record. Provenance describes the training inputs and producer; it does not
validate accuracy or biological meaning. Broader numerical training domains remain pending.

## Binary image/mask sampling API

`seedvision.cuda.plantcv_naive_bayes_sampling.sample_naive_bayes_image(bgr, mask)`
returns two CUDA uint8 Nx3 sample tensors for plant/background, a bounded plan,
immutable before/post random states and ordered draw identity. Pass matching
contiguous CUDA uint8 BGR HWC and mask HW. Only mask255 and mask0 participate;
intermediate values are ignored. Each HSV column is sampled independently using
PlantCV's exact legacy random calls. The default consumes the existing global
NumPy state; `rng=` can receive a captured legacy RandomState. No seed is added.

The return can feed `fit_naive_bayes_hsv(result.samples, result.class_names)`.
Zero samples are a valid sampler outcome but cannot fit a KDE. The source's
last-pixel exclusion and invalid-background failures are preserved. Cancellation
and source errors publish no sample result; earlier random draws stay consumed.
The separate directory adapter enumerates and decodes inputs; the native
training dialog provides directory selection and atomic TSV saving. Sampling value
conformance and empirical PDF comparisons do not validate training accuracy.


## Optional Naive Bayes probability plots

After training in either input mode, select **Save probability density plots
(SVG) beside the model**, then **Save model…**. The background renderer creates
`hue_pdf.svg`, `saturation_pdf.svg` and `value_pdf.svg` with ordered class curves
at byte values0..255 and the source's automatic legend placement. This option
does not require a new fit. Stop or closing drains rendering and writes no files
when rendering is cancelled. Rendering errors leave the trained model available
for retry, including saving with the option cleared.

All SVG payloads render before any output is saved. The exact TSV is committed
and reread first; the three sibling SVGs are then replaced atomically one file
at a time. A later SVG failure reports actual saved/failed filenames and keeps
the verified TSV selected. Each failed atomic write preserves its previous file;
the four outputs are not one filesystem transaction. Choose a model filename
that differs from the three fixed SVG names. SVGs are separate presentation
files. Their default-style source comparisons do not establish training accuracy
or cross-platform byte identity.

## Replay a saved native stage

Select a supported raster node in the native calculation dialog, click
**Replay saved stage…**, and choose a directory created by **Export run…** with
the same recipe settings and current calculation runtime. An image need not be
selected because execution uses the saved input product. The worker verifies
the whole archive, reruns only that node and displays match/mismatch plus a fresh
preview. **Stop** cancels; closing waits for the worker to drain. The active run
and its graph status stay unchanged. Select a node again to inspect that run.

Median, binary/mean/Otsu threshold, fill, erosion/dilation, watershed distance,
maxima, connected markers, negative distance, masked flood, marker count and the
debug image are supported in source or canonical calibrated seed-ROI graphs.
Gaussian, triangle, texture and dual-channel threshold replay use their separate
standalone source-frame recipes. Each operation retains its documented limits.
Maxima replay previews its painted peak mask; count replay shows the estimated
count above detailed facts and explicitly has no image. Completion and count
records are compared alongside all raster/coordinate outputs. Local replay facts
retain both frames and their translation. Other operations/cropped graphs or old
runtime bindings report unavailable, unsupported or invalid outcomes as appropriate. Replay currently rejects UHD-sized selected rasters; conservative general-fill
work admission can reject smaller inputs too. A computational match does not
certify source conformance, biological identity or laboratory accuracy.
