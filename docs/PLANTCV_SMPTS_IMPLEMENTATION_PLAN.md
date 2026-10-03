# SeedFiddle: PlantCV–SMPTS peer-functionality implementation and benchmark plan

**Prepared for:** Eric Fedosejevs
**Original external plan:** 1 October 2026
**Revised and adopted direction:** 3 October 2026
**Status:** Accepted project direction; peer implementations and benchmark infrastructure are not yet implemented or validated. This revision integrates documentation only; no comparative performance experiments have been run.
**Intended application:** Soybean and lupin seed-image analysis within the existing Python-based SeedFiddle application.

## 1. Decision and relationship to the comparison report

This is the canonical delivery specification for the new direction in
[PLAN.md](../PLAN.md), accompanied by the revised
[comparison report](PLANTCV_SMPTS_COMPARISON.md). It extends the existing engineering
baseline; it does not reclassify proposed functionality as implemented. Current
capabilities belong to [CURRENT_STATUS.md](CURRENT_STATUS.md), architecture contracts
to [ARCHITECTURE.md](ARCHITECTURE.md), and empirical requirements to
[SCIENTIFIC_VALIDATION_PROTOCOL.md](SCIENTIFIC_VALIDATION_PROTOCOL.md). The proposed
backend names, paths and YAML below are design contracts, not runnable features.

The supplied external comparison motivated this direction. Its recommendation to replace the application backbone and its fixed splitter cascade are superseded by the revised comparison and this plan. The integration targets peer functionality within SeedFiddle, with explicit evidence for every selected capability.

**Retain SeedFiddle as the application, pipeline coordinator, review interface and source of authoritative results. Treat PlantCV and SMPTS as reference implementations and sources of individually replaceable algorithms—not as replacement application frameworks.**

The objective has four distinct parts:

1. **Achieve peer functionality:** deliver the selected seed-focused capabilities
   and outputs with usable desktop/batch integration. Functional coverage can reuse
   existing native code; it does not imply exact reference-output equivalence.

2. **Reproduce specified behaviour:** establish computational compatibility with selected, version-pinned reference operations.
3. **Improve scientific performance:** reduce independently measured counting, boundary and trait errors on representative soybean and lupin images.
4. **Improve operational performance:** reduce compute, memory and correction costs without relaxing the agreed quality requirements.

These four claims require separate status and evidence. A numerically faithful implementation can reproduce a reference algorithm's weaknesses. A better algorithm can intentionally differ from reference outputs. A faster implementation is not necessarily a more accurate one. Record and test each claim separately.

### Repository basis and limits of this review

The connected repository is `efedo/SeedFiddle`. This plan uses its current README, architecture, status and scientific validation documents, plus the implementation roadmap and a targeted inspection of `seedvision/learning/metrics.py`. Code-search results and explicitly pinned reads identified snapshot `094aa7dd587099ac0444dfec02f556a1cbf53108`. The documentation itself describes a review through revision `f244df8`, dated 29 September 2026. These are different identifiers and should not be conflated. [R1–R6]

This was not an exhaustive source audit. Consequently, “add or verify” below means that exact compatibility or coverage was not established by the inspected material; it does not prove that equivalent code is absent elsewhere. The local checkout at documentation integration is the same `094aa7d` snapshot. The established shape, learning and operational records were also checked for overlap. Freeze and inspect the actual target branch again before algorithm implementation; this remains a documentation and targeted-source review, not a complete code audit.

## 2. Scope: a seed-focused union of functionality

Scope is the union of the **seed-relevant processing and analysis functions** selected from PlantCV and SMPTS: calibration, image/mask preparation, foreground segmentation, connected-object analysis, touching-seed separation, object extraction, geometry, colour, diagnostic output and reproducible batch execution. It is not a commitment to reproduce unrelated PlantCV capabilities such as photosynthetic measurements, whole-plant temporal tracking or root-specific analysis.

Use two coverage tiers. The first delivery slice must name an exact PlantCV seed
recipe, its selected preprocessing/watershed/measurement outputs, and the recoverable
SMPTS extraction stages. F01–F21 define capability families, not a claim to clone all
PlantCV functions. Expand and approve scope through the project roadmap; do not
silently shrink core coverage to make a completion claim.

- **Core peer coverage:** every operation and output used by the selected PlantCV seed-analysis recipe, and every recoverable operation and output in the selected SMPTS implementation.
- **Extended capability coverage:** additional seed-relevant PlantCV methods—such as alternative foreground classifiers—and SeedFiddle improvements, each with its own requirement and test. Do not describe these as complete merely because a general-purpose extension hook exists.

Build `docs/plantcv_smpts_features.json` during WP0 before algorithm development
(the register is a planned deliverable, not present functionality). Track functional
coverage separately from compatibility and empirical validation, with states such
as unaudited, reuse-confirmed, gap, implemented, verified, blocked and explicitly
out-of-scope. Every exclusion needs a rationale and scope revision. No aggregate
peer-functionality claim is valid while a core capability is blocked or unverified.
Build the register before development. For each feature record: source function or paper section, reference version, exact input/output contract, native implementation location, whether behaviour is reused/ported/enhanced, test IDs, validation status, and any unresolved ambiguity. A claim of functional peer coverage requires all core capability acceptance tests to pass; a separate claim of exact compatibility requires every selected conformance row to pass; an unavailable reference remains explicitly unavailable.

## 3. Existing SeedFiddle capabilities to preserve and reuse

The repository already describes a native PySide6 desktop application, a typed computation graph, CUDA-first evidence processing, procedural instance proposals, optional U-Net/watershed and StarDist branches, reviewed geometry, immutable result revisions, batch execution and a scientific evaluation protocol. Learned branches are implemented but not established as validated laboratory models. The application deliberately withholds validated physical units and unvalidated biological traits. [R1–R5]

| Existing subsystem | Integration decision | Main work remaining for this plan |
|---|---|---|
| Typed graph, node-local caching and settings translation | Reuse; do not introduce a second workflow engine. | Add backend identities, compatibility parameters and intermediate checkpoints to the existing contracts. |
| Material references, corrected coordinates and calibration diagnostics | Reuse their source identity and eligibility rules. | Establish exact reference preprocessing and calibration compatibility; independently validate physical and colour accuracy. |
| Procedural segmentation | Retain as the frozen current baseline and production candidate. | Expose separable foreground, marker and partition stages for controlled comparison. |
| U-Net/watershed and StarDist | Retain as optional existing comparison arms. | Supply reviewed development data and frozen checkpoints before scientific comparisons. |
| Reviewed geometry and pose/visibility metadata | Reuse the measurement and eligibility layers. | Add a definition-by-definition PlantCV compatibility mapping, rather than duplicating all geometry code. |
| Existing instance and probability metrics | Extend, not replace. | Add step replay, matched-ID audit output, reference adapters, paired uncertainty estimates and resource measurements. |
| Review, JSON/CSV/image export and portable snapshots | Reuse. | Expose comparison records and backend provenance without changing authoritative-result semantics. |

### Non-negotiable integration constraints

Preserve the single `seed_vision.py` launcher, native Qt interface, on-premises operation and the documented Python baseline. Production processing should retain reusable full-resolution arrays on the CUDA device; CPU reference comparisons are a separately labelled benchmark facility, not a reason to copy all production rasters repeatedly to the host. [R1–R3]

Keep analytical settings in the existing graph/settings pathway. New controls must affect real calculations, have optimization coverage or an explicit exemption, and appear in the generated node catalogue. Preserve source-bound corrected/deskewed coordinates, annotation eligibility, cache invalidation, cancellation, project migrations and snapshot identities. Do not reinterpret unknown coordinate frames by resizing masks or changing schema numbers. [R2–R4]

Do not use the legacy `BaselineAnalysis.count` compatibility property for new authoritative reporting. Preserve the distinction between an unavailable analysis and a valid empty result. Experimental physical measurements and unvalidated traits must not bypass the existing release gates. [R2, R4]

## 4. Reference implementations and naming

PlantCV is modular, so “compare against PlantCV” is insufficiently specified. Pin the actual seed recipe, library release or commit, transitive dependencies, parameters, image decoding and output definitions. Its documented watershed function consumes an RGB image and binary mask and exposes a minimum local-maximum distance; its size analysis consumes labelled objects. [R7–R9]

SMPTS is a narrower reference. Its 2024 preprint describes two threshold-derived masks, filtering and combination, bounding-rectangle localization, targeted erosion and individual-seed extraction. The authors report soybean separation results, not a validation of lupin morphometry. They link an MIT-licensed CodeOcean capsule, but that original implementation was not retrieved or executed in this review. A paper-derived port must therefore remain distinguishable from a verified original-code run. [R12]

| Proposed backend identity | Purpose | Conditions for use |
|---|---|---|
| `seedfiddle_current` | Frozen pre-change baseline. | Pin repository revision, saved graph, references and settings. |
| `plantcv_reference` | Run the original selected PlantCV functions/recipe. | Isolated reference environment; exact versions and workflow retained. |
| `plantcv_compatible` | Native implementation of specified PlantCV behaviour. | Stage-level conformance tests; documented tolerances and supported domain. |
| `smpts_reference` | Run the original SMPTS code/capsule. | Pending access, licence-file inspection, build reproduction and interface audit. |
| `smpts_paper_port` | Explicit reconstruction from the available description. | Record unresolved choices; do not claim original-code parity. |
| `smpts_enhanced` | Deliberate improvements to the recovered method. | Separate identity and configuration; independent tests for each change. |
| Existing learned method identities | Optional U-Net/watershed and StarDist comparisons. | Development-only training and frozen eligible checkpoints. |

Never substitute `smpts_paper_port` for `smpts_reference` silently. If the original capsule remains inaccessible, the deliverable can still be useful and scientifically evaluated, but the corresponding claim is “a documented SMPTS-inspired reimplementation,” not “verified reproduction of SMPTS.”

### Reference audit checklist

For PlantCV, inspect the chosen release's actual source rather than assuming that current documentation and `main` are identical. For SMPTS, resolve the paper's ambiguous description of its Otsu/fixed-threshold branch, threshold polarity, channel conversion, local-threshold constants, border handling, bounding-rectangle decision rules, kernel shape, erosion iterations and extraction semantics before marking those rows compatible. Standard automatic Otsu threshold selection and a manually fixed cutoff are different operations. [R12, R13]

Audit source licences and dependencies before copying implementation code. SeedFiddle declares MIT, PlantCV declares MPL-2.0, and the SMPTS authors describe their capsule as MIT; the capsule's actual licence and included components still need inspection. Keep provenance and notices with any reused code, and review distribution obligations rather than assuming identical licensing. [R1, R12, R17]

## 5. Proposed processing architecture

### 5.1 Shared stages, interchangeable methods

Use the following proposed extension of the current processing contract:

**Immutable source and acquisition metadata → coordinate/calibration preparation → foreground evidence → mask preparation → instance proposal method → instance validity/review → common measurement layer → existing result export.**

Reference adapters can expose comparable checkpoints alongside this path. They must not own project persistence, the user interface or canonical result definitions.

Do not enforce the earlier report's fixed cascade of watershed, then erosion, then AI. For development, run alternative splitters from the **same unchanged component mask** and retain their outputs independently. Select or route methods using a policy chosen on development data. If a production cascade is later justified by accuracy and cost, each fallback should receive an explicitly defined original input—not silently inherit damage from a previous failed split.

### 5.2 Intermediate contracts

Extend existing records where possible rather than introducing duplicate data models. Each checkpoint should identify:

| Contract element | Required content |
|---|---|
| Identity | Source hash, coordinate-frame identity, upstream checkpoint hash, graph/settings identity, backend and dependency versions. |
| Image semantics | RGB versus BGR, dtype/range, channel conversion, geometric transformation and interpolation method. |
| Mask semantics | Foreground polarity, boolean versus 0/255 representation, connectivity, validity/ignore region and border policy. |
| Instance semantics | Integer labels with background zero, label-ID mapping, split-boundary ownership, visibility and eligibility. |
| Algorithm state | Distance map, seed-centre markers, suspected-cluster IDs, erosion cores, crop rectangles and decision reasons where applicable. |
| Measurement provenance | Mask type, metric-definition version, units, calibration status and uncertainty/eligibility. |
| Execution status | Available/empty/failed/unsupported, warnings, elapsed time, memory and transfer measurements. |

Use explicit representation adapters at the benchmark boundary. Comparing an RGB implementation with a BGR reference, a bool mask with a 0/255 assumption, or differently rectified coordinates would test adapters rather than algorithms.

Do not require numerically identical label IDs. Two arrays can represent the same partition with permuted IDs. Conversely, identical object counts do not establish equivalent partitions.

### 5.3 Proposed source organization

Keep new backend-specific logic outside the already substantial orchestration module. Proposed locations, subject to the first source audit, are:

- `seedvision/segmentation/`: small compatibility and enhanced splitter modules, called by existing orchestration.
- `seedvision/benchmarking/`: checkpoint contracts, reference adapters, replay, conformance and comparative reporting.
- `seedvision/learning/metrics.py`: shared scientific metrics; extend cautiously rather than create competing definitions.
- `tests/`: deterministic and integration regressions using the existing unittest conventions.
- `scripts/`: a benchmark entry point that consumes saved production recipes and locked manifests.
- `docs/`: feature register, reference recipes, benchmark specification and result interpretation.

External reference environments should be isolated from the desktop environment. Running an older dependency stack for reproduction must not downgrade the application's normal runtime.

## 6. Feature implementation and test matrix

“Compatibility” below refers to an audited, pinned operation. “Enhancement” refers to new behaviour that should not alter the compatible mode.

| ID | Stage or feature | Implementation approach | Required comparison or acceptance evidence |
|---|---|---|---|
| F01 | Image decoding, ROI and coordinate preparation | Reuse source identity and corrected-frame contracts; add explicit reference conversions. | Same pixels/ROI at stage entry; transform round-trip and mask alignment tests. |
| F02 | Spatial calibration | Reuse geometry diagnostics and scale evidence. | Synthetic coordinate checks plus independent physical references; remain pixel-only until validated. |
| F03 | Neutral balance and colour-card correction | Keep neutral balance distinct; add or verify the selected PlantCV correction path. | Same patch correspondences for parity; independent held-out patches and sessions for colour accuracy. |
| F04 | Foreground thresholds | Expose selected global, range/channel and local-threshold recipes as explicit methods. | Exact threshold and mask comparisons under fixed dtype, polarity and border rules. |
| F05 | Alternative foreground classifiers | Audit existing material models against selected PlantCV classifier capabilities; add only genuine gaps. | Same training/reference access and frozen preprocessing; no implied equivalence between different classifiers. |
| F06 | SMPTS two-branch foreground construction | Implement the recovered branches, filtering and mask combination separately. | Checkpoints for both masks, filtered mask and combined foreground; unresolved source choices remain flagged. |
| F07 | Mask cleaning | Reuse or add median, component filtering, hole policy and morphology as required. | Pixel-exact primitive tests where semantics match; evaluate damage to legitimate small seeds and coat features. |
| F08 | Connected components and suspected contacts | Expose component geometry and the recovered rectangle-based screening rule. | Component agreement, contact-candidate precision/recall and retained atypical seeds. |
| F09 | Distance transform | Provide a reference-compatible implementation and separately named alternatives. | Numerical map comparison, edge cases and downstream marker sensitivity. |
| F10 | Peak and marker generation | Expose peak spacing, plateau/tie handling and marker connectivity. | Marker matching and marker count on identical distance maps; downstream split/merge effects. |
| F11 | Watershed partitioning | Reuse procedural infrastructure; add exact-reference mode where feasible. | Partition agreement with identical masks and markers; then end-to-end splitter accuracy. |
| F12 | Targeted erosion splitting | Add or verify a recovered SMPTS-compatible operation with all kernel semantics recorded. | Component/core masks, disappearance, residual contacts and false splits. |
| F13 | Contour-preserving recovery | Enhancement: use eroded interiors as markers to partition the pre-erosion foreground. | Boundary, area and dimension error versus erosion-only and other splitters; ambiguous contacts retained as uncertain. |
| F14 | Individual extraction | Retain instance masks and original-image crops with coordinate mappings. | Correct crop identity, truncation, padding, foreign-seed contamination and image/mask alignment. |
| F15 | Size and shape compatibility | Map selected PlantCV definitions to existing geometry; implement missing definitions in a compatibility namespace. | Same-mask, definition-matched tests; distinguish pixel area, contour area, rectangle dimensions, fitted ellipse and Feret metrics. |
| F16 | Canonical research geometry | Reuse reviewed SeedFiddle geometry and visibility rules. | Independent eligible-seed measurements, repeatability and joint detection-plus-measurement coverage. |
| F17 | Colour distributions/statistics | Add or verify RGB, HSV and Lab summaries/histograms and circular hue statistics. | Same image/mask/binning tests; preserve colour-space and exclusion semantics. |
| F18 | Hilum, pattern and condition features | Reuse annotations and diagnostic branches; classify extra biological features as extensions. | Expert label definitions, adjudication, independent accuracy and abstention; not a claimed SMPTS classifier. |
| F19 | QC, method selection and export | Reuse review and result revisions; add comparison overlays and provenance fields. | Desktop/batch agreement, authoritative method selection and lossless serialization. |
| F20 | Batch execution and resource control | Reuse existing runner, caching, cancellation and snapshots. | Same-recipe repeatability, bounded resources, safe cancellation and reproducible restored runs. |
| F21 | Scientific and conformance evaluation | Reuse existing metrics; add reference runs and stage replay. | Metric fixtures, leakage audit, confidence intervals and failure accounting. |

The PlantCV colour-statistics, colour-correction and classifier documentation provide the reference scope for F03, F05 and F17. [R10, R11, R18]

The inventory must expand F04–F05 and F15–F17 into individual selected functions and outputs before anyone signs off “all features implemented.” Broad table rows are planning units, not sufficient evidence of complete parity.

### 6.1 A critical enhancement: do not measure eroded cores

Erosion removes foreground boundary pixels. It can help separate contacts while making the remaining object too small for direct area or dimensional measurement. [R14]

Keep at least three distinct products: the pre-erosion foreground, the eroded separation cores, and the final instance measurement masks. For an enhanced route, use the cores as markers in a constrained partition of the original foreground. A marker-controlled watershed or competitive expansion is a candidate implementation, not an assumed improvement; watershed semantics must be fixed in its own configuration. [R15]

Test whether recovery preserves isolated-seed outer contours and improves contact-case measurements without merging adjacent seeds or absorbing debris. Retain an explicit uncertainty region where the shared boundary is not directly observable. If the original foreground itself is incomplete, expansion within it cannot restore missing evidence.

For actual overlap, a complete hidden outline cannot be directly measured from a single view. Keep visible-region measurements separate from any inferred complete shape. Existing outline, pose and full-length eligibility should determine which traits are exported.

### 6.2 Geometry definitions must not be silently substituted

The inspected PlantCV size implementation uses particular pixel, contour, rectangle and ellipse operations. For example, its bounding-box width and height are not interchangeable with orientation-invariant Feret diameters. A contour-fitted ellipse is not automatically equivalent to a second-moment ellipse. [R9]

Maintain two explicitly named sets where necessary: `plantcv_compatible.*` for reference reproduction and `seedfiddle.*` for canonical research measurements. Every metric should state its formula/estimator, coordinate frame, units and valid domain. A changed estimator may be a scientific improvement, but it is not numerical parity.

Synthetic raster shapes should test implemented definitions, while independently measured specimens should test biological usefulness. Do not require the perimeter of a rasterized circle to equal the analytic circumference without accounting for discretization.

## 7. Comparative testing: four complementary levels

### Level A — Primitive and numerical conformance

Give the reference and native operation the same serialized input, exact parameters and defined output contract. Compare threshold masks, filters, morphological operations, distance maps, markers, partitions and measurements separately.

Require exact equality where semantics are deterministic and identical. For floating-point operations, declare justified absolute/relative tolerances and their valid domain. Differences caused by dependency versions, tie breaking, interpolation or approximation must be investigated and classified—not hidden by a generous global tolerance.

For watershed, test the distance transform, marker generation and flooding independently. PlantCV's inspected source specifically combines an OpenCV distance transform, local maxima, connected marker labels and watershed on the negative distance map. Replacing one of these with a different operation may change boundaries even when the broad algorithm name is unchanged. [R8]

Fixtures should include blank images, one-pixel and tiny objects, equal-distance plateaus, border contacts, thin necks, unequal seed sizes, holes, touching chains and dense clusters. Debug-overlay colours are not scientific outputs and need not be identical.

**Claim supported:** “This implementation reproduces this specified operation within its declared tolerance.”

### Level B — Discrete-stage scientific accuracy

Use independently reviewed real-image truth, but hold all other stages fixed. Examples:

| Stage being tested | Identical upstream input | Primary evidence |
|---|---|---|
| Foreground segmentation | Same decoded/corrected image and permitted references | Foreground error, missed small/dark seeds and resulting downstream error. |
| Marker generation | Same foreground and distance map | Missed/duplicate centres and common-watershed split/merge errors. |
| Watershed flooding | Same foreground, distance map and markers | Instance partitions and boundaries, with tie-policy differences documented. |
| Complete contact splitter | Same original connected components | Object precision/recall, partition quality, split/merge and eligible trait error. |
| Geometry | Same instance masks and coordinate frame | Formula agreement; independent dimensional error where a physical reference exists. |
| Colour statistics/correction | Same pixels, masks, patches and colour-space contract | Statistic agreement; independent patch/seed repeatability and colour differences. |

Using a ground-truth foreground or perfect markers can be valuable to isolate errors. Label this an **oracle-input diagnostic**; do not present it as deployable end-to-end accuracy.

**Claim supported:** “With identical upstream evidence, this stage yields equal or better independently measured outcomes.”

### Level C — End-to-end pipeline comparison

Give each complete recipe the same raw images and the same permitted information under the declared deployment protocol. Preserve each reference's native preprocessing where end-to-end reproduction requires it. Equalize development/tuning opportunities and record operator input.

Report original reference output performance, native compatible output performance and enhanced output performance separately. In addition, send each splitter's valid instance masks through one common measurement layer to isolate segmentation effects from differing geometry formulas.

If SMPTS exposes only crops or rectangles, benchmark those native outputs directly: count, successful isolation, crop truncation, contamination and extraction completeness. Compute dense-mask metrics only when actual instance masks are available, or explicitly label any derived-mask procedure. Do not invent an original mask output merely to fill a comparison table.

**Claim supported:** “This complete configuration meets the required quality and usability on the stated deployment population.”

### Level D — Computational and operational performance

Measure identical workloads on the intended hardware. Separate computation-only time from decoding, host/device transfers, model loading, serialization and review/export time. Use warm and cold runs, documented cache state, median and tail latency, peak CPU/GPU memory and failure rate.

Synchronize CUDA when measuring completed GPU work; otherwise timing can primarily measure asynchronous dispatch. PyTorch's benchmarking guidance addresses warm-up and accelerator synchronization. [R16]

Compare original CPU code and CPU ports on the same CPU when making implementation-efficiency claims. Report CUDA-native results as a distinct hardware/execution mode. A speed difference between a CPU reference and a GPU implementation is useful operational information, but is not evidence that the underlying algorithm alone is faster.

Measure operator correction time at a fixed acceptance standard. A small compute saving is not useful if it creates substantially more manual corrections.

**Claim supported:** “This configuration has the stated resource and correction costs at the reported quality.”

## 8. Reuse and extend the existing evaluator

The inspected `seedvision/learning/metrics.py` already provides object precision/recall/F1, matched IoU, panoptic quality, count error, split/merge indicators and boundary scores. It uses global matching with maximum cardinality prioritized before summed IoU. Its boundary tolerance and overlap thresholds are explicit parameters. [R6]

Extend this foundation with:

1. **Auditable matched identities.** Add an optional matched-ID record alongside aggregate results, enabling per-seed trait comparison, disputed-match review and paired error analysis without changing existing output meaning.
2. **Intermediate replay.** Load an immutable checkpoint and replace exactly one stage. Verify coordinate and content identity before execution.
3. **Reference metric adapters.** Preserve published SMPTS-style rates only where their counting definitions and annotations are recoverable. Do not rename the existing split/merge metrics as those rates.
4. **Grouped uncertainty and resource reports.** Add paired resampling, per-stratum results, latency/memory records and complete failure/coverage accounting.

Keep detection and partition quality as primary classical-method metrics. COCO-style average precision can be an additional learned-method metric when meaningful scored instances and the exact evaluation protocol are available; it should not be fabricated by assigning arbitrary confidence rankings to classical outputs.

Matched-object mean IoU or dimensional error alone can be misleading if difficult seeds are missed. Also report recall, eligibility, rejection/abstention and a joint success measure such as the fraction of eligible true seeds both detected and measured within tolerance. Define the denominator before evaluation.

Choose boundary tolerance in relation to image resolution and intended use. Retain pixel-domain results; use physical-domain tolerances only with validated calibration. Report empty-image false positives separately so numerous empty images cannot inflate a pooled quality score.

## 9. Benchmark corpus and protection against leakage

Extend the existing scientific validation protocol, rather than introducing a competing protocol. It already distinguishes image-local adaptation, frozen inference and reference-assisted inference, and calls for biological/acquisition grouping and independent measurements. [R5]

### Population and strata

Use the project's stated initial species: soybean, *Lupinus mutabilis*, *L. polyphyllus* and *L. mexicanus*. Retain operator-supplied, verified species metadata and the current one-species-per-photograph contract. [R3]

For each species, deliberately include isolated seeds, touching pairs, chains, multi-seed clusters, broad contacts, true overlap, size variation, different coat patterns and hilum appearances, damage, debris, glare, poor contrast, frame-edge objects and valid empty scenes. Sample biological lots and acquisition sessions, not merely many crops from one image.

Include repeat acquisitions across the image plane, illumination conditions and relevant resolution/geometry settings. A useful controlled subset photographs identified seeds both separated and touching. Keep every view of the same seed and related source imagery in the same evaluation group; these repeated views are not independent replicates.

### Ground truth and allowed inputs

Retain full-resolution instance masks, explicit evaluation/ignore regions, contact ambiguity, visibility, reviewer identity, adjudication and physical seed identity where available. Independently review a prespecified stratified subset; retain disagreements rather than erasing uncertainty.

Construct development, tuning/validation and final-test groups before training, feature-library extraction or parameter selection. Related biological and acquisition identities may require connected grouping to prevent indirect leakage. Synthetic shapes and composited seeds belong in engineering tests or development, not as substitutes for representative real-image final testing.

For reference-assisted deployment, define exactly which external references may influence a test image. Evaluation annotations cannot be used to tune foreground, choose marker spacing, select a splitter or choose the winning per-image method. A test mask used for review is not automatically a permissible production input.

Choose final sample size from pilot estimates of group-level variability, the non-inferiority margin and the frequency of important difficult strata. Do not assume that thousands of seeds in a handful of photographs provide thousands of independent tests.

## 10. What “at least as performant” should mean

### 10.1 Deterministic compatibility

For a defined deterministic operation, parity is an output-conformance question, not a statistical significance test. Require identical results or a justified predeclared tolerance. A failure becomes an implementation defect, a reference defect, or a deliberate enhanced-mode divergence; classify it explicitly.

### 10.2 Scientific non-inferiority

For higher-is-better quality, define the paired difference as:

`dQ = quality_native − quality_reference`

A proposed non-inferiority gate is:

`lower one-sided 95% confidence bound(dQ) >= −delta_Q`

For lower-is-better error:

`dE = error_native − error_reference`

and:

`upper one-sided 95% confidence bound(dE) <= delta_E`

The margins are laboratory decisions, declared before opening the final test set. A zero margin expresses a stricter objective than practical non-inferiority. Failure to find a statistically significant difference does **not** establish equivalence or non-inferiority.

Estimate uncertainty with paired resampling of independent biological/acquisition groups, using the same resampled groups for both methods. Prespecify primary metrics and the rule for requiring all critical gates to pass; do not choose a favourable metric after seeing results. Insufficient data for a critical stratum means inconclusive, not passed.

Require both comparison against the reference and an absolute fitness-for-purpose gate. Matching an inaccurate reference is insufficient. Do not let a pooled soybean result conceal a lupin-specific failure.

### 10.3 Proposed gate register

| Gate | What must be specified before final evaluation |
|---|---|
| Primitive compatibility | Exact outputs, valid domain, floating-point tolerance and allowed tie differences. |
| Instance quality | Primary object/partition metrics, absolute threshold and non-inferiority margin. |
| Boundary/geometry | Boundary tolerance, eligible visible-shape traits, bias and error limits. |
| Contact robustness | Residual merges, false splits and disappearance limits by critical stratum. |
| Negative and unavailable cases | False positives on empty/debris scenes; explicit failed/unsupported accounting. |
| Coverage | Minimum detection, eligible measurement and accepted-result coverage; abstention policy. |
| Colour/biological traits | Independent reference measurements and labels; no automatic release from numerical parity alone. |
| Resources/usability | Hardware, workload, latency/memory budget and correction-time standard. |

Published results obtained with different datasets, definitions and hardware are context, not pass thresholds for SeedFiddle. The relevant evidence comes from the locked comparisons above.

## 11. Enhancement experiments after compatible baselines exist

Run one-change ablations before promoting a combined enhanced recipe:

| Experiment | Question | Control |
|---|---|---|
| Foreground method × splitter | Is the gain due to cleaner foreground or better instance separation? | Factorial runs using identical permissible preprocessing and common measurements. |
| Fixed-pixel versus scale-normalized morphology | Does normalization transfer across resolution and seed-size variation? | Fixed reference parameters plus a development-selected normalized variant. |
| Reference versus alternate marker generation | Are missed centres or flooding decisions the main limitation? | Same mask and common partition stage. |
| Erosion-only versus recovered measurement masks | Does boundary recovery reduce size bias without introducing merges? | Same original foreground and erosion cores. |
| Shape/size screening versus evidence-only screening | Are small, irregular or damaged seeds being excluded by priors? | Rare/atypical strata and rejected-object audit. |
| Procedural versus existing learned methods | Does learned segmentation reduce residual failures enough to justify its costs? | Frozen checkpoints, equal target access and operator correction measurement. |
| Fixed method versus routing policy | Can selective use improve cost without sacrificing difficult cases? | Frozen development-selected policy; no test-label-driven oracle selection. |

Express adaptive parameters in a recorded scale convention. Physical units require validated calibration; otherwise use a transparent pixel- or reference-size-based convention with its provenance. Do not silently copy a kernel expressed in pixels into photographs acquired at another scale.

Do not treat all scores on a 0–1 scale as interchangeable probabilities. Any routing confidence used to abstain or select a method needs independent reliability assessment. Keep method disagreement as diagnostic evidence rather than proof that either method is correct.

## 12. Implementation work packages and release sequence

**Delivery status on adoption:** WP0–WP6 are pending. Documentation integration is
not completion of WP0: the operation-level register, pinned reference environments
and baseline replay do not yet exist. Existing native infrastructure is reused
where the audit confirms its contracts. WP0/WP1 engineering can proceed while
laboratory inputs are collected; final scientific promotion cannot.

### WP0 — Freeze scope, source and baseline

**Deliverables:** target SeedFiddle commit; current saved recipe and fixture outputs; expanded feature register; pinned PlantCV seed recipe; SMPTS source-access/ambiguity register; dependency and licence inventory.

**Exit:** the machine-readable register expands F01–F21 into named operations/outputs, maps existing implementations and tests, records gaps and exclusions, and freezes the first delivery slice. Each core feature has a source, contract and proposed test. Original SMPTS availability is explicitly resolved or remains an acknowledged block on original-code parity.

### WP1 — Build the comparison infrastructure first

**Deliverables:** immutable checkpoint interface; reference-environment runner; matched-ID audit records; stage replay; conformance reports; dataset/group manifest checks; resource-timing support; unittest fixtures for adapters and metrics.

**Exit:** the harness detects deliberate mask, coordinate, marker and metric-definition errors; a frozen current-baseline replay is reproducible. Existing application tests, catalogue checks and desktop/batch behaviour remain intact.

### WP2 — Establish PlantCV reference and compatible operations

**Deliverables:** original-function reference runs and native equivalents for selected threshold/morphology, watershed and measurement operations; documented unsupported/degenerate cases; same-mask geometry and colour tests.

**Exit:** every selected operation has a conformance result. Scientific accuracy is still a separate gate. Native optimization is not allowed to silently change compatible semantics.

### WP3 — Recover SMPTS and implement its distinct stages

**Deliverables:** original executable adapter when available; otherwise an explicitly provisional paper port; exposed threshold branches, filtering, mask combination, contact screening, erosion and crop outputs.

**Exit:** recoverable intermediate outputs are compared against the original where possible. Unresolved choices and absent dense-mask outputs are clearly reported. No reference name is substituted silently.

### WP4 — Optimize and add justified enhancements

**Deliverables:** CUDA-native implementations consistent with production residency contracts; scale-aware variants; contour recovery; controlled alternative marker/splitter experiments; optional comparisons with existing learned branches.

**Exit:** gains survive one-change ablations and development-set quality/resource comparisons. Compatibility mode remains available and stable.

### WP5 — Validate measurement, colour and result integration

**Deliverables:** explicit metric-definition namespace; eligibility-preserving outputs; comparison overlays; independent geometry/colour studies; complete provenance through desktop, batch and snapshots.

**Exit:** validated outputs are distinguished from experimental diagnostics. No scientific gate is bypassed by a successful software test or manual review checkbox.

### WP6 — Run the locked comparative study and promote defaults

**Deliverables:** frozen final manifest/configuration; per-species/contact/visibility results with paired uncertainty; conformance and speed report; correction-time study; failure gallery; release decision record.

**Exit:** required absolute and comparative gates pass for the intended deployment population. Inconclusive or failing features remain experimental, disabled or limited to their supported scope.

Do not carry forward the earlier greenfield development estimate as an estimate for this extension. Much of the application, annotation, geometry and evaluation infrastructure already exists. Estimate incremental work only after WP0 identifies genuine gaps and original-reference access.

## 13. Benchmark record and reporting contract

A benchmark record should contain the fields below. This is a **proposed schema**, not an existing supported SeedFiddle configuration or runnable command.

```yaml
schema: proposed.seedfiddle.comparison.v1
study_id: REQUIRED
status: design_only
seedfiddle_commit: REQUIRED
reference_versions:
  plantcv_recipe_hash: REQUIRED
  plantcv_environment_hash: REQUIRED
  smpts_source_hash: null
  smpts_status: original_source_not_yet_verified
protocol:
  mode: frozen_inference  # or a separately specified reference-assisted study
  source_manifest_hash: REQUIRED
  group_manifest_hash: REQUIRED
  truth_revision_hash: REQUIRED
  permitted_reference_manifest_hash: REQUIRED
  final_split_locked: false
comparisons:
  - level: stage
    stage: instance_partition
    shared_input_checkpoint_hash: REQUIRED
    reference_backend: plantcv_reference
    candidate_backend: plantcv_compatible
  - level: end_to_end
    reference_backend: seedfiddle_current
    candidate_backend: smpts_enhanced
acceptance:
  primary_quality_metrics: REQUIRED
  absolute_quality_limits: REQUIRED
  noninferiority_margins: REQUIRED
  critical_strata: REQUIRED
  uncertainty_method: paired_group_resampling
  physical_units_enabled: false
resources:
  hardware_manifest_hash: REQUIRED
  include_transfer_time: true
  cold_and_warm_runs: true
  accelerator_synchronization: true
```

A complete study report should distinguish four tables: numerical conformance, independent scientific quality, resource/correction costs, and availability/coverage. Retain per-image and per-group outputs, difference overlays and a failure gallery. Archive the exact references, environments, input manifests and result revisions in the existing snapshot/provenance system.

## 14. Definition of completion

Functional peer coverage is complete only when every core feature has an implemented, tested usable path, blocked features are resolved, and the selected feature register is fully accounted for. Original versus reconstructed versus enhanced backends remain distinguishable, the benchmark can replay individual stages, measurement definitions are explicit, and production behaviour preserves existing project contracts.

A claim that SeedFiddle is at least as accurate as a reference additionally requires the prespecified independent comparison—not merely passing unit tests or agreeing with the reference on synthetic examples. A claim of laboratory-ready physical or biological measurements additionally requires the relevant independent calibration and trait-validation gates.

**Next implementation target:** WP0 and WP1, alongside laboratory acceptance decisions and reviewed-corpus collection in the root roadmap. Building the reference adapters and replayable comparisons before tuning new algorithms provides the evidence needed to decide what to replace, what to retain and what actually improves soybean and lupin analysis.

## 15. Review performed and remaining uncertainties

The original external review checked the retrieved SeedFiddle architecture, roadmap, validation protocol and selected metric implementation against its cited external sources. The integration review checked the local documentation and the primary sources identified below. It intentionally reuses existing learned branches, geometry and evaluation infrastructure rather than proposing them as new application features.

The following distinctions were explicitly checked: compatibility versus accuracy; original SMPTS versus a paper reconstruction; eroded cores versus measurement masks; axis-aligned dimensions versus Feret measurements; image-local adaptation versus independent testing; and documented test results versus tests actually run for this response.

**External review did not perform:** a complete repository audit, original SMPTS
execution, native implementation changes, automated tests, laboratory calibration
or comparative experiments. The 3 October integration adds repository documentation
checks and the required existing test-suite run, recorded in HANDOFF.md; those are
engineering checks, not reference conformance or scientific evidence. No numerical
performance guarantee is made.

## References and source register

Repository links below are pinned to the inspected snapshot where possible. Documentation pages and external source branches can change; archive exact versions during WP0. The external review records source access on 1 October 2026. During integration on 3 October, PlantCV watershed/size documentation and the SMPTS preprint were rechecked; other external links remain source leads requiring WP0 verification. References R1–R6 preserve the pre-direction baseline, while local documents linked above govern current scope.

**[R1]** SeedFiddle. README: application, runtime, workflow and licence.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/README.md

**[R2]** SeedFiddle. Current architecture and analysis contracts.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/docs/ARCHITECTURE.md

**[R3]** SeedFiddle. Implementation and validation roadmap.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/PLAN.md

**[R4]** SeedFiddle. Current project status, documented 29 September 2026.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/docs/CURRENT_STATUS.md

**[R5]** SeedFiddle. Scientific validation and required laboratory inputs.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/docs/SCIENTIFIC_VALIDATION_PROTOCOL.md

**[R6]** SeedFiddle. Instance, boundary and count metric implementation; targeted inspection of lines 1–260.
https://github.com/efedo/SeedFiddle/blob/094aa7dd587099ac0444dfec02f556a1cbf53108/seedvision/learning/metrics.py

**[R7]** PlantCV. Watershed segmentation documentation.
https://docs.plantcv.org/en/stable/watershed/

**[R8]** PlantCV. Watershed implementation; development source inspected by the external review, not a frozen release benchmark.
https://github.com/danforthcenter/plantcv/blob/main/plantcv/plantcv/watershed.py

**[R9]** PlantCV. Size/shape documentation and implementation.
https://docs.plantcv.org/en/stable/analyze_size/
https://github.com/danforthcenter/plantcv/blob/main/plantcv/plantcv/analyze/size.py

**[R10]** PlantCV. Labelled-object colour analysis.
https://docs.plantcv.org/en/stable/analyze_color2/

**[R11]** PlantCV. Reference-image colour correction.
https://docs.plantcv.org/en/stable/transform_correct_color/

**[R12]** Lin, W., Fang, J., Su, Q., Liao, H., Liu, S., Yao, H., and Xu, P. (2024). *SMPTS: Segmentation Method for Physically Touching Soybean Images.* EasyChair Preprint 13071, version 2, 15 July 2024. The authors' linked code capsule was not obtained or executed in this review.
https://easychair.org/publications/preprint/dd5x2
https://easychair.org/publications/preprint/dd5x2/open
Author-linked capsule: https://codeocean.com/capsule/9219546/tree/v1

**[R13]** OpenCV. Image thresholding: fixed, adaptive and Otsu methods.
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

**[R14]** OpenCV. Morphological transformations.
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

**[R15]** scikit-image. Segmentation API, including marker-controlled watershed.
https://scikit-image.org/docs/stable/api/skimage.segmentation.html

**[R16]** PyTorch. Benchmarking recipe.
https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html

**[R17]** PlantCV project metadata and licence.
https://pypi.org/project/plantcv/
https://github.com/danforthcenter/plantcv

**[R18]** PlantCV. Workflow tools, including classifier training.
https://docs.plantcv.org/en/stable/tools/
