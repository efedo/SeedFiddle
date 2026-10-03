# Scientific validation and required laboratory inputs

**Protocol extended 3 October 2026.** The existing engineering workflow is implemented;
none of the software/UI completion summarized in
[Current project status](CURRENT_STATUS.md) closes the empirical requirements in
this protocol.

The current application supports image-local adaptation and pixel-space review.
It does not establish counting accuracy, physical metric accuracy or trait validity.
Neutral white/gray balancing is not a colorimetric calibration. Heuristic evidence,
prototype compatibility, model confidence, disagreement and shape-fit scores are
uncalibrated; their common 0–1 range does not give them a common interpretation.

## Locked benchmark

Record immutable source hashes, original/corrected coordinate transforms, complete
evaluation-region masks, annotation authors and revisions, species, biological lot,
physical seed identity where available, acquisition session, density, contact and
occlusion, pattern, glare and size strata. Include empty dishes, debris, broken,
small and atypical seeds, and rejected candidates. Assign related biological AND
capture groups to a single split before any tuning. Do not infer groups from filenames.
Hash and freeze the final test manifest, all components, preprocessing recipe,
checkpoint, decoder and library membership. A second expert independently reviews
a stratified subset; retain both originals and adjudication with reasons.

Declare deployment before acquiring final results:

- **Image-local adaptation:** regions from the analyzed image can influence colour,
  material, boundaries, scale or shape. Scores against those regions are tuning
  diagnostics. GUI learning exports explicitly record this mode.
- **Frozen inference:** held-out targets cannot affect any upstream calculation,
  feature scaling, preprocessing, library extraction or parameter choice.
- **Reference-assisted inference:** record separate deployment references and their
  hashes, their sampling protocol and allowed influence. Keep evaluation targets
  inaccessible to all adaptation. Holding out only a decoder is insufficient.

The evaluation CLI defaults to development diagnostics. Independent protocols reject
unknown or image-local preprocessing provenance. Their metadata audit is a guardrail,
not proof that a human-authored provenance declaration is true. No code path promotes
a reviewed split to automatic scientific validation.

Report global object matching, detection precision/recall, segmentation IoU/PQ,
boundary quality, absolute count error, empty-dish false positives, micro totals and
per-group/per-stratum results. Bootstrap biological/capture groups, not pixels or
repeated views of the same seed. Declare sample-size and acceptable-error thresholds
before opening the final split. Repeated optimization belongs to validation data.

## PlantCV–SMPTS comparative extension

The accepted [implementation plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) applies
this same protocol; it does not create a second route to scientific approval.
Before final testing, freeze selected reference versions/recipes, the functional
feature register, allowed inputs, primary endpoints, critical strata, absolute
fitness thresholds and comparative margins. Pending original SMPTS access blocks
original-code parity, not an explicitly labelled paper-port experiment.

Report numerical conformance, stage-isolated quality, end-to-end quality and
resource/operator costs separately. Equalize permitted information and development
opportunity; retain native preprocessing for complete-reference runs and common
upstream checkpoints for isolated-stage runs. Never select a per-image winning
method from evaluation targets. Label oracle-mask/marker diagnostics separately.

Use paired resampling of independent biological/acquisition groups for comparative
uncertainty. For higher-is-better quality, require the lower one-sided 95% confidence
bound of candidate-minus-reference to meet the prespecified negative margin; for
lower-is-better error, require its upper bound to meet the positive margin. Also
require absolute fitness and coverage gates; no significant difference is not proof
of non-inferiority. Inadequately sampled critical strata remain inconclusive.

Report detection, misses, exclusions, abstention and eligible measurement coverage
alongside matched-seed errors. Reference extraction-only outputs require crop/count
metrics; do not invent dense masks or arbitrary confidence rankings for AP. Preserve
published metric definitions only when recoverable. Physical/colour/trait outputs
still require the independent evidence below; conformance never enables them alone.

For operational comparisons, record hardware, warm/cold and cache state, synchronized
GPU completion, transfers, peak memory, failures and correction time at a fixed
quality standard. CPU-versus-GPU timings describe execution modes, not algorithm-only
speed. Keep the final test set inaccessible during routing and parameter selection.

## Acquisition and physical measurement

Provide measured card dimensions/aspect ratio or validated camera intrinsics and
distortion coefficients. Record camera/lens, distance, pose, focus, exposure, lighting,
reference plane, ruler/card coplanarity and seed height. Check independent horizontal
and vertical lengths at the center and at least two peripheral positions, with
repeat captures. Quantify lens, perspective, scale, pose and elevation contributions.
Compare seed lengths/areas to a traceable independent measurement or reference image.
The `assess_metric_geometry` helper accepts laboratory-selected error bounds; it
does not fabricate an acceptable threshold or automatically enable millimetres.
Exported visible 2-D pixel measurements remain separate from inferred full shapes.

For colour, record illumination spectrum/geometry, white balance, exposure, RAW/JPEG
processing and patch reference values. Use held-out patches and colour differences
before claiming colorimetric calibration. Do not interpret neutral gain correction
as validated colour-class accuracy across capture sessions.

## Traits, scores and operator study

An expert must define operational labels with positive, negative and ambiguous
examples for coat pattern, damage, wrinkling and condition. Review per-seed labels,
visibility and exclusions separately from drawing progress. Measure inter-annotator
agreement, abstention coverage and error by biological/acquisition stratum. The
results table abstains from unvalidated trait assignments. `score_reliability`
produces bins/Brier/ECE for independent binary outcomes; calibration and selection
must occur outside the final test set. `stratified_errors` retains rare and rejected
cases for prior-bias review.

Recruit representative operators to perform acquisition → references → analysis →
review → export at laptop and desktop sizes and 125%, 150%, 200% scaling. Measure
correction time and errors, keyboard-only completion, screen-reader names/order,
colour-vision accessibility and recovery from interrupted saves/cancellation.

## Decisions/data needed from the laboratory

1. Intended decisions and prespecified tolerances for counts, dimensions and traits.
2. Reviewed independent source images/labels and biological/capture grouping.
3. Card/camera geometry and independent dimensional/colour reference measurements.
4. Expert trait definitions, adjudicators and operator-study participants.

The missing historical fixture has been recovered byte-for-byte from Git; no
replacement photograph or weakened fixture assertion is required.
