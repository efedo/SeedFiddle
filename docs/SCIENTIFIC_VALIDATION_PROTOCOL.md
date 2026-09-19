# Scientific validation and required laboratory inputs

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
