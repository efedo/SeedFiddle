# Reference seed dimensions and shape: implementation plan

Status: implemented on 2026-09-01 for observed, pose-conditioned 2-D morphology.
The explicitly deferred intrinsic 3-D research extension remains out of scope
because a single silhouette cannot identify seed thickness.

## Implemented result

The former scale-only node is now **Reference seed dimensions and shape**. Its
implemented contract includes:

- persisted per-seed shape review, outline visibility, flat/oblique/side/
  uncertain pose, physical-seed identity, exclusion reason, hilum point, and
  optional hilum direction, with old archives migrating to unknown/unreviewed;
- exact maximum-Feret span, a robust main-body ellipse, body length and width,
  ovality, area and ellipse-relative area, non-ellipticity, asymmetry, solidity,
  exterior concavity, protrusion, neck/thin-extension diagnostics, localized
  contour-feature geometry, support, and coherent-boundary measurement
  uncertainty;
- separate calibrated physical and dimensionless measurements so uncalibrated
  observations can improve shape without ever inventing millimetres;
- repeated-view grouping by physical-seed ID, pose-conditioned robust joint
  components, regularized smooth contour modes, species → lineage → accession
  → lot partial pooling, broader-level fallback, predictive compatibility, and
  explicit out-of-family abstention;
- diagnostic `SpeciesShapeSummary` and inference-capable
  `SpeciesDimensionsShapeBank` products published through the one immutable
  species-library path with source exclusion and grouped validation;
- overlays for reviewed geometry, maximum spans, size–ovality distribution,
  pose families, mean contour/mode atlas, and measurement uncertainty;
- a legacy scalar-diameter compatibility port plus explicit opt-in use of the
  richer prior by oval candidates, Shape fill, and procedural candidate
  plausibility; and
- cache identities and invalidation scoped to shape metadata and downstream
  geometry consumers, without feeding automatic instances back into their own
  prior.

The statistical prior is deliberately soft. Existing hard width, area,
concavity, protrusion, solidity, axis-ratio, and topology protections remain in
force. Complete reviewed masks remain authoritative user annotations, while
parameter fitting withholds them from procedural inference and uses them only as
scoring targets.

This plan expands the existing **Seed scale estimate** node into **Reference
seed dimensions & shape**. It is intended to complement the immutable,
source-balanced library design in
[`SPECIES_REFERENCE_LIBRARIES.md`](SPECIES_REFERENCE_LIBRARIES.md), not create a
second library system.

## Shared library contract

This plan uses the species-library framework exactly as follows:

- `SPECIES_REFERENCE_LIBRARIES.md` owns immutable manifests, content hashes,
  catalogues, import/export, project pins, source exclusion, publication, and
  the `Species reference library` graph node.
- This plan owns shape-review metadata, measurements, uncertainty, pose,
  hierarchical shape statistics, and geometry consumers.
- `SpeciesShapeSummary` is the basic diagnostic product delivered by the
  measurement milestone. `SpeciesDimensionsShapeBank` is the joint,
  uncertainty-aware inference product. Neither type may masquerade as the other.
- Shape uses three modes: `Current image only`, `Species library only`, and
  `Species-library prior + current reviewed observations`. The last mode is a
  hierarchical update, not a union of samples or probability rasters, and the
  generic library `Current-image reference weight` does not apply.
- In combined mode the pinned bank excludes the current source before local
  reviewed observations are added exactly once. The UI marks that result
  in-sample, and validation never reports it as held-out performance.
- `capture_group_id` describes the imaging context. `seed_lot_id` describes the
  biological/material source. The fields are never interchangeable.
- One species-library version may contain several lineage/accession/lot shape
  components. A project still pins one exact immutable version for the species,
  and runtime selection falls back within that version from lot to accession to
  lineage to species when a compatible narrower component is unavailable.

## Outcome

Seed Fiddle will estimate a joint, uncertainty-aware distribution of seed
dimensions and shape rather than one preferred diameter. The model will:

- distinguish maximum silhouette span from the dimensions of the main oval body;
- estimate ovality and controlled departures from ovality;
- represent a localized hilum-associated nub or indentation when the evidence
  supports one;
- distinguish flat, oblique, side-lying, and uncertain views;
- separate measurement uncertainty, biological variation, and uncertainty in
  the estimated population;
- partially pool information across species, optional lineage groups,
  accessions, and seed lots; and
- expose an explicit compatibility adapter for existing consumers that still
  require one scalar seed diameter.

The first implementation will model observed two-dimensional silhouettes. It
will not claim to recover thickness or a unique three-dimensional seed from one
photograph. A later 3-D extension requires paired views or physical thickness
measurements.

## Existing behaviour and migration constraints

The current scale estimator measures exact maximum-Feret spans from annotated
instance masks. It marks a mask complete when it is connected and does not touch
an image edge, then selects the largest configured fraction, currently the top
25%, when at least two eligible masks exist. That is a useful conservative
processing-scale heuristic, but it is deliberately biased upward and cannot be
treated as a population sample.

The current completeness check also cannot determine whether a neighbouring
seed obscures part of the outline. Therefore:

1. the current top-fraction estimate remains available as **Legacy processing
   diameter** during migration;
2. it must not supply a learned population distribution;
3. pre-existing masks migrate with unknown visibility and pose rather than being
   silently declared complete and flat; and
4. downstream consumers migrate individually because one scalar diameter
   currently sets many semantically different kernel, radius, and normalization
   scales.

The existing ellipse fitter, curved-edge oval proposals, Shape fill prior, and
procedural shape diagnostics are useful consumers and sources of candidate
geometry. Their automatic results do not become reference observations unless
a user explicitly reviews and promotes them.

## Measurement contract

Every reviewed seed observation will retain the actual visible mask plus a
compact measurement record. Pixel and calibrated physical measurements are
stored separately; missing or unreliable ruler calibration never fabricates
millimetres.

| Measurement | Definition | Purpose |
| --- | --- | --- |
| Maximum silhouette span | Maximum Feret diameter of the visible outline, including a nub | Preserves the existing maximum-width meaning |
| Body length | Full major-axis diameter of a robust main-body ellipse | Main size and orientation |
| Body width | Full minor-axis diameter of the same ellipse | Transverse size and ovality |
| Ovality | Body length divided by body width | Separates round from elongated bodies |
| Projected area | Area of the actual visible mask | Measurement and candidate plausibility |
| Ellipse-relative area | Actual area divided by fitted-ellipse area | Bulk departure from an ellipse |
| Non-ellipticity | Robust contour residual after scale, translation, and rotation alignment | Separates a clean oval from an irregular seed |
| Asymmetry | Difference between corresponding halves of the aligned contour | Detects one-sided broadening or flattening |
| Local feature geometry | Angular position, arc width, signed height/depth, and support of localized protrusions or indentations | Hilum-associated or anonymous local features |
| Existing topology measures | Solidity, exterior-connected concavity, protrusion fraction, neck width, and thin-extension length | Prevents permissive shape models from legitimizing merged or leaking masks |

Maximum span and body length are intentionally different. A small broad nub can
increase the former while robust ellipse fitting preserves the main body. The
true mask, rather than the ellipse, remains authoritative for visible projected
area.

Near-round seeds have weakly identified orientation. Their records must retain
orientation uncertainty rather than assigning biological meaning to an
arbitrary major axis. Axial orientation has a 180-degree ambiguity unless an
optional landmark resolves it.

## Reference annotation additions

Add the following optional fields to each reference seed instance:

- **Shape reference reviewed:** explicit permission to use the observation for
  shape modelling, separate from coat-pattern and condition review.
- **Outline visibility:** complete, partly occluded, image-cutoff, or uncertain.
- **Pose:** flat, oblique, side, or uncertain.
- **Hilum landmark:** optional point and, when useful, an outward direction.
- **Physical seed ID:** optional identifier associating multiple photographs of
  the same physical seed.
- **Shape exclusion reason:** damage, split, severe wrinkle, segmentation
  uncertainty, or another recorded reason. Exclusion is auditable and never
  silently deletes the annotation.

Condition labels such as Split or Wrinkled remain semantic observations; they
do not automatically exclude a seed. The reviewer decides whether the observed
outline is suitable for the particular shape product.

Complete reviewed masks are the first population-fitting source. Partial masks
can still train existing material and edge evidence. A later censored-outline
model may use visible arcs without interpreting the missing outline as a small
whole seed.

## Three separate uncertainties

The application must not collapse all variability into one error bar.

1. **Per-seed measurement uncertainty** represents boundary ambiguity,
   resolution, occlusion, ellipse-fit stability, and calibration sensitivity for
   one observation.
2. **Population variation** is the genuine predictive range expected among
   different seeds from the selected biological group and pose.
3. **Population-parameter uncertainty** expresses how well the available
   observations establish the population mean, covariance, and shape modes.

The node will initially display 95% intervals with their type written out. A
narrow interval for the mean must not be presented as a narrow expected range
for the next seed.

Boundary uncertainty will use spatially coherent contour perturbations or
alternative supported boundaries, not independent random noise at every pixel.
Ruler and homography error is shared by every seed in an image and must be a
shared image-level uncertainty term. Measuring many seeds cannot average away
one incorrect ruler scale.

Until held-out coverage is measured, combined edge/shape compatibility remains
a score rather than a calibrated posterior probability. Correlated area,
ovality, and contour-residual terms must not be multiplied as though they were
independent observations.

## Shape representation

Use an interpretable, bounded representation:

```text
robust body ellipse
    + a few smooth population deformation modes
    + an optional localized contour feature
    + explicit residual/out-of-family evidence
```

### Robust body ellipse

Fit centre, major and minor axes, and in-plane orientation while down-weighting
a bounded local feature. The main ovality therefore cannot be inflated by a
small nub. The fit must still fail or report low support when too little of the
body outline is visible.

### Smooth contour modes

Align complete reviewed contours by centre, scale, and rotation, resample them
at consistent perimeter locations, and learn only a small number of smooth
deformation modes. These can represent a broader end, a mildly flattened side,
rounded triangularity, or general asymmetry.

A low-order periodic basis or landmark point-distribution model is appropriate.
The retained dimensionality is selected by held-out reconstruction and
plausibility, not by training fit alone. Scale and ovality are removed from the
residual basis so that the same biological variation is not represented twice.

The first version must be strongly regularized. An unconstrained high-frequency
outline model could explain merged seeds, narrow leaks, and jagged annotation
noise instead of learning seed shape.

### Local hilum-associated feature

Represent one optional local feature by:

- signed height or depth relative to body length;
- angular centre around the fitted body;
- perimeter arc width;
- asymmetry or local shape coefficients;
- presence/support confidence; and
- whether a reviewer supplied a hilum landmark.

Without a visible or annotated hilum, call it a **local contour feature**, not a
biologically identified hilum. Do not align every contour to an automatically
guessed nub: that would create false phase alignment. Preserve the 180-degree
and mirror alternatives where necessary.

A supported broad nub may receive a soft population prior, but it cannot exempt
the rest of the outline from thin-extension, neck-width, concavity, or topology
constraints.

### Out-of-family handling

The model must be allowed to abstain. A split seed, a major occlusion, a merged
candidate, or a shape not represented by the library should receive explicit
out-of-family evidence and competing hypotheses rather than being forced into
the nearest oval. Visible and inferred/amodal outlines remain separate products.

## Pose model

### First release: pose-conditioned 2-D families

Learn related silhouette distributions for flat, oblique, side, and uncertain
views. Inference evaluates several plausible pose hypotheses and retains their
relative support. A narrow outline alone is not authoritative evidence of a
side view because it could instead be a narrow accession, occlusion, damage, or
segmentation error.

Colour pattern, hilum visibility, and shading may provide weak auxiliary pose
evidence. They remain suggestive and may not override a contradictory physical
boundary.

The pose frequencies in an ordinary dish are estimated only from representative
dish images. A deliberately balanced pose-training collection must not imply
that side views are common in normal analysis images.

### Deferred extension: intrinsic 3-D body

A later model may represent intrinsic length, breadth, and thickness plus a
localized hilum deformation, with a projection model for observed pose. This
requires paired flat/side views of identified physical seeds, preferably with
some intermediate orientations and occasional thickness measurements.

One silhouette is insufficient to identify a unique hidden three-dimensional
shape. Until paired data supports the extension, the application reports only
observed projected dimensions and pose-conditioned alternatives.

## Hierarchical transfer model

Use partial pooling through this biological structure:

```text
species -> optional lineage/group -> accession -> seed lot
```

Capture session, image, ruler calibration, and annotation revision are separate
observation effects, not biological descendants. Accession is not assumed to
equal genetic lineage; lineage/group is explicit optional project metadata.

The joint model includes transformed positive dimensions, ovality, selected
smooth contour coefficients, and supported local-feature parameters. It learns
their covariance rather than fitting independent marginals.

- A new accession begins with a broad species or lineage predictive
  distribution.
- Local reviewed references update the accession while retaining appropriate
  uncertainty.
- Well-supported accession differences are allowed to dominate rather than
  being permanently shrunk to the species mean.
- Absolute dimensions may adapt faster than dimensionless shape if held-out
  evidence supports that transfer policy.
- Missing lineage or lot metadata skips that level; ancestry is never inferred
  from appearance.
- Repeated views of one physical seed do not increase the effective biological
  sample count.

Robust tails protect the fit from unusual genuine seeds. More than one
population component is added only when independent sources and held-out checks
support it; a mixture must not be introduced merely to fit annotation errors.

## Species-library integration

Publish two explicitly distinct products through the species-library framework.
`SpeciesShapeSummary` contains source-aware audited measurements for diagnostics
only. `SpeciesDimensionsShapeBank` is the versioned shape prior containing:

- per-source compact reviewed measurements and uncertainty summaries;
- biological hierarchy and capture-group identifiers;
- physical-seed identifiers for repeated-view grouping;
- pose and visibility metadata;
- fitted joint population parameters and shape modes;
- measurement/descriptor schema hashes;
- validation tier, effective sample counts, and source balance; and
- complete source-image and annotation provenance.

Published versions are immutable and projects pin an exact content hash.
Applying or saving an annotation never publishes to a library. Promotion is an
explicit reviewed operation. Library-only evaluation excludes contributions
from the current image, and grouped validation excludes the entire physical seed
and image rather than random contour points.

Pixel-only references can contribute dimensionless shape but cannot establish
physical size. A library with insufficient support returns a broad provisional
prior or marks the relevant product unavailable; it does not invent precision.
Uncalibrated pixel length/area remains source-local provenance and is never
pooled across images as a physical population distribution.

The basic summary becomes available only after Milestone 1 defines explicit
shape review and visibility. The dimensions/shape bank requires Milestones 1--2
and their uncertainty gates. Other species-library products are not blocked by
shape readiness.

## Pipeline-node contract

**Seed scale estimate** is renamed to **Reference seed dimensions and shape**.
The node receives the existing corrected image, vessel
geometry, absolute scale, and project annotation inputs plus the pinned
`SpeciesDimensionsShapeBank`. A `SpeciesShapeSummary` can
populate diagnostics but cannot provide the procedural prior.

Implemented compact outputs:

- `SeedDimensionsShapeModel`
- `SeedMeasurementSummary`
- `SeedPoseShapeFamilies`
- `SeedShapeProvenance`
- `SeedDiameter` compatibility output

The compatibility output initially preserves the current processing diameter
exactly. Every consumer is then audited and migrated to the semantically correct
quantity:

- edge and texture support windows may need body width or an equivalent-area
  diameter;
- centre separation may need a pose-conditioned body span;
- candidate maximum-width checks need the predictive maximum-span distribution;
- area limits need the joint projected-area distribution; and
- learned-model normalization retains its existing scalar contract until any
  changed normalization is independently trained and validated.

There must be no feedback loop from a graph's automatic oval or procedural
instances into that graph's reference prior. Only explicitly reviewed and saved
annotations can affect a later local fit or published library version.

## Downstream inference

### Curved-edge ovals and centre probability

Use existing reverse edge voting to propose centres and orientations. Score a
bounded CUDA bank of plausible ellipse dimensions and pose families, then apply
richer contour deformation only to the best proposals. Avoid a full-resolution
tensor for every shape, pose, and centre combination.

### Shape fill

Use the selected body and pose distribution as a soft outer prior. Actual edge
and colour evidence remains authoritative, and the existing inward freedom for
divots or occlusion is retained. A dashed inferred completion may be shown, but
it is never inserted into the visible annotation mask without an explicit edit.

### Procedural instances

Score candidates with separate, inspectable terms for boundary evidence,
material evidence, population shape compatibility, pose compatibility, and
topology. Statistical priors are soft. Explicit user hard limits remain hard,
and existing concavity, solidity, protrusion, and thin-extension safeguards stay
active until a validated replacement improves held-out results.

Multiple candidate hypotheses are still combined through the non-overlap
selection stage. An annotation marker may supply a trusted centre but must not
make the automatically grown candidate immune to shape plausibility checks.

## User interface and overlays

Organize inspector controls under:

1. **Reference sources**
2. **Measurement eligibility**
3. **Dimensions and uncertainty**
4. **Pose families**
5. **Contour variation and local feature**
6. **Library transfer**
7. **Downstream prior use**

Every exposed control must affect its documented calculation. Controls for an
unimplemented pose or 3-D path remain absent rather than disabled placeholders.

Proposed diagnostics:

- annotated outlines with robust body ellipses, body axes, maximum-Feret chords,
  optional hilum landmarks, and visibility/pose labels;
- a joint size-ovality density/scatter view with marginal distributions;
- separate flat, oblique, side, and uncertain-view distributions;
- a mean-shape atlas with representative learned deformation modes;
- separate measurement, next-seed predictive, and population-mean intervals;
- source/accession/lot balance and effective sample counts;
- per-instance measurements, boundary sensitivity, alternative poses, local
  feature support, and out-of-family reason; and
- solid observed versus dashed inferred outline conventions.

The node overlay, overlay-menu heading, pipeline-card title, and right-inspector
title must use exactly the same owner name. Each selectable overlay needs an
identically labelled output connector, consistent with the existing UI contract.

## Cache and cancellation contract

The direct node cache identity includes:

- local annotation content and review metadata hashes;
- source image and calibrated-coordinate identity;
- pinned library content hash and self-source exclusion hash;
- accession, optional lineage, lot, and capture-group identity;
- measurement and contour-schema versions; and
- every enabled model parameter.

Changing shape review or pose metadata invalidates the shape node and geometry
consumers, not unrelated colour/noise evidence when the masks did not change.
Changing a library pin invalidates only the library node, direct consumers, and
descendants. Compact model data is uploaded once per content hash/device/dtype;
full-resolution reusable rasters remain on CUDA. Image switching cooperatively
aborts extraction, inference, and display work, and a stale revision can never
install its result into the new image.

## Delivery sequence and gates

### Milestone 1: measurements and annotations

- Add visibility, pose, shape-review, physical-seed ID, exclusion reason, and
  optional hilum landmark persistence.
- Implement robust ellipse, maximum-span, area, asymmetry, residual, topology,
  and measurement-sensitivity calculations.
- Display diagnostics only; preserve all current downstream calculations.

Gate: known synthetic shapes measure correctly across rotation, translation,
resolution, and image boundaries; old archives migrate without invented review
state.

### Milestone 2: joint distributions and uncertainty

- Fit the compact joint dimension/ovality model from representative complete
  reviewed instances.
- Separate all three uncertainty types and shared image calibration error.
- Retain the legacy processing-diameter adapter.

Gate: duplicated masks/images cannot increase effective weight; held-out
interval coverage and source balance are reported; the upper-quartile heuristic
cannot enter the population fit.

### Milestone 3: hierarchical library transfer

- Build on species-library Milestones 1--2 rather than adding another catalogue,
  bundle, pin, or publication path.
- Add lineage/accession/lot and physical-seed metadata through the shared
  project/library contracts.
- Publish immutable source-aware shape banks with exact project pins.
- Implement image/seed/capture-group holdouts and self-source exclusion.

Gate: rebuilding in another input order is deterministic; a new accession
receives an appropriately broad prior; real supported differences survive
partial pooling.

### Milestone 4: ellipse-aware downstream inference

- Add opt-in pose-conditioned dimensions to edge-oval confirmation and centre
  voting.
- Add opt-in priors to Shape fill and procedural candidate scoring.
- Audit and migrate scalar-diameter consumers individually.

Gate: held-out boundary/count performance improves without increasing merged,
  leaking, overconcave, or invalid instances; disabled mode is regression-
  equivalent to the current path.

### Milestone 5: smooth contour and local-feature modes

- Learn a small regularized contour basis from complete reviewed shapes.
- Add landmark-aware or phase-ambiguous local-feature modelling.
- Preserve explicit residual and abstention paths.

Gate: held-out contours improve over ellipses, while synthetic leaks and merged
  candidates remain rejected. Adding modes must not merely reduce training loss.

### Milestone 6: pose-conditioned families

- Learn flat, oblique, side, and uncertain 2-D silhouette families.
- Evaluate multiple pose alternatives and calibrate confidence/abstention.
- Keep staged-pose collection frequencies separate from ordinary-dish priors.

Gate: pose-specific held-out measurements improve and genuinely narrow flat
seeds are not systematically mislabeled as side views.

### Deferred milestone: 3-D intrinsic model

Proceed only if paired same-seed views and physical measurements make intrinsic
dimensions identifiable and materially useful. This milestone needs a separate
approved design and validation plan.

## Required validation

### Geometry and invariance

- Synthetic circles, ellipses, asymmetric bodies, nubs, indentations, partial
  outlines, occlusions, narrow leaks, and merged pairs.
- Rotation, translation, reflection/axial ambiguity, scale, source resolution,
  and calibrated-unit consistency.
- Near-round orientation uncertainty and optional-landmark alignment.

### Statistical and transfer validation

- Held-out images, accessions, capture groups, and physical seeds.
- Separate results by pose, visibility, size range, condition, and source.
- Coverage checks for measurement and predictive intervals.
- Posterior/predictive checks for covariance and out-of-family rate.
- Duplication invariance at pixel, seed, repeated-view, and image levels.
- Representative-sampling and deliberately enriched-pose audits.

### Anti-cheating

- No held-out mask, centre, contour, calibration statistic, or library
  contribution can enter its own evaluation.
- Identical geometry at different coordinates receives identical model scores.
- Reviewed pixels use the same equations as every other pixel.
- Automatic candidate outputs cannot update the current graph's prior.
- Local annotations affect a library only through explicit publish/rebuild.

### Pipeline and performance

- Typed graph dependencies, overlay/connector ownership, settings migration,
  cache signatures, descendant-only invalidation, and image-switch cancellation.
- Full-resolution tensor/device tests and bounded compact CPU metadata.
- CUDA memory/time checks for the ellipse bank and top-candidate contour stage.
- Existing material, edge, learned-segmentation, and scalar-scale regressions.

## Data-collection guidance

For an initial pilot, aim for roughly 20--30 representative complete flat seeds
per accession across at least several images, plus a smaller deliberately paired
flat/side collection. Across a species, several accessions are necessary to
estimate transferable versus accession-specific variation. These are collection
targets, not claims of statistical sufficiency.

Record image/accession/lot/capture metadata before adding model complexity.
Where feasible, photograph the same identified seed flat and on its side, add an
occasional oblique view, and annotate the hilum only when it is actually
recognizable. Do not use repeated views as independent population seeds.

## Resolved implementation decisions

1. The current schema uses one biological context per image, with image-level
   overrides of project lineage/accession/lot defaults. Mixed-context images are
   intentionally not inferred from appearance and should be split into separate
   source records until a reviewed per-seed workflow is requested.
2. Paired flat/side observations are optional. When available, the persisted
   physical-seed ID groups them so they contribute one biological seed rather
   than two independent samples; ordinary unpaired observations remain valid.
3. The delivered model is observed 2-D morphology. Intrinsic thickness/3-D
   recovery remains the explicit deferred milestone and requires paired views
   or physical measurements plus a separately approved design.

## Definition of done

The redesign is complete when reviewed observations produce auditable joint
dimension/shape distributions with honest uncertainty; sparse accessions borrow
appropriate information without erasing supported differences; pose and partial
visibility are not confused with biological size; no automatic result leaks into
its own prior; consumers use the correct shape quantity rather than one implicit
diameter; and held-out, cache, cancellation, GPU, persistence, migration, and UI
contracts pass the full test suite.

## Supporting references

- Tanabata et al., *SmartGrain: High-Throughput Phenotyping Software for
  Measuring Seed Shape through Image Analysis*, Plant Physiology 160 (2012),
  <https://doi.org/10.1104/pp.112.205120>.
- Cootes et al., *Active Shape Models—Their Training and Application*, Computer
  Vision and Image Understanding 61 (1995),
  <https://doi.org/10.1006/cviu.1995.1004>.
- Iwata and Ukai, *SHAPE: A Computer Program Package for Quantitative Evaluation
  of Biological Shapes Based on Elliptic Fourier Descriptors*, Journal of
  Heredity 93 (2002), <https://doi.org/10.1093/jhered/93.5.384>.
- Carpenter, *Hierarchical Partial Pooling for Repeated Binary Trials*, Stan case
  study, <https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html>.
- Pinheiro et al., *Domain-Adaptive Single-View 3D Reconstruction*, ICCV 2019,
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Pinheiro_Domain-Adaptive_Single-View_3D_Reconstruction_ICCV_2019_paper.html>.
