# PlantCV and SMPTS: seed-analysis peers for SeedFiddle

**Revised and adopted as project direction: 3 October 2026.** This document
adapts the external `plantcv_smpts_comparison_report.md` supplied in
`SeedFiddle_Codex_Handoff.zip`. It is research rationale, not a record of
implemented integrations or measured SeedFiddle performance. The accompanying
[implementation and benchmark plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) defines
the work; [current status](CURRENT_STATUS.md) defines what exists today.

## Decision and scope

Develop SeedFiddle toward peer functionality with the selected seed-analysis
capabilities of PlantCV and SMPTS. Keep SeedFiddle as the application, graph,
review interface and authority for results. Reuse its existing annotation,
calibration diagnostics, geometry, procedural and learned branches, optimization,
batch processing and provenance. External tools supply comparison recipes and
algorithm candidates, not a replacement framework.

Peer functionality means accountable coverage of the agreed seed-focused feature
register. It does not mean reproducing all PlantCV applications, copying its UI,
or claiming numerical equivalence between different algorithms. Track functional
coverage, numerical compatibility, scientific accuracy and resource/correction
cost separately. Implementing a capability does not validate it for laboratory use.

## Comparison with the current project

| Area | Peer contribution | SeedFiddle baseline and direction |
|---|---|---|
| Foreground preparation | Selected PlantCV threshold/classifier recipes; SMPTS two-branch mask construction. | Existing material colour, noise and reference evidence remains the baseline. Compare explicit alternatives under equal permitted reference access. |
| Touching-seed separation | PlantCV marker watershed; SMPTS targeted contact erosion and extraction. | Existing procedural watershed and optional U-Net/watershed and StarDist remain independent comparison arms. Add missing capabilities only after auditing current seams. |
| Geometry | PlantCV labelled-object size/shape definitions provide a reference contract. | Reuse reviewed geometry, including maximum-Feret span and robust body ellipse; distinguish compatible definitions from SeedFiddle estimators. |
| Colour | Selected PlantCV object colour statistics and correction operations provide reference candidates. | Neutral balance and trait diagnostics already exist; colour correction and biological accuracy need their own independent evidence. |
| Review and reproducibility | Reference outputs expose comparison checkpoints. | Retain native Qt review, authoritative result revisions, saved-recipe batch parity, source identities and portable snapshots. |
| Validation | Published studies motivate methods worth testing. | Use the existing grouped, leakage-controlled laboratory protocol; published scores are not acceptance thresholds. |

PlantCV's documented watershed accepts an RGB image and binary foreground mask,
returns labels and exposes local-maximum spacing. Its size-analysis API accepts
labelled objects and reports geometry. Pin actual versions and function semantics
before claiming compatibility. [PlantCV watershed](https://docs.plantcv.org/en/stable/watershed/),
[size analysis](https://docs.plantcv.org/en/stable/analyze_size/).

## What the SMPTS evidence supports

The SMPTS preprint describes adaptive thresholding, filtering, mask combination,
rectangle-based contact screening, erosion and seed extraction. In its experiment,
the selected erosion configuration reports RIS 99.17%, RPT 0.83% and ROS 0.39%.
These are study-specific extraction/separation measures, not SeedFiddle accuracy,
dense-mask IoU or validated dimensional error. Its soybean evidence motivates a
candidate method; transfer to lupins and SeedFiddle photographs remains untested.
[Lin et al., SMPTS, 2024](https://easychair.org/publications/preprint/dd5x2/open).

Do not transfer a fixed pixel kernel unchanged across acquisition scales. Recover
the original implementation and resolve threshold and extraction ambiguities
before claiming original-code parity. The external report notes differing timing
contexts and inconsistent defect-category reporting; these figures are excluded
from project targets. No original SMPTS execution was performed for this revision.

## Algorithm and measurement policy

Compare alternative splitters on the same unchanged input components before
choosing a production route. A fixed watershed-to-erosion-to-AI cascade is not an
accepted default. Any routing policy must be selected on development data and
evaluated without test-target access. Existing learned methods are peers in the
comparison, not restricted in advance to a residual fallback role.

Keep original foreground, eroded separation cores and final measurement masks
distinct. Test recovery of contours from original foreground; never assume an
eroded core measures the seed's full area or width. Preserve complete/partial
outline, pose and full-length eligibility. Visible silhouettes and inferred hidden
shapes remain different products. Physical units and unvalidated traits remain
withheld under the [scientific protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md).

Keep reference environments isolated from the desktop runtime. Production retains
the single launcher, native PySide6, honest graph-owned controls and node-local
CUDA caches. CPU reference runs and serialized comparison checkpoints need explicit
transfer accounting and must not impose repeated host copies on normal analysis.

## Evidence and delivery

The [implementation plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) supplies F01–F21
capability families and WP0–WP6 delivery gates. WP0 expands families into exact
selected operations and outputs with source versions, existing-code mappings,
tests and unresolved gaps. WP1 establishes reproducible comparisons before tuning.
Laboratory corpus collection and acceptance decisions proceed alongside this work.

Keep four reports: functional availability/coverage, numerical conformance,
independent scientific performance, and resource/operator costs. Require both
absolute fitness for intended use and prespecified comparative gates. Matching a
reference's weaknesses does not establish laboratory readiness.

The external report's greenfield estimate of 90–170 developer-days is not an
estimate for this extension. Estimate incremental work after the feature audit.
Unrelated leaf/time-series results and unverified AI comparison figures are not
carried forward as project evidence or as reasons to add another learned backend.

## Sources and revision boundary

- [PlantCV watershed](https://docs.plantcv.org/en/stable/watershed/) and
  [size/shape](https://docs.plantcv.org/en/stable/analyze_size/): documentation checked
  on 3 October 2026; these moving pages are not frozen benchmark versions.
- [SMPTS preprint](https://easychair.org/publications/preprint/dd5x2/open): primary
  method and overall reported rates checked on 3 October 2026. The
  [author-linked capsule](https://codeocean.com/capsule/9219546/tree/v1) has not been
  recovered, built or tested in this project review.
- The [implementation plan source register](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md#references-and-source-register)
  retains additional source leads from the external plan. WP0 must pin and inspect
  their versions, dependencies and actual licences before reuse.
- [Architecture](ARCHITECTURE.md), [reviewed shape implementation](REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md),
  [roadmap](../PLAN.md) and [validation protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md)
  govern the local integration. The original sandbox download links and claims of
  a separate complete report have been removed.
