# Learned instance-segmentation engineering report

Date: 2026-08-11
Branch: `codex/learned-instance-segmentation`

## Verdict

Both requested native-PyTorch methods are implemented, tested, checkpointed,
and integrated as disabled-by-default DAG nodes. The multi-head U-Net includes
independent seed-interior, physical-boundary, apparent coat-pattern-boundary,
centre/distance, and auxiliary error-probability outputs. StarDist predicts
object probability, 32 radial distances, and auxiliary radial-error
probability.

The software implementation works and both models solve the controlled
simulator task. Neither model is ready for scientific use on the real fixture
domain. The repository has eleven real photographs but no complete
human-reviewed instance masks, so real precision, recall, count error, and
measurement error cannot yet be estimated. Visual inspection demonstrates a
large simulator-to-photograph domain gap. Additional simulator-only tuning is
therefore not a scientifically defensible substitute for annotation.

## Reproducible controlled dataset

The harder deterministic simulator corpus contains 72 complete images and
1,474 labelled patterned instances:

- 48 image groups for training;
- 12 image groups for validation and decoder selection; and
- 12 untouched image groups for the final engineering test.

It varies density, contact, ellipse geometry, four pattern families, pale and
dark coats, broad illumination, hard shadow, glare-like blobs, background
colour, and sensor noise. It is explicitly marked `provenance: synthetic` and
`scientific_validation_eligible: false`.

The selected models use a 24-channel-base, depth-three residual U-Net backbone,
32 StarDist rays, canonical 40-pixel seed diameter, deterministic augmentation,
mixed-precision CUDA training, and early stopping. Validation alone selected
decoder settings; the test split was opened with settings frozen.

## Quantitative engineering results

| Method and split | Instance F1 | PQ | matched IoU | relative count error | boundary F1 | mean model inference |
|---|---:|---:|---:|---:|---:|---:|
| U-Net, validation | 1.000 | 0.811 | 0.811 | 0.000 | 0.966 | 0.036 s |
| U-Net, locked test | 1.000 | 0.820 | 0.820 | 0.000 | 0.987 | 0.038 s |
| StarDist, validation | 0.926 | 0.759 | 0.816 | 0.162 | 0.743 | 0.036 s |
| StarDist, locked test | 0.996 | 0.825 | 0.829 | 0.009 | 0.776 | 0.037 s |
| U-Net-gated StarDist hybrid, validation | 0.998 | 0.809 | 0.810 | 0.005 | 0.966 | 0.051 s |

The StarDist validation/test variance is itself a warning that this synthetic
corpus is too small and artificial for model selection claims. The hybrid did
not beat the U-Net baseline and remains experimental, outside the active DAG.

On the locked synthetic test, the U-Net dense heads achieved:

| Dense head | F1 at 0.5 | ROC AUC | average precision | Brier score |
|---|---:|---:|---:|---:|
| Physical boundary | 0.989 | 0.9999 | 0.9990 | 0.0011 |
| Apparent coat-pattern boundary | 0.839 | 0.980 | 0.941 | 0.0427 |

A short pattern-weight fine-tune improved validation pattern F1 from 0.855 to
0.867 and AP from 0.939 to 0.944, but degraded validation instance F1 from
1.000 to 0.988. The original checkpoint was retained because the tradeoff did
not improve the primary topology objective.

## Visual analysis of controlled outputs

The U-Net contact sheets show every pale, dark, striped, spotted, mottled, and
bicoloured test seed assigned exactly once. Internal coat transitions are not
used as watershed cuts. The remaining error is contour calibration: predicted
edges are often one or two pixels outside the sharp simulator label, which is a
large area fraction for a roughly 40-pixel object and limits PQ. Validation
search over interior thresholds and zero-to-two-pixel erosion retained zero
erosion, indicating that the discrepancy is not a uniform offset.

Corrected StarDist polygons no longer show alternating radial spikes. On most
test images their shape prior gives compact, visually plausible contours and
ignores internal patterns. Its failure mode is false star-convex objects on
seed-shaped glare/background structures; two of twelve test images contain one
extra object each, while several validation images contain larger false-object
clusters. U-Net interior/distance gating removes most of these, but remaining
duplicate markers slightly degrade an otherwise perfect U-Net partition.

## Qualitative review on the eleven real fixtures

The review command used the neutral `unknown` species plane because reliable
per-image species metadata is not stored. Counts are predictions, not truth:

| Fixture | U-Net | StarDist | Existing procedural diagnostic |
|---|---:|---:|---:|
| `IMG_0002c.JPG` | 65 | 1,171 | 621 |
| `IMG_9632c.JPG` | 16 | 235 | 130 |
| `IMG_9636c.JPG` | 63 | 191 | 92 |
| `IMG_9641c.JPG` | 43 | 139 | 90 |
| `IMG_9666c.JPG` | 42 | 209 | 129 |
| `IMG_9667c.JPG` | 51 | 203 | 138 |
| `IMG_9668c.JPG` | 40 | 150 | 84 |
| `IMG_9670c.JPG` | 15 | 56 | 18 (16 visibly present) |
| `IMG_9685c.JPG` | 19 | 214 | 116 |
| `IMG_9689c.JPG` | 12 | 103 | 58 |
| `IMG_9974c.JPG` | 141 | 855 | 563 |

Detailed examination finds:

- The U-Net generally rejects the dish rim and background and rarely splits a
  seed at a strong bicolour transition. It nevertheless misses most seeds in
  dense soybean dishes, many pale touching seeds, and many elongated lupins.
- `IMG_9670c.JPG` is the encouraging exception: 15 predicted versus 16 seeds
  visibly present, with mostly plausible contours.
- StarDist finds many objects missed by U-Net, but patterns, glass edges, and
  seed-like background/glare produce pervasive false polygons. Its output is
  not a usable count.
- Forcing the wrong `soybean` condition materially changed U-Net counts (for
  example 165 instead of 65 on `IMG_0002c.JPG`). Species conditioning must be
  ablated using correct real metadata; synthetic conditioning is not evidence.

Complete analysis plus U-Net review averaged 6.42 seconds per fixture on the
RTX 3070 (70.63 seconds total). Spatially indexed/local StarDist NMS reduced its
worst fixture from 69.5 to 19.4 seconds; the optimized complete StarDist review
averaged 7.73 seconds (85.07 seconds total). These are batch-review timings, not
model-forward-only timings.

The annotation-bootstrap coordinate path was subsequently exercised on
`IMG_9670c.JPG`. It expanded the bounded 854 x 854 procedural crop at offset
`(1509, 542)` into a 3243 x 2129 corrected-image label draft without shifting
the visible seed contours. Visual inspection also preserved the known failure:
18 proposal identities for 16 visible dish seeds, including obvious rim
fragments and imperfect contours. That is the intended honest workflow: the
automatic result saves initial tracing effort while presenting its errors for
human correction rather than silently promoting them to labels.

## Required path to scientific use

1. Produce complete, edge-accurate instance masks for representative real
   images; interior scribbles are not labels. Export is now available from the
   desktop File menu and always marks new samples unreviewed.
2. Independently review annotations and explicitly label apparent pattern
   boundaries with a validity mask. Measure inter-annotator disagreement on a
   subset.
3. Assign train/validation/test by lot and capture session, stratified by
   species, crowding, colour/pattern, damage, and lighting. Lock the test set
   before model selection.
4. Train colour-only and rich-evidence input ablations, with and without correct
   species conditioning. Pretraining may use simulation; final selection may
   not.
5. Optimize and calibrate on validation data only, repeat training with at
   least three random seeds, and report confidence intervals and per-species/
   per-crowding failures.
6. Open the locked test once. The provisional publication gates in
   `LEARNED_INSTANCE_SEGMENTATION.md` remain unmet until real test metrics,
   measurement errors, annotation reliability, and a predeclared protocol are
   available.
