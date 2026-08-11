# Learned seed-instance segmentation plan

## Scientific objective

Develop and compare two independently decodable, native-PyTorch instance
segmentation methods for the calibrated Seed Fiddle photographs:

1. a multi-head U-Net followed by marker-controlled watershed; and
2. a two-dimensional StarDist model using star-convex polygons and
   non-maximum suppression.

The models must separate touching soybean and lupin seeds without interpreting
normal coat pattern transitions as physical seed boundaries. They are review
aids until they pass the locked, image-level validation gate below.

## Non-negotiable implementation constraints

- Preserve `seed_vision.py` as the only application/command launch point and
  preserve the native PySide6 desktop application.
- Use PyTorch and keep full-resolution tensors, tiles, and reusable model
  intermediates on CUDA until compact topology, metrics, or Qt display requires
  a CPU transfer.
- Cache learned-node outputs independently. A threshold change may rerun only
  its decoder; a checkpoint or input change invalidates the model forward pass.
- Never download weights, upload images, or silently train from annotations.
- Keep all exposed parameters computationally effective.
- Never call pseudo-label, synthetic, or visually inspected results scientific
  validation.

## Ground-truth data contract

Each sample consists of:

- the original image;
- a `uint16` or `int32` instance-label raster of the same dimensions, where
  zero is background and each visible seed has one positive identifier;
- optional pattern-boundary labels with separate positive and validity masks;
- image metadata including species, lot/capture group, annotation author,
  review state, and revision;
- optional per-instance flags for partial occlusion, truncation, damage, and
  measurement validity.

Train, validation, and test partitions are assigned by complete image and,
where available, by lot/capture group. Tiles or seeds from one photograph may
never cross partitions. Empty or partial instance annotations are excluded from
supervised loss unless their validity region explicitly identifies reviewed
pixels.

Pattern-boundary supervision is deliberately independent from physical
boundaries. A missing pattern annotation is unknown and receives zero loss
weight. A weak pseudo-target may be generated from strong within-instance image
gradients, away from physical boundaries, but remains flagged as weak and is
never used in the locked test set.

## Shared model inputs

The initial reproducible input stack is assembled without using annotations:

- corrected CIE Lab image channels;
- valid-dish mask;
- foreground colour and foreground-noise likelihoods;
- inverse background colour/noise evidence;
- shared edge magnitude and sensor/noise likelihood;
- flattened grayscale, shadow, and highlight likelihoods; and
- an optional one-hot species condition, including an unknown-species channel.

Every channel definition and normalization is stored with the checkpoint.
Ablations will determine whether diagnostic channels improve held-out results
over corrected colour alone.

## U-Net plus watershed

The compact residual U-Net has five logical output heads:

1. seed-interior logit;
2. physical seed-boundary logit;
3. apparent seed-pattern-boundary logit;
4. a joint seed-centre heatmap and normalized interior distance regression;
5. calibrated auxiliary error-probability outputs for the supervised dense
   tasks.

Losses use class-balanced BCE and Dice for interior/boundaries, class-balanced
BCE for sparse centres, robust regression for distance, and masked auxiliary
error prediction for uncertainty. Uncertainty cannot reduce the primary task
loss; this avoids the degenerate solution in which a model declares difficult
pixels uncertain instead of learning them. The physical-boundary loss
explicitly upweights difficult contact pixels. Pattern positives are hard
negatives for the physical-boundary head and vice versa, without forcing the
two probabilities to be complements.

The decoder constructs foreground from the interior prediction, derives one
marker per accepted centre peak with distance-map support, and floods an
elevation surface that rewards physical boundaries while discounting predicted
pattern boundaries. Painted instance identifiers can replace or supplement
automatic markers. Decoder parameters are optimized on validation data only.

## StarDist

The StarDist path predicts:

- object probability at eligible interior pixels;
- radial distance to the instance boundary along a configurable fixed ray bank;
- auxiliary radial-error probability used for review ranking.

Targets are derived from reviewed instance masks. Invalid and truncated rays
are masked rather than clipped into false supervision. Dense candidates are
decoded into polygons in corrected-image coordinates and filtered with
score/overlap non-maximum suppression. The implementation supports tiled CUDA
inference with overlap at least as large as the maximum supported seed radius.
Radial regression combines log-distance, relative-error, and angular
second-derivative losses so adjacent rays form smooth, faithful boundaries
rather than alternating spikes.

Initial experiments use 32 and 64 rays. Polygon masks are evaluated both
directly and after optional local refinement by the U-Net physical-boundary
surface.

## Training and optimization protocol

1. Record immutable dataset manifests and hashes.
2. Audit labels for duplicate IDs, disconnected instances, unlabeled holes,
   boundary ambiguity, and image/label dimension mismatch.
3. Establish deterministic synthetic geometry tests and an overfit-one-tile
   training test for each architecture.
4. Use group-held-out image splits stratified by species and crowding. When the
   dataset becomes large enough, retain a never-tuned test partition.
5. Tune preprocessing, crop size, augmentation, loss weights, channel
   ablations, U-Net width/depth, marker decoding, ray count, NMS, and confidence
   thresholds using validation data only.
6. Repeat final candidate training with at least three seeds and report the
   distribution, not only the best run.
7. Calibrate confidence on held-out validation data and define automatic-review
   thresholds before opening the test set.
8. Run full-resolution CUDA timing and peak-memory benchmarks on every fixture.
9. Render raw-image overlays, physical/pattern boundaries, centres, uncertainty,
   instance identities, split/merge flags, and side-by-side error crops for
   manual examination.

## Metrics and validation gate

Report per image, per species, per crowding class, and overall:

- true/false positives and negatives under one-to-one IoU matching;
- precision, recall, F1, average precision, panoptic quality, and matched IoU;
- exact count agreement and absolute/relative count error;
- explicit split and merge rates;
- physical-boundary precision/recall/F1 at pixel and calibrated-distance
  tolerances;
- centre localization error in pixels and millimetres;
- area, Feret-diameter, perimeter, and circularity error on complete seeds;
- uncertainty calibration and error-detection AUROC/AUPRC;
- inference time, peak CUDA memory, and correction time per image.

The following are provisional engineering targets, not by themselves a claim of
publication readiness: held-out instance F1 at IoU 0.5 at least 0.95; panoptic
quality at least 0.90; median absolute image count error at most 1%, with no
image above 3%; split and merge rates each below 1%; and boundary F1 at a
predeclared calibrated tolerance at least 0.90. Confidence intervals and
inter-annotator/reference uncertainty must accompany the final report.

Publication-ready use additionally requires a sufficiently diverse,
human-reviewed, locked test set representing all intended species, lots,
capture sessions, crowding levels, pattern types, damage states, and relevant
occlusions. The current eleven unlabelled fixtures cannot satisfy that
requirement, regardless of apparent visual quality.

## Delivery milestones

1. Baseline audit, written protocol, and brainstorm log.
2. Dataset manifest, label persistence/import/export, target generation, and
   quantitative metrics.
3. Multi-head U-Net, losses, watershed decoder, checkpointing, and tests.
4. Native PyTorch StarDist, losses, polygon/NMS decoder, and tests.
5. Learned DAG nodes, checkpoint discovery, caching, controls, and overlays.
6. Synthetic correctness and overfit tests, followed by fixture inference.
7. Reproducible optimization experiments and side-by-side visual failure audit.
8. Full regression, performance report, and explicit validation verdict.
