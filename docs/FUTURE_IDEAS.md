# Running future-ideas brainstorm

This is a dated, append-only working list. Ideas are not commitments and should
be promoted into the implementation plan only after evidence or a concrete use
case justifies them.

## 2026-08-11 — learned instance-segmentation kickoff

- Use StarDist centres/polygons as watershed markers and the U-Net physical
  boundary prediction as the elevation surface; this may combine reliable
  object counting with pixel-accurate measurement contours.
- Mine hard negatives for the physical-boundary head from strong gradients
  wholly inside reviewed seed masks, especially bicolour coat transitions.
- Add an explicit mutually informative loss between physical- and
  pattern-boundary heads while still permitting both to be uncertain at glare,
  damage, and occlusion.
- Condition models by species through small one-hot feature planes first;
  compare this with separate species models only after enough data exists.
- Use calibrated seed diameter to normalize crop scale and StarDist ray range,
  reducing the amount of geometric variation the network must learn.
- Train a synthetic packed-seed generator with controllable ellipse shape,
  overlap, coat mottling, shadows, highlights, rim intrusion, and sensor noise
  for topology pretraining—not for final accuracy claims.
- Turn the existing procedural result into an editable annotation proposal,
  then preserve every accepted correction as versioned training data.
- Add active-learning review queues ranked by ensemble disagreement, split/merge
  risk, boundary uncertainty, and species/capture novelty.
- Compare deep ensembles, checkpoint ensembles, Monte Carlo dropout, and simple
  test-time augmentation for uncertainty calibration before selecting one.
- Refine polygon edges only in a narrow band using graph cuts, dynamic
  programming, or the existing magnetic-edge machinery to preserve full image
  resolution without a full-resolution network pass.
- Evaluate topology-aware and boundary-distance losses if ordinary BCE/Dice
  leaves contact gaps or produces broken physical boundaries.
- Record two independent human annotations for a subset to measure the ceiling
  imposed by ambiguous contacts and to define calibrated boundary tolerance.
- Include unseen capture sessions and lots in external validation; random seed
  crops from the same dish are not independent test data.
- Investigate polarized or cross-polarized capture as a future acquisition
  improvement if glass glare remains a dominant error source.
- Preserve original logits and model/version hashes with exported measurements
  so downstream scientific analyses are auditable and reproducible.

## 2026-08-11 â€” findings from the first controlled model runs

- Use U-Net interior *and predicted distance-core* support before admitting a
  StarDist marker. Interior-only gating still permits duplicate pattern peaks.
- Keep the hybrid experimental unless it beats the U-Net on real validation;
  it nearly matched but did not improve the controlled U-Net result.
- Replace synthetic species planes with `unknown`, or ablate them, unless the
  generator implements real species-specific shape/coat distributions.
- Mine StarDist hard negatives from actual dish rims, glare, inter-seed gaps,
  ruler/card fragments, and pale empty background. Artificial Gaussian blobs
  exposed the failure but cannot reproduce the real distribution.
- Pretrain the encoder using self-supervised crops from all real photographs,
  then fine-tune supervised heads on reviewed masks; compare against ordinary
  ImageNet initialization without letting unlabeled fixtures enter test tuning.
- Consider a small U-Net centre head trained from StarDist-consistent centroid
  targets as a cheaper alternative to running two complete backbones.
- Add a reviewer-facing pattern-boundary paint/import mode with an explicit
  validity brush; unknown pattern regions must continue to receive zero loss.
- Calibrate dense physical/pattern thresholds and uncertainty independently;
  a higher pattern loss improved that head modestly but harmed instance
  topology, demonstrating a multi-objective tradeoff.
- Use spatially indexed polygon NMS or batched GPU box/polygon suppression for
  any dense StarDist deployment; exact all-pairs polygon IoU is not interactive.
- Measure real boundary error in millimetres and relative to seed diameter. A
  one-pixel discrepancy on the canonical simulator had a disproportionate PQ
  cost and should not be interpreted without physical scale.

## 2026-08-11 - annotation-bootstrap review

- Record correction operations and active editing time separately for
  procedural, U-Net, and StarDist initial drafts. Accuracy alone does not reveal
  which proposal minimizes human annotation cost.
- Turn corrected proposal errors into explicit hard-example tags: missing seed,
  false object, split, merge, rim fragment, coat-pattern split, and contour
  correction. These tags can drive balanced sampling and per-failure reporting.
- Use the calibrated inner/outer dish geometry as a hard-negative context
  channel and for proposal-review warnings, not as an unconditional learned-mask
  override; seeds near the wall must remain representable.
- Preserve the initializing method, checkpoint hash, decoder parameters, and
  elapsed correction time in annotation revision provenance. The first desktop
  bootstrap records the method, but complete revision history still needs a
  durable annotation project format.
- Add reviewer-facing pattern-boundary and pattern-validity brushes only after
  instance-draft persistence is durable; otherwise the independent pattern
  supervision can be lost when switching images or restarting the application.
