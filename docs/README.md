# Documentation index

## Start here

- [Repository README](../README.md): what Seed Fiddle is, installation, a first run.
- [Current status](CURRENT_STATUS.md): what is implemented, what is validated (nothing
  yet, scientifically) and what blocks laboratory use.
- [Operator guide](OPERATOR_GUIDE.md): day-to-day use of every part of the
  application. Opens with `F1`.

## For developers

- [HANDOFF.md](../HANDOFF.md) and [AGENTS.md](../AGENTS.md): orientation, binding
  rules and known pitfalls.
- [Architecture and contracts](ARCHITECTURE.md): packages, caching, persistence and
  extension rules.
- [Testing](TESTING.md): runner, tiers, prerequisites and CI. Generated
  [test inventory](testing/TEST_INVENTORY.md); the
  [18 October consolidation](archive/TEST_SUITE_REDESIGN_2026_10_18.md); the
  [frozen recipe profile coverage](testing/RECIPE_PROFILE_COVERAGE.md).
- [Node catalogue](NODE_CATALOGUE.md): generated list of every node, control,
  default and optimisation status. Regenerate with `scripts/generate_node_catalogue.py`.
- [Roadmap](../PLAN.md): engineering phases, the peer-functionality direction and the
  remaining validation phases.

## Methods and science

- [Seed-detection evaluation](SEED_DETECTION_EVALUATION.md): how the current
  detection pipelines compare with the literature, and recommended directions.
- [Scientific validation protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md): evidence
  required before any scientific or laboratory claim (gates L01–L04).
- [Node optimisation](NODE_OPTIMIZATION.md): objectives, eligibility and exemptions.
- [Learned instance segmentation](LEARNED_INSTANCE_SEGMENTATION.md) and
  [seed-trait training](TRAIT_CLASSIFIER_TRAINING.md).
- [Species reference libraries](SPECIES_REFERENCE_LIBRARIES.md),
  [reference seed dimensions and shape](REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md),
  [procedural reference matching and costs](PROCEDURAL_REFERENCE_MATCHING_AND_COSTS.md),
  [local edge normalisation](LOCAL_EDGE_NORMALIZATION.md) and
  [ruler evidence detection](RULER_EVIDENCE_DETECTION.md): implementation records for
  specific calculations. Their UI descriptions may predate the current window.

## PlantCV–SMPTS peer functionality

- [Comparison report](PLANTCV_SMPTS_COMPARISON.md): rationale and scope.
- [Implementation and benchmark plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md): the
  design specification (families F01–F22, work packages WP0–WP6).
- [Method and input-domain inventory](PLANTCV_SMPTS_METHOD_DOMAIN_MATRIX.md) and its
  [JSON](plantcv_smpts_method_domain_matrix.json); the operation-level
  [feature register](plantcv_smpts_features.json) (validated by
  `tests/test_peer_feature_register.py`); the
  [acceptance audit](PLANTCV_SMPTS_ACCEPTANCE_AUDIT.json).
- [Benchmark reports](BENCHMARK_REPORTS.md): command-line benchmark actions and the
  reference-environment setup.
- Third-party algorithm notices: [OpenCV](OPENCV_MATH_NOTICES.md),
  [scikit-image watershed](SCIKIT_IMAGE_WATERSHED_NOTICES.md),
  [SciPy k-means](SCIPY_KMEANS_NOTICES.md), [scikit-learn k-means](SKLEARN_KMEANS_NOTICES.md),
  [Bayes KDE](BAYES_KDE_NOTICES.md).

## Ideas and history

- [Future ideas](FUTURE_IDEAS.md): an uncommitted brainstorm; nothing there is
  implemented unless the roadmap says so.
- [archive/](archive/): dated reviews, audits and redesign records (August–October
  2026). They describe the revision they examined; findings marked open there may
  since have been fixed.
- Until October 2026 the root documents and several guides carried a long,
  repeated chronological implementation log. It was removed in the documentation
  cleanup and remains in git history, for example `git show 2e5b398:HANDOFF.md`.

## Directory notes

- [Seed-instance reference masks](../seed-instance-references/README.md)
- [Seed-instance benchmark](INSTANCE_BENCHMARK.md): scoring predicted masks against the
  unreviewed reference pre-annotations, plus the experimental Cellpose-SAM spike.
- [Learned checkpoints](../models/README.md)
- [Offline wheelhouse](../wheels/README.md)
- [Development prompt notes](../seedfiddle_prompts.md): dated requests, not a defect
  list or roadmap.
