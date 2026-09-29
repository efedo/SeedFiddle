# Documentation index

Start with [Current project status](CURRENT_STATUS.md). It states the implemented
capabilities, current annotation behavior, latest verification, and remaining
laboratory work as of 29 September 2026.

## Current operational documents

- [Repository README](../README.md): setup, main workflow, compatibility, and checks.
- [Operator guide](OPERATOR_GUIDE.md): day to day image, annotation, result, project,
  training, and evaluation workflow.
- [Architecture and contracts](ARCHITECTURE.md): ownership, caching, persistence,
  evaluation, and extension rules.
- [Implementation roadmap](../PLAN.md): completed engineering phases and remaining
  validation milestones.
- [Node catalogue](NODE_CATALOGUE.md): generated card, control, default, and
  optimization inventory.
- [Node optimization](NODE_OPTIMIZATION.md): current fitting workflow and objectives.
- [Scientific validation protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md): evidence that
  is still required before scientific or laboratory claims.
- [Critical review remediation](CRITICAL_REVIEW_REMEDIATION.md): disposition of the
  30 findings and the boundary between software repair and empirical validation.

## Current implementation records

- [Learned instance segmentation](LEARNED_INSTANCE_SEGMENTATION.md) and
  [engineering results](LEARNED_SEGMENTATION_RESULTS.md)
- [Species reference libraries](SPECIES_REFERENCE_LIBRARIES.md)
- [Reference seed dimensions and shape](REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md)
- [Reference matching and costs](PROCEDURAL_REFERENCE_MATCHING_AND_COSTS.md)
- [Edge normalization](LOCAL_EDGE_NORMALIZATION.md)
- [Material evidence redesign](MATERIAL_EVIDENCE_REDESIGN.md)
- [Ruler evidence detection](RULER_EVIDENCE_DETECTION.md)
- [Annotation and shape review](ANNOTATION_SHAPE_AND_EDGE_REVIEW_2026_09_05.md)

These documents explain implemented calculations and migration decisions. Their
dated UI descriptions may precede the compact palette described in the current
status and operator guide.

## Historical audits and redesign records

- [Critical application review, 8 September](CRITICAL_APPLICATION_REVIEW_2026_09_08.md)
- [Node optimization compliance review, 9 September](NODE_OPTIMIZATION_COMPLIANCE_REVIEW_2026_09_09.md)
- [Comprehensive node calculation audit](COMPREHENSIVE_NODE_CALCULATION_FORENSIC_AUDIT.md)
- [Evidence and analytical chain audit](EVIDENCE_AND_ANALYSIS_FORENSIC_AUDIT.md)
- [Pipeline consolidation audit](PIPELINE_CONSOLIDATION_AND_EVIDENCE_AUDIT.md)
- [Node control audit](NODE_CONTROL_AUDIT.md)
- [Analysis reliability redesign](analysis-redesign-2026-08-22.md)
- [Historical repository README](../README_HISTORY.md)

These are retained as evidence for the revision they examined. Findings marked
open there may have been repaired later; use the current status and remediation
register for present behavior.

## Research backlog

[Future ideas](FUTURE_IDEAS.md) is an append-only brainstorm. Items there are not
commitments, current controls, or evidence of implementation.

## Directory specific notes

- [Seed-instance reference masks](../seed-instance-references/README.md)
- [Learned checkpoints](../models/README.md)
- [Offline wheelhouse](../wheels/README.md)
- [Pinned runtime requirements](../requirements-runtime.txt)
- [Archived prompt scratchpad](../seedfiddle_prompts.md): dated requests, not a
  current defect list or roadmap.
- [Agent handoff and chronological implementation log](../HANDOFF.md)
