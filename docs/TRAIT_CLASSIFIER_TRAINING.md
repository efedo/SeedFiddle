# Native seed-trait training

The **Object-level seed traits** node classifies each accepted seed into one of a
supplied set of trait classes, or abstains. Training, saving and activation are
separate steps, and nothing here establishes biological validity: definitions,
truth, permissions and grouping are supplied by you and are only checked for
consistency and split isolation. Independent locked testing (laboratory gate L04)
is still required.

## Train and activate

Use **Learning → Train object-level seed traits…** to choose an evidence manifest, output file and training controls. The job shares the desktop's serial CUDA worker. Cancel and Stop request a stop before the next calculation or file commit. **Learning → Activate frozen seed-trait model…** also activates an existing model. Activation validates the artifact and graph before changing the Object-level seed traits node.

Saving and activation are separate. The default saves without activating. Save to a new file when a model is already active. A failed or cancelled job before publication preserves the previous output; cancellation after publication keeps the valid saved file and skips activation. A graph change during training also skips automatic activation. Only the classifier and its dependents are invalidated on successful activation; reusable upstream caches remain available.

Definitions, truth, permissions and biological grouping must be supplied. The program checks their consistency and isolation; it does not establish their biological authority. Calibration diagnostics and predictive probabilities do not establish scientific validity. Independent locked testing remains required.

## Training manifest

The UTF-8 JSON manifest has exactly these root fields:

```json
{
  "format": "seedfiddle-trait-training",
  "version": 1,
  "feature_schema": "seed-classifier-pixel-v1",
  "vocabulary": {
    "revision": "your-definition-revision",
    "definition_source": "your supplied definition reference",
    "definition_source_sha256": "replace with 64 lowercase hexadecimal characters",
    "traits": [
      {"identifier": "class_a", "definition": "Your explicit definition", "evidence_reference": "Your definition evidence"},
      {"identifier": "class_b", "definition": "Your second explicit definition", "evidence_reference": "Your definition evidence"}
    ]
  },
  "examples": []
}
```

This shows the envelope, not a ready-to-train dataset. Supply 2–64 unique trait definitions and 2–10,000 examples, including permitted examples in both splits and every defined class in training. Definition IDs use lowercase letters, digits and underscores, beginning with a letter. Replace the digest placeholder with your supplied definition-source digest. That digest is recorded as supplied provenance, not independently verified expert approval.

Each example has exactly these fields:

```json
{
  "example_id": "unique-example",
  "physical_id": "your-stable-physical-seed-id",
  "group_id": "your-biological-lot-and-capture-group",
  "trait_id": "class_a",
  "truth_provenance": "your adjudicated label reference",
  "permitted_use": true,
  "split": "train",
  "input": {
    "kind": "reference_seed",
    "image_path": "capture.png",
    "image_sha256": "replace with the actual source-file SHA256",
    "archive_path": "projects/reference-regions/your-archive.npz",
    "archive_sha256": "replace with the actual archive-file SHA256",
    "seed_id": 1
  }
}
```

Splits are `train` and `calibration`. Example, physical-seed and group identities cannot overlap splits. All captures from the same biological group must retain that group identity. The program also rejects identical actual source/selected-mask evidence across splits, even when declared identities differ. It cannot independently establish whether supplied groups represent independent biological samples.

Paths resolve relative to the manifest directory; absolute paths also work. The reference archive must contain the requested seed, a recorded source-to-corrected transform and a reviewed complete outline without an exclusion. Known physical-seed annotations must agree with the example. Canvas-border seeds and pixels outside decoded source support reject. Archive/source identity and known annotation-generation identity must match the exact consumed source snapshot; absent generation history remains unknown.

Alternatively, `input` can contain exactly:

```json
{
  "kind": "compact_features",
  "values": [0.1, 0, 0.1, 0.1, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.3, 0, 0.3, 0.3, 0.3, 2.772588722239781, 0, 1, 0, 4],
  "provenance": "Your explicit feature-source reference"
}
```

This illustrative row describes a uniform-colour 4×4 square. The twenty finite values must follow the exact schema: red, green and blue each contribute mean, population standard deviation, q10, q50 and q90; then log pixel area, log bounding-box aspect, bounding-box fill, moment anisotropy and grid perimeter divided by square-root area. Colours are unit RGB; area and perimeter refer to visible corrected-frame pixels. Geometry is resolution-dependent and carries no physical scale. Supplied compact rows record supplied provenance without claiming verified extractor lineage.

Set `permitted_use` to false to retain an excluded fact in the audit. Its `input` can be null; the program never interprets or opens denied input contents. Fact identities still participate in split-isolation checks.

Manifest input is bounded to 16 MiB. Each compressed image snapshot is bounded to 512 MiB and each compressed archive to 256 MiB; oversized inputs reject without resizing. Full-resolution decoded images use the existing CPU decoder, while correction, feature rasters and fitting stay on CUDA. The frozen finite JSON model is bounded to 4 MiB and contains definitions, controls, split/input audits, evidence identities, weights, normalization, calibration and development diagnostics. It contains no executable checkpoint. Loading never retrains it.

## Portable project snapshots

A selected Object-level seed traits model is included even while its node is disabled. Create Snapshot validates the exact finite JSON bytes within 4 MiB and records byte count, raw SHA256 and canonical model SHA256. The snapshot contains a content-addressed object; Restore Snapshot verifies it and changes the stored selection to a path within the restored workspace. Neither operation retrains the model or changes the original model. Model paths resolve relative to the learning/workspace root, matching native and saved-project batch analysis.

An explicit `None` selection creates no model asset. Missing, invalid, oversized, changed or inconsistent selected models stop completion. Earlier snapshots without a selected trait model remain readable. Earlier snapshots that selected a trait model without storing it in the new inventory cannot supply complete portable closure and are rejected. Restored supplied definitions and audit facts remain supplied provenance; they do not establish expert approval or biological accuracy.

## Reading stored predictions in review

The results table’s Predicted trait column shows the stored supplied class ID or an explicit abstention reason. Hover for the supplied definition/evidence, model identity and maximum predictive probability where available. That probability does not establish correctness. The summary counts assigned, abstained and unavailable stored rows. Reviewed annotation traits remain independent and are never replaced by these predictions.

Opening review does not fit or run the classifier. Changing the abstention threshold and recomputing can change assignments while preserving probabilities. Saved decisions apply only to their exact prediction revision; earlier review bytes remain historical. A probability exactly equal to the selected threshold qualifies, including an exact 1 at threshold 1.
