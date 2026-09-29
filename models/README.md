# Learned model checkpoints

The U-Net/watershed and StarDist engineering paths are implemented but disabled
by default. No committed checkpoint is an independently validated laboratory
model. See [current status](../docs/CURRENT_STATUS.md) and the
[scientific validation protocol](../docs/SCIENTIFIC_VALIDATION_PROTOCOL.md).

Place self-describing Seed Fiddle checkpoints here by default:

- `unet_seed_instances.pt`
- `stardist_seed_instances.pt`

Checkpoint binaries are intentionally ignored because trained weights must be
versioned together with their immutable dataset manifest, training report,
license/provenance record, and validation report in an appropriate model/data
registry. The desktop node accepts another project-relative or absolute path.

Neither node is enabled merely because a file exists. Enable it explicitly
after checking that the checkpoint's feature specification and model family are
appropriate. A checkpoint is not evidence of scientific validity; consult its
held-out evaluation report.
