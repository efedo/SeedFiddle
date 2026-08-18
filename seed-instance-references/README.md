# Seed-instance reference masks

This directory contains lossless, source-coordinate `uint16` instance-ID masks
for the committed images in `images/`. Pixel value `0` is background; each
positive value identifies one visible seed. The isolated reference seeds below
each dish are included and counted separately in `manifest.json`.

The masks deliberately remain in the original photograph's coordinates. Seed
Fiddle verifies the image and mask SHA-256 digests, then applies the current
source-to-corrected calibration transform with nearest-neighbour interpolation
when **Load reference mask…** is selected in the seed-annotation panel. The
loaded labels are an unapplied, undoable draft so they can be inspected and
corrected before **Apply + save**. Use **Choose mask file…** when an explicit
corrected-coordinate PNG, TIFF, or NPZ should replace the bundled starting
point.

`reviewed: false` means the mask was generated and visually audited by Codex,
but has not been independently accepted as human ground truth. It must not be
treated as a validated learning/test label until a person has reviewed every
instance at full resolution.
