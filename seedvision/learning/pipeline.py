"""Seed Fiddle pipeline adapters for checkpointed learned models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from seedvision.learning.checkpoint import load_checkpoint
from seedvision.learning.contracts import ModelFamily
from seedvision.learning.decode import (
    StarDistDecodeSettings,
    UNetWatershedSettings,
    decode_stardist,
    decode_unet_watershed,
)
from seedvision.learning.features import assemble_feature_stack
from seedvision.learning.inference import tiled_predict


@dataclass(frozen=True, slots=True)
class UNetPipelineSettings:
    checkpoint_path: str = "models/unet_seed_instances.pt"
    tile_size: int = 512
    tile_overlap: int = 96
    interior_threshold: float = 0.50
    centre_threshold: float = 0.30
    centre_minimum_separation_fraction: float = 0.32
    minimum_instance_area_fraction: float = 0.15
    physical_boundary_weight: float = 0.72
    distance_topography_weight: float = 0.28
    pattern_boundary_discount: float = 0.80
    uncertainty_penalty: float = 0.20
    foreground_erosion_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise ValueError("A U-Net checkpoint path is required.")
        if self.tile_size < 128 or self.tile_overlap < 0 or self.tile_overlap * 2 >= self.tile_size:
            raise ValueError("U-Net tile overlap must be less than half its tile size.")
        if self.centre_minimum_separation_fraction <= 0 or self.minimum_instance_area_fraction <= 0:
            raise ValueError("U-Net scale-relative decoder limits must be positive.")
        if not 0.0 <= self.foreground_erosion_fraction <= 0.10:
            raise ValueError("U-Net foreground erosion must be between zero and 0.10 diameter.")

    @property
    def inference_signature(self):
        return (self.checkpoint_path, self.tile_size, self.tile_overlap)

    @property
    def decoder_signature(self):
        return (
            self.interior_threshold,
            self.centre_threshold,
            self.centre_minimum_separation_fraction,
            self.minimum_instance_area_fraction,
            self.physical_boundary_weight,
            self.distance_topography_weight,
            self.pattern_boundary_discount,
            self.uncertainty_penalty,
            self.foreground_erosion_fraction,
        )

    def decoder_settings(self, seed_diameter_px: float) -> UNetWatershedSettings:
        return UNetWatershedSettings(
            interior_threshold=self.interior_threshold,
            centre_threshold=self.centre_threshold,
            centre_minimum_separation_px=max(
                1.0, seed_diameter_px * self.centre_minimum_separation_fraction
            ),
            minimum_instance_area_px=max(
                1, round(seed_diameter_px * seed_diameter_px * self.minimum_instance_area_fraction)
            ),
            physical_boundary_weight=self.physical_boundary_weight,
            distance_topography_weight=self.distance_topography_weight,
            pattern_boundary_discount=self.pattern_boundary_discount,
            uncertainty_penalty=self.uncertainty_penalty,
            foreground_erosion_px=max(
                0, round(seed_diameter_px * self.foreground_erosion_fraction)
            ),
        )


@dataclass(frozen=True, slots=True)
class StarDistPipelineSettings:
    checkpoint_path: str = "models/stardist_seed_instances.pt"
    tile_size: int = 512
    tile_overlap: int = 96
    object_threshold: float = 0.45
    nms_iou_threshold: float = 0.35
    local_maximum_radius_fraction: float = 0.08
    minimum_instance_area_fraction: float = 0.15
    maximum_candidates: int = 4096

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise ValueError("A StarDist checkpoint path is required.")
        if self.tile_size < 128 or self.tile_overlap < 0 or self.tile_overlap * 2 >= self.tile_size:
            raise ValueError("StarDist tile overlap must be less than half its tile size.")
        if self.local_maximum_radius_fraction <= 0 or self.minimum_instance_area_fraction <= 0:
            raise ValueError("StarDist scale-relative decoder limits must be positive.")

    @property
    def inference_signature(self):
        return (self.checkpoint_path, self.tile_size, self.tile_overlap)

    @property
    def decoder_signature(self):
        return (
            self.object_threshold,
            self.nms_iou_threshold,
            self.local_maximum_radius_fraction,
            self.minimum_instance_area_fraction,
            self.maximum_candidates,
        )

    def decoder_settings(self, seed_diameter_px: float) -> StarDistDecodeSettings:
        return StarDistDecodeSettings(
            object_threshold=self.object_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            local_maximum_radius_px=max(
                1, round(seed_diameter_px * self.local_maximum_radius_fraction)
            ),
            minimum_instance_area_px=max(
                1, round(seed_diameter_px * seed_diameter_px * self.minimum_instance_area_fraction)
            ),
            maximum_candidates=self.maximum_candidates,
        )


_MODEL_CACHE: dict[tuple[str, int, str], tuple[object, object, dict]] = {}
_MODEL_CACHE_LOCK = RLock()


def _checkpoint(root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def cached_checkpoint(root: Path, configured: str, *, device):
    path = _checkpoint(root, configured)
    if not path.is_file():
        raise FileNotFoundError(
            f"Learned-model checkpoint not found: {path}. Train or copy a compatible checkpoint first."
        )
    key = (str(path).casefold(), path.stat().st_mtime_ns, str(device))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            cached = load_checkpoint(path, device=device)
            for old_key in tuple(_MODEL_CACHE):
                if old_key[0] == key[0] and old_key != key:
                    del _MODEL_CACHE[old_key]
            _MODEL_CACHE[key] = cached
        return path, cached


def predict_pipeline_model(
    family: ModelFamily,
    *,
    root: Path,
    settings: UNetPipelineSettings | StarDistPipelineSettings,
    source_bgr,
    valid_mask,
    evidence,
    species: str,
    seed_diameter_px: float,
):
    """Run the learned forward pass in the checkpoint's canonical seed scale.

    Scale normalization is part of the forward-pass signature: changing the
    measured seed diameter invalidates logits, while decoder-only controls can
    still reuse those logits.  Radial StarDist distances are converted back to
    source-image pixels after inference.
    """

    import torch.nn.functional as functional

    device = source_bgr.device
    checkpoint_path, (model, feature_spec, payload) = cached_checkpoint(
        root, settings.checkpoint_path, device=device
    )
    if ModelFamily(payload["family"]) is not family:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains {payload['family']}, not {family.value}."
        )
    features = assemble_feature_stack(
        source_bgr,
        valid_mask,
        evidence,
        spec=feature_spec,
        species=species,
    )
    actual_diameter = max(1.0, float(seed_diameter_px))
    scale = float(feature_spec.nominal_seed_diameter_px) / actual_diameter
    # Avoid pathological memory use if a corrupted scale estimate reaches this
    # optional branch.  The generous range still covers a 16-fold area change.
    scale = min(4.0, max(0.25, scale))
    source_height, source_width = features.shape[-2:]
    inference_height = max(16, round(source_height * scale))
    inference_width = max(16, round(source_width * scale))
    if (inference_height, inference_width) != (source_height, source_width):
        features = functional.interpolate(
            features,
            size=(inference_height, inference_width),
            mode="bilinear",
            align_corners=False,
        )
    outputs = tiled_predict(
        model,
        features,
        tile_size=settings.tile_size,
        overlap=settings.tile_overlap,
        use_mixed_precision=True,
    )
    if (inference_height, inference_width) != (source_height, source_width):
        outputs = {
            name: functional.interpolate(
                values,
                size=(source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
            for name, values in outputs.items()
        }
        if family is ModelFamily.STARDIST:
            outputs["radial_distances"] = outputs["radial_distances"] / scale
    return outputs, str(checkpoint_path)


def decode_pipeline_model(
    family: ModelFamily,
    outputs,
    *,
    settings: UNetPipelineSettings | StarDistPipelineSettings,
    seed_diameter_px: float,
    painted_instances=None,
    valid_mask=None,
    checkpoint_id: str = "",
):
    if family is ModelFamily.UNET_WATERSHED:
        return decode_unet_watershed(
            outputs,
            valid_mask=valid_mask,
            painted_instances=painted_instances,
            settings=settings.decoder_settings(seed_diameter_px),
            checkpoint_id=checkpoint_id,
        )
    return decode_stardist(
        outputs,
        valid_mask=valid_mask,
        settings=settings.decoder_settings(seed_diameter_px),
        checkpoint_id=checkpoint_id,
    )
