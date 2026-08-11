"""Stable contracts shared by training, inference, evaluation, and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

import cv2
import numpy as np


class ModelFamily(StrEnum):
    UNET_WATERSHED = "unet_watershed"
    STARDIST = "stardist"


DEFAULT_SPECIES = (
    "unknown",
    "soybean",
    "lupinus_mutabilis",
    "lupinus_polyphyllus",
    "lupinus_mexicanus",
)


@dataclass(frozen=True, slots=True)
class FeatureStackSpec:
    """Ordered, checkpointed description of every model input channel."""

    channels: tuple[str, ...] = (
        "lab_l",
        "lab_a",
        "lab_b",
        "valid_dish",
        "foreground_colour",
        "foreground_noise",
        "inverse_background_colour",
        "inverse_background_noise",
        "edge_magnitude",
        "sensor_noise",
        "flattened_grayscale",
        "shadow",
        "highlight",
    )
    species: tuple[str, ...] = DEFAULT_SPECIES
    include_species_planes: bool = True
    nominal_seed_diameter_px: float = 48.0
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(str(item) for item in self.channels))
        object.__setattr__(self, "species", tuple(str(item) for item in self.species))
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("Feature channels must be non-empty and unique.")
        if not self.species or self.species[0] != "unknown":
            raise ValueError("The species vocabulary must begin with 'unknown'.")
        if len(set(self.species)) != len(self.species):
            raise ValueError("Species vocabulary entries must be unique.")
        if self.version != 1:
            raise ValueError("Unsupported feature-stack specification version.")
        if not 8.0 <= float(self.nominal_seed_diameter_px) <= 256.0:
            raise ValueError("Nominal seed diameter must be between 8 and 256 pixels.")

    @property
    def input_channels(self) -> int:
        return len(self.channels) + (
            len(self.species) if self.include_species_planes else 0
        )

    def species_index(self, species: str | None) -> int:
        normalized = str(species or "unknown").strip().lower().replace(" ", "_")
        aliases = {
            "soy": "soybean",
            "l._mutabilis": "lupinus_mutabilis",
            "l._polyphyllus": "lupinus_polyphyllus",
            "l._mexicanus": "lupinus_mexicanus",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return self.species.index(normalized)
        except ValueError:
            return 0


@dataclass(slots=True)
class LearnedInstanceResult:
    """Common decoded result returned by either learned model family."""

    family: ModelFamily
    labels: np.ndarray
    centres_xy: np.ndarray
    instance_confidences: np.ndarray
    rasters: Mapping[str, object] = field(default_factory=dict)
    method: str = ""
    checkpoint_id: str = ""
    scientifically_validated: bool = False
    validation_summary: str = "Not validated on locked human-reviewed data"

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int32)
        self.centres_xy = np.asarray(self.centres_xy, dtype=np.float32).reshape(-1, 2)
        self.instance_confidences = np.asarray(
            self.instance_confidences, dtype=np.float32
        ).reshape(-1)
        if self.labels.ndim != 2:
            raise ValueError("Learned instance labels must be a two-dimensional raster.")
        if np.any(self.labels < 0):
            raise ValueError("Learned instance labels cannot be negative.")
        if len(self.centres_xy) != self.count:
            raise ValueError("One centre is required for every decoded instance.")
        if len(self.instance_confidences) != self.count:
            raise ValueError("One confidence is required for every decoded instance.")

    @property
    def count(self) -> int:
        return int(self.labels.max(initial=0))

    def instance_rgba(self) -> np.ndarray:
        """Render deterministic, distinct instance colours for Qt overlays."""

        labels = self.labels
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        if self.count == 0:
            return rgba
        identifiers = np.arange(self.count + 1, dtype=np.uint32)
        hue = np.uint8((identifiers * 137 + 29) % 180)
        hsv = np.column_stack(
            (hue, np.full_like(hue, 205), np.full_like(hue, 255))
        ).reshape(-1, 1, 3)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
        rgba[..., :3] = rgb[np.clip(labels, 0, self.count)]
        rgba[..., 3] = np.uint8(labels > 0) * 205
        return rgba

    def confidence_raster(self) -> np.ndarray:
        lookup = np.zeros(self.count + 1, dtype=np.float32)
        if self.count:
            lookup[1:] = self.instance_confidences
        return np.uint8(np.rint(np.clip(lookup[self.labels], 0.0, 1.0) * 255.0))
