"""Patterned packed-seed simulator for correctness tests and pretraining only."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from seedvision.learning.contracts import DEFAULT_SPECIES, FeatureStackSpec
from seedvision.learning.data import LearningManifest, LearningSample, write_label_image
from seedvision.learning.features import colour_only_feature_spec


def _feature_stack(image_bgr: np.ndarray, species_index: int, spec: FeatureStackSpec):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    base = [
        lab[..., 0] / 127.5 - 1.0,
        lab[..., 1] / 127.5 - 1.0,
        lab[..., 2] / 127.5 - 1.0,
        np.ones(image_bgr.shape[:2], dtype=np.float32),
    ]
    if spec.include_species_planes:
        base.extend(
            np.full(image_bgr.shape[:2], float(index == species_index), np.float32)
            for index in range(len(spec.species))
        )
    return np.stack(base, axis=0).astype(np.float32)


def _scene(rng: np.random.Generator, size: int, instance_count: int):
    base_colour = rng.uniform((175, 178, 172), (242, 245, 240))
    yy, xx = np.indices((size, size), dtype=np.float32)
    linear = rng.uniform(-22.0, 22.0) * (xx / size - 0.5)
    linear += rng.uniform(-22.0, 22.0) * (yy / size - 0.5)
    background = np.broadcast_to(base_colour, (size, size, 3)).copy()
    background += linear[..., None]
    # Broad illumination/glare/shadow structures are deliberately unlabelled
    # hard negatives. They prevent a model from treating brightness alone as
    # objectness, as happened in the first simulator smoke run.
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(0, size, size=2)
        sigma = rng.uniform(size * 0.10, size * 0.38)
        amplitude = rng.uniform(-48.0, 55.0)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        tint = rng.uniform(0.75, 1.25, size=3)
        background += amplitude * blob[..., None] * tint
    image = np.uint8(np.clip(background, 0, 255))
    labels = np.zeros((size, size), dtype=np.uint16)
    pattern_boundary = np.zeros((size, size), dtype=np.uint8)
    centres = []
    grid = int(np.ceil(np.sqrt(instance_count)))
    spacing = size / (grid + 1)
    for identifier in range(1, instance_count + 1):
        row, column = divmod(identifier - 1, grid)
        cx = int(round((column + 1) * spacing + rng.normal(0, spacing * 0.12)))
        cy = int(round((row + 1) * spacing + rng.normal(0, spacing * 0.12)))
        axis_a = int(round(spacing * rng.uniform(0.38, 0.52)))
        axis_b = int(round(axis_a * rng.uniform(0.58, 0.92)))
        angle = float(rng.uniform(-80, 80))
        mask = np.zeros(labels.shape, dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (axis_a, axis_b), angle, 0, 360, 1, -1)
        shadow = np.zeros(labels.shape, dtype=np.uint8)
        shadow_offset = (
            cx + int(round(rng.uniform(-0.10, 0.16) * axis_a)),
            cy + int(round(rng.uniform(0.04, 0.20) * axis_b)),
        )
        cv2.ellipse(
            shadow,
            shadow_offset,
            (max(1, axis_a + 2), max(1, axis_b + 2)),
            angle,
            0,
            360,
            1,
            -1,
        )
        shadow_only = (shadow > 0) & (mask == 0) & (labels == 0)
        image[shadow_only] = np.uint8(
            np.clip(image[shadow_only].astype(np.float32) * rng.uniform(0.72, 0.93), 0, 255)
        )
        colour_a = np.asarray(
            rng.choice(
                (
                    (74, 124, 176),
                    (102, 151, 188),
                    (128, 172, 202),
                    (68, 91, 116),
                    (155, 175, 184),
                )
            ),
            dtype=np.float32,
        )
        colour_b = np.clip(colour_a * rng.uniform(0.35, 1.55), 25, 235)
        phase = rng.uniform(0, 2 * np.pi)
        pattern_mode = int(rng.integers(0, 4))
        if pattern_mode == 0:
            frequency = rng.uniform(0.08, 0.24)
            direction = rng.uniform(0, np.pi)
            wave = np.sin(
                (xx * np.cos(direction) + yy * np.sin(direction)) * frequency + phase
            )
            patterned = wave > rng.uniform(-0.25, 0.35)
        elif pattern_mode == 1:
            noise = rng.normal(0, 1, labels.shape).astype(np.float32)
            sigma = rng.uniform(1.5, 5.0)
            smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
            patterned = smooth > np.quantile(smooth[mask > 0], rng.uniform(0.48, 0.72))
        elif pattern_mode == 2:
            direction = rng.uniform(0, np.pi)
            projection = (xx - cx) * np.cos(direction) + (yy - cy) * np.sin(direction)
            patterned = projection > rng.uniform(-0.25, 0.25) * axis_a
        else:
            spots = np.zeros(labels.shape, dtype=np.uint8)
            for _spot in range(int(rng.integers(2, 8))):
                spot_x = int(round(cx + rng.normal(0, axis_a * 0.45)))
                spot_y = int(round(cy + rng.normal(0, axis_b * 0.45)))
                cv2.circle(
                    spots,
                    (spot_x, spot_y),
                    max(1, int(round(rng.uniform(0.10, 0.32) * axis_b))),
                    1,
                    -1,
                )
            patterned = spots > 0
        transition = cv2.morphologyEx(
            np.uint8(patterned), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        ) > 0
        local_noise = rng.normal(0, 7.0, image.shape[:2])[..., None]
        seed_colour = np.where(patterned[..., None], colour_a, colour_b) + local_noise
        visible = mask > 0
        image[visible] = np.uint8(np.clip(seed_colour[visible], 0, 255))
        labels[visible] = identifier
        pattern_boundary[visible] = 0
        pattern_boundary[visible & transition] = 255
        centres.append((cx, cy))
    # Mild broad illumination and sensor noise make the synthetic task nontrivial.
    lighting = 0.86 + 0.18 * (0.6 * xx / size + 0.4 * yy / size)
    image = np.uint8(
        np.clip(image.astype(np.float32) * lighting[..., None] + rng.normal(0, 2.0, image.shape), 0, 255)
    )
    return image, labels, pattern_boundary


def create_synthetic_dataset(
    destination: Path | str,
    *,
    train_count: int = 48,
    validation_count: int = 12,
    test_count: int = 12,
    image_size: int = 256,
    random_seed: int = 20260811,
) -> Path:
    """Create a manifest-labelled simulator dataset outside committed fixtures."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    spec = colour_only_feature_spec(
        include_species_planes=True, nominal_seed_diameter_px=40.0
    )
    rng = np.random.default_rng(random_seed)
    samples = []
    split_counts = (
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    )
    sequence = 0
    for split, count in split_counts:
        for _ in range(count):
            sequence += 1
            identifier = f"synthetic_{sequence:04d}"
            species_index = int(rng.integers(1, len(DEFAULT_SPECIES)))
            image, labels, pattern = _scene(
                rng,
                int(image_size),
                int(rng.integers(max(8, (image_size // 64) ** 2), max(12, (image_size // 48) ** 2) + 1)),
            )
            feature_path = root / f"{identifier}.features.npz"
            label_path = root / f"{identifier}.instances.png"
            pattern_path = root / f"{identifier}.pattern.png"
            pattern_valid_path = root / f"{identifier}.pattern_valid.png"
            image_path = root / f"{identifier}.png"
            np.savez_compressed(
                feature_path,
                features=_feature_stack(image, species_index, spec).astype(np.float16),
            )
            cv2.imwrite(str(image_path), image)
            write_label_image(label_path, labels)
            write_label_image(pattern_path, pattern)
            write_label_image(pattern_valid_path, np.uint8(labels > 0) * 255)
            samples.append(
                LearningSample(
                    identifier=identifier,
                    features=feature_path.name,
                    instances=label_path.name,
                    species=DEFAULT_SPECIES[species_index],
                    group=identifier,
                    split=split,
                    reviewed=True,
                    pattern_boundary=pattern_path.name,
                    pattern_valid=pattern_valid_path.name,
                    image=image_path.name,
                    annotation_author="Seed Fiddle deterministic simulator",
                    annotation_revision="1",
                )
            )
    manifest = LearningManifest(
        dataset_id=f"synthetic-patterned-seeds-{random_seed}",
        feature_spec=spec,
        samples=tuple(samples),
        provenance="synthetic",
        scientific_validation_eligible=False,
    )
    manifest_path = root / "manifest.json"
    manifest.save(manifest_path)
    (root / "README.json").write_text(
        json.dumps(
            {
                "warning": "Synthetic data is for software verification/pretraining only, never scientific validation.",
                "feature_spec": asdict(spec),
                "seed": random_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
