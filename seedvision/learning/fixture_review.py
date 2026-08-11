"""Unlabelled fixture inference and visual domain-shift review."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from seedvision.learning.checkpoint import load_checkpoint
from seedvision.learning.contracts import ModelFamily
from seedvision.learning.pipeline import StarDistPipelineSettings, UNetPipelineSettings
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation import analyze_path
from seedvision.ui.image_view import SUPPORTED_SUFFIXES


def _panel(result, learned, title: str, maximum_width: int = 1500) -> np.ndarray:
    offset_x, offset_y = result.crop_offset
    height, width = learned.labels.shape
    image = np.asarray(
        result.calibration.corrected_bgr[
            offset_y : offset_y + height, offset_x : offset_x + width
        ],
        dtype=np.uint8,
    )
    rgba = learned.instance_rgba()
    colours_bgr = rgba[..., :3][..., ::-1]
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0) * 0.52
    identities = np.uint8(
        np.clip(image.astype(np.float32) * (1.0 - alpha) + colours_bgr * alpha, 0, 255)
    )
    boundary = np.zeros(learned.labels.shape, dtype=bool)
    boundary[:, 1:] |= learned.labels[:, 1:] != learned.labels[:, :-1]
    boundary[:, :-1] |= learned.labels[:, 1:] != learned.labels[:, :-1]
    boundary[1:] |= learned.labels[1:] != learned.labels[:-1]
    boundary[:-1] |= learned.labels[1:] != learned.labels[:-1]
    boundary &= learned.labels > 0
    outlined = image.copy()
    outlined[boundary] = (255, 0, 255)
    panel = np.hstack((image, outlined, identities))
    header = np.full((44, panel.shape[1], 3), 245, np.uint8)
    cv2.putText(
        header,
        title,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    panel = np.vstack((header, panel))
    if panel.shape[1] > maximum_width:
        scale = maximum_width / panel.shape[1]
        panel = cv2.resize(
            panel,
            (maximum_width, max(1, round(panel.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return panel


def _sheet(panels: list[np.ndarray]) -> np.ndarray:
    width = max(panel.shape[1] for panel in panels)
    rows = []
    for panel in panels:
        rows.append(
            cv2.copyMakeBorder(
                panel,
                0,
                0,
                0,
                width - panel.shape[1],
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
        )
    return np.vstack(rows)


def review_fixtures(
    checkpoint_path: Path | str,
    image_directory: Path | str,
    output_directory: Path | str,
    *,
    learning_root: Path | str,
    species: str = "unknown",
) -> dict:
    """Run one checkpoint on every fixture and render unscored review panels."""

    checkpoint_path = Path(checkpoint_path).resolve()
    image_directory = Path(image_directory).resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _model, _spec, payload = load_checkpoint(checkpoint_path, device="cpu")
    family = ModelFamily(payload["family"])
    graph = build_default_pipeline()
    node_id = (
        "unet_instances"
        if family is ModelFamily.UNET_WATERSHED
        else "stardist_instances"
    )
    graph.set_enabled(node_id, True)
    enabled_nodes = {
        identifier
        for identifier, node in graph.nodes.items()
        if node.enabled and identifier not in graph.unused_nodes
    }
    unet_settings = UNetPipelineSettings(checkpoint_path=str(checkpoint_path))
    star_settings = StarDistPipelineSettings(checkpoint_path=str(checkpoint_path))
    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported fixture images found in {image_directory}.")
    records = []
    panels = []
    for path in paths:
        started = perf_counter()
        result = analyze_path(
            path,
            enabled_nodes=enabled_nodes,
            unet_settings=unet_settings,
            stardist_settings=star_settings,
            learning_root=Path(learning_root),
            species=species,
        )
        learned = (
            result.unet_instances
            if family is ModelFamily.UNET_WATERSHED
            else result.stardist_instances
        )
        if learned is None:
            raise RuntimeError(f"The learned branch returned no result for {path.name}.")
        elapsed = perf_counter() - started
        title = (
            f"{path.name} | predicted {learned.count} | "
            f"seed diameter {result.estimated_seed_diameter_px:.1f}px | {elapsed:.2f}s"
        )
        panel = _panel(result, learned, title)
        panels.append(panel)
        cv2.imwrite(str(output_directory / f"{path.stem}.review.jpg"), panel)
        records.append(
            {
                "image": path.name,
                "predicted_instances": learned.count,
                "estimated_seed_diameter_px": result.estimated_seed_diameter_px,
                "elapsed_seconds": elapsed,
                "warnings": list(result.warnings),
            }
        )
    cv2.imwrite(str(output_directory / "contact_sheet.jpg"), _sheet(panels))
    report = {
        "family": family.value,
        "checkpoint": str(checkpoint_path),
        "image_directory": str(image_directory),
        "fixture_count": len(records),
        "scientifically_validated": False,
        "validation_note": (
            "These fixtures have no human instance masks. Panels support qualitative "
            "domain-shift review only and cannot estimate accuracy."
        ),
        "records": records,
    }
    (output_directory / "fixture_review.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
