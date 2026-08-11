"""Image-level learned-model evaluation, decoder tuning, and visual reports."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from seedvision.learning.checkpoint import load_checkpoint
from seedvision.learning.contracts import ModelFamily
from seedvision.learning.data import LearningManifest, read_label_image, resolve_sample_path
from seedvision.learning.decode import (
    StarDistDecodeSettings,
    UNetWatershedSettings,
    decode_stardist,
    decode_unet_watershed,
)
from seedvision.learning.inference import tiled_predict
from seedvision.learning.metrics import evaluate_binary_probability, evaluate_instances
from seedvision.learning.targets import physical_boundary_mask


def _load_features(manifest_path: Path, relative_path: str, device):
    import torch

    with np.load(resolve_sample_path(manifest_path, relative_path), allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(features))[None].to(device=device)


def _load_display_image(manifest_path: Path, sample, shape):
    image = None
    if sample.image:
        image = cv2.imread(
            str(resolve_sample_path(manifest_path, sample.image)), cv2.IMREAD_COLOR
        )
    if image is None:
        image = np.full((*shape, 3), 220, dtype=np.uint8)
    if image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return image


def _render_comparison(image, truth, prediction, title: str):
    truth_boundary = physical_boundary_mask(truth, width=1) > 0
    predicted_boundary = physical_boundary_mask(prediction, width=1) > 0
    overlay = image.copy()
    overlay[truth_boundary] = (50, 220, 50)
    overlay[predicted_boundary] = (220, 60, 220)
    both = truth_boundary & predicted_boundary
    overlay[both] = (40, 220, 240)
    colour = np.zeros_like(image)
    count = int(prediction.max(initial=0))
    for identifier in range(1, count + 1):
        selected = prediction == identifier
        hue = np.uint8((identifier * 137 + 29) % 180)
        pixel = cv2.cvtColor(
            np.asarray([[[hue, 210, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        colour[selected] = pixel
    identities = cv2.addWeighted(image, 0.52, colour, 0.48, 0.0)
    panel = np.hstack((image, overlay, identities))
    header = np.full((34, panel.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def _contact_sheet(panels: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    target_width = min(1500, max(panel.shape[1] for panel in panels))
    resized = []
    for panel in panels:
        scale = target_width / panel.shape[1]
        resized.append(
            cv2.resize(
                panel,
                (target_width, max(1, round(panel.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    row_height = max(panel.shape[0] for panel in resized)
    rows = []
    for start in range(0, len(resized), columns):
        row_panels = resized[start : start + columns]
        while len(row_panels) < columns:
            row_panels.append(np.full_like(resized[0], 245))
        padded = [
            cv2.copyMakeBorder(
                panel,
                0,
                row_height - panel.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
            for panel in row_panels
        ]
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def _decode(family, outputs, settings, checkpoint_id):
    if family is ModelFamily.UNET_WATERSHED:
        return decode_unet_watershed(
            outputs, settings=settings, checkpoint_id=checkpoint_id
        )
    return decode_stardist(outputs, settings=settings, checkpoint_id=checkpoint_id)


def _score(metrics: list[dict]) -> float:
    if not metrics:
        return -float("inf")
    return mean(
        item["panoptic_quality"]
        + 0.35 * item["f1"]
        + 0.15 * item["boundary_f1"]
        - 0.50 * item["relative_count_error"]
        for item in metrics
    )


def decoder_search(family: ModelFamily, predictions, truths):
    """Small deterministic validation-only grid search over decoder parameters."""

    best_settings = None
    best_score = -float("inf")
    if family is ModelFamily.UNET_WATERSHED:
        candidates = (
            UNetWatershedSettings(
                interior_threshold=interior,
                centre_threshold=centre,
                centre_minimum_separation_px=separation,
                pattern_boundary_discount=discount,
                foreground_erosion_px=erosion,
            )
            for interior in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
            for centre in (0.20, 0.35, 0.50)
            for separation in (8.0, 12.0, 16.0)
            for discount in (0.0, 0.5, 0.8)
            for erosion in (0, 1, 2)
        )
    else:
        candidates = (
            StarDistDecodeSettings(
                object_threshold=threshold,
                nms_iou_threshold=nms,
                local_maximum_radius_px=radius,
            )
            for threshold in (0.30, 0.40, 0.50, 0.60)
            for nms in (0.20, 0.35, 0.50)
            for radius in (2, 3, 4)
        )
    for settings in candidates:
        current = []
        for outputs, truth in zip(predictions, truths, strict=True):
            result = _decode(family, outputs, settings, "validation-search")
            current.append(evaluate_instances(truth, result.labels).to_dict())
        value = _score(current)
        if value > best_score:
            best_score = value
            best_settings = settings
    return best_settings, best_score


def load_decoder_settings(path: Path | str, family: ModelFamily):
    """Load frozen decoder settings from an evaluation report or plain JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("decoder_settings", payload)
    if not isinstance(values, dict):
        raise ValueError("Decoder settings JSON must contain an object.")
    setting_type = (
        UNetWatershedSettings
        if family is ModelFamily.UNET_WATERSHED
        else StarDistDecodeSettings
    )
    return setting_type(**values)


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    split: str = "test",
    tile_size: int = 512,
    overlap: int = 96,
    decoder_settings=None,
    optimize_decoder: bool = False,
    device: str = "cuda",
) -> dict:
    """Evaluate complete held-out images and render auditable comparisons."""

    import torch

    resolved_device = torch.device(
        "cuda" if device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
    )
    if device == "cuda" and resolved_device.type != "cuda":
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    model, feature_spec, payload = load_checkpoint(checkpoint_path, device=resolved_device)
    family = ModelFamily(payload["family"])
    manifest = LearningManifest.load(manifest_path)
    if feature_spec != manifest.feature_spec:
        raise ValueError("Checkpoint and manifest feature specifications differ.")
    samples = [item for item in manifest.samples if item.split == split]
    if not samples:
        raise ValueError(f"The manifest has no samples in split {split!r}.")
    predictions = []
    truths = []
    timings = []
    for sample in samples:
        features = _load_features(manifest_path, sample.features, resolved_device)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        outputs = tiled_predict(
            model,
            features,
            tile_size=tile_size,
            overlap=overlap,
            use_mixed_precision=True,
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(perf_counter() - started)
        predictions.append({name: value.cpu() for name, value in outputs.items()})
        truths.append(read_label_image(resolve_sample_path(manifest_path, sample.instances)))
    optimized_score = None
    if optimize_decoder:
        if split != "validation":
            raise ValueError("Decoder optimization is permitted only on the validation split.")
        decoder_settings, optimized_score = decoder_search(family, predictions, truths)
    if decoder_settings is None:
        decoder_settings = (
            UNetWatershedSettings()
            if family is ModelFamily.UNET_WATERSHED
            else StarDistDecodeSettings()
        )
    panels = []
    records = []
    for sample, outputs, truth, elapsed in zip(
        samples, predictions, truths, timings, strict=True
    ):
        result = _decode(family, outputs, decoder_settings, checkpoint_path.name)
        metrics = evaluate_instances(truth, result.labels).to_dict()
        record = {
            "sample_id": sample.identifier,
            "species": sample.species,
            "group": sample.group,
            "reviewed": sample.reviewed,
            "elapsed_seconds": elapsed,
            **metrics,
        }
        if family is ModelFamily.UNET_WATERSHED:
            physical_probability = outputs["physical_boundary_logits"][0, 0].sigmoid().numpy()
            physical_truth = physical_boundary_mask(truth, width=2)
            physical_metrics = evaluate_binary_probability(
                physical_probability, physical_truth
            ).to_dict()
            record.update(
                {f"physical_head_{name}": value for name, value in physical_metrics.items()}
            )
            if sample.pattern_boundary and sample.pattern_valid:
                pattern_truth = read_label_image(
                    resolve_sample_path(manifest_path, sample.pattern_boundary)
                )
                pattern_valid = read_label_image(
                    resolve_sample_path(manifest_path, sample.pattern_valid)
                )
                pattern_probability = outputs["pattern_boundary_logits"][0, 0].sigmoid().numpy()
                pattern_metrics = evaluate_binary_probability(
                    pattern_probability,
                    pattern_truth,
                    valid_mask=pattern_valid,
                ).to_dict()
                record.update(
                    {f"pattern_head_{name}": value for name, value in pattern_metrics.items()}
                )
        records.append(record)
        image = _load_display_image(manifest_path, sample, truth.shape)
        title = (
            f"{sample.identifier} | true {metrics['true_instances']} / predicted "
            f"{metrics['predicted_instances']} | F1 {metrics['f1']:.3f} | "
            f"PQ {metrics['panoptic_quality']:.3f}"
        )
        panel = _render_comparison(image, truth, result.labels, title)
        panels.append(panel)
        cv2.imwrite(str(output_directory / f"{sample.identifier}.comparison.jpg"), panel)
        write_path = output_directory / f"{sample.identifier}.prediction.png"
        from seedvision.learning.data import write_label_image

        write_label_image(write_path, result.labels)
        diagnostic_directory = output_directory / f"{sample.identifier}.rasters"
        diagnostic_directory.mkdir(parents=True, exist_ok=True)
        for name, raster in result.rasters.items():
            values = raster.cpu_array() if hasattr(raster, "cpu_array") else np.asarray(raster)
            values = np.squeeze(values)
            if values.ndim == 2:
                cv2.imwrite(
                    str(diagnostic_directory / f"{name}.png"),
                    np.asarray(values, dtype=np.uint8),
                )
    contact_sheet = _contact_sheet(panels)
    cv2.imwrite(str(output_directory / "contact_sheet.jpg"), contact_sheet)
    aggregate = {
        key: mean(float(item[key]) for item in records)
        for key in (
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "panoptic_quality",
            "relative_count_error",
            "boundary_precision",
            "boundary_recall",
            "boundary_f1",
        )
    }
    dense_aggregate = {}
    if family is ModelFamily.UNET_WATERSHED:
        for prefix in ("physical_head_", "pattern_head_"):
            for name in ("precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"):
                key = prefix + name
                available = [float(item[key]) for item in records if key in item and np.isfinite(item[key])]
                if available:
                    dense_aggregate[key] = mean(available)
    report = {
        "family": family.value,
        "checkpoint": str(checkpoint_path),
        "dataset_id": manifest.dataset_id,
        "split": split,
        "decoder_settings": asdict(decoder_settings),
        "decoder_search_score": optimized_score,
        "sample_count": len(records),
        "reviewed_sample_count": sum(item["reviewed"] for item in records),
        "aggregate_mean": aggregate,
        "dense_head_aggregate_mean": dense_aggregate,
        "mean_inference_seconds": mean(timings),
        "samples": records,
        "scientifically_validated": bool(
            manifest.scientific_validation_eligible
            and split == "test"
            and all(item["reviewed"] for item in records)
        ),
        "validation_caveat": (
            "A reviewed test split is necessary but scientific publication also "
            "requires representative sampling, annotation reliability, confidence "
            "intervals, and a predeclared analysis protocol."
        ),
        "visual_legend": (
            "Panels are raw image, boundary comparison, and predicted identities. "
            "Truth is green, prediction magenta, and agreement yellow."
        ),
    }
    (output_directory / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
