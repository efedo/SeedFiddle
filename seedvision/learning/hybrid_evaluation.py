"""Controlled evaluation of U-Net boundaries with gated StarDist markers."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2

from seedvision.learning.checkpoint import load_checkpoint
from seedvision.learning.contracts import ModelFamily
from seedvision.learning.data import LearningManifest, read_label_image, resolve_sample_path
from seedvision.learning.decode import (
    HybridDecodeSettings,
    StarDistDecodeSettings,
    UNetWatershedSettings,
    decode_unet_stardist_hybrid,
)
from seedvision.learning.evaluation import (
    _contact_sheet,
    _load_display_image,
    _load_features,
    _render_comparison,
    _score,
)
from seedvision.learning.inference import tiled_predict
from seedvision.learning.evaluation_protocol import aggregate_records, decoder_identity, audit_decoder_source
from seedvision.learning.resources import PredictionSpool, LabelSequence, append_thumbnail
from seedvision.learning.metrics import evaluate_instances
from seedvision.learning.data import write_label_image
from seedvision.learning.data import file_sha256
from seedvision.learning.evaluation_protocol import audit_evaluation, dataset_identity, checkpoint_membership


def evaluate_hybrid_checkpoints(
    unet_checkpoint: Path | str,
    stardist_checkpoint: Path | str,
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    split: str,
    unet_settings: UNetWatershedSettings = UNetWatershedSettings(),
    stardist_settings: StarDistDecodeSettings = StarDistDecodeSettings(),
    gate_threshold: float = 0.50,
    distance_threshold: float = 0.35,
    optimize_decoder: bool = False,
    tile_size: int = 512,
    overlap: int = 96,
    device: str = "cuda",
    maximum_prediction_bytes: int = 8*1024**3,
    protocol: str = "development",
    decoder_source_path: Path | str | None = None,
) -> dict:
    import torch

    resolved_device = torch.device(
        "cuda" if device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
    )
    if device == 'cuda' and resolved_device.type != 'cuda':
        raise RuntimeError('CUDA evaluation was requested but CUDA is unavailable.')
    unet_checkpoint = Path(unet_checkpoint).resolve()
    stardist_checkpoint = Path(stardist_checkpoint).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    audit = audit_evaluation(manifest_path, split, optimize_decoder, protocol)
    identity = dataset_identity(manifest_path, splits={split})
    unet_model, unet_spec, unet_payload = load_checkpoint(
        unet_checkpoint, device=resolved_device
    )
    star_model, star_spec, star_payload = load_checkpoint(
        stardist_checkpoint, device=resolved_device
    )
    membership = [checkpoint_membership(payload, identity) for payload in (unet_payload, star_payload)] if split == 'test' else ['Development evaluation']
    if protocol != "development" and split == "test" and any(value.startswith("Unknown") for value in membership):
        raise ValueError("Independent evaluation requires checkpoint development membership.")
    if ModelFamily(unet_payload["family"]) is not ModelFamily.UNET_WATERSHED:
        raise ValueError("The first hybrid checkpoint must be U-Net + watershed.")
    if ModelFamily(star_payload["family"]) is not ModelFamily.STARDIST:
        raise ValueError("The second hybrid checkpoint must be StarDist.")
    manifest = LearningManifest.load(manifest_path)
    if unet_spec != star_spec or unet_spec != manifest.feature_spec:
        raise ValueError("Hybrid checkpoints and manifest must share a feature specification.")
    samples = [sample for sample in manifest.samples if sample.split == split]
    if not samples:
        raise ValueError(f"No manifest samples use split {split!r}.")
    settings = HybridDecodeSettings(
        unet=unet_settings,
        stardist=stardist_settings,
        stardist_gate_interior_threshold=gate_threshold,
        stardist_gate_distance_threshold=distance_threshold,
    )
    decoder_provenance = None
    if decoder_source_path is not None:
        values = json.loads(Path(decoder_source_path).read_text(encoding='utf-8'))['decoder_settings']
        settings = HybridDecodeSettings(**dict(values,
            unet=UNetWatershedSettings(**values['unet']),stardist=StarDistDecodeSettings(**values['stardist'])))
        decoder_provenance = audit_decoder_source(decoder_source_path,settings,identity,
            independent=protocol!='development' and split=='test')
        unet_settings, stardist_settings = settings.unet, settings.stardist
    elif protocol != 'development' and split == 'test':
        raise ValueError('Independent hybrid evaluation requires its frozen decoder selection report.')
    predictions = PredictionSpool(maximum_prediction_bytes)
    try:
        truths = LabelSequence([resolve_sample_path(manifest_path,sample.instances) for sample in samples])
        timings = []
        for sample in samples:
            features = _load_features(manifest_path, sample.features, resolved_device)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            unet_outputs = tiled_predict(
                unet_model, features, tile_size=tile_size, overlap=overlap
            )
            star_outputs = tiled_predict(
                star_model, features, tile_size=tile_size, overlap=overlap
            )
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(perf_counter() - started)
            predictions.append(
                (
                    {name: value.cpu() for name, value in unet_outputs.items()},
                    {name: value.cpu() for name, value in star_outputs.items()},
                )
            )


        search_score = None
        search_trials = []
        if optimize_decoder:
            if split != "validation":
                raise ValueError("Hybrid decoder optimization is restricted to validation data.")
            candidates = (
                HybridDecodeSettings(
                    unet=unet_settings,
                    stardist=replace(
                        stardist_settings,
                        object_threshold=object_threshold,
                        nms_iou_threshold=nms,
                        local_maximum_radius_px=radius,
                    ),
                    stardist_gate_interior_threshold=gate,
                    stardist_gate_distance_threshold=distance,
                )
                for gate in (0.45, 0.60, 0.75)
                for distance in (0.20, 0.35, 0.50, 0.65)
                for object_threshold in (0.30, 0.50, 0.70)
                for nms in (0.20, 0.35)
                for radius in (4, 8, 12)
            )
            best_value = -float("inf")
            for candidate in candidates:
                metrics = []
                for (unet_outputs, star_outputs), truth in zip(
                    predictions, truths, strict=True
                ):
                    result = decode_unet_stardist_hybrid(
                        unet_outputs, star_outputs, settings=candidate
                    )
                    metrics.append(evaluate_instances(truth, result.labels).to_dict())
                value = _score(metrics)
                search_trials.append({"score": value, "settings": asdict(candidate)})
                if value > best_value:
                    best_value = value
                    settings = candidate
            search_score = best_value
            search_trials.sort(key=lambda item: item["score"], reverse=True)

        records = []
        panels = []
        for sample, (unet_outputs, star_outputs), truth, elapsed in zip(
            samples, predictions, truths, timings, strict=True
        ):
            result = decode_unet_stardist_hybrid(
                unet_outputs,
                star_outputs,
                settings=settings,
                checkpoint_id=f"{unet_checkpoint.name}+{stardist_checkpoint.name}",
            )
            metrics = evaluate_instances(truth, result.labels).to_dict()
            records.append(
                {
                    "sample_id": sample.identifier,
                    "species": sample.species,
                    "group": sample.group,
                    "reviewed": sample.reviewed,
                    "elapsed_seconds": elapsed,
                    **metrics,
                }
            )
            image = _load_display_image(manifest_path, sample, truth.shape)
            panel = _render_comparison(
                image,
                truth,
                result.labels,
                f"{sample.identifier} | true {metrics['true_instances']} / predicted "
                f"{metrics['predicted_instances']} | F1 {metrics['f1']:.3f} | "
                f"PQ {metrics['panoptic_quality']:.3f}",
            )
            append_thumbnail(panels,panel)
            cv2.imwrite(str(output_directory / f"{sample.identifier}.comparison.jpg"), panel)
            write_label_image(
                output_directory / f"{sample.identifier}.prediction.png", result.labels
            )
        cv2.imwrite(str(output_directory / "contact_sheet.jpg"), _contact_sheet(panels))
        metric_names = (
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
        report = {
            "family": "unet_stardist_hybrid",
            "unet_checkpoint": str(unet_checkpoint),
            "stardist_checkpoint": str(stardist_checkpoint),
            "dataset_id": manifest.dataset_id,
            "split": split,
            "decoder_settings": asdict(settings),
            "decoder_sha256": decoder_identity(settings),
            "decoder_search_score": search_score,
            "decoder_search_top_trials": search_trials[:10],
            "sample_count": len(records),
            "aggregate_mean": {
                name: (mean(float(record[name]) for record in records if record[name] is not None) if any(record[name] is not None for record in records) else None)
                for name in metric_names
            },
            "mean_inference_seconds": mean(timings),
            "samples": records,
            "aggregate_grouped": aggregate_records(records),
            "scientifically_validated": False,
            "evaluation_protocol": protocol,
            "decoder_selection_provenance": decoder_provenance,
            "contact_sheet_limit": 12,
            "prediction_spool_bytes": predictions.bytes,
            'evaluation_state': 'reviewed split evaluated' if all(record['reviewed'] for record in records) else 'unreviewed development evaluation',
            'dataset_audit': audit,
            'dataset_identity': identity,
            'checkpoint_sha256': [file_sha256(path) for path in (unet_checkpoint,stardist_checkpoint)],
            'checkpoint_membership_audit': membership,
            "validation_caveat": (
                "Synthetic or unreviewed data cannot establish scientific validity."
            ),
        }
        (output_directory / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return report
    finally:
        predictions.close()
