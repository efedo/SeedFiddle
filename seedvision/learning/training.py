"""Deterministic training and validation orchestration for learned models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from seedvision.learning.checkpoint import load_checkpoint, save_checkpoint
from seedvision.learning.contracts import ModelFamily
from seedvision.learning.data import (
    LearningManifest,
    SeedTileDataset,
    audit_manifest,
    file_sha256,
)
from seedvision.learning.losses import multi_head_unet_loss, stardist_loss
from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D


class TrainingCancelled(RuntimeError):
    """Raised when an interactive training run is cancelled safely."""


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    family: str
    epochs: int = 40
    batch_size: int = 4
    tile_size: int = 256
    tiles_per_sample: int = 16
    ray_count: int = 32
    base_channels: int = 24
    depth: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    random_seed: int = 20260811
    patience: int = 8
    minimum_improvement: float = 1e-4
    device: str = "cuda"
    mixed_precision: bool = True
    allow_unreviewed: bool = False
    pattern_boundary_loss_weight: float = 0.8
    initial_checkpoint: str | None = None

    def __post_init__(self) -> None:
        ModelFamily(self.family)
        if self.epochs < 1 or self.batch_size < 1 or self.tiles_per_sample < 1:
            raise ValueError("Epoch, batch, and tile counts must be positive.")
        if self.tile_size < 32 or self.patience < 1:
            raise ValueError("Tile size must be >=32 and patience must be positive.")
        if self.base_channels < 8:
            raise ValueError("Base channels must be at least 8.")
        if not 1 <= self.depth <= 5:
            raise ValueError("Network depth must be between 1 and 5.")
        if not 8 <= self.ray_count <= 256 or self.ray_count % 4:
            raise ValueError("StarDist ray count must be 8-256 and divisible by four.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("Training device must be cuda, cpu, or auto.")
        if self.pattern_boundary_loss_weight < 0:
            raise ValueError("Pattern-boundary loss weight cannot be negative.")


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("CUDA training was requested but PyTorch cannot access CUDA.")
    return torch.device("cpu")


def _build_model(configuration: TrainingConfiguration, input_channels: int):
    family = ModelFamily(configuration.family)
    if family is ModelFamily.UNET_WATERSHED:
        return MultiHeadSeedUNet(
            input_channels,
            base_channels=configuration.base_channels,
            depth=configuration.depth,
        )
    return SeedStarDist2D(
        input_channels,
        ray_count=configuration.ray_count,
        base_channels=configuration.base_channels,
        depth=configuration.depth,
    )


def _loss(configuration: TrainingConfiguration, family: ModelFamily, outputs, targets):
    return (
        multi_head_unet_loss(
            outputs,
            targets,
            weights={"pattern_boundary": configuration.pattern_boundary_loss_weight},
        )
        if family is ModelFamily.UNET_WATERSHED
        else stardist_loss(outputs, targets)
    )


def _move_targets(targets, device):
    return {
        name: values.to(device=device, dtype=None, non_blocking=True)
        for name, values in targets.items()
    }


def _epoch(
    model,
    loader,
    *,
    configuration,
    family,
    device,
    optimizer=None,
    scaler=None,
    mixed_precision=False,
    cancellation_requested: Callable[[], bool] | None = None,
):
    import torch

    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            if cancellation_requested is not None and cancellation_requested():
                raise TrainingCancelled("Learned-model training was cancelled.")
            features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                outputs = model(features)
                total, terms = _loss(configuration, family, outputs, targets)
            if training:
                if scaler is None:
                    total.backward()
                    optimizer.step()
                else:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
            batch_size = int(features.shape[0])
            sample_count += batch_size
            totals["total"] = totals.get("total", 0.0) + float(total.detach()) * batch_size
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / max(1, sample_count) for name, value in totals.items()}


def _train_from_manifest_impl(
    manifest_path: Path | str,
    output_path: Path | str,
    configuration: TrainingConfiguration,
    *,
    report_path: Path | str | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> dict:
    """Train one model and save only the best validation-loss checkpoint."""

    import torch

    family = ModelFamily(configuration.family)
    audit = audit_manifest(manifest_path)
    if not audit["valid"]:
        raise ValueError("Learning dataset audit failed: " + "; ".join(audit["errors"]))
    manifest = LearningManifest.load(manifest_path)
    supervised = [item for item in manifest.samples if item.split in {"train", "validation"}]
    if not configuration.allow_unreviewed and any(not item.reviewed for item in supervised):
        raise ValueError(
            "Training and validation samples must be marked human-reviewed; "
            "set allow_unreviewed only for an explicitly non-scientific experiment."
        )
    _seed_everything(configuration.random_seed)
    device = _resolve_device(configuration.device)
    train_dataset = SeedTileDataset(
        manifest_path,
        split="train",
        family=family,
        tile_size=configuration.tile_size,
        tiles_per_sample=configuration.tiles_per_sample,
        ray_count=configuration.ray_count,
        augment=True,
        random_seed=configuration.random_seed,
    )
    validation_dataset = SeedTileDataset(
        manifest_path,
        split="validation",
        family=family,
        tile_size=configuration.tile_size,
        tiles_per_sample=max(2, configuration.tiles_per_sample // 2),
        ray_count=configuration.ray_count,
        augment=False,
        random_seed=configuration.random_seed + 1,
    )
    generator = torch.Generator().manual_seed(configuration.random_seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = _build_model(configuration, manifest.feature_spec.input_channels).to(device)
    if configuration.initial_checkpoint:
        initial_model, initial_spec, initial_payload = load_checkpoint(
            configuration.initial_checkpoint, device=device
        )
        if ModelFamily(initial_payload["family"]) is not family:
            raise ValueError("Initial checkpoint belongs to a different model family.")
        if initial_spec != manifest.feature_spec:
            raise ValueError("Initial checkpoint and manifest feature specifications differ.")
        if initial_model.configuration != model.configuration:
            raise ValueError("Initial checkpoint architecture differs from the training configuration.")
        model.load_state_dict(initial_model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, configuration.epochs)
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if configuration.mixed_precision and device.type == "cuda"
        else None
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = perf_counter()
    for epoch in range(1, configuration.epochs + 1):
        if cancellation_requested is not None and cancellation_requested():
            raise TrainingCancelled("Learned-model training was cancelled.")
        train_terms = _epoch(
            model,
            train_loader,
            family=family,
            configuration=configuration,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=configuration.mixed_precision,
            cancellation_requested=cancellation_requested,
        )
        validation_terms = _epoch(
            model,
            validation_loader,
            family=family,
            configuration=configuration,
            device=device,
            mixed_precision=configuration.mixed_precision,
            cancellation_requested=cancellation_requested,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_terms,
            "validation": validation_terms,
        }
        history.append(record)
        validation_loss = validation_terms["total"]
        if validation_loss < best_loss - configuration.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_path,
                model,
                family=family,
                feature_spec=manifest.feature_spec,
                training_metadata={
                    "dataset_id": manifest.dataset_id,
                    "manifest_sha256": file_sha256(manifest_path),
                    "configuration": asdict(configuration),
                    "best_epoch": epoch,
                    "best_validation_loss": validation_loss,
                    "scientifically_validated": False,
                },
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:03d}: train={train_terms['total']:.5f}; "
            f"validation={validation_loss:.5f}; best={best_loss:.5f}",
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(epoch, configuration.epochs, record)
        if epochs_without_improvement >= configuration.patience:
            break
    report = {
        "family": family.value,
        "dataset_audit": audit,
        "configuration": asdict(configuration),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": perf_counter() - started,
        "checkpoint": str(Path(output_path).resolve()),
        "history": history,
        "scientifically_validated": False,
        "validation_note": (
            "Training/validation loss is software evidence only; a locked, "
            "human-reviewed image-level test evaluation is still required."
        ),
    }
    destination = Path(report_path) if report_path else Path(output_path).with_suffix(".training.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train_from_manifest(
    manifest_path: Path | str,
    output_path: Path | str,
    configuration: TrainingConfiguration,
    *,
    report_path: Path | str | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> dict:
    """Train reproducibly without leaking deterministic mode into live analysis."""

    import torch

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        return _train_from_manifest_impl(
            manifest_path,
            output_path,
            configuration,
            report_path=report_path,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
