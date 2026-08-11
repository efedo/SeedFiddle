"""Versioned, self-describing learned-model checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from seedvision.learning.contracts import FeatureStackSpec, ModelFamily
from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D


CHECKPOINT_VERSION = 1


def _model_configuration(model) -> dict[str, Any]:
    configuration = getattr(model, "configuration", None)
    if configuration is None:
        raise ValueError("The model does not expose a checkpoint configuration.")
    return asdict(configuration)


def save_checkpoint(
    path: Path | str,
    model,
    *,
    family: ModelFamily | str,
    feature_spec: FeatureStackSpec,
    training_metadata: dict[str, Any] | None = None,
    optimizer=None,
) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "family": str(ModelFamily(family)),
        "feature_spec": asdict(feature_spec),
        "model_configuration": _model_configuration(model),
        "model_state": model.state_dict(),
        "training_metadata": dict(training_metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(path: Path | str, *, device="cpu"):
    import torch

    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported learned-model checkpoint: {source}")
    family = ModelFamily(payload["family"])
    feature_spec = FeatureStackSpec(**payload["feature_spec"])
    configuration = dict(payload["model_configuration"])
    if int(configuration["input_channels"]) != feature_spec.input_channels:
        raise ValueError("Checkpoint feature and model input-channel counts disagree.")
    if family is ModelFamily.UNET_WATERSHED:
        model = MultiHeadSeedUNet(
            configuration["input_channels"],
            base_channels=configuration["base_channels"],
            depth=configuration["depth"],
        )
    else:
        model = SeedStarDist2D(
            configuration["input_channels"],
            ray_count=configuration["ray_count"],
            base_channels=configuration["base_channels"],
            depth=configuration["depth"],
        )
    model.load_state_dict(payload["model_state"])
    model.to(device=device)
    model.eval()
    return model, feature_spec, payload
