"""Pretrained Cellpose-SAM seed instances, an optional third-party backend.

Cellpose-SAM predicts, for every pixel, a flow towards its object's centre and
a cell probability, using a SAM ViT-L backbone; masks are recovered by
following the flows. The pretrained ``cpsam_v2`` weights run zero-shot on the
calibrated RGB crop, so no project training is needed.

The ``cellpose`` package is optional. It is installed into the project
environment without its ``opencv-python-headless`` requirement, which would
replace the OpenCV build the application uses (see docs/OPERATOR_GUIDE.md).

Network inference and mask recovery are separate cache stages, so threshold
edits reuse the flows. Cellpose's API accepts and returns NumPy arrays, so the
corrected crop crosses to the CPU at this library boundary and the cached flows
stay in Cellpose's NumPy form; the network and flow following run on the
analysis device.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from threading import RLock

import numpy as np

from seedvision.learning.contracts import LearnedInstanceResult, PretrainedModelFamily
from seedvision.learning.targets import relabel_consecutive
from seedvision.timing import AnalysisCancelled


DEFAULT_MODEL = "cpsam_v2"
# Cellpose resizes an image so a supplied object diameter becomes 30 px.
_CELLPOSE_DIAMETER_PX = 30.0
_FLOW_ITERATIONS = 200
INSTALL_HINT = (
    "Install it into the project environment without its headless OpenCV "
    "dependency; see Cellpose-SAM in docs/OPERATOR_GUIDE.md."
)


@dataclass(frozen=True, slots=True)
class CellposeSamSettings:
    pretrained_model: str = DEFAULT_MODEL
    rescale_to_seed_diameter: bool = False
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    minimum_instance_area_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not str(self.pretrained_model).strip():
            raise ValueError("A Cellpose model name or weights path is required.")
        if not 0.0 <= self.flow_threshold <= 3.0:
            raise ValueError("Cellpose flow-error threshold must be between 0 and 3.")
        if not -6.0 <= self.cellprob_threshold <= 6.0:
            raise ValueError("Cellpose cell-probability threshold must be between -6 and 6.")
        if not 0.0 < self.minimum_instance_area_fraction <= 1.5:
            raise ValueError("Cellpose minimum area must be a positive seed-relative fraction.")

    @property
    def inference_signature(self):
        return (self.pretrained_model, self.rescale_to_seed_diameter)

    @property
    def decoder_signature(self):
        return (
            self.flow_threshold,
            self.cellprob_threshold,
            self.minimum_instance_area_fraction,
        )


def cellpose_installed() -> bool:
    """Cheap check that never imports the package or its PyTorch models."""

    return importlib.util.find_spec("cellpose") is not None


def model_directory() -> Path:
    """Cellpose's own weights cache, honouring its environment override."""

    configured = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    return Path(configured) if configured else Path.home() / ".cellpose" / "models"


def resolve_weights(selection: str, root: Path | str | None = None) -> Path:
    """Weights file a selection names: a path, else a name in Cellpose's cache."""

    candidate = Path(str(selection).strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if root is not None and (Path(root) / candidate).is_file():
        return (Path(root) / candidate).resolve()
    return (model_directory() / candidate).resolve()


_LOCK = RLock()
_INFERENCE_LOCK = RLock()
_DIGESTS: dict[tuple[str, int, int], str] = {}
_MODEL: dict[tuple[str, str, str], object] = {}


def weights_sha256(path: Path) -> str:
    """Content digest, hashed once per file version (the weights are ~1.2 GB)."""

    from seedvision.learning.data import file_sha256

    stat = Path(path).stat()
    key = (str(path).casefold(), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        digest = _DIGESTS.get(key)
    if digest is None:
        digest = file_sha256(path)
        with _LOCK:
            _DIGESTS[key] = digest
    return digest


def _weights_for_run(selection: str, root: Path) -> Path:
    path = resolve_weights(selection, root)
    if path.is_file():
        return path
    from cellpose import models

    name = str(selection).strip()
    if name not in models.MODEL_NAMES:
        # Cellpose would silently substitute its default model for an unknown name.
        raise FileNotFoundError(
            f"Cellpose weights not found: {path}. Use a pretrained name such as "
            f"{DEFAULT_MODEL!r} or the path of a Cellpose weights file."
        )
    # A pretrained name downloads once into Cellpose's cache (about 1.2 GB).
    return Path(models.cache_model_path(name)).resolve()


def _model(path: Path, digest: str, device):
    key = (str(path).casefold(), digest, str(device))
    with _LOCK:
        model = _MODEL.get(key)
        if model is None:
            from cellpose import models

            # Keep one model resident: each copy holds about 1.2 GB of GPU memory.
            _MODEL.clear()
            model = models.CellposeModel(pretrained_model=str(path), device=device)
            _MODEL[key] = model
        return model


def predict_cellpose_sam(
    settings: CellposeSamSettings,
    *,
    root: Path,
    source_bgr,
    seed_diameter_px: float,
    cancellation_requested=None,
):
    """Run the network once; return its flows and the weights producer id."""

    import torch

    if cancellation_requested is not None and cancellation_requested():
        raise AnalysisCancelled("Analysis superseded before Cellpose-SAM inference.")
    path = _weights_for_run(settings.pretrained_model, root)
    digest = weights_sha256(path)
    model = _model(path, digest, source_bgr.device)
    tensor = source_bgr[0] if source_bgr.ndim == 4 else source_bgr
    # BGR float NCHW on the analysis device -> uint8 RGB HWC for Cellpose.
    rgb = (
        tensor.flip(0).permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8)
        .contiguous().cpu().numpy()
    )
    diameter = float(seed_diameter_px) if settings.rescale_to_seed_diameter else None
    # One shared model; concurrent image analyses take turns.
    with _INFERENCE_LOCK:
        _, flows, _ = model.eval(
            rgb, channel_axis=2, diameter=diameter, compute_masks=False, batch_size=8
        )
    if cancellation_requested is not None and cancellation_requested():
        raise AnalysisCancelled("Analysis superseded after Cellpose-SAM inference.")
    outputs = {
        "flows": np.asarray(flows[1], dtype=np.float32),
        "cell_probability_logits": np.asarray(flows[2], dtype=np.float32),
        "diameter_px": diameter,
        "device": str(model.device),
    }
    return outputs, f"{path}#sha256={digest}"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def decode_cellpose_sam(
    outputs,
    *,
    settings: CellposeSamSettings,
    seed_diameter_px: float,
    valid_mask: np.ndarray | None = None,
    checkpoint_id: str = "",
) -> LearnedInstanceResult:
    """Follow the cached flows into masks clipped to the detected dish."""

    import torch
    from cellpose import dynamics

    flows = outputs["flows"]
    logits = outputs["cell_probability_logits"]
    shape = logits.shape
    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask) > 0
    if flows.shape != (2, *shape) or valid.shape != shape:
        raise ValueError("Cellpose flows, probability and dish mask must share one frame.")
    minimum_area = max(
        1, round(seed_diameter_px * seed_diameter_px * settings.minimum_instance_area_fraction)
    )
    diameter = outputs["diameter_px"]
    # Cellpose scales flow-following steps with its diameter rescaling.
    iterations = (
        _FLOW_ITERATIONS
        if diameter is None
        else max(1, int(_FLOW_ITERATIONS * float(diameter) / _CELLPOSE_DIAMETER_PX))
    )
    labels = dynamics.compute_masks(
        flows,
        logits,
        niter=iterations,
        cellprob_threshold=settings.cellprob_threshold,
        flow_threshold=settings.flow_threshold,
        min_size=minimum_area,
        device=torch.device(outputs["device"]),
    ).astype(np.int32)
    probability = _sigmoid(logits).astype(np.float32)
    labels = relabel_consecutive(labels)
    count = int(labels.max(initial=0))
    rows, columns = np.indices(shape, dtype=np.float32)

    def moments(identifiers):
        area = np.bincount(identifiers, minlength=count + 1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            centre_x = np.bincount(identifiers, weights=columns.ravel(), minlength=count + 1) / area
            centre_y = np.bincount(identifiers, weights=rows.ravel(), minlength=count + 1) / area
            confidence = np.bincount(identifiers, weights=probability.ravel(), minlength=count + 1) / area
        return area, np.nan_to_num(centre_x), np.nan_to_num(centre_y), confidence

    # Judge each object by its whole-mask centre: clipping first would pull
    # every rim fragment's centre inside the dish.
    _, whole_x, whole_y, _ = moments(labels.ravel())
    inside = valid[
        np.clip(np.rint(whole_y), 0, shape[0] - 1).astype(np.intp),
        np.clip(np.rint(whole_x), 0, shape[1] - 1).astype(np.intp),
    ]
    labels[~valid] = 0
    area, centre_x, centre_y, confidence = moments(labels.ravel())
    # Objects centred outside the dish, or clipped below the minimum area, are
    # not dish seeds.
    keep = (area >= minimum_area) & inside
    keep[0] = False
    mapping = np.zeros(count + 1, dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labels = mapping[labels]
    centres = np.column_stack((centre_x[keep], centre_y[keep])).astype(np.float32)
    return LearnedInstanceResult(
        family=PretrainedModelFamily.CELLPOSE_SAM,
        labels=labels,
        centres_xy=centres,
        instance_confidences=np.clip(confidence[keep], 0.0, 1.0).astype(np.float32),
        rasters={"cell_probability": np.uint8(np.rint(probability * 255.0))},
        method="Cellpose-SAM zero-shot flow-field instances (review required)",
        checkpoint_id=checkpoint_id,
    )
