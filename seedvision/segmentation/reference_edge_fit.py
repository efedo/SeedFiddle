"""Deterministic annotation-guided fitting for reference edge probabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from seedvision.visualization.layers import AnalysisLayerSettings


@dataclass(frozen=True, slots=True)
class ReferenceEdgeFitParameter:
    name: str
    minimum: float
    maximum: float
    initial_step: float


DEFAULT_REFERENCE_EDGE_FIT_PARAMETERS = (
    ReferenceEdgeFitParameter("reference_edge_similarity_scale", 0.35, 2.5, 0.30),
    ReferenceEdgeFitParameter("reference_edge_strip_normal_offset_fraction", 0.02, 0.14, 0.02),
    ReferenceEdgeFitParameter("reference_edge_strip_tangent_half_length_fraction", 0.025, 0.18, 0.025),
    ReferenceEdgeFitParameter("reference_edge_ridge_weight", 0.0, 0.85, 0.15),
    ReferenceEdgeFitParameter("reference_texture_instance_interior_buffer_fraction", 0.03, 0.24, 0.035),
)


@dataclass(frozen=True, slots=True)
class ReferenceEdgeFitScore:
    loss: float
    physical_recall: float
    nonphysical_recall: float
    physical_cross_match: float
    nonphysical_cross_match: float
    physical_samples: int
    nonphysical_samples: int


@dataclass(frozen=True, slots=True)
class ReferenceEdgeFitTrial:
    evaluation_number: int
    pass_number: int
    parameter_name: str
    parameter_value: float
    score: ReferenceEdgeFitScore


@dataclass(frozen=True, slots=True)
class ReferenceEdgeFitResult:
    initial_settings: AnalysisLayerSettings
    proposed_settings: AnalysisLayerSettings
    initial_score: ReferenceEdgeFitScore
    proposed_score: ReferenceEdgeFitScore
    trials: tuple[ReferenceEdgeFitTrial, ...]
    cancelled: bool = False

    @property
    def evaluations(self) -> int:
        return len(self.trials)

    @property
    def improved(self) -> bool:
        return self.proposed_score.loss < self.initial_score.loss - 1e-9


def score_reference_edge_probabilities(
    physical_probability: np.ndarray,
    nonphysical_probability: np.ndarray,
    physical_target: np.ndarray,
    nonphysical_target: np.ndarray,
) -> ReferenceEdgeFitScore:
    """Score correct class support and symmetric cross-matches on reviewed pixels."""

    physical = np.asarray(physical_probability, dtype=np.float32)
    nonphysical = np.asarray(nonphysical_probability, dtype=np.float32)
    if physical.size and float(np.max(physical)) > 1.0:
        physical = physical / 255.0
    if nonphysical.size and float(np.max(nonphysical)) > 1.0:
        nonphysical = nonphysical / 255.0
    physical_target = np.asarray(physical_target, dtype=bool)
    nonphysical_target = np.asarray(nonphysical_target, dtype=bool)
    if not (
        physical.shape
        == nonphysical.shape
        == physical_target.shape
        == nonphysical_target.shape
    ):
        raise ValueError("Edge probabilities and annotation targets must share one shape.")
    physical_count = int(np.count_nonzero(physical_target))
    nonphysical_count = int(np.count_nonzero(nonphysical_target))
    if physical_count == 0 or nonphysical_count == 0:
        raise ValueError(
            "Edge fitting needs both annotated physical contours and internal edge "
            "candidates. Add complete patterned seed instances before fitting."
        )
    epsilon = 1e-5
    physical_values = np.clip(physical[physical_target], epsilon, 1.0 - epsilon)
    physical_wrong = np.clip(nonphysical[physical_target], epsilon, 1.0 - epsilon)
    nonphysical_values = np.clip(
        nonphysical[nonphysical_target], epsilon, 1.0 - epsilon
    )
    nonphysical_wrong = np.clip(
        physical[nonphysical_target], epsilon, 1.0 - epsilon
    )
    # Balanced class loss: correct support and cross-match rejection receive
    # equal authority regardless of contour length or internal-pattern density.
    loss = 0.25 * (
        -float(np.mean(np.log(physical_values)))
        - float(np.mean(np.log1p(-physical_wrong)))
        - float(np.mean(np.log(nonphysical_values)))
        - float(np.mean(np.log1p(-nonphysical_wrong)))
    )
    return ReferenceEdgeFitScore(
        loss=loss,
        physical_recall=float(np.mean(physical_values)),
        nonphysical_recall=float(np.mean(nonphysical_values)),
        physical_cross_match=float(np.mean(physical_wrong)),
        nonphysical_cross_match=float(np.mean(nonphysical_wrong)),
        physical_samples=physical_count,
        nonphysical_samples=nonphysical_count,
    )


def evaluate_reference_edge_settings(
    crop: np.ndarray,
    gradients,
    ridges,
    frequency_noise,
    seed_diameter_px: float,
    annotations: np.ndarray,
    settings: AnalysisLayerSettings,
    *,
    cuda_context=None,
) -> ReferenceEdgeFitScore:
    """Fit on one deterministic instance fold and score a disjoint fold.

    Pixels from a held-out instance never enter the prototype bank whose
    settings are being evaluated. This prevents the former resubstitution
    score from rewarding a descriptor for memorizing its own contour and coat
    pattern.
    """

    import cv2
    import torch

    from seedvision.annotation.instance_references import (
        instance_boundary_references_from_edge_evidence,
    )
    from seedvision.cuda.layers import reference_texture_probabilities
    from seedvision.cuda.ops import CudaContext, image_to_tensor

    context = cuda_context or CudaContext.resolve()
    annotations = np.asarray(annotations, dtype=np.uint16)
    identifiers = np.unique(annotations)
    identifiers = identifiers[identifiers > 0]
    if len(identifiers) < 2:
        raise ValueError(
            "Leakage-safe edge fitting needs at least two annotated seed "
            "instances so one can be withheld from prototype training."
        )
    # Alternate sorted IDs to make the split stable across runs and ensure
    # arbitrary label magnitudes cannot affect fold membership.
    holdout_ids = identifiers[1::2]
    if not len(holdout_ids):
        holdout_ids = identifiers[-1:]
    training_ids = identifiers[~np.isin(identifiers, holdout_ids)]
    training_annotations = np.where(
        np.isin(annotations, training_ids), annotations, 0
    ).astype(np.uint16, copy=False)
    holdout_annotations = np.where(
        np.isin(annotations, holdout_ids), annotations, 0
    ).astype(np.uint16, copy=False)
    products = reference_texture_probabilities(
        crop,
        gradients,
        ridges,
        frequency_noise,
        seed_diameter_px,
        settings,
        seed_instance_annotations=training_annotations,
        cuda_context=context,
    )
    physical = products.physical_edge_field.gpu_tensor(
        device=context.device, dtype=torch.float32
    ).clamp(1e-5, 1.0 - 1e-5)
    nonphysical = products.non_edge_field.gpu_tensor(
        device=context.device, dtype=torch.float32
    ).clamp(1e-5, 1.0 - 1e-5)
    height, width = physical.shape[-2:]
    source_height, source_width = holdout_annotations.shape
    if (height, width) == (source_height, source_width):
        working_annotations = holdout_annotations
    else:
        working_annotations = cv2.resize(
            np.asarray(holdout_annotations, dtype=np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint16)
    edge = gradients.strength
    if edge.shape[-2:] != (height, width):
        edge = torch.nn.functional.interpolate(
            edge.float(), (height, width), mode="bilinear", align_corners=False
        )
    ridge = image_to_tensor(ridges, context) / 255.0
    if ridge.shape[-2:] != (height, width):
        ridge = torch.nn.functional.interpolate(
            ridge.float(), (height, width), mode="bilinear", align_corners=False
        )
    working_diameter = float(seed_diameter_px) * min(
        width / max(1, source_width), height / max(1, source_height)
    )
    references = instance_boundary_references_from_edge_evidence(
        working_annotations,
        max(1.0, working_diameter),
        edge.detach().squeeze().cpu().numpy(),
        ridge.detach().squeeze().cpu().numpy(),
        interior_buffer_fraction=float(
            settings.reference_texture_instance_interior_buffer_fraction
        ),
    )
    physical_target = image_to_tensor(
        references.physical_edge.astype(np.uint8), context
    ).bool()
    nonphysical_target = image_to_tensor(
        references.non_edge.astype(np.uint8), context
    ).bool()
    physical_count = int(physical_target.sum().item())
    nonphysical_count = int(nonphysical_target.sum().item())
    if physical_count == 0 or nonphysical_count == 0:
        raise ValueError(
            "Edge fitting needs both contour pixels and internal patterned edges in "
            "the applied seed instances."
        )
    correct_physical = physical[physical_target]
    wrong_physical = nonphysical[physical_target]
    correct_nonphysical = nonphysical[nonphysical_target]
    wrong_nonphysical = physical[nonphysical_target]
    loss = 0.25 * (
        -torch.log(correct_physical).mean()
        - torch.log1p(-wrong_physical).mean()
        - torch.log(correct_nonphysical).mean()
        - torch.log1p(-wrong_nonphysical).mean()
    )
    return ReferenceEdgeFitScore(
        loss=float(loss.item()),
        physical_recall=float(correct_physical.mean().item()),
        nonphysical_recall=float(correct_nonphysical.mean().item()),
        physical_cross_match=float(wrong_physical.mean().item()),
        nonphysical_cross_match=float(wrong_nonphysical.mean().item()),
        physical_samples=physical_count,
        nonphysical_samples=nonphysical_count,
    )


def fit_reference_edge_parameters(
    initial_settings: AnalysisLayerSettings,
    evaluate: Callable[[AnalysisLayerSettings], ReferenceEdgeFitScore],
    *,
    maximum_evaluations: int = 21,
    passes: int = 2,
    step_decay: float = 0.5,
    progress_callback: Callable[[int, int, ReferenceEdgeFitScore], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    parameters: tuple[ReferenceEdgeFitParameter, ...] = DEFAULT_REFERENCE_EDGE_FIT_PARAMETERS,
) -> ReferenceEdgeFitResult:
    """Return a bounded coordinate-search proposal without mutating node state."""

    if maximum_evaluations < 1 or passes < 1:
        raise ValueError("Edge fitting requires positive evaluation and pass counts.")
    if not 0.0 < step_decay <= 1.0:
        raise ValueError("Edge fit step decay must be in (0, 1].")
    current = initial_settings
    initial_score = evaluate(current)
    best_score = initial_score
    trials = [
        ReferenceEdgeFitTrial(1, 0, "initial", 0.0, initial_score)
    ]
    if progress_callback is not None:
        progress_callback(1, maximum_evaluations, initial_score)
    cancelled = False
    for pass_index in range(passes):
        if len(trials) >= maximum_evaluations:
            break
        pass_improved = False
        for parameter in parameters:
            if len(trials) >= maximum_evaluations:
                break
            if cancellation_requested is not None and cancellation_requested():
                cancelled = True
                break
            base_value = float(getattr(current, parameter.name))
            step = parameter.initial_step * (step_decay**pass_index)
            candidates = []
            for direction in (-1.0, 1.0):
                value = float(
                    np.clip(
                        base_value + direction * step,
                        parameter.minimum,
                        parameter.maximum,
                    )
                )
                if isfinite(value) and abs(value - base_value) > 1e-12:
                    candidates.append(value)
            coordinate_best = current
            coordinate_score = best_score
            for value in dict.fromkeys(candidates):
                if len(trials) >= maximum_evaluations:
                    break
                proposal = replace(current, **{parameter.name: value})
                score = evaluate(proposal)
                trials.append(
                    ReferenceEdgeFitTrial(
                        len(trials) + 1,
                        pass_index + 1,
                        parameter.name,
                        value,
                        score,
                    )
                )
                if progress_callback is not None:
                    progress_callback(len(trials), maximum_evaluations, score)
                if score.loss < coordinate_score.loss - 1e-9:
                    coordinate_best = proposal
                    coordinate_score = score
            if coordinate_best is not current:
                current = coordinate_best
                best_score = coordinate_score
                pass_improved = True
        if cancelled or not pass_improved:
            break
    return ReferenceEdgeFitResult(
        initial_settings=initial_settings,
        proposed_settings=current,
        initial_score=initial_score,
        proposed_score=best_score,
        trials=tuple(trials),
        cancelled=cancelled,
    )


__all__ = (
    "DEFAULT_REFERENCE_EDGE_FIT_PARAMETERS",
    "ReferenceEdgeFitParameter",
    "ReferenceEdgeFitResult",
    "ReferenceEdgeFitScore",
    "ReferenceEdgeFitTrial",
    "fit_reference_edge_parameters",
    "evaluate_reference_edge_settings",
    "score_reference_edge_probabilities",
)
