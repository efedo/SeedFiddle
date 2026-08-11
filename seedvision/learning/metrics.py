"""Instance, boundary, and count metrics without optional dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from seedvision.learning.targets import physical_boundary_mask, relabel_consecutive


@dataclass(frozen=True, slots=True)
class InstanceMetrics:
    true_instances: int
    predicted_instances: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    panoptic_quality: float
    count_error: int
    absolute_count_error: int
    relative_count_error: float
    split_instances: int
    merged_instances: int
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinaryProbabilityMetrics:
    positive_pixels: int
    evaluated_pixels: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    brier_score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_binary_probability(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.50,
) -> BinaryProbabilityMetrics:
    """Evaluate a dense probability map with grouped-threshold ROC/PR curves."""

    probability = np.asarray(probability, dtype=np.float32)
    truth = np.asarray(truth) > 0
    if probability.shape != truth.shape:
        raise ValueError("Probability and binary truth dimensions must agree.")
    valid = np.ones(truth.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask) > 0
    if valid.shape != truth.shape:
        raise ValueError("Binary metric validity mask dimensions must agree.")
    scores = np.clip(probability[valid], 0.0, 1.0)
    expected = truth[valid]
    if not len(scores):
        return BinaryProbabilityMetrics(0, 0, 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"))
    predicted = scores >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    positives = int(np.count_nonzero(expected))
    negatives = len(expected) - positives
    roc_auc = float("nan")
    average_precision = float("nan")
    if positives and negatives:
        order = np.argsort(scores, kind="stable")[::-1]
        ordered_scores = scores[order]
        ordered_truth = expected[order]
        cumulative_true = np.cumsum(ordered_truth)
        cumulative_false = np.cumsum(~ordered_truth)
        endpoints = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
        true_positive_rate = np.r_[0.0, cumulative_true[endpoints] / positives]
        false_positive_rate = np.r_[0.0, cumulative_false[endpoints] / negatives]
        roc_auc = float(np.trapezoid(true_positive_rate, false_positive_rate))
        grouped_recall = cumulative_true[endpoints] / positives
        grouped_precision = cumulative_true[endpoints] / (
            cumulative_true[endpoints] + cumulative_false[endpoints]
        )
        average_precision = float(
            np.sum(np.diff(np.r_[0.0, grouped_recall]) * grouped_precision)
        )
    return BinaryProbabilityMetrics(
        positive_pixels=positives,
        evaluated_pixels=len(scores),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=average_precision,
        brier_score=float(np.mean((scores - expected.astype(np.float32)) ** 2)),
    )


def _overlap_iou_pairs(truth: np.ndarray, prediction: np.ndarray):
    truth_count = int(truth.max(initial=0))
    prediction_count = int(prediction.max(initial=0))
    combined = truth.astype(np.int64) * (prediction_count + 1) + prediction
    contingency = np.bincount(
        combined.reshape(-1), minlength=(truth_count + 1) * (prediction_count + 1)
    ).reshape(truth_count + 1, prediction_count + 1)
    truth_area = contingency.sum(axis=1)
    prediction_area = contingency.sum(axis=0)
    pairs = []
    for true_id, predicted_id in np.argwhere(contingency[1:, 1:] > 0) + 1:
        intersection = int(contingency[true_id, predicted_id])
        union = int(truth_area[true_id] + prediction_area[predicted_id] - intersection)
        pairs.append((intersection / max(1, union), int(true_id), int(predicted_id), intersection))
    return pairs, contingency


def _boundary_scores(truth: np.ndarray, prediction: np.ndarray, tolerance_px: float):
    true_boundary = physical_boundary_mask(truth, width=1) > 0
    predicted_boundary = physical_boundary_mask(prediction, width=1) > 0
    if not np.any(true_boundary) and not np.any(predicted_boundary):
        return 1.0, 1.0, 1.0
    true_distance = cv2.distanceTransform(np.uint8(~true_boundary), cv2.DIST_L2, 5)
    predicted_distance = cv2.distanceTransform(
        np.uint8(~predicted_boundary), cv2.DIST_L2, 5
    )
    precision = (
        float(np.mean(true_distance[predicted_boundary] <= tolerance_px))
        if np.any(predicted_boundary)
        else 0.0
    )
    recall = (
        float(np.mean(predicted_distance[true_boundary] <= tolerance_px))
        if np.any(true_boundary)
        else 0.0
    )
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def evaluate_instances(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    iou_threshold: float = 0.50,
    contact_overlap_fraction: float = 0.10,
    boundary_tolerance_px: float = 2.0,
) -> InstanceMetrics:
    truth = relabel_consecutive(truth)
    prediction = relabel_consecutive(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction label rasters must have the same shape.")
    pairs, contingency = _overlap_iou_pairs(truth, prediction)
    accepted = []
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    for iou, true_id, predicted_id, _intersection in sorted(pairs, reverse=True):
        if iou < iou_threshold:
            break
        if true_id in used_truth or predicted_id in used_prediction:
            continue
        accepted.append(iou)
        used_truth.add(true_id)
        used_prediction.add(predicted_id)
    true_count = int(truth.max(initial=0))
    predicted_count = int(prediction.max(initial=0))
    true_positives = len(accepted)
    false_positives = predicted_count - true_positives
    false_negatives = true_count - true_positives
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    sum_iou = float(np.sum(accepted))
    panoptic_quality = sum_iou / max(
        1e-12, true_positives + 0.5 * false_positives + 0.5 * false_negatives
    )

    true_areas = contingency.sum(axis=1)
    predicted_areas = contingency.sum(axis=0)
    splits = 0
    for true_id in range(1, true_count + 1):
        overlaps = contingency[true_id, 1:] / max(1, true_areas[true_id])
        splits += int(np.count_nonzero(overlaps >= contact_overlap_fraction) > 1)
    merges = 0
    for predicted_id in range(1, predicted_count + 1):
        overlaps = contingency[1:, predicted_id] / max(1, predicted_areas[predicted_id])
        merges += int(np.count_nonzero(overlaps >= contact_overlap_fraction) > 1)
    boundary_precision, boundary_recall, boundary_f1 = _boundary_scores(
        truth, prediction, boundary_tolerance_px
    )
    count_error = predicted_count - true_count
    return InstanceMetrics(
        true_instances=true_count,
        predicted_instances=predicted_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=float(np.mean(accepted)) if accepted else 0.0,
        panoptic_quality=panoptic_quality,
        count_error=count_error,
        absolute_count_error=abs(count_error),
        relative_count_error=abs(count_error) / max(1, true_count),
        split_instances=splits,
        merged_instances=merges,
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
    )
