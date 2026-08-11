"""Masked multi-task losses for the learned instance models."""

from __future__ import annotations


def _masked_mean(values, mask):
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    weighted = values * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def _balanced_bce(logits, target, valid):
    import torch
    import torch.nn.functional as functional

    positives = (target * valid).sum()
    negatives = ((1.0 - target) * valid).sum()
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    loss = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    return _masked_mean(loss, valid)


def _dice_loss(logits, target, valid):
    probabilities = logits.sigmoid() * valid
    target = target * valid
    intersection = (probabilities * target).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def multi_head_unet_loss(outputs, targets, *, weights=None):
    """Return total loss and named terms for the five logical U-Net heads."""

    import torch
    import torch.nn.functional as functional

    weights = {
        "interior": 1.0,
        "physical_boundary": 1.5,
        "pattern_boundary": 0.8,
        "centre": 1.0,
        "distance": 0.7,
        "separation": 0.3,
        "uncertainty": 0.20,
        **(weights or {}),
    }
    valid = targets["valid"]
    interior = targets["interior"]
    physical = targets["physical_boundary"]
    pattern = targets["pattern_boundary"]
    pattern_valid = targets["pattern_valid"] * valid
    centre = targets["centre"]
    distance = targets["distance"]
    foreground_valid = valid * interior

    interior_loss = _balanced_bce(outputs["interior_logits"], interior, valid)
    interior_loss = interior_loss + _dice_loss(
        outputs["interior_logits"], interior, valid
    )
    physical_loss = _balanced_bce(
        outputs["physical_boundary_logits"], physical, valid
    ) + _dice_loss(outputs["physical_boundary_logits"], physical, valid)
    if float(pattern_valid.sum().detach().item()) > 0:
        pattern_loss = _balanced_bce(
            outputs["pattern_boundary_logits"], pattern, pattern_valid
        ) + _dice_loss(outputs["pattern_boundary_logits"], pattern, pattern_valid)
    else:
        pattern_loss = outputs["pattern_boundary_logits"].sum() * 0.0
    centre_loss = _balanced_bce(outputs["centre_logits"], centre, foreground_valid)
    distance_error = functional.smooth_l1_loss(
        outputs["distance"], distance, reduction="none"
    )
    distance_loss = _masked_mean(distance_error, foreground_valid)

    physical_probability = outputs["physical_boundary_logits"].sigmoid()
    pattern_probability = outputs["pattern_boundary_logits"].sigmoid()
    separation_loss = _masked_mean(
        physical_probability * pattern + pattern_probability * physical,
        torch.maximum(pattern_valid, valid * physical),
    )
    uncertainty_targets = torch.cat(
        (
            (outputs["interior_logits"].sigmoid() - interior).abs().detach(),
            (outputs["physical_boundary_logits"].sigmoid() - physical).abs().detach(),
            (outputs["pattern_boundary_logits"].sigmoid() - pattern).abs().detach(),
            (outputs["centre_logits"].sigmoid() - centre).abs().detach(),
            (outputs["distance"] - distance).abs().detach(),
        ),
        dim=1,
    ).clamp(0.0, 1.0)
    uncertainty_masks = torch.cat(
        (valid, valid, pattern_valid, foreground_valid, foreground_valid), dim=1
    )
    uncertainty_error = functional.binary_cross_entropy_with_logits(
        outputs["uncertainty_logits"], uncertainty_targets, reduction="none"
    )
    uncertainty_regularizer = _masked_mean(uncertainty_error, uncertainty_masks)

    terms = {
        "interior": interior_loss,
        "physical_boundary": physical_loss,
        "pattern_boundary": pattern_loss,
        "centre": centre_loss,
        "distance": distance_loss,
        "separation": separation_loss,
        "uncertainty": uncertainty_regularizer,
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


def stardist_loss(
    outputs,
    targets,
    *,
    object_weight=1.0,
    radial_weight=1.0,
    angular_weight=0.20,
    uncertainty_weight=0.20,
):
    """Objectness plus uncertainty-weighted radial regression."""

    import torch.nn.functional as functional

    valid = targets["valid"]
    object_target = targets["object_probability"]
    object_loss = _balanced_bce(outputs["object_logits"], object_target, valid)
    object_loss = object_loss + _dice_loss(
        outputs["object_logits"], (object_target > 0).to(object_target.dtype), valid
    )
    radial_valid = targets["radial_valid"] * (object_target > 0).to(valid.dtype)
    predicted = outputs["radial_distances"].clamp_min(0.25)
    expected = targets["radial_distances"].clamp_min(0.25)
    log_error = functional.smooth_l1_loss(
        predicted.log(), expected.log(), reduction="none"
    )
    relative_error = ((predicted - expected).abs() / expected).clamp_max(4.0)
    radial_loss = _masked_mean(log_error + 0.20 * relative_error, radial_valid)
    predicted_second = (
        predicted.roll(1, dims=1) - 2.0 * predicted + predicted.roll(-1, dims=1)
    )
    expected_second = (
        expected.roll(1, dims=1) - 2.0 * expected + expected.roll(-1, dims=1)
    )
    angular_loss = _masked_mean(
        functional.smooth_l1_loss(
            predicted_second / expected.clamp_min(1.0),
            expected_second / expected.clamp_min(1.0),
            reduction="none",
        ),
        radial_valid,
    )
    uncertainty_target = relative_error.mean(dim=1, keepdim=True).detach().clamp(0.0, 1.0)
    uncertainty_error = functional.binary_cross_entropy_with_logits(
        outputs["radial_uncertainty_logits"], uncertainty_target, reduction="none"
    )
    uncertainty = _masked_mean(uncertainty_error, valid * (object_target > 0))
    terms = {
        "object": object_loss,
        "radial": radial_loss,
        "angular": angular_loss,
        "uncertainty": uncertainty,
    }
    total = (
        object_weight * object_loss
        + radial_weight * radial_loss
        + angular_weight * angular_loss
        + uncertainty_weight * uncertainty
    )
    return total, terms
