"""Colour correction against a chart's published reference colours.

The measured swatch grid is matched to every known chart in each orientation
whose shape fits. The best match is chosen by fit residual. An affine
transform in linear sRGB then maps all measured patches onto their reference
colours, so colour balance is corrected as well as the grey balance and
levels. Neutral patches anchor the fit with extra weight. The fit is small
(24 patches, 12 parameters) and runs on the CPU. Applying it to the image
runs on the tensor's device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ChartColours:
    """A known chart in its upright orientation, patches in reading order."""

    name: str
    rows: int
    columns: int
    patch_names: tuple[str, ...]
    lab_d50: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class ColourCorrectionFacts:
    """What the colour correction did, compactly enough to export."""

    model: str  # "chart_reference_fit", "neutral_balance" or "none"
    chart: str | None
    rotation_degrees: int
    matrix_linear_rgb: tuple[tuple[float, float, float], ...]
    offset_linear_rgb: tuple[float, float, float]
    used_patch_count: int
    anchor_patch_ids: tuple[int, ...]
    mean_delta_e: float | None
    max_delta_e: float | None
    held_out_mean_delta_e: float | None
    patch_delta_e: tuple[float, ...]
    reason: str = ""


# ---------------------------------------------------------------- references

def _srgb8_to_lab_d50(values: Iterable[tuple[int, int, int]]):
    rgb = np.asarray(tuple(values), np.float64) / 255.0
    return tuple(
        tuple(float(v) for v in row) for row in linear_rgb_to_lab_d50(srgb_to_linear(rgb))
    )


# Datacolor SpyderCHECKR 24: columns E-H of the 48-patch SpyderCHECKR, rows 1-6.
# Lab (D50) values as published for the 48-patch card
# (https://www.bartneck.de/?p=3807). Converting them with a Bradford D50->D65
# adaptation reproduces that table's sRGB values to about one 8-bit unit,
# except its G2 "Blueprint" sRGB entry, which is a typo (Lab gives 58/88/159).
# Upright: white top left, greys down the left column, skin tones bottom right.
_SPYDER_COLUMNS = (
    (  # E: grey ramp
        ("Card White", (96.04, 2.16, 2.6)),
        ("20% Gray", (80.44, 1.17, 2.05)),
        ("40% Gray", (65.52, 0.69, 1.86)),
        ("60% Gray", (49.62, 0.58, 1.56)),
        ("80% Gray", (33.55, 0.35, 1.4)),
        ("Card Black", (16.91, 1.43, -0.81)),
    ),
    (  # F: primaries
        ("Primary Cyan", (47.12, -32.5, -28.75)),
        ("Primary Magenta", (50.49, 53.45, -13.55)),
        ("Primary Yellow", (83.61, 3.36, 87.02)),
        ("Primary Red", (41.05, 60.75, 31.17)),
        ("Primary Green", (54.14, -40.8, 34.75)),
        ("Primary Blue", (24.75, 13.78, -49.48)),
    ),
    (  # G
        ("Primary Orange", (60.94, 38.21, 61.31)),
        ("Blueprint", (37.8, 7.3, -43.04)),
        ("Pink", (49.81, 48.5, 15.76)),
        ("Violet", (28.88, 19.36, -24.48)),
        ("Apple Green", (72.45, -23.6, 60.47)),
        ("Sunflower", (71.65, 23.74, 72.28)),
    ),
    (  # H
        ("Aqua", (70.19, -31.9, 1.98)),
        ("Lavender", (54.38, 8.84, -25.71)),
        ("Evergreen", (42.03, -15.8, 22.93)),
        ("Steel Blue", (48.82, -5.11, -23.08)),
        ("Classic Light Skin", (65.1, 18.14, 18.68)),
        ("Classic Dark Skin", (36.13, 14.15, 15.78)),
    ),
)
SPYDERCHECKR_24 = ChartColours(
    "Datacolor SpyderCHECKR 24",
    6,
    4,
    tuple(_SPYDER_COLUMNS[c][r][0] for r in range(6) for c in range(4)),
    tuple(_SPYDER_COLUMNS[c][r][1] for r in range(6) for c in range(4)),
)

# X-Rite (now Calibrite) ColorChecker Classic: published sRGB values. The
# Passport and Mini share this layout. Upright: brown "dark skin" top left,
# greys along the bottom row from white to black.
_COLORCHECKER_SRGB = (
    ("Dark Skin", (115, 82, 68)), ("Light Skin", (194, 150, 130)),
    ("Blue Sky", (98, 122, 157)), ("Foliage", (87, 108, 67)),
    ("Blue Flower", (133, 128, 177)), ("Bluish Green", (103, 189, 170)),
    ("Orange", (214, 126, 44)), ("Purplish Blue", (80, 91, 166)),
    ("Moderate Red", (193, 90, 99)), ("Purple", (94, 60, 108)),
    ("Yellow Green", (157, 188, 64)), ("Orange Yellow", (224, 163, 46)),
    ("Blue", (56, 61, 150)), ("Green", (70, 148, 73)),
    ("Red", (175, 54, 60)), ("Yellow", (231, 199, 31)),
    ("Magenta", (187, 86, 149)), ("Cyan", (8, 133, 161)),
    ("White", (243, 243, 242)), ("Neutral 8", (200, 200, 200)),
    ("Neutral 6.5", (160, 160, 160)), ("Neutral 5", (122, 122, 121)),
    ("Neutral 3.5", (85, 85, 85)), ("Black", (52, 52, 52)),
)

# ------------------------------------------------------------ colour science

_D50 = np.array((0.96422, 1.0, 0.82521))
_D65 = np.array((0.95047, 1.0, 1.08883))
_BRADFORD = np.array(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296))
)
_XYZ_TO_RGB = np.array(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    )
)
_RGB_TO_XYZ = np.linalg.inv(_XYZ_TO_RGB)


def _adaptation(source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    source = _BRADFORD @ source_white
    target = _BRADFORD @ target_white
    return np.linalg.inv(_BRADFORD) @ np.diag(target / source) @ _BRADFORD


_D50_TO_D65 = _adaptation(_D50, _D65)
_D65_TO_D50 = _adaptation(_D65, _D50)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_rgb_to_lab_d50(linear: np.ndarray) -> np.ndarray:
    xyz = np.asarray(linear, np.float64) @ (_D65_TO_D50 @ _RGB_TO_XYZ).T / _D50
    epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(np.maximum(xyz, 0.0)), (kappa * xyz + 16.0) / 116.0)
    return np.stack(
        (116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])),
        axis=-1,
    )


def lab_d50_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, np.float64)
    epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def inverse(f):
        cube = f**3
        return np.where(cube > epsilon, cube, (116.0 * f - 16.0) / kappa)

    y = np.where(lab[..., 0] > kappa * epsilon, fy**3, lab[..., 0] / kappa)
    xyz = np.stack((inverse(fx), y, inverse(fz)), axis=-1) * _D50
    return xyz @ (_XYZ_TO_RGB @ _D50_TO_D65).T


def delta_e_2000(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference between Lab rows."""

    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_mean = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_mean**7 / (c_mean**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    dhh = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2.0))
    l_mean = (l1 + l2) / 2.0
    cp_mean = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    hp_mean = np.where(
        c1p * c2p == 0.0,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(hp_mean - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_mean))
        + 0.32 * np.cos(np.radians(3.0 * hp_mean + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_mean - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((hp_mean - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt(cp_mean**7 / (cp_mean**7 + 25.0**7))
    sl = 1.0 + 0.015 * (l_mean - 50.0) ** 2 / np.sqrt(20.0 + (l_mean - 50.0) ** 2)
    sc = 1.0 + 0.045 * cp_mean
    sh = 1.0 + 0.015 * cp_mean * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    return np.sqrt(
        (dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh)
    )


COLORCHECKER_CLASSIC_24 = ChartColours(
    "X-Rite ColorChecker Classic 24",
    4,
    6,
    tuple(name for name, _ in _COLORCHECKER_SRGB),
    _srgb8_to_lab_d50(rgb for _, rgb in _COLORCHECKER_SRGB),
)

KNOWN_CHARTS = (SPYDERCHECKR_24, COLORCHECKER_CLASSIC_24)

# ------------------------------------------------------------------- fitting

# A chart is accepted when the best orientation fits its patches with a mean
# CIEDE2000 below this, and clearly better than any other candidate.
IDENTIFICATION_MEAN_DELTA_E = 9.0
IDENTIFICATION_MARGIN = 0.75
MINIMUM_FIT_PATCHES = 12
ANCHOR_WEIGHT = 4.0
# Reference greys have almost no chroma; these anchor the fit when the
# operator leaves neutral patches on automatic.
_NEUTRAL_CHROMA = 4.0
# A measured channel at or above this (8-bit) is treated as clipped.
_CLIPPED_LEVEL = 250.0


@dataclass(frozen=True, slots=True)
class _Candidate:
    chart: ChartColours
    rotation: int
    pairs: tuple[tuple[int, int], ...]  # (measured index, reference index)
    mean_delta_e: float


def _reference_grid(chart: ChartColours, quarter_turns: int) -> np.ndarray:
    """Reference patch indices laid out as the chart appears after rotation."""

    grid = np.arange(chart.rows * chart.columns).reshape(chart.rows, chart.columns)
    # np.rot90 turns anticlockwise; a chart photographed turned clockwise by
    # 90 degrees appears as the upright grid turned clockwise.
    return np.rot90(grid, k=-quarter_turns)


def _weighted_affine(measured: np.ndarray, reference: np.ndarray, weights: np.ndarray):
    """Least-squares affine map from measured to reference linear RGB."""

    design = np.column_stack((measured, np.ones(len(measured))))
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, reference * root, rcond=None)
    return solution[:3].T, solution[3]


def _apply_affine(matrix: np.ndarray, offset: np.ndarray, linear: np.ndarray) -> np.ndarray:
    return linear @ matrix.T + offset


def _fit_delta_e(matrix, offset, measured, reference_lab) -> np.ndarray:
    corrected = np.clip(_apply_affine(matrix, offset, measured), 0.0, None)
    return delta_e_2000(linear_rgb_to_lab_d50(corrected), reference_lab)


def fit_chart_correction(
    patches: Sequence[tuple[int, int, int, tuple[float, float, float]]],
    *,
    chart_name: str | None = None,
    rotation_degrees: int = 0,
    anchor_patch_numbers: Sequence[int] = (),
    charts: Sequence[ChartColours] = KNOWN_CHARTS,
) -> tuple[ColourCorrectionFacts, np.ndarray, np.ndarray] | str:
    """Fit an affine linear-sRGB correction to a known chart.

    ``patches`` holds ``(number, row, column, mean_bgr)`` per measured swatch.
    When the detector identified the chart (``chart_name``), ``number`` is
    the patch's position in that chart's upright reading order and is used
    directly. Otherwise ``row`` and ``column`` locate the patch in image
    reading order and every known chart is tried in each orientation whose
    shape matches the complete grid. ``anchor_patch_numbers`` names
    operator-selected neutral patches; without them the chart's own greys
    anchor the fit. Returns the facts with the matrix and offset, or a reason
    string when no chart can be fitted.
    """

    if not patches:
        return "No colour chart swatches were measured."
    numbers = [int(number) for number, _row, _column, _bgr in patches]
    cells = [(int(row), int(column)) for _number, row, column, _bgr in patches]
    measured_bgr = np.asarray([bgr for *_rest, bgr in patches], np.float64)
    valid = np.isfinite(measured_bgr).all(axis=1) & (measured_bgr.max(axis=1) < _CLIPPED_LEVEL)
    measured = srgb_to_linear(np.clip(measured_bgr[:, ::-1], 0.0, 255.0) / 255.0)

    def candidate(chart, rotation, pairs):
        if len(pairs) < MINIMUM_FIT_PATCHES:
            return None
        reference_lab = np.asarray(chart.lab_d50, np.float64)
        reference_linear = np.clip(lab_d50_to_linear_rgb(reference_lab), 0.0, None)
        measured_index = np.asarray([pair[0] for pair in pairs])
        reference_index = np.asarray([pair[1] for pair in pairs])
        matrix, offset = _weighted_affine(
            measured[measured_index], reference_linear[reference_index], np.ones(len(pairs))
        )
        errors = _fit_delta_e(
            matrix, offset, measured[measured_index], reference_lab[reference_index]
        )
        return _Candidate(chart, rotation, tuple(pairs), float(np.mean(errors)))

    by_name = {chart.name: chart for chart in charts}
    candidates: list[_Candidate] = []
    if chart_name is not None:
        chart = by_name.get(chart_name)
        if chart is None:
            return f"No reference colours are known for {chart_name}."
        count = chart.rows * chart.columns
        pairs = [
            (index, number - 1)
            for index, number in enumerate(numbers)
            if valid[index] and 1 <= number <= count
        ]
        found = candidate(chart, int(rotation_degrees) % 360, pairs)
        if found is None:
            return (
                f"Only {len(pairs)} unclipped {chart.name} patches were measured; "
                f"the fit needs {MINIMUM_FIT_PATCHES}."
            )
        candidates.append(found)
    else:
        rows = 1 + max(row for row, _column in cells)
        columns = 1 + max(column for _row, column in cells)
        for chart in charts:
            for quarter_turns in range(4):
                grid = _reference_grid(chart, quarter_turns)
                if grid.shape != (rows, columns):
                    continue
                pairs = [
                    (index, int(grid[row, column]))
                    for index, (row, column) in enumerate(cells)
                    if valid[index]
                ]
                found = candidate(chart, quarter_turns * 90, pairs)
                if found is not None:
                    candidates.append(found)
        if not candidates:
            return (
                f"The {rows} x {columns} swatch grid does not match a known chart "
                f"with at least {MINIMUM_FIT_PATCHES} unclipped patches."
            )
    candidates.sort(key=lambda item: item.mean_delta_e)
    best = candidates[0]
    if best.mean_delta_e > IDENTIFICATION_MEAN_DELTA_E:
        return (
            f"The swatches do not match {best.chart.name} closely enough "
            f"(mean ΔE00 {best.mean_delta_e:.1f})."
        )
    if len(candidates) > 1 and best.mean_delta_e > IDENTIFICATION_MARGIN * candidates[1].mean_delta_e:
        return (
            f"The chart orientation is ambiguous ({best.chart.name} at "
            f"{best.rotation}° fits about as well as {candidates[1].chart.name} at "
            f"{candidates[1].rotation}°)."
        )

    chart = best.chart
    reference_lab = np.asarray(chart.lab_d50, np.float64)
    reference_linear = np.clip(lab_d50_to_linear_rgb(reference_lab), 0.0, None)
    measured_index = np.asarray([pair[0] for pair in best.pairs])
    reference_index = np.asarray([pair[1] for pair in best.pairs])
    requested = {int(value) for value in anchor_patch_numbers}
    if requested:
        anchors = np.asarray([numbers[index] in requested for index in measured_index])
    else:
        anchors = (
            np.hypot(reference_lab[reference_index, 1], reference_lab[reference_index, 2])
            < _NEUTRAL_CHROMA
        )
    weights = np.where(anchors, ANCHOR_WEIGHT, 1.0)
    matrix, offset = _weighted_affine(
        measured[measured_index], reference_linear[reference_index], weights
    )
    determinant = float(np.linalg.det(matrix))
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(matrix).all() or determinant <= 0.0 or condition > 60.0:
        return (
            f"The {chart.name} fit is ill-conditioned (determinant {determinant:.3g}, "
            f"condition {condition:.1f})."
        )
    errors = _fit_delta_e(matrix, offset, measured[measured_index], reference_lab[reference_index])

    held_out = []
    everything = np.arange(len(measured_index))
    for left_out in everything:
        keep = everything != left_out
        loo_matrix, loo_offset = _weighted_affine(
            measured[measured_index[keep]], reference_linear[reference_index[keep]], weights[keep]
        )
        held_out.append(
            float(
                _fit_delta_e(
                    loo_matrix,
                    loo_offset,
                    measured[measured_index[left_out : left_out + 1]],
                    reference_lab[reference_index[left_out : left_out + 1]],
                )[0]
            )
        )

    facts = ColourCorrectionFacts(
        model="chart_reference_fit",
        chart=chart.name,
        rotation_degrees=best.rotation,
        matrix_linear_rgb=tuple(tuple(float(v) for v in row) for row in matrix),
        offset_linear_rgb=tuple(float(v) for v in offset),
        used_patch_count=len(measured_index),
        anchor_patch_ids=tuple(
            sorted(int(numbers[index]) for index, anchored in zip(measured_index, anchors) if anchored)
        ),
        mean_delta_e=float(np.mean(errors)),
        max_delta_e=float(np.max(errors)),
        held_out_mean_delta_e=float(np.mean(held_out)),
        patch_delta_e=tuple(float(value) for value in errors),
    )
    return facts, matrix, offset


def neutral_correction_facts(reason: str, *, applied: bool) -> ColourCorrectionFacts:
    """Facts for the neutral-balance fallback (or no correction at all)."""

    return ColourCorrectionFacts(
        model="neutral_balance" if applied else "none",
        chart=None,
        rotation_degrees=0,
        matrix_linear_rgb=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        offset_linear_rgb=(0.0, 0.0, 0.0),
        used_patch_count=0,
        anchor_patch_ids=(),
        mean_delta_e=None,
        max_delta_e=None,
        held_out_mean_delta_e=None,
        patch_delta_e=(),
        reason=reason,
    )


def apply_linear_rgb_correction(tensor_bgr, matrix: np.ndarray, offset: np.ndarray):
    """Apply an affine linear-sRGB correction to a 0-255 BGR NCHW tensor.

    Out-of-gamut pixels are scaled down along their own hue rather than
    clipped channel by channel, so bright saturated colours keep their hue.
    """

    import torch

    dtype = tensor_bgr.dtype if tensor_bgr.is_floating_point() else torch.float32
    rgb = tensor_bgr.flip(1).to(dtype) / 255.0
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    del rgb
    weights = torch.as_tensor(matrix, device=linear.device, dtype=linear.dtype)
    shift = torch.as_tensor(offset, device=linear.device, dtype=linear.dtype)
    corrected = torch.einsum("ij,njhw->nihw", weights, linear) + shift.reshape(1, 3, 1, 1)
    del linear
    peak = corrected.amax(dim=1, keepdim=True).clamp_min(1.0)
    corrected = (corrected / peak).clamp_(0.0, 1.0)
    del peak
    encoded = torch.where(
        corrected <= 0.0031308,
        corrected * 12.92,
        1.055 * corrected.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055,
    )
    return (encoded * 255.0).clamp_(0.0, 255.0).flip(1)


def correction_description(facts: ColourCorrectionFacts | None) -> str:
    if facts is None:
        return "colour correction unrecorded"
    if facts.model == "chart_reference_fit":
        return (
            f"{facts.chart} reference fit ({facts.used_patch_count} patches, "
            f"mean ΔE00 {facts.mean_delta_e:.1f}, held out {facts.held_out_mean_delta_e:.1f})"
        )
    if facts.model == "neutral_balance":
        return "grey balance only: " + facts.reason if facts.reason else "grey balance only"
    return "no colour correction" + (f": {facts.reason}" if facts.reason else "")


def colour_correction_fell_back(calibration) -> bool:
    """Whether a chart fit was wanted for a detected chart but not achieved."""

    settings = getattr(calibration, "calibration_settings", None)
    facts = getattr(calibration, "colour_correction", None)
    return bool(
        settings is not None
        and settings.apply_colour_balance
        and settings.colour_correction_model == "Chart reference fit"
        and getattr(calibration, "colour_card", None) is not None
        and (facts is None or facts.model != "chart_reference_fit")
    )


def colour_correction_summary(calibration) -> str:
    """One status line for the colour correction an image received."""

    from seedvision.calibration.neutral import neutral_balance_description

    facts = getattr(calibration, "colour_correction", None)
    if facts is not None and facts.model == "chart_reference_fit":
        return correction_description(facts)
    settings = getattr(calibration, "calibration_settings", None)
    if settings is not None and not settings.apply_colour_balance:
        return "colour correction off"
    gains = calibration.channel_gains_bgr
    text = (
        neutral_balance_description(getattr(calibration, "neutral_balance", None))
        + f" (BGR gains {gains[0]:.3f}/{gains[1]:.3f}/{gains[2]:.3f})"
    )
    if colour_correction_fell_back(calibration) and facts is not None and facts.reason:
        text = f"grey balance only, {facts.reason[0].lower()}{facts.reason[1:]} " + text
    return text
