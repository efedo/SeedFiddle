"""CUDA-first colour-reference, ruler, deskew, and scale calibration.

Pixel-level work in this module is implemented with PyTorch tensors.  NumPy is
used only for the small sets of detected corners, endpoints, and homographies
that form the public calibration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seedvision.calibration.geometry import CalibrationDetectionError
from seedvision.cuda import (
    CudaContext,
    bgr_to_gray,
    bilinear_sample,
    binary_close,
    connected_components,
    gaussian_blur,
    gradient_magnitude,
    image_to_tensor,
    resize,
    tensor_to_image,
    warp_perspective,
)


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """User-adjustable assumptions for the controlled calibration layout."""

    ruler_length_mm: float = 150.0
    minor_tick_mm: float = 1.0
    max_deskew_degrees: float = 10.0
    apply_colour_balance: bool = True
    apply_perspective_correction: bool = True
    max_perspective_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 10.0 <= self.ruler_length_mm <= 1000.0:
            raise ValueError("ruler_length_mm must be between 10 and 1000.")
        if not 0.1 <= self.minor_tick_mm <= 10.0:
            raise ValueError("minor_tick_mm must be between 0.1 and 10.")
        if not 0.0 <= self.max_deskew_degrees <= 30.0:
            raise ValueError("max_deskew_degrees must be between 0 and 30.")
        if not 0.0 <= self.max_perspective_fraction <= 0.50:
            raise ValueError("max_perspective_fraction must be between 0 and 0.50.")


@dataclass(frozen=True, slots=True)
class ColourSwatch:
    row: int
    column: int
    center_x: float
    center_y: float
    corners: tuple[tuple[float, float], ...]
    mean_bgr: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ColourCardDetection:
    swatches: tuple[ColourSwatch, ...]
    bounds: tuple[tuple[float, float], ...]
    angle_degrees: float
    confidence: float
    detected_swatch_count: int


@dataclass(frozen=True, slots=True)
class RulerDetection:
    endpoint_a: tuple[float, float]
    endpoint_b: tuple[float, float]
    center_x: float
    center_y: float
    length_px: float
    width_px: float
    angle_degrees: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ImageCalibration:
    """Deskewed, colour-balanced image and its detected references."""

    corrected_bgr: np.ndarray
    colour_card: ColourCardDetection | None
    ruler: RulerDetection | None
    affine_matrix: np.ndarray
    deskew_degrees: float
    perspective_corrected: bool
    perspective_strength: float
    channel_gains_bgr: tuple[float, float, float]
    pixels_per_mm: float | None
    scale_confidence: float
    ruler_tick_spacing_px: float | None
    scale_bar_mm: float
    warnings: tuple[str, ...]
    ruler_in_corrected_coordinates: bool = True
    gpu_corrected_bgr: object | None = None

    def transform_points(
        self, points: tuple[tuple[float, float], ...] | list[tuple[float, float]]
    ) -> np.ndarray:
        return _transform_points(np.asarray(points, dtype=np.float64), self.affine_matrix)

    def ruler_endpoints_corrected(self) -> np.ndarray:
        """Return ruler endpoints in the displayed, corrected image space."""

        if self.ruler is None:
            return np.empty((0, 2), dtype=np.float32)
        points = np.asarray(
            (self.ruler.endpoint_a, self.ruler.endpoint_b), dtype=np.float64
        )
        if self.ruler_in_corrected_coordinates:
            return points.astype(np.float32)
        return _transform_points(points, self.affine_matrix)


def calibrate_image(
    image: np.ndarray,
    settings: CalibrationSettings | None = None,
    *,
    cuda_context: CudaContext | None = None,
    timing_recorder=None,
) -> ImageCalibration:
    """Detect references and perform all raster calibration on one GPU."""

    _validate_image(image)
    settings = settings or CalibrationSettings()
    context = cuda_context or CudaContext.resolve()
    colour_timing = (
        None
        if timing_recorder is None
        else timing_recorder.start("colour_reference")
    )
    tensor = image_to_tensor(image, context)
    warnings: list[str] = []
    try:
        colour_card = detect_colour_card(image, cuda_context=context, image_tensor=tensor)
    except CalibrationDetectionError as error:
        colour_card = None
        warnings.append(str(error))
    gains = (
        _neutral_balance_gains(colour_card)
        if colour_card is not None and settings.apply_colour_balance
        else np.ones(3, np.float32)
    )
    if colour_timing is not None:
        timing_recorder.stop(colour_timing)
    deskew_timing = (
        None
        if timing_recorder is None
        else timing_recorder.start("deskew_colour")
    )
    # Geometric correction is deliberately based only on the card. The ruler
    # is detected later in this corrected coordinate system.
    deskew_degrees = _combined_reference_angle(colour_card, None)
    correction_allowed = abs(deskew_degrees) <= settings.max_deskew_degrees
    if not correction_allowed:
        warnings.append(
            f"Reference angle {deskew_degrees:.2f}° exceeds the deskew limit; "
            "geometric correction was not applied."
        )
        deskew_degrees = 0.0

    height, width = image.shape[:2]
    transform = np.eye(3, dtype=np.float32)
    perspective_corrected = False
    perspective_strength = 0.0
    if correction_allowed:
        if colour_card is not None and settings.apply_perspective_correction:
            projective, perspective_strength = _projective_deskew_matrix(
                image.shape[:2], colour_card
            )
            if perspective_strength <= settings.max_perspective_fraction:
                transform = projective.astype(np.float32)
                perspective_corrected = True
            else:
                warnings.append(
                    f"Perspective strength {perspective_strength:.3f} exceeds the "
                    "configured limit; rotation-only correction was used."
                )
        if not perspective_corrected:
            transform = _rotation_matrix(
                width * 0.5, height * 0.5, deskew_degrees
            ).astype(np.float32)

    # Translate and enlarge the destination so every transformed source corner
    # remains visible. Keeping the original canvas size clipped the colour card
    # whenever deskew moved its already-near-border swatches out of frame.
    transform, output_size = _expanded_warp_canvas(transform, height, width)
    corrected_tensor = warp_perspective(tensor, transform, output_size)
    import torch

    gain_tensor = torch.as_tensor(
        gains, device=context.device, dtype=corrected_tensor.dtype
    ).reshape(1, 3, 1, 1)
    balanced = corrected_tensor * gain_tensor
    # Gamut-preserving highlight compression avoids clipping bright colour
    # swatches channel-by-channel after neutral balance.
    maximum_channel = balanced.amax(dim=1, keepdim=True).clamp_min(1.0)
    gamut_scale = torch.minimum(
        torch.ones_like(maximum_channel), 255.0 / maximum_channel
    )
    corrected_tensor = (balanced * gamut_scale).clamp(0.0, 255.0)
    corrected = tensor_to_image(corrected_tensor)
    if deskew_timing is not None:
        timing_recorder.stop(deskew_timing)

    ruler_timing = (
        None
        if timing_recorder is None
        else timing_recorder.start("ruler_detection")
    )
    try:
        ruler = detect_ruler(
            corrected,
            settings.ruler_length_mm,
            minor_tick_mm=settings.minor_tick_mm,
            cuda_context=context,
            image_tensor=corrected_tensor,
        )
    except CalibrationDetectionError as error:
        ruler = None
        warnings.append(str(error))
    if ruler_timing is not None:
        timing_recorder.stop(ruler_timing)

    scale_timing = (
        None
        if timing_recorder is None
        else timing_recorder.start("scale_calibration")
    )
    pixels_per_mm: float | None = None
    tick_spacing: float | None = None
    scale_confidence = 0.0
    if ruler is not None:
        pixels_per_mm = ruler.length_px / settings.ruler_length_mm
        tick_spacing = pixels_per_mm * settings.minor_tick_mm
        scale_confidence = ruler.confidence
    if scale_timing is not None:
        timing_recorder.stop(scale_timing)

    if colour_card is None and ruler is None:
        warnings.append("Deskew could not be estimated because neither reference was found.")
    elif colour_card is None:
        warnings.append("The ruler was detected without colour-card deskew.")
    elif ruler is None:
        warnings.append("Deskew succeeded, but ruler scale could not be assigned.")

    return ImageCalibration(
        corrected_bgr=corrected,
        colour_card=colour_card,
        ruler=ruler,
        affine_matrix=transform,
        deskew_degrees=float(deskew_degrees),
        perspective_corrected=perspective_corrected,
        perspective_strength=float(perspective_strength),
        channel_gains_bgr=tuple(float(value) for value in gains),
        pixels_per_mm=pixels_per_mm,
        scale_confidence=float(scale_confidence),
        ruler_tick_spacing_px=tick_spacing,
        scale_bar_mm=50.0,
        warnings=tuple(warnings),
        ruler_in_corrected_coordinates=True,
        gpu_corrected_bgr=corrected_tensor,
    )


def detect_colour_card(
    image: np.ndarray,
    *,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
) -> ColourCardDetection:
    """Detect the card as a dark bordered quadrilateral on the GPU."""

    _validate_image(image)
    import torch

    context = cuda_context or CudaContext.resolve()
    source = image_tensor if image_tensor is not None else image_to_tensor(image, context)
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small_height = max(32, round(height * scale))
    small_width = max(32, round(width * scale))
    small = resize(source, (small_height, small_width), mode="area") if scale < 1.0 else source
    gray = bgr_to_gray(small)
    search_width = max(16, round(small_width * 0.58))
    search = gray[:, :, :, :search_width]
    median = torch.median(search)
    lower_quartile = torch.quantile(search, 0.28)
    threshold = torch.minimum(median - 22.0, lower_quartile + 15.0)
    dark = search < threshold
    # Only bridge small breaks in the printed border.  A larger close can join
    # the card to nearby ruled paper in real laboratory layouts and shear the
    # fitted top edge.
    kernel = max(5, round(min(small_height, small_width) * 0.007))
    if kernel % 2 == 0:
        kernel += 1
    closed = binary_close(dark, kernel)
    labels, stats = connected_components(closed)
    component_count = int(labels.max().item()) + 1
    candidates: list[tuple[float, int]] = []
    total = small_height * small_width
    for index in range(1, component_count):
        area = float(stats["area"][index])
        component_width = float(stats["width"][index])
        component_height = float(stats["height"][index])
        left = float(stats["left"][index])
        top = float(stats["top"][index])
        if component_width <= 0 or component_height <= 0:
            continue
        aspect = component_height / component_width
        fill = area / (component_width * component_height)
        plausible = (
            area > total * 0.035
            and 0.13 * small_width < component_width < 0.58 * small_width
            and 0.30 * small_height < component_height < 0.99 * small_height
            and 1.05 < aspect < 3.4
            and left < small_width * 0.30
            and top < small_height * 0.25
            and fill > 0.34
        )
        if plausible:
            score = area * fill * (1.0 - left / max(small_width, 1))
            candidates.append((score, index))
    if not candidates:
        raise CalibrationDetectionError("Colour calibration card was not detected.")

    _, selected = max(candidates)
    mask = labels[0, 0] == selected
    coordinates = torch.nonzero(mask, as_tuple=False).float()
    quad_small = _quadrilateral_from_component(coordinates).cpu().numpy()
    quad = _order_quad(quad_small / scale)
    if _polygon_area(quad) < height * width * 0.025:
        raise CalibrationDetectionError("Colour calibration card outline was too small.")

    unit = np.float64(((0, 0), (1, 0), (1, 1), (0, 1)))
    unit_to_image = _homography(unit, quad.astype(np.float64))
    vertical_edges, horizontal_edges = _refine_swatch_grid_edges(
        source, unit_to_image
    )
    swatches: list[ColourSwatch] = []
    for row in range(6):
        for column in range(4):
            left, right = vertical_edges[column]
            top_edge, bottom_edge = horizontal_edges[row]
            normalized_corners = np.asarray(
                (
                    (left / 420.0, top_edge / 720.0),
                    (right / 420.0, top_edge / 720.0),
                    (right / 420.0, bottom_edge / 720.0),
                    (left / 420.0, bottom_edge / 720.0),
                ),
                np.float64,
            )
            corners = _transform_points(normalized_corners, unit_to_image)
            center = np.asarray(
                (
                    (left + right) / (2.0 * 420.0),
                    (top_edge + bottom_edge) / (2.0 * 720.0),
                ),
                np.float64,
            )
            center_image = _transform_points(center.reshape(1, 2), unit_to_image)[0]
            mean_bgr = _sample_quadrilateral_mean(source, corners)
            swatches.append(
                ColourSwatch(
                    row=row,
                    column=column,
                    center_x=float(center_image[0]),
                    center_y=float(center_image[1]),
                    corners=tuple((float(x), float(y)) for x, y in corners),
                    mean_bgr=mean_bgr,
                )
            )

    top = quad[1] - quad[0]
    angle = float(np.degrees(np.arctan2(top[1], top[0])))
    selected_area = float(stats["area"][selected]) / (scale * scale)
    confidence = float(np.clip(0.60 + selected_area / max(_polygon_area(quad), 1.0) * 0.35, 0.0, 0.98))
    return ColourCardDetection(
        swatches=tuple(swatches),
        bounds=tuple((float(x), float(y)) for x, y in quad),
        angle_degrees=angle,
        confidence=confidence,
        detected_swatch_count=24,
    )


def detect_ruler(
    image: np.ndarray,
    ruler_length_mm: float = 150.0,
    *,
    minor_tick_mm: float = 1.0,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
) -> RulerDetection:
    """Detect ruler orientation and the terminal printed scale dashes."""

    _validate_image(image)
    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    source = image_tensor if image_tensor is not None else image_to_tensor(image, context)
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small_height = max(32, round(height * scale))
    small_width = max(32, round(width * scale))
    small = resize(source, (small_height, small_width), mode="area") if scale < 1.0 else source
    gray = gaussian_blur(bgr_to_gray(small), 0.65)
    # The card occupies the left side of the controlled layout.  Starting at
    # 40% keeps its long border out of the ruler-line accumulator.
    roi_x0 = round(small_width * 0.40)
    roi_y0 = round(small_height * 0.62)
    roi = gray[:, :, roi_y0:, roi_x0:]
    gradient, _, _ = gradient_magnitude(roi, scharr=True)
    threshold = torch.quantile(gradient, 0.92)
    if float(threshold.item()) < 1.0 or float(gradient.max().item()) < 4.0:
        raise CalibrationDetectionError("Ruler was not detected.")
    edge = gradient[0, 0] > torch.maximum(
        threshold, torch.as_tensor(2.0, device=context.device)
    )
    points = torch.nonzero(edge, as_tuple=False).float()
    if points.shape[0] < 80:
        raise CalibrationDetectionError("Ruler was not detected.")
    point_y = points[:, 0] + roi_y0
    point_x = points[:, 1] + roi_x0

    angles = torch.linspace(-10.0, 10.0, 81, device=context.device)
    best_score = -1.0
    best_angle = 0.0
    best_v = 0.0
    best_histogram = None
    for angle_tensor in angles:
        angle = torch.deg2rad(angle_tensor)
        v = -torch.sin(angle) * point_x + torch.cos(angle) * point_y
        bins = torch.round(v).long().clamp(0, small_height + small_width - 1)
        histogram = torch.bincount(bins, minlength=small_height + small_width).float()
        score, index = torch.max(histogram, dim=0)
        score_value = float(score.item())
        if score_value > best_score:
            best_score = score_value
            best_angle = float(angle_tensor.item())
            best_v = float(index.item())
            best_histogram = histogram
    if best_histogram is None or best_score < small_width * 0.12:
        raise CalibrationDetectionError("Ruler baseline was not detected.")

    angle_radians = np.deg2rad(best_angle)
    cosine, sine = float(np.cos(angle_radians)), float(np.sin(angle_radians))
    point_u = cosine * point_x + sine * point_y
    point_v = -sine * point_x + cosine * point_y
    histogram = best_histogram
    minimum_width = max(12, round(small_height * 0.025))
    maximum_width = max(minimum_width + 2, round(small_height * 0.23))
    candidate_v = torch.topk(histogram, min(80, histogram.numel())).indices
    best_pair: tuple[float, float] | None = None
    pair_score = -1.0
    for first in candidate_v.detach().cpu().tolist():
        for second in candidate_v.detach().cpu().tolist():
            separation = abs(second - first)
            if not minimum_width <= separation <= maximum_width:
                continue
            score = float(min(histogram[first], histogram[second]).item())
            if score > pair_score:
                pair_score = score
                best_pair = (float(min(first, second)), float(max(first, second)))
    if best_pair is None:
        half_width = max(10.0, small_height * 0.06)
        best_pair = (best_v - half_width, best_v + half_width)
    top_v, bottom_v = best_pair
    ruler_width = bottom_v - top_v
    in_body = (point_v >= top_v - 2.0) & (point_v <= bottom_v + 2.0)
    body_u = point_u[in_body]
    if body_u.numel() < 40:
        raise CalibrationDetectionError("Ruler outline was incomplete.")
    u_min = float(torch.quantile(body_u, 0.002).item())
    u_max = float(torch.quantile(body_u, 0.998).item())
    strip_length = max(16, round(u_max - u_min) + 1)
    strip_height = max(8, round(ruler_width) + 1)
    sample_u = torch.linspace(u_min, u_max, strip_length, device=context.device)
    sample_v = torch.linspace(top_v + 2.0, bottom_v - 2.0, strip_height, device=context.device)
    grid_v, grid_u = torch.meshgrid(sample_v, sample_u, indexing="ij")
    sample_x = cosine * grid_u - sine * grid_v
    sample_y = sine * grid_u + cosine * grid_v
    sampled = bilinear_sample(gray, sample_x, sample_y)
    local_median = torch.median(sampled)
    darkness = (local_median - sampled).clamp_min(0.0)
    # Remove broad shade changes along the ruler while preserving thin dashes.
    profile = darkness.mean(dim=0)[None, None]
    broad = functional.avg_pool1d(profile, 25, stride=1, padding=12)
    profile = (profile - broad * 0.45).clamp_min(0.0)[0, 0]
    profile_cpu = profile.detach().cpu().numpy()
    expected_intervals = max(2, round(ruler_length_mm / minor_tick_mm))
    start_index, end_index, pitch, periodicity = _regular_tick_span(
        profile_cpu, expected_intervals
    )
    if periodicity < 0.04:
        raise CalibrationDetectionError("Ruler scale dashes were not detected reliably.")

    endpoint_u_a = u_min + start_index / max(strip_length - 1, 1) * (u_max - u_min)
    endpoint_u_b = u_min + end_index / max(strip_length - 1, 1) * (u_max - u_min)
    # Put the vector through the tick field, not through either plastic edge.
    tick_v = top_v + ruler_width * 0.35
    endpoint_a_small = np.asarray(
        (cosine * endpoint_u_a - sine * tick_v,
         sine * endpoint_u_a + cosine * tick_v),
        np.float64,
    )
    endpoint_b_small = np.asarray(
        (cosine * endpoint_u_b - sine * tick_v,
         sine * endpoint_u_b + cosine * tick_v),
        np.float64,
    )
    endpoint_a = endpoint_a_small / scale
    endpoint_b = endpoint_b_small / scale
    length = float(np.linalg.norm(endpoint_b - endpoint_a))
    if length < width * 0.18:
        raise CalibrationDetectionError("Detected ruler scale was implausibly short.")
    confidence = float(np.clip(0.55 + periodicity * 0.40 + min(0.12, pair_score / max(small_width, 1)), 0.0, 0.99))
    return RulerDetection(
        endpoint_a=(float(endpoint_a[0]), float(endpoint_a[1])),
        endpoint_b=(float(endpoint_b[0]), float(endpoint_b[1])),
        center_x=float((endpoint_a[0] + endpoint_b[0]) * 0.5),
        center_y=float((endpoint_a[1] + endpoint_b[1]) * 0.5),
        length_px=length,
        width_px=float(ruler_width / scale),
        angle_degrees=float(best_angle),
        confidence=confidence,
    )


def _quadrilateral_from_component(coordinates):
    """Fit lines to the four outer silhouettes of a card component."""

    import torch

    xy = coordinates[:, (1, 0)]
    x = xy[:, 0].long()
    y = xy[:, 1].long()
    width = int(x.max().item()) + 1
    height = int(y.max().item()) + 1
    large = torch.iinfo(torch.int64).max
    left_x = torch.full((height,), large, device=xy.device, dtype=torch.int64)
    right_x = torch.full((height,), -1, device=xy.device, dtype=torch.int64)
    top_y = torch.full((width,), large, device=xy.device, dtype=torch.int64)
    bottom_y = torch.full((width,), -1, device=xy.device, dtype=torch.int64)
    left_x.scatter_reduce_(0, y, x, reduce="amin", include_self=True)
    right_x.scatter_reduce_(0, y, x, reduce="amax", include_self=True)
    top_y.scatter_reduce_(0, x, y, reduce="amin", include_self=True)
    bottom_y.scatter_reduce_(0, x, y, reduce="amax", include_self=True)

    def central_indices(valid, low_fraction: float, high_fraction: float):
        indices = torch.nonzero(valid, as_tuple=False)[:, 0]
        low = torch.quantile(indices.float(), low_fraction)
        high = torch.quantile(indices.float(), high_fraction)
        return indices[(indices.float() >= low) & (indices.float() <= high)]

    side_rows = central_indices(right_x >= 0, 0.08, 0.92)
    # A projectively tilted card has sloping left/right sides. Columns near a
    # corner trace those sides rather than the top/bottom border, so fit the
    # horizontal silhouettes only through the central card span.
    side_columns = central_indices(bottom_y >= 0, 0.18, 0.82)

    def fit_dependent(independent, dependent, dependent_is_x: bool):
        design = torch.stack((independent.float(), torch.ones_like(independent).float()), dim=1)
        slope, intercept = torch.linalg.lstsq(design, dependent.float()).solution
        if dependent_is_x:  # x = slope*y + intercept
            return torch.as_tensor(1.0, device=xy.device), -slope, -intercept
        # y = slope*x + intercept
        return -slope, torch.as_tensor(1.0, device=xy.device), -intercept

    left = fit_dependent(side_rows, left_x[side_rows], True)
    right = fit_dependent(side_rows, right_x[side_rows], True)
    top = fit_dependent(side_columns, top_y[side_columns], False)
    bottom = fit_dependent(side_columns, bottom_y[side_columns], False)

    def intersection(first, second):
        matrix = torch.stack(
            (torch.stack(first[:2]), torch.stack(second[:2]))
        )
        target = -torch.stack((first[2], second[2]))
        return torch.linalg.solve(matrix, target)

    return torch.stack(
        (
            intersection(left, top),
            intersection(right, top),
            intersection(right, bottom),
            intersection(left, bottom),
        )
    )


def _refine_swatch_grid_edges(
    source, unit_to_image: np.ndarray
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Snap the projected 4x6 template to shared RGB edge-supported grid lines.

    The outer card homography first rectifies the source entirely on its tensor
    device. Each vertical swatch edge is then supported by all six swatches in
    that column, and each horizontal edge by all four swatches in that row.
    Only the compact 20-by-25 response-profile table is downloaded.
    """

    import torch
    import torch.nn.functional as functional

    canonical_width = 421
    canonical_height = 721
    device = source.device
    dtype = source.dtype
    axis_x = torch.linspace(0.0, 1.0, canonical_width, device=device, dtype=dtype)
    axis_y = torch.linspace(0.0, 1.0, canonical_height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    homogeneous = torch.stack(
        (grid_x, grid_y, torch.ones_like(grid_x)), dim=-1
    )
    transform = torch.as_tensor(
        unit_to_image, device=device, dtype=dtype
    )
    mapped = homogeneous @ transform.T
    denominator = mapped[..., 2].clamp_min(1e-8)
    image_x = mapped[..., 0] / denominator
    image_y = mapped[..., 1] / denominator
    source_height, source_width = source.shape[-2:]
    sample_grid = torch.stack(
        (
            image_x / max(source_width - 1, 1) * 2.0 - 1.0,
            image_y / max(source_height - 1, 1) * 2.0 - 1.0,
        ),
        dim=-1,
    )[None]
    rectified = functional.grid_sample(
        source,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]

    difference_step = 2
    vertical_difference = torch.sqrt(
        (
            rectified[:, :, difference_step * 2 :]
            - rectified[:, :, : -difference_step * 2]
        ).square().mean(dim=0)
        + 1e-6
    )
    vertical_response = functional.pad(
        vertical_difference,
        (difference_step, difference_step, 0, 0),
    )
    horizontal_difference = torch.sqrt(
        (
            rectified[:, difference_step * 2 :, :]
            - rectified[:, : -difference_step * 2, :]
        ).square().mean(dim=0)
        + 1e-6
    )
    horizontal_response = functional.pad(
        horizontal_difference,
        (0, 0, difference_step, difference_step),
    )

    search_radius = 12
    vertical_expected = tuple(
        edge
        for column in range(4)
        for edge in (22 + column * 100, 98 + column * 100)
    )
    horizontal_expected = tuple(
        edge
        for row in range(6)
        for edge in (22 + row * 118, 108 + row * 118)
    )
    profiles = []
    for expected in vertical_expected:
        candidate_x = torch.arange(
            expected - search_radius,
            expected + search_radius + 1,
            device=device,
        )
        row_support = []
        for row in range(6):
            center_y = 65 + row * 118
            row_support.append(
                vertical_response[
                    center_y - 35 : center_y + 36, candidate_x
                ].mean(dim=0)
            )
        profiles.append(torch.stack(row_support).mean(dim=0))
    for expected in horizontal_expected:
        candidate_y = torch.arange(
            expected - search_radius,
            expected + search_radius + 1,
            device=device,
        )
        column_support = []
        for column in range(4):
            center_x = 60 + column * 100
            column_support.append(
                horizontal_response[
                    candidate_y, center_x - 30 : center_x + 31
                ].mean(dim=1)
            )
        profiles.append(torch.stack(column_support).mean(dim=0))
    compact_profiles = torch.stack(profiles).detach().cpu().numpy()

    refined: list[float] = []
    all_expected = vertical_expected + horizontal_expected
    for expected, profile in zip(all_expected, compact_profiles, strict=True):
        peak_index = int(np.argmax(profile))
        baseline = float(np.quantile(profile, 0.30))
        if float(profile[peak_index]) < baseline + 1.0:
            refined.append(float(expected))
            continue
        low = max(0, peak_index - 2)
        high = min(len(profile), peak_index + 3)
        weights = np.clip(profile[low:high] - baseline, 0.0, None)
        candidate_positions = np.arange(
            expected - search_radius + low,
            expected - search_radius + high,
            dtype=np.float64,
        )
        refined.append(
            float(np.average(candidate_positions, weights=weights))
            if float(weights.sum()) > 1e-6
            else float(expected - search_radius + peak_index)
        )

    vertical_values = refined[: len(vertical_expected)]
    horizontal_values = refined[len(vertical_expected) :]
    vertical_edges = []
    for column in range(4):
        left, right = vertical_values[column * 2 : column * 2 + 2]
        if not 56.0 <= right - left <= 94.0:
            left, right = 22.0 + column * 100, 98.0 + column * 100
        vertical_edges.append((left, right))
    horizontal_edges = []
    for row in range(6):
        top, bottom = horizontal_values[row * 2 : row * 2 + 2]
        if not 66.0 <= bottom - top <= 106.0:
            top, bottom = 22.0 + row * 118, 108.0 + row * 118
        horizontal_edges.append((top, bottom))
    return tuple(vertical_edges), tuple(horizontal_edges)


def _sample_quadrilateral_mean(source, corners: np.ndarray) -> tuple[float, float, float]:
    import torch

    top_left, top_right, bottom_right, bottom_left = (
        torch.as_tensor(corner, device=source.device, dtype=torch.float32)
        for corner in corners
    )
    axis = torch.linspace(0.12, 0.88, 11, device=source.device)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    points = (
        top_left * ((1.0 - uu) * (1.0 - vv))[..., None]
        + top_right * (uu * (1.0 - vv))[..., None]
        + bottom_right * (uu * vv)[..., None]
        + bottom_left * ((1.0 - uu) * vv)[..., None]
    )
    means = [
        float(bilinear_sample(source[:, channel : channel + 1], points[..., 0], points[..., 1]).mean().item())
        for channel in range(3)
    ]
    return tuple(means)


def _regular_tick_span(profile: np.ndarray, intervals: int) -> tuple[float, float, float, float]:
    """Fit a regularly spaced tick train to a downloaded 1-D diagnostic profile."""

    values = np.asarray(profile, np.float64)
    count = values.size
    if count < intervals + 8 or float(values.max()) <= 1e-6:
        return 0.0, float(max(0, count - 1)), 0.0, 0.0
    body_span = count - 1
    nominal = body_span / intervals
    pitches = np.linspace(nominal * 0.88, nominal * 1.01, 100)
    best: tuple[float, float, float, float] | None = None
    baseline = float(np.quantile(values, 0.45))
    scale = max(float(np.quantile(values, 0.97) - baseline), 1e-6)
    for pitch in pitches:
        maximum_margin = max(1.0, body_span - intervals * pitch)
        # Ruler-body outlines can be much darker than the actual scale.  The
        # terminal printed dashes are inset from those plastic edges, so do not
        # allow a single outline peak to anchor the periodic fit.
        inset = min(maximum_margin * 0.35, max(pitch * 1.25, body_span * 0.008))
        start_low = min(inset, maximum_margin * 0.48)
        start_high = max(start_low, maximum_margin - inset)
        for start in np.linspace(start_low, start_high, 80):
            positions = start + np.arange(intervals + 1) * pitch
            tick_values = np.interp(positions, np.arange(count), values)
            halfway = np.interp(
                positions[:-1] + pitch * 0.5, np.arange(count), values
            )
            score = float(np.mean(tick_values) - 0.55 * np.mean(halfway))
            if best is None or score > best[0]:
                best = (score, start, positions[-1], pitch)
    assert best is not None
    confidence = float(np.clip((best[0] - baseline * 0.45) / scale, 0.0, 1.0))
    return best[1], best[2], best[3], confidence


def _neutral_balance_gains(card: ColourCardDetection) -> np.ndarray:
    neutral = [
        swatch.mean_bgr
        for swatch in card.swatches
        if max(swatch.mean_bgr) - min(swatch.mean_bgr) < 18.0
        and 35.0 < float(np.mean(swatch.mean_bgr)) < 240.0
    ]
    if len(neutral) < 2:
        ordered = sorted(card.swatches, key=lambda item: max(item.mean_bgr) - min(item.mean_bgr))
        neutral = [item.mean_bgr for item in ordered[:6]]
    channels = np.median(np.asarray(neutral, np.float32), axis=0)
    target = float(np.mean(channels))
    return np.clip(target / np.maximum(channels, 1.0), 0.72, 1.35).astype(np.float32)


def _combined_reference_angle(
    card: ColourCardDetection | None, ruler: RulerDetection | None
) -> float:
    values: list[tuple[float, float]] = []
    if card is not None:
        values.append((card.angle_degrees, card.confidence))
    if ruler is not None:
        values.append((ruler.angle_degrees, ruler.confidence))
    if not values:
        return 0.0
    return float(sum(value * weight for value, weight in values) / sum(weight for _, weight in values))


def _projective_deskew_matrix(
    image_shape: tuple[int, int], card: ColourCardDetection
) -> tuple[np.ndarray, float]:
    source = _order_quad(np.asarray(card.bounds, np.float64))
    top_length = np.linalg.norm(source[1] - source[0])
    bottom_length = np.linalg.norm(source[2] - source[3])
    right_length = np.linalg.norm(source[2] - source[1])
    left_length = np.linalg.norm(source[3] - source[0])
    target_width = (top_length + bottom_length) * 0.5
    target_height = (left_length + right_length) * 0.5
    center = source.mean(axis=0)
    target = np.asarray(
        (
            (center[0] - target_width * 0.5, center[1] - target_height * 0.5),
            (center[0] + target_width * 0.5, center[1] - target_height * 0.5),
            (center[0] + target_width * 0.5, center[1] + target_height * 0.5),
            (center[0] - target_width * 0.5, center[1] + target_height * 0.5),
        ),
        np.float64,
    )
    adjacent = source[1] - source[0], source[3] - source[0]
    cosine = abs(float(np.dot(*adjacent))) / max(
        float(np.linalg.norm(adjacent[0]) * np.linalg.norm(adjacent[1])), 1e-6
    )
    strength = max(
        abs(top_length - bottom_length) / max((top_length + bottom_length) * 0.5, 1.0),
        abs(left_length - right_length) / max((left_length + right_length) * 0.5, 1.0),
        cosine,
    )
    return _homography(source, target).astype(np.float32), float(strength)


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(float(u))
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(float(v))
    solution = np.linalg.solve(np.asarray(rows, np.float64), np.asarray(values, np.float64))
    return np.append(solution, 1.0).reshape(3, 3)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(points.shape[0], np.float64)))
    transformed = homogeneous @ np.asarray(matrix, np.float64).T
    return (transformed[:, :2] / transformed[:, 2:3]).astype(np.float32)


def _expanded_warp_canvas(
    matrix: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Translate a homography into a canvas containing all source corners."""

    corners = np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0),
         (width - 1.0, height - 1.0), (0.0, height - 1.0)),
        dtype=np.float64,
    )
    transformed = _transform_points(corners, matrix).astype(np.float64)
    if not np.all(np.isfinite(transformed)):
        return np.asarray(matrix, np.float32), (height, width)
    minimum = np.floor(transformed.min(axis=0))
    maximum = np.ceil(transformed.max(axis=0))
    output_width = max(1, int(maximum[0] - minimum[0] + 1.0))
    output_height = max(1, int(maximum[1] - minimum[1] + 1.0))
    translation = np.asarray(
        ((1.0, 0.0, -minimum[0]),
         (0.0, 1.0, -minimum[1]),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    translated = translation @ np.asarray(matrix, np.float64)
    return translated.astype(np.float32), (output_height, output_width)


def _rotation_matrix(center_x: float, center_y: float, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        (
            (cosine, sine, (1.0 - cosine) * center_x - sine * center_y),
            (-sine, cosine, sine * center_x + (1.0 - cosine) * center_y),
            (0.0, 0.0, 1.0),
        ),
        np.float64,
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _pleasant_scale_bar_length(pixels_per_mm: float | None, width: int) -> float:
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return 10.0
    target_mm = width * 0.12 / pixels_per_mm
    exponent = 10.0 ** np.floor(np.log10(max(target_mm, 0.1)))
    candidates = np.asarray((1.0, 2.0, 5.0, 10.0)) * exponent
    return float(candidates[np.argmin(np.abs(candidates - target_mm))])


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image must be an H×W BGR NumPy array")
    if image.size == 0:
        raise ValueError("image cannot be empty")
