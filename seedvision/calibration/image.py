"""CUDA-first colour-reference, ruler, deskew, and scale calibration.

Pixel-level work in this module is implemented with PyTorch tensors.  NumPy is
used only for the small sets of detected corners, endpoints, and homographies
that form the public calibration metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

from seedvision.calibration.neutral import (
    NeutralBalanceFacts, NeutralPatchWitness, normalize_neutral_patch_selection,
)

import numpy as np

from seedvision.calibration.colour_chart import identify_chart, locate_chart_grid
from seedvision.calibration.chart_colour import (
    ColourCorrectionFacts,
    apply_linear_rgb_correction,
    fit_chart_correction,
    neutral_correction_facts,
)
from seedvision.calibration.geometry import CalibrationDetectionError
from seedvision.calibration.ruler_search import RulerCandidate, ruler_candidates
from seedvision.timing import raise_if_analysis_cancelled
from seedvision.cuda import (
    CudaContext,
    bgr_to_gray,
    bilinear_sample,
    gaussian_blur,
    image_to_tensor,
    resize,
    tensor_to_image,
    warp_perspective,
)


CHART_REFERENCE_FIT = "Chart reference fit"
GREY_BALANCE_ONLY = "Grey balance only"
COLOUR_CORRECTION_MODELS = (CHART_REFERENCE_FIT, GREY_BALANCE_ONLY)


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """User-adjustable assumptions for the controlled calibration layout."""

    ruler_length_mm: float = 150.0
    minor_tick_mm: float = 1.0
    max_deskew_degrees: float = 10.0
    apply_colour_balance: bool = True
    colour_correction_model: str = "Chart reference fit"
    apply_perspective_correction: bool = True
    max_perspective_fraction: float = 0.25
    neutral_patch_selection: str = "automatic"
    neutral_sample_frame: str = "source"

    def __post_init__(self) -> None:
        object.__setattr__(self, "neutral_patch_selection",
                           normalize_neutral_patch_selection(self.neutral_patch_selection))
        if not isinstance(self.neutral_sample_frame, str) or self.neutral_sample_frame not in ("source", "corrected"):
            raise ValueError("neutral_sample_frame must be source or corrected.")
        if not 10.0 <= self.ruler_length_mm <= 1000.0:
            raise ValueError("ruler_length_mm must be between 10 and 1000.")
        if not 0.1 <= self.minor_tick_mm <= 10.0:
            raise ValueError("minor_tick_mm must be between 0.1 and 10.")
        if not 0.0 <= self.max_deskew_degrees <= 30.0:
            raise ValueError("max_deskew_degrees must be between 0 and 30.")
        if self.colour_correction_model not in COLOUR_CORRECTION_MODELS:
            raise ValueError(
                "colour_correction_model must be one of "
                + ", ".join(COLOUR_CORRECTION_MODELS) + "."
            )
        if not 0.0 <= self.max_perspective_fraction <= 0.50:
            raise ValueError("max_perspective_fraction must be between 0 and 0.50.")


@dataclass(frozen=True, slots=True)
class ColourSwatch:
    """One chart patch, numbered and placed in the chart's own upright layout.

    ``corners`` run clockwise from the patch's top-left corner in the image.
    """

    number: int
    row: int
    column: int
    center_x: float
    center_y: float
    corners: tuple[tuple[float, float], ...]
    mean_bgr: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ColourCardDetection:
    """A located colour chart.

    ``bounds`` outlines the swatch grid, clockwise from its top-left corner in
    the image, and ``angle_degrees`` is its rotation, the mean of its sides'
    residual angles. ``chart`` names the identified chart, whose swatches are
    numbered in its upright reading order; it is None for an unrecognised
    grid, numbered in image reading order. ``rotation_degrees`` is the
    chart's clockwise rotation from upright. ``rows`` and ``columns`` give
    the located grid as it lies in the image, which is part of the chart when
    the image edge cuts it, and ``patch_count`` the patches on the whole chart.
    """

    swatches: tuple[ColourSwatch, ...]
    bounds: tuple[tuple[float, float], ...]
    angle_degrees: float
    confidence: float
    detected_swatch_count: int
    chart: str | None
    rotation_degrees: int
    rows: int
    columns: int
    patch_count: int


@dataclass(frozen=True, slots=True)
class RulerScaleLabel:
    """Semantic ruler text associated with one inferred scale divider.

    ``observed`` means that one or more dark printed glyph components were
    found beside the divider.  The numeric value is still anchored by the tick
    lattice, so a damaged digit cannot change the physical calibration.
    """

    text: str
    unit: str
    tick_index: int
    position: tuple[float, float]
    observed: bool
    confidence: float
    kind: str = "number"


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
    tick_pitch_px: float = 0.0
    matched_tick_count: int = 0
    supported_tick_points: tuple[tuple[float, float], ...] = ()
    outline_points: tuple[tuple[float, float], ...] = ()
    metric_tick_points: tuple[tuple[float, float], ...] = ()
    imperial_tick_points: tuple[tuple[float, float], ...] = ()
    metric_tick_segments: tuple[
        tuple[tuple[float, float], tuple[float, float]], ...
    ] = ()
    imperial_tick_segments: tuple[
        tuple[tuple[float, float], tuple[float, float]], ...
    ] = ()
    metric_number_boxes: tuple[tuple[tuple[float, float], ...], ...] = ()
    imperial_number_boxes: tuple[tuple[tuple[float, float], ...], ...] = ()
    imperial_tick_pitch_px: float = 0.0
    metric_tick_classes: tuple[int, ...] = ()
    imperial_tick_classes: tuple[int, ...] = ()
    metric_unit_divider_indices: tuple[int, ...] = ()
    imperial_unit_divider_indices: tuple[int, ...] = ()
    metric_labels: tuple[RulerScaleLabel, ...] = ()
    imperial_labels: tuple[RulerScaleLabel, ...] = ()
    metric_pixels_per_mm: float = 0.0
    imperial_pixels_per_mm: float = 0.0
    scale_disagreement_percent: float | None = None
    metric_hierarchy_consistency: float = 0.0
    imperial_hierarchy_consistency: float = 0.0
    metric_hierarchy_reliable: bool = False
    imperial_hierarchy_reliable: bool = False
    imperial_scale_reliable: bool = False
    scale_reliable: bool = True


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
    calibration_settings: CalibrationSettings | None = None
    source_shape_hw: tuple[int, int] | None = None
    neutral_balance: NeutralBalanceFacts | None = None
    colour_correction: ColourCorrectionFacts | None = None
    _geometry: _CalibrationGeometry | None = None
    _run_key: str | None = None

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

    def ruler_tick_points_corrected(self) -> np.ndarray:
        """Return supported ruler-tick centres in the corrected image space."""

        if self.ruler is None or not self.ruler.supported_tick_points:
            return np.empty((0, 2), dtype=np.float32)
        points = np.asarray(self.ruler.supported_tick_points, dtype=np.float64)
        if self.ruler_in_corrected_coordinates:
            return points.astype(np.float32)
        return _transform_points(points, self.affine_matrix)

    def imperial_measurement_endpoints_corrected(self) -> np.ndarray:
        """Outer roots of the first/last independently detected inch dividers.

        Falls back to the observed tick span, never extrapolates a metric line.
        """
        ruler = self.ruler
        if ruler is None or len(ruler.imperial_tick_segments) < 2:
            return np.empty((0, 2), dtype=np.float32)
        segments = ruler.imperial_tick_segments
        indices = [i for i in ruler.imperial_unit_divider_indices if 0 <= i < len(segments)]
        if len(indices) < 2:
            indices = [0, len(segments) - 1]
        points = np.asarray((segments[min(indices)][0], segments[max(indices)][0]), float)
        return (points.astype(np.float32) if self.ruler_in_corrected_coordinates
                else _transform_points(points, self.affine_matrix))


@dataclass(frozen=True, slots=True)
class _CalibrationGeometry:
    source_key: tuple
    geometry_settings: tuple
    corrected_bgr: np.ndarray
    corrected_tensor: object
    colour_card: ColourCardDetection | None
    transform: np.ndarray
    deskew_degrees: float
    perspective_corrected: bool
    perspective_strength: float
    gains: tuple[float, float, float]
    neutral_facts: NeutralBalanceFacts
    warnings: tuple[str, ...]
    product_key: tuple | None
    metadata_key: str | None = None
    colour_correction: ColourCorrectionFacts | None = None


def _array_key(value):
    array = np.ascontiguousarray(value)
    return array.shape, array.dtype.str, sha256(memoryview(array)).hexdigest()


def _calibration_source_key(image, context):
    import torch

    device = torch.device(context.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _array_key(image), str(device), "source.uint8-bgr-hwc"


def _geometry_settings(settings):
    return (settings.max_deskew_degrees, settings.apply_colour_balance,
            settings.apply_perspective_correction, settings.max_perspective_fraction,
            settings.neutral_patch_selection, settings.neutral_sample_frame,
            settings.colour_correction_model)


def _compact_metadata_key(value):
    try:
        return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                 separators=(',', ':')).encode('utf-8')).hexdigest()
    except (TypeError, ValueError):
        return None


def _geometry_metadata_key(geometry):
    return _compact_metadata_key((
        None if geometry.colour_card is None else asdict(geometry.colour_card),
        asdict(geometry.neutral_facts),
        None if geometry.colour_correction is None else asdict(geometry.colour_correction),
        geometry.geometry_settings,
        geometry.deskew_degrees, geometry.perspective_corrected,
        geometry.perspective_strength, geometry.gains, geometry.warnings))


def _calibration_run_key(calibration):
    return _compact_metadata_key((
        None if calibration.ruler is None else asdict(calibration.ruler),
        asdict(calibration.calibration_settings), calibration.source_shape_hw,
        calibration.deskew_degrees, calibration.perspective_corrected,
        calibration.perspective_strength, calibration.pixels_per_mm,
        calibration.scale_confidence, calibration.ruler_tick_spacing_px,
        calibration.scale_bar_mm, calibration.ruler_in_corrected_coordinates,
        calibration.warnings))


def _geometry_product_key(corrected, transform, tensor):
    # Inference tensors do not track in-place changes. Keep their fresh result
    # available, but never treat them as a reusable guarded product.
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (_array_key(corrected), _array_key(transform),
            (str(tensor.device), str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()),
             tensor.storage_offset(), tensor.untyped_storage().data_ptr(),
             tensor.untyped_storage().nbytes(), version))


def _reusable_geometry(previous, source_key, settings):
    geometry = getattr(previous, '_geometry', None)
    if not isinstance(geometry, _CalibrationGeometry):
        return None
    if geometry.source_key != source_key or geometry.geometry_settings != _geometry_settings(settings):
        return None
    if geometry.metadata_key is None or geometry.metadata_key != _geometry_metadata_key(geometry):
        return None
    if (previous.corrected_bgr is not geometry.corrected_bgr
            or previous.gpu_corrected_bgr is not geometry.corrected_tensor
            or previous.affine_matrix is not geometry.transform
            or previous.colour_card is not geometry.colour_card
            or previous.neutral_balance is not geometry.neutral_facts
            or previous.colour_correction is not geometry.colour_correction
            or previous.channel_gains_bgr != geometry.gains):
        return None
    if geometry.product_key is None or geometry.product_key != _geometry_product_key(
            geometry.corrected_bgr, geometry.transform, geometry.corrected_tensor):
        return None
    return geometry


def calibration_cache_matches(previous, image, settings, context):
    """Require actual settings, source pixels/device and retained products."""
    return (getattr(previous, 'calibration_settings', None) == settings
            and getattr(previous, '_run_key', None) is not None
            and previous._run_key == _calibration_run_key(previous)
            and _reusable_geometry(previous, _calibration_source_key(image, context), settings) is not None)


def _calibrate_geometry(image, settings, context, timing_recorder, source_key):
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
    gains, neutral_facts = _neutral_balance(colour_card, settings, corrected_tensor, transform)
    correction = _chart_colour_correction(colour_card, settings, corrected_tensor, transform)
    if isinstance(correction, str):
        # Grey balance is the fallback whenever no known chart can be fitted.
        colour_facts = neutral_correction_facts(
            correction, applied=neutral_facts.state == "applied"
        )
        if (
            settings.apply_colour_balance
            and settings.colour_correction_model == CHART_REFERENCE_FIT
            and colour_card is not None
        ):
            warnings.append(f"Colour correction used grey balance only. {correction}")
        if neutral_facts.state == "unavailable":
            warnings.append(neutral_facts.reason)
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
    else:
        # The chart fit corrects every channel against the reference colours,
        # greys included, so no separate channel gains are applied.
        colour_facts, matrix, offset = correction
        gains = np.ones(3, np.float32)
        corrected_tensor = apply_linear_rgb_correction(corrected_tensor, matrix, offset)
    corrected = tensor_to_image(corrected_tensor)
    if deskew_timing is not None:
        timing_recorder.stop(deskew_timing)

    geometry = _CalibrationGeometry(source_key, _geometry_settings(settings), corrected,
        corrected_tensor, colour_card, transform, float(deskew_degrees), perspective_corrected,
        float(perspective_strength), tuple(float(value) for value in gains), neutral_facts,
        tuple(warnings), _geometry_product_key(corrected, transform, corrected_tensor),
        colour_correction=colour_facts)
    return replace(geometry, metadata_key=_geometry_metadata_key(geometry))


def calibrate_image(
    image: np.ndarray,
    settings: CalibrationSettings | None = None,
    *,
    cuda_context: CudaContext | None = None,
    timing_recorder=None,
    _previous_geometry: ImageCalibration | None = None,
    _kept_ruler: ImageCalibration | None = None,
) -> ImageCalibration:
    """Detect references and perform all raster calibration on one GPU.

    ``_kept_ruler`` holds a previous ruler for a run that leaves the ruler node
    out. A previous failure to find one is retried, so that its warning recurs.
    """

    _validate_image(image)
    settings = settings or CalibrationSettings()
    context = cuda_context or CudaContext.resolve()
    source_key = _calibration_source_key(image, context)
    geometry = _reusable_geometry(_previous_geometry, source_key, settings)
    if geometry is None:
        geometry = _calibrate_geometry(image, settings, context, timing_recorder, source_key)
    corrected, corrected_tensor = geometry.corrected_bgr, geometry.corrected_tensor
    colour_card, transform = geometry.colour_card, geometry.transform
    deskew_degrees, perspective_corrected = geometry.deskew_degrees, geometry.perspective_corrected
    perspective_strength, gains = geometry.perspective_strength, geometry.gains
    warnings = list(geometry.warnings)

    kept_ruler = None if _kept_ruler is None else _kept_ruler.ruler
    ruler = kept_ruler
    if ruler is None:
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
                excluded_regions=(
                    () if colour_card is None
                    else (_chart_exclusion_region(colour_card, transform),)
                ),
            )
        except CalibrationDetectionError as error:
            ruler = None
            warnings.append(str(error))
        if ruler_timing is not None:
            timing_recorder.stop(ruler_timing)

    scale_timing = (
        None
        if timing_recorder is None or kept_ruler is not None
        else timing_recorder.start("scale_calibration")
    )
    pixels_per_mm: float | None = None
    tick_spacing: float | None = None
    scale_confidence = 0.0
    if ruler is not None and ruler.scale_reliable:
        # The regular metric tick family is more local and robust than the two
        # terminal endpoints, which may sit inside a clipped/rounded plastic
        # outline. Use its measured pitch for absolute scale.
        pixels_per_mm = (
            ruler.metric_pixels_per_mm
            if ruler.metric_pixels_per_mm > 0.0
            else ruler.tick_pitch_px / settings.minor_tick_mm
        )
        tick_spacing = ruler.tick_pitch_px
        disagreement = ruler.scale_disagreement_percent
        consistency_factor = (
            1.0
            if disagreement is None or disagreement <= 1.0
            else float(np.clip(1.0 - (disagreement - 1.0) / 18.0, 0.35, 1.0))
        )
        scale_confidence = ruler.confidence * consistency_factor
    if scale_timing is not None:
        timing_recorder.stop(scale_timing)

    if colour_card is None and ruler is None:
        warnings.append("Deskew could not be estimated because neither reference was found.")
    elif colour_card is None:
        warnings.append("The ruler was detected without colour-card deskew.")
    if colour_card is not None and (ruler is None or not ruler.scale_reliable):
        warnings.append("Deskew succeeded, but ruler scale could not be assigned.")
    elif colour_card is None and ruler is not None and not ruler.scale_reliable:
        warnings.append("The ruler outline was found, but metric scale could not be assigned.")
    if (
        ruler is not None
        and ruler.scale_disagreement_percent is not None
        and ruler.scale_disagreement_percent > 3.0
    ):
        warnings.append(
            "Metric and imperial tick scales disagree by "
            f"{ruler.scale_disagreement_percent:.2f}%; metric remains authoritative."
        )

    if _calibration_source_key(image, context) != source_key:
        raise ValueError("Source pixels changed during calibration.")
    result = ImageCalibration(
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
        calibration_settings=settings,
        source_shape_hw=tuple(image.shape[:2]),
        neutral_balance=geometry.neutral_facts,
        colour_correction=geometry.colour_correction,
        _geometry=geometry,
    )
    return replace(result, _run_key=_calibration_run_key(result))


def detect_colour_card(
    image: np.ndarray,
    *,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
) -> ColourCardDetection:
    """Find a grid colour chart at any position and orientation on the GPU.

    The chart is located from its swatch lattice and identified from its
    colours (see ``seedvision.calibration.colour_chart``). Swatches are
    returned in patch-number order.
    """

    _validate_image(image)
    context = cuda_context or CudaContext.resolve()
    source = image_tensor if image_tensor is not None else image_to_tensor(image, context)
    grid = locate_chart_grid(source)
    identity = identify_chart(grid)
    chart = identity.chart
    layout_columns = chart.columns if chart is not None else grid.columns
    swatches = []
    for row in range(grid.rows):
        for column in range(grid.columns):
            index = int(identity.canonical_index[row, column])
            chart_row, chart_column = divmod(index, layout_columns)
            centre = grid.centres[row, column]
            swatches.append(
                ColourSwatch(
                    number=index + 1,
                    row=chart_row,
                    column=chart_column,
                    center_x=float(centre[0]),
                    center_y=float(centre[1]),
                    corners=tuple((float(x), float(y)) for x, y in grid.corners[row, column]),
                    mean_bgr=tuple(float(value) for value in grid.mean_bgr[row, column]),
                )
            )
    swatches.sort(key=lambda swatch: swatch.number)
    # Rotation is the mean of all four sides' residual angles: a keystoned
    # chart's top edge alone also carries its perspective.
    sides = np.roll(grid.outline, -1, axis=0) - grid.outline
    residual = np.arctan2(sides[:, 1], sides[:, 0]) - np.radians((0.0, 90.0, 180.0, 270.0))
    angle = float(np.degrees(np.arctan2(np.sin(residual).mean(), np.cos(residual).mean())))
    detected = int(grid.detected.sum())
    # Missed swatches and a poor lattice fit both lower confidence; a
    # residual of a quarter pitch leaves none.
    confidence = float(np.clip(
        0.98 * detected / grid.detected.size * (1.0 - 4.0 * grid.residual), 0.0, 0.98))
    return ColourCardDetection(
        swatches=tuple(swatches),
        bounds=tuple((float(x), float(y)) for x, y in grid.outline),
        angle_degrees=angle,
        confidence=confidence,
        detected_swatch_count=detected,
        chart=None if chart is None else chart.name,
        rotation_degrees=int(identity.rotation_degrees),
        rows=int(grid.rows),
        columns=int(grid.columns),
        patch_count=len(chart.rgb) if chart is not None else int(grid.rows * grid.columns),
    )


_RULER_CANDIDATES = 4
# A ruler measured from its visible tick run needs at least 3 cm in view, and
# dashes no closer than 2.5 pixels, below which neighbours blur together.
_VISIBLE_RUN_MINIMUM_INTERVALS = 30
_SHORTEST_TICK_PITCH = 2.5
# A strip narrower than this many dashes may hold only the minor dashes, so
# it is widened across the centimetre marks before their lengths are read.
_VISIBLE_RUN_STRIP_PITCHES = 12.0
# Grid units added around the colour chart's swatch outline where ruler edges
# may not vote and the ruler strip may not extend.
_CHART_EXCLUSION_MARGIN = 0.35


def detect_ruler(
    image: np.ndarray,
    ruler_length_mm: float = 150.0,
    *,
    minor_tick_mm: float = 1.0,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
    excluded_regions: tuple[np.ndarray, ...] = (),
) -> RulerDetection:
    """Detect the ruler body plus independent metric/imperial tick evidence.

    The ruler may lie anywhere and at any angle. Ruler-width pairs of
    parallel edges are found over the whole image, outside
    ``excluded_regions`` (convex image polygons such as the colour chart),
    and the strongest few are measured in turn until one carries a reliable
    metric tick train of ``ruler_length_mm``. If none does, each is measured
    again over its longest visible run of regular ticks, for a ruler cut by
    the frame or the chart, or shorter than the configured span.

    A band that carries a regular metric tick train is returned even when
    that train is not yet safe for absolute scale, so the diagnostic overlay
    can explain which stage failed. A band without one is not a ruler.
    """

    _validate_image(image)
    context = cuda_context or CudaContext.resolve()
    source = image_tensor if image_tensor is not None else image_to_tensor(image, context)
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small_height = max(32, round(height * scale))
    small_width = max(32, round(width * scale))
    small = resize(source, (small_height, small_width), mode="area") if scale < 1.0 else source
    gray = gaussian_blur(bgr_to_gray(small), 0.65)
    excluded = tuple(np.asarray(region, np.float64).reshape(-1, 2) for region in excluded_regions)
    candidates = ruler_candidates(
        gray, tuple(region * scale for region in excluded), count=_RULER_CANDIDATES
    )
    if not candidates:
        raise CalibrationDetectionError("Ruler was not detected.")
    # Ticks are measured on a strip resampled from the full-resolution image.
    full_gray = gaussian_blur(bgr_to_gray(source), 0.65) if scale < 1.0 else gray
    best: RulerDetection | None = None
    for candidate in candidates:
        raise_if_analysis_cancelled()
        try:
            ruler = _measure_ruler_candidate(
                full_gray,
                candidate,
                scale,
                excluded,
                ruler_length_mm=ruler_length_mm,
                minor_tick_mm=minor_tick_mm,
                device=context.device,
            )
        except CalibrationDetectionError:
            continue
        if ruler.scale_reliable:
            return ruler
        # Without a periodic tick family a long straight band is not evidence
        # of a ruler: box walls, label edges and paper edges form such bands,
        # and the strokes of a printed label count as a few dozen loose ticks.
        if ruler.metric_tick_segments and (
            best is None
            or (ruler.matched_tick_count, ruler.confidence)
            > (best.matched_tick_count, best.confidence)
        ):
            best = ruler
    # No band holds the configured span: measure the visible tick run instead.
    for candidate in candidates:
        raise_if_analysis_cancelled()
        try:
            ruler = _measure_ruler_candidate(
                full_gray,
                candidate,
                scale,
                excluded,
                ruler_length_mm=ruler_length_mm,
                minor_tick_mm=minor_tick_mm,
                device=context.device,
                visible_span=True,
            )
        except CalibrationDetectionError:
            continue
        if ruler.scale_reliable:
            return ruler
    if best is None:
        raise CalibrationDetectionError("Ruler was not detected.")
    return best


def _excluded_strip_span(
    regions: tuple[np.ndarray, ...],
    cosine: float,
    sine: float,
    v_low: float,
    v_high: float,
) -> list[tuple[float, float]]:
    """Ranges of u where each convex region crosses the band v_low..v_high."""

    spans = []
    for region in regions:
        u = cosine * region[:, 0] + sine * region[:, 1]
        v = -sine * region[:, 0] + cosine * region[:, 1]
        polygon = list(zip(u.tolist(), v.tolist()))
        for limit, keep_below in ((v_high, True), (v_low, False)):
            clipped = []
            for index, (u0, v0) in enumerate(polygon):
                u1, v1 = polygon[(index + 1) % len(polygon)]
                inside0 = v0 <= limit if keep_below else v0 >= limit
                inside1 = v1 <= limit if keep_below else v1 >= limit
                if inside0:
                    clipped.append((u0, v0))
                if inside0 != inside1:
                    fraction = (limit - v0) / (v1 - v0)
                    clipped.append((u0 + fraction * (u1 - u0), limit))
            polygon = clipped
            if not polygon:
                break
        if polygon:
            values = [point[0] for point in polygon]
            spans.append((min(values), max(values)))
    return spans


def _measure_ruler_candidate(
    gray,
    candidate: RulerCandidate,
    candidate_scale: float,
    excluded: tuple[np.ndarray, ...],
    *,
    ruler_length_mm: float,
    minor_tick_mm: float,
    device,
    visible_span: bool = False,
    widened: bool = False,
) -> RulerDetection:
    """Measure the tick families and outline of one ruler-width band.

    With ``visible_span`` the metric train spans the longest visible run of
    regular ticks rather than ``ruler_length_mm``. Its ends need not be whole
    centimetres where the frame, the chart or an object cuts the ruler, but
    must be where blank ruler continues beyond them. The imperial family is
    not sought, and labels are given only from a visible zero.
    """

    import torch

    height, width = gray.shape[-2:]
    best_angle = candidate.angle_degrees
    pair_score = candidate.pair_score
    working_side = max(height, width) * candidate_scale
    angle_radians = np.deg2rad(best_angle)
    cosine, sine = float(np.cos(angle_radians)), float(np.sin(angle_radians))
    # Orientation and body-line discovery are safely coarse, but 1 mm dashes
    # on the six-megapixel fixtures are only about 18 pixels apart, so all
    # tick and outline work uses a strip resampled at full resolution.
    scale = 1.0
    u_min = candidate.u_min / candidate_scale
    u_max = candidate.u_max / candidate_scale
    top_v = candidate.top_v / candidate_scale
    bottom_v = candidate.bottom_v / candidate_scale
    ruler_width = bottom_v - top_v
    # A visible run's ends are judged on the ruler beyond them: a band at least
    # 12 dashes wide extends three dashes past its ends.
    sampling_margin = max(8.0, ruler_width * (0.25 if visible_span else 0.16))
    image_corner_u = np.asarray(
        (
            0.0,
            cosine * (width - 1),
            sine * (height - 1),
            cosine * (width - 1) + sine * (height - 1),
        ),
        dtype=np.float64,
    )
    # The band's extent comes from edge points on a coarse image; extend it
    # at both ends so terminal ticks and outline edges are sampled, but never
    # across the colour chart, whose swatch edges mimic half-inch dividers.
    centre_u = (u_min + u_max) * 0.5
    u_min = max(float(np.min(image_corner_u)), u_min - sampling_margin)
    u_max = min(float(np.max(image_corner_u)), u_max + sampling_margin)
    outline_v_margin = max(10.0, ruler_width * 0.24)
    for low, high in _excluded_strip_span(
        excluded, cosine, sine, top_v - outline_v_margin, bottom_v + outline_v_margin
    ):
        if high < centre_u:
            u_min = max(u_min, high)
        elif low > centre_u:
            u_max = min(u_max, low)
    # One sample per pixel along the ruler, so strip pitch is pixel pitch.
    strip_length = max(16, int(np.floor(u_max - u_min)) + 1)
    u_max = u_min + strip_length - 1
    strip_height = max(8, round(ruler_width) + 1)
    sample_u = torch.linspace(u_min, u_max, strip_length, device=device)
    sample_v = torch.linspace(top_v + 2.0, bottom_v - 2.0, strip_height, device=device)
    grid_v, grid_u = torch.meshgrid(sample_v, sample_u, indexing="ij")
    sample_x = cosine * grid_u - sine * grid_v
    sample_y = sine * grid_u + cosine * grid_v
    sampled = bilinear_sample(gray, sample_x, sample_y)
    outline_margin = max(4.0, ruler_width * 0.045)
    outline_offsets = torch.linspace(
        -outline_margin,
        outline_margin,
        max(11, min(65, round(outline_margin * 2.0) + 1)),
        device=device,
    )
    outline_grid_u = sample_u[None, :].expand(outline_offsets.numel(), -1)

    def transverse_edge_support(base_v: float):
        outline_grid_v = base_v + outline_offsets[:, None]
        outline_x = cosine * outline_grid_u - sine * outline_grid_v
        outline_y = sine * outline_grid_u + cosine * outline_grid_v
        values = bilinear_sample(gray, outline_x, outline_y)
        return torch.max(torch.abs(values[1:] - values[:-1]), dim=0).values

    # Both long ruler sides must begin/end at the same transverse edge.  Taking
    # the minimum rejects the supplied image's colour-card bottom edge, which
    # happens to align with only the ruler's lower side.
    long_edge_support = torch.minimum(
        transverse_edge_support(top_v),
        transverse_edge_support(bottom_v),
    ).detach().to(device="cpu", dtype=torch.float32).numpy()
    sampled_host = sampled.detach().to(device="cpu", dtype=torch.float32).numpy()
    if visible_span:
        run = _visible_tick_run(sampled_host)
        if run is None:
            raise CalibrationDetectionError("No regular tick run was found.")
        consensus_pitch, expected_intervals = run
        reach = _VISIBLE_RUN_STRIP_PITCHES * consensus_pitch - ruler_width
        if reach > 0.0 and not widened:
            # A clear ruler's far edge is faint, so the band found may end at
            # the minor dashes. Widen it across the longer marks, on whichever
            # side they hang, before measuring the band as found.
            for top_shift, bottom_shift in ((0.0, reach), (-reach, 0.0)):
                try:
                    ruler = _measure_ruler_candidate(
                        gray,
                        replace(
                            candidate,
                            top_v=candidate.top_v + top_shift * candidate_scale,
                            bottom_v=candidate.bottom_v + bottom_shift * candidate_scale,
                        ),
                        candidate_scale,
                        excluded,
                        ruler_length_mm=ruler_length_mm,
                        minor_tick_mm=minor_tick_mm,
                        device=device,
                        visible_span=True,
                        widened=True,
                    )
                except CalibrationDetectionError:
                    continue
                if ruler.scale_reliable:
                    return ruler
    else:
        expected_intervals = max(2, round(ruler_length_mm / minor_tick_mm))
        consensus_pitch = _whole_strip_regular_tick_pitch(
            sampled, expected_intervals
        )
    (
        start_index,
        end_index,
        pitch,
        periodicity,
        matched_tick_count,
        tick_row_fraction,
        matched_tick_indices,
    ) = _detect_regular_tick_train(
        sampled,
        expected_intervals,
        fixed_pitch=consensus_pitch,
    )
    metric_reliable = bool(
        periodicity >= 0.16
        and matched_tick_count >= max(12, expected_intervals * 0.60)
        and pitch > 0.0
    )

    # A typical dual 15 cm / 6 inch ruler has a separately printed 1/16-inch
    # family on the opposite edge. Detect it independently: it is useful
    # corroborating evidence and must never be averaged into the metric pitch.
    imperial_intervals = max(16, round(ruler_length_mm / 25.4) * 16)
    (
        imperial_start_index,
        _imperial_end_index,
        imperial_pitch,
        imperial_periodicity,
        imperial_matched_tick_count,
        imperial_tick_row_fraction,
        imperial_matched_tick_indices,
    ) = (
        (0.0, 0.0, 0.0, 0.0, 0, 1.0 - float(tick_row_fraction), ())
        if visible_span
        else _detect_regular_tick_train(
            sampled,
            imperial_intervals,
            # The whole strip also contains the denser metric family and
            # barcode; let the independently localized lower-edge band refine
            # its own pitch.
            fixed_pitch=0.0,
            excluded_row_fraction=float(tick_row_fraction),
        )
    )
    imperial_reliable = bool(
        imperial_periodicity >= 0.14
        and imperial_matched_tick_count >= max(10, imperial_intervals * 0.52)
        and imperial_pitch > 0.0
        and abs(imperial_tick_row_fraction - tick_row_fraction) >= 0.25
    )

    (
        metric_segments_strip,
        metric_supported_indices,
        metric_tick_classes,
        metric_unit_dividers,
        metric_hierarchy_consistency,
    ) = (
        _map_complete_tick_segments(
            sampled_host,
            start=float(start_index),
            pitch=float(pitch),
            intervals=expected_intervals,
            row_fraction=float(tick_row_fraction),
            family="metric",
        )
        if metric_reliable
        else ((), (), (), (), 0.0)
    )
    (
        imperial_segments_strip,
        imperial_supported_indices,
        imperial_tick_classes,
        imperial_unit_dividers,
        imperial_hierarchy_consistency,
    ) = (
        _map_complete_tick_segments(
            sampled_host,
            start=float(imperial_start_index),
            pitch=float(imperial_pitch),
            intervals=imperial_intervals,
            row_fraction=float(imperial_tick_row_fraction),
            family="imperial",
        )
        if imperial_reliable
        else ((), (), (), (), 0.0)
    )

    ends_anchored = bool(
        metric_unit_dividers
        and metric_unit_dividers[0] == 0
        and metric_unit_dividers[-1] == expected_intervals
    )
    zero_visible = True
    if visible_span:
        ends_anchored = zero_visible = False
        if metric_segments_strip and len(metric_unit_dividers) >= 3:
            first = float(metric_segments_strip[0][0])
            last = float(metric_segments_strip[-1][0])
            run_pitch = (last - first) / max(len(metric_segments_strip) - 1, 1)
            start_whole = metric_unit_dividers[0] == 0
            end_whole = metric_unit_dividers[-1] == expected_intervals
            start_blank = _run_end_is_blank_ruler(sampled_host, first, run_pitch, -1)
            end_blank = _run_end_is_blank_ruler(sampled_host, last, run_pitch, 1)
            ends_anchored = (start_whole or not start_blank) and (end_whole or not end_blank)
            zero_visible = (
                start_blank and start_whole
                if tick_row_fraction <= 0.5
                else end_blank and end_whole
            )
    metric_hierarchy_reliable = bool(
        metric_tick_classes
        and metric_hierarchy_consistency >= 0.55
        and ends_anchored
    )
    imperial_hierarchy_reliable = bool(
        imperial_tick_classes
        and imperial_hierarchy_consistency >= 0.62
        and imperial_unit_dividers
        and imperial_unit_dividers[0] == 0
        and imperial_unit_dividers[-1] == imperial_intervals
    )
    metric_reliable = bool(metric_reliable and metric_hierarchy_reliable)
    imperial_reliable = bool(imperial_reliable and imperial_hierarchy_reliable)

    if metric_segments_strip:
        metric_positions = np.asarray(
            [segment[0] for segment in metric_segments_strip],
            dtype=np.float64,
        )
        start_index = float(metric_positions[0])
        end_index = float(metric_positions[-1])
        pitch = _lattice_pitch(metric_positions)
        matched_tick_indices = metric_supported_indices
        matched_tick_count = len(metric_supported_indices)
    if imperial_segments_strip:
        imperial_positions = np.asarray(
            [segment[0] for segment in imperial_segments_strip],
            dtype=np.float64,
        )
        imperial_start_index = float(imperial_positions[0])
        _imperial_end_index = float(imperial_positions[-1])
        imperial_pitch = _lattice_pitch(imperial_positions)
        imperial_matched_tick_indices = imperial_supported_indices
        imperial_matched_tick_count = len(imperial_supported_indices)

    # The ruler ends must be inferred from the terminal tick rows, not from a
    # broad quantile over every dark edge in the strip.  The old quantile let
    # the adjacent colour-card border become the ruler's left edge.  Search for
    # a coherent transverse edge only just outside the union of the fitted tick
    # trains, and retain the broad bounds solely as a last-resort fallback.
    fitted_families = tuple(
        family
        for family in (
            (
                float(start_index),
                float(end_index),
                float(pitch),
            )
            if metric_reliable
            else None,
            (
                float(imperial_start_index),
                float(_imperial_end_index),
                float(imperial_pitch),
            )
            if imperial_reliable
            else None,
        )
        if family is not None
    )
    outline_start_index, outline_end_index = _tick_constrained_outline_span(
        sampled_host,
        fitted_families,
        long_edge_support=long_edge_support,
    )
    outline_v_min = top_v - outline_v_margin
    outline_v_max = bottom_v + outline_v_margin
    outline_height = max(12, round(outline_v_max - outline_v_min) + 1)
    outline_sample_v = torch.linspace(
        outline_v_min, outline_v_max, outline_height, device=device
    )
    outline_grid_v, outline_grid_u = torch.meshgrid(
        outline_sample_v, sample_u, indexing="ij"
    )
    outline_x = cosine * outline_grid_u - sine * outline_grid_v
    outline_y = sine * outline_grid_u + cosine * outline_grid_v
    sampled_outline_host = (
        bilinear_sample(gray, outline_x, outline_y)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    outline_strip_points = _fit_ruler_outline_quadrilateral(
        sampled_outline_host,
        fitted_families,
        coarse_top_row=(top_v - outline_v_min)
        / max(outline_v_max - outline_v_min, 1e-6)
        * (outline_height - 1),
        coarse_bottom_row=(bottom_v - outline_v_min)
        / max(outline_v_max - outline_v_min, 1e-6)
        * (outline_height - 1),
        initial_left=float(outline_start_index),
        initial_right=float(outline_end_index),
    )

    endpoint_u_a = u_min + start_index / max(strip_length - 1, 1) * (u_max - u_min)
    endpoint_u_b = u_min + end_index / max(strip_length - 1, 1) * (u_max - u_min)
    # Put the vector through the tick field, not through either plastic edge.
    tick_v = top_v + ruler_width * tick_row_fraction

    def image_point_at_u(strip_index: float) -> np.ndarray:
        coordinate_u = u_min + strip_index / max(strip_length - 1, 1) * (
            u_max - u_min
        )
        return np.asarray(
            (
                cosine * coordinate_u - sine * tick_v,
                sine * coordinate_u + cosine * tick_v,
            ),
            np.float64,
        )

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
    supported_tick_points = tuple(
        tuple(
            float(value)
            for value in (
                image_point_at_u(metric_segments_strip[index][0]) / scale
            )
        )
        for index in matched_tick_indices
        if 0 <= index < len(metric_segments_strip)
    )

    def image_point_at_uv(coordinate_u: float, coordinate_v: float) -> tuple[float, float]:
        return (
            float((cosine * coordinate_u - sine * coordinate_v) / scale),
            float((sine * coordinate_u + cosine * coordinate_v) / scale),
        )

    def image_point_at_strip(
        strip_index: float, row_index: float
    ) -> tuple[float, float]:
        coordinate_u = u_min + strip_index / max(strip_length - 1, 1) * (
            u_max - u_min
        )
        coordinate_v = (top_v + 2.0) + row_index / max(strip_height - 1, 1) * (
            max(ruler_width - 4.0, 1.0)
        )
        return image_point_at_uv(coordinate_u, coordinate_v)

    def transformed_tick_segments(segments):
        return tuple(
            (
                image_point_at_strip(strip_index, outer_row),
                image_point_at_strip(strip_index, inner_row),
            )
            for strip_index, outer_row, inner_row in segments
        )

    metric_tick_segments = transformed_tick_segments(metric_segments_strip)
    imperial_tick_segments = transformed_tick_segments(imperial_segments_strip)

    def segment_centres(segments):
        return tuple(
            (
                float((outer[0] + inner[0]) * 0.5),
                float((outer[1] + inner[1]) * 0.5),
            )
            for outer, inner in segments
        )

    metric_tick_points = segment_centres(metric_tick_segments)
    imperial_tick_points = segment_centres(imperial_tick_segments)

    outline_points = tuple(
        image_point_at_uv(
            u_min
            + float(strip_column) / max(strip_length - 1, 1) * (u_max - u_min),
            outline_v_min
            + float(outline_row) / max(outline_height - 1, 1)
            * (outline_v_max - outline_v_min),
        )
        for strip_column, outline_row in outline_strip_points
    )
    fitted_ruler_width = float(
        (
            (outline_strip_points[2][1] + outline_strip_points[3][1])
            - (outline_strip_points[0][1] + outline_strip_points[1][1])
        )
        * 0.5
        / max(outline_height - 1, 1)
        * (outline_v_max - outline_v_min)
    )
    metric_number_boxes_uv = _likely_ruler_number_boxes(
        sampled_host,
        u_min=u_min,
        u_max=u_max,
        top_v=top_v,
        bottom_v=bottom_v,
        tick_row_fraction=tick_row_fraction,
    )
    imperial_number_boxes_uv = (
        _likely_ruler_number_boxes(
            sampled_host,
            u_min=u_min,
            u_max=u_max,
            top_v=top_v,
            bottom_v=bottom_v,
            tick_row_fraction=imperial_tick_row_fraction,
        )
        if imperial_reliable
        else ()
    )

    def transformed_boxes(boxes):
        return tuple(
            tuple(image_point_at_uv(coordinate_u, coordinate_v) for coordinate_u, coordinate_v in box)
            for box in boxes
        )

    def strip_u_coordinates(segments):
        return tuple(
            float(
                u_min
                + strip_index / max(strip_length - 1, 1) * (u_max - u_min)
            )
            for strip_index, _outer, _inner in segments
        )

    metric_labels = _semantic_ruler_labels(
        family="metric",
        tick_coordinates_u=strip_u_coordinates(metric_segments_strip),
        tick_segments=metric_tick_segments,
        unit_divider_indices=metric_unit_dividers if zero_visible else (),
        glyph_boxes_uv=metric_number_boxes_uv,
        minor_tick_mm=float(minor_tick_mm),
        row_fraction=float(tick_row_fraction),
    )
    imperial_labels = _semantic_ruler_labels(
        family="imperial",
        tick_coordinates_u=strip_u_coordinates(imperial_segments_strip),
        tick_segments=imperial_tick_segments,
        unit_divider_indices=imperial_unit_dividers,
        glyph_boxes_uv=imperial_number_boxes_uv,
        minor_tick_mm=float(minor_tick_mm),
        row_fraction=float(imperial_tick_row_fraction),
    )

    metric_pixels_per_mm = (
        float(pitch / scale / max(float(minor_tick_mm), 1e-9))
        if metric_reliable
        else 0.0
    )
    imperial_pixels_per_mm = (
        float(imperial_pitch / scale * 16.0 / 25.4)
        if imperial_reliable
        else 0.0
    )
    scale_disagreement_percent = None
    if metric_pixels_per_mm > 0.0 and imperial_pixels_per_mm > 0.0:
        midpoint_scale = (metric_pixels_per_mm + imperial_pixels_per_mm) * 0.5
        scale_disagreement_percent = float(
            abs(metric_pixels_per_mm - imperial_pixels_per_mm)
            / max(midpoint_scale, 1e-9)
            * 100.0
        )

    length = float(np.linalg.norm(endpoint_b - endpoint_a))
    if length < max(height, width) * 0.18:
        raise CalibrationDetectionError("Detected ruler scale was implausibly short.")
    confidence = float(
        np.clip(
            0.36
            + periodicity * 0.42
            + min(0.12, pair_score / max(working_side, 1.0))
            + (0.08 if imperial_reliable else 0.0),
            0.0,
            0.99,
        )
    )
    return RulerDetection(
        endpoint_a=(float(endpoint_a[0]), float(endpoint_a[1])),
        endpoint_b=(float(endpoint_b[0]), float(endpoint_b[1])),
        center_x=float((endpoint_a[0] + endpoint_b[0]) * 0.5),
        center_y=float((endpoint_a[1] + endpoint_b[1]) * 0.5),
        length_px=length,
        width_px=float(max(fitted_ruler_width, ruler_width * 0.75) / scale),
        angle_degrees=float(best_angle),
        confidence=confidence,
        tick_pitch_px=float(pitch / scale) if metric_reliable else 0.0,
        matched_tick_count=int(matched_tick_count),
        supported_tick_points=supported_tick_points,
        outline_points=outline_points,
        metric_tick_points=metric_tick_points,
        imperial_tick_points=imperial_tick_points,
        metric_tick_segments=metric_tick_segments,
        imperial_tick_segments=imperial_tick_segments,
        metric_number_boxes=transformed_boxes(metric_number_boxes_uv),
        imperial_number_boxes=transformed_boxes(imperial_number_boxes_uv),
        imperial_tick_pitch_px=(
            float(imperial_pitch / scale) if imperial_reliable else 0.0
        ),
        metric_tick_classes=metric_tick_classes,
        imperial_tick_classes=imperial_tick_classes,
        metric_unit_divider_indices=metric_unit_dividers,
        imperial_unit_divider_indices=imperial_unit_dividers,
        metric_labels=metric_labels,
        imperial_labels=imperial_labels,
        metric_pixels_per_mm=metric_pixels_per_mm,
        imperial_pixels_per_mm=imperial_pixels_per_mm,
        scale_disagreement_percent=scale_disagreement_percent,
        metric_hierarchy_consistency=metric_hierarchy_consistency,
        imperial_hierarchy_consistency=imperial_hierarchy_consistency,
        metric_hierarchy_reliable=metric_hierarchy_reliable,
        imperial_hierarchy_reliable=imperial_hierarchy_reliable,
        imperial_scale_reliable=imperial_reliable,
        scale_reliable=metric_reliable,
    )


def _likely_ruler_number_boxes(
    sampled_ruler: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    top_v: float,
    bottom_v: float,
    tick_row_fraction: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return compact likely digit components on the inside of one tick row.

    This is diagnostic evidence rather than OCR. Tick periodicity assigns the
    metric/imperial family; connected dark glyph components on that family's
    inward side show whether printed scale numbers corroborate the ruler body.
    """

    import cv2

    gray = np.asarray(sampled_ruler, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 6:
        return ()
    height, width = gray.shape
    if tick_row_fraction <= 0.5:
        row_start = round(height * min(0.76, tick_row_fraction + 0.12))
        row_stop = round(height * 0.86)
    else:
        row_start = round(height * 0.14)
        row_stop = round(height * max(0.24, tick_row_fraction - 0.12))
    row_start = max(0, min(height - 1, row_start))
    row_stop = max(row_start + 1, min(height, row_stop))
    band = gray[row_start:row_stop]
    if not band.size or float(np.ptp(band)) < 8.0:
        return ()
    normalized = np.uint8(
        np.clip(
            np.rint((band - float(band.min())) / max(float(np.ptp(band)), 1e-6) * 255.0),
            0,
            255,
        )
    )
    thresholded = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        9,
        3,
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        thresholded, 8
    )
    minimum_height = max(2, round(height * 0.07))
    maximum_height = max(minimum_height, round(height * 0.38))
    maximum_width = max(3, round(height * 0.55))
    boxes: list[tuple[tuple[float, float], ...]] = []
    for identifier in range(1, count):
        x, local_y, component_width, component_height, area = (
            int(value) for value in stats[identifier]
        )
        if not minimum_height <= component_height <= maximum_height:
            continue
        if not 1 <= component_width <= maximum_width:
            continue
        if area < max(2, component_height * component_width * 0.10):
            continue
        y = row_start + local_y
        x0 = u_min + x / max(width - 1, 1) * (u_max - u_min)
        x1 = u_min + (x + component_width) / max(width - 1, 1) * (u_max - u_min)
        y0 = top_v + y / max(height - 1, 1) * (bottom_v - top_v)
        y1 = top_v + (y + component_height) / max(height - 1, 1) * (
            bottom_v - top_v
        )
        boxes.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    # Keep the overlay legible if barcode fragments or ruler branding survived
    # the geometric filters.
    return tuple(boxes[:48])


def _semantic_ruler_labels(
    *,
    family: str,
    tick_coordinates_u: tuple[float, ...],
    tick_segments: tuple[
        tuple[tuple[float, float], tuple[float, float]], ...
    ],
    unit_divider_indices: tuple[int, ...],
    glyph_boxes_uv: tuple[tuple[tuple[float, float], ...], ...],
    minor_tick_mm: float,
    row_fraction: float,
) -> tuple[RulerScaleLabel, ...]:
    """Associate printed glyph evidence and physical values with long ticks.

    Digits are deliberately *not* allowed to establish scale.  Their presence
    corroborates a unit divider, while the value comes from the independently
    fitted tick lattice. On the dual-scale fixture ruler each scale counts
    from zero in the direction it reads when its ticks hang from the upper
    edge, so the edge carrying a family's ticks (``row_fraction`` across the
    strip) gives its direction at any ruler orientation. Nearby glyphs
    corroborate a label but cannot reverse that orientation or change
    calibration.
    """

    if family not in {"metric", "imperial"}:
        raise ValueError(f"Unknown ruler family {family!r}.")
    if (
        not tick_coordinates_u
        or len(tick_coordinates_u) != len(tick_segments)
        or not unit_divider_indices
    ):
        return ()
    coordinates = np.asarray(tick_coordinates_u, dtype=np.float64)
    box_centres = np.asarray(
        [
            float(np.mean([point[0] for point in box]))
            for box in glyph_boxes_uv
            if box
        ],
        dtype=np.float64,
    )
    first_u = float(coordinates[unit_divider_indices[0]])
    last_u = float(coordinates[unit_divider_indices[-1]])
    divider_spacing = abs(last_u - first_u) / max(
        len(unit_divider_indices) - 1, 1
    )
    terminal_band = max(divider_spacing * 0.72, abs(last_u - first_u) * 0.08)
    start_glyphs = int(np.count_nonzero(np.abs(box_centres - first_u) <= terminal_band))
    end_glyphs = int(np.count_nonzero(np.abs(box_centres - last_u) <= terminal_band))
    # Raw glyph density is not a safe orientation signal: branding at the
    # imperial zero end previously outweighed the small zero/unit string and
    # reversed every number label. Tick geometry owns scale; the edge holding
    # the ticks owns label order.
    zero_at_start = row_fraction <= 0.5

    labels: list[RulerScaleLabel] = []
    total_divisions = len(unit_divider_indices) - 1
    for order, tick_index in enumerate(unit_divider_indices):
        coordinate = float(coordinates[tick_index])
        observed = bool(
            box_centres.size
            and np.any(
                np.abs(box_centres - coordinate)
                <= max(2.0, divider_spacing * 0.42)
            )
        )
        physical_order = order if zero_at_start else total_divisions - order
        if family == "metric":
            millimetres = physical_order * 10.0 * float(minor_tick_mm)
            value = millimetres / 10.0
            unit = "cm"
        else:
            value = float(physical_order)
            unit = "in"
        text = f"{value:g}"
        labels.append(
            RulerScaleLabel(
                text=text,
                unit=unit,
                tick_index=int(tick_index),
                position=tuple(float(value) for value in tick_segments[tick_index][1]),
                observed=observed,
                confidence=0.94 if observed else 0.68,
            )
        )

    zero_tick = unit_divider_indices[0 if zero_at_start else -1]
    unit_text = "mm / cm" if family == "metric" else '1/16 in'
    labels.append(
        RulerScaleLabel(
            text=unit_text,
            unit="",
            tick_index=int(zero_tick),
            position=tuple(float(value) for value in tick_segments[zero_tick][1]),
            observed=bool(start_glyphs if zero_at_start else end_glyphs),
            confidence=(
                0.88 if (start_glyphs if zero_at_start else end_glyphs) else 0.62
            ),
            kind="unit",
        )
    )
    return tuple(labels)


def _quadrilateral_sample_points(corners, device):
    import torch

    top_left, top_right, bottom_right, bottom_left = (
        torch.as_tensor(corner, device=device, dtype=torch.float32)
        for corner in corners
    )
    axis = torch.linspace(0.12, 0.88, 11, device=device)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    points = (
        top_left * ((1.0 - uu) * (1.0 - vv))[..., None]
        + top_right * (uu * (1.0 - vv))[..., None]
        + bottom_right * (uu * vv)[..., None]
        + bottom_left * ((1.0 - uu) * vv)[..., None]
    )
    return points


def _sample_quadrilateral_mean(source, corners: np.ndarray) -> tuple[float, float, float]:
    points = _quadrilateral_sample_points(corners, source.device)
    means = [
        float(bilinear_sample(source[:, channel : channel + 1], points[..., 0], points[..., 1]).mean().item())
        for channel in range(3)
    ]
    return tuple(means)


def _expected_tick_level(index: int, family: str, *, phase: int = 0) -> int:
    """Return the printed hierarchy level used only for validation/fallback."""

    offset = int(index + phase)
    if family == "metric":
        return 2 if offset % 10 == 0 else 1 if offset % 5 == 0 else 0
    if family != "imperial":
        raise ValueError(f"Unknown ruler family {family!r}.")
    if offset % 16 == 0:
        return 4
    if offset % 8 == 0:
        return 3
    if offset % 4 == 0:
        return 2
    if offset % 2 == 0:
        return 1
    return 0


def _measured_tick_hierarchy(
    measured: dict[int, tuple[float, float, float]],
    *,
    tick_count: int,
    family: str,
    ruler_height: int,
    minor_and_major_only: bool = False,
) -> tuple[
    dict[int, int],
    tuple[int, ...],
    np.ndarray,
    tuple[int, ...],
    float,
    int,
]:
    """Classify measured tick runs and validate their printed increment phase.

    Length clustering is independent of index.  The known 10-way metric or
    16-way imperial hierarchy is consulted only after classification to select
    phase, quantify consistency, and infer a length for an actually missing
    tick.  This prevents semantic expectations from rewriting contradictory
    image evidence.

    ``minor_and_major_only`` fits a metric scale printed without 5 mm marks:
    two length classes, millimetre and centimetre (levels 0 and 2).
    """

    if family not in {"metric", "imperial"}:
        raise ValueError(f"Unknown ruler family {family!r}.")
    level_count = 3 if family == "metric" else 5
    period = 10 if family == "metric" else 16
    two_level = bool(minor_and_major_only and family == "metric")
    clusters = 2 if two_level else level_count

    def expected_level(index: int, phase: int) -> int:
        if two_level:
            return 2 if (index + phase) % 10 == 0 else 0
        return _expected_tick_level(index, family, phase=phase)

    default_fractions = (
        (0.15, 0.25, 0.38)
        if family == "metric"
        else (0.11, 0.16, 0.22, 0.29, 0.38)
    )
    default_lengths = np.asarray(default_fractions, dtype=np.float64) * float(
        ruler_height
    )
    if not measured:
        inferred = tuple(
            _expected_tick_level(index, family) for index in range(tick_count)
        )
        return {}, inferred, default_lengths, (), 0.0, 0

    indices = np.asarray(sorted(measured), dtype=np.int32)
    lengths = np.asarray(
        [abs(measured[int(index)][1] - measured[int(index)][0]) for index in indices],
        dtype=np.float64,
    )
    strengths = np.asarray(
        [max(0.0, measured[int(index)][2]) for index in indices],
        dtype=np.float64,
    )
    plausible = (
        np.isfinite(lengths)
        & (lengths >= max(2.0, ruler_height * 0.025))
        & (lengths <= ruler_height * 0.55)
    )
    indices = indices[plausible]
    lengths = lengths[plausible]
    strengths = strengths[plausible]
    if lengths.size < max(level_count * 2, 8):
        inferred = tuple(
            _expected_tick_level(index, family) for index in range(tick_count)
        )
        return {}, inferred, default_lengths, (), 0.0, 0

    # A few glare-erased dashes leave short dark stubs. They are supported tick
    # positions but not a new printed length class; the lower decile remains
    # safely inside the overwhelmingly common minor-tick population.
    lower = float(np.quantile(lengths, 0.08))
    upper = float(np.quantile(lengths, 0.98))
    if upper - lower < max(2.0, ruler_height * 0.04):
        inferred = tuple(
            _expected_tick_level(index, family) for index in range(tick_count)
        )
        return {}, inferred, default_lengths, (), 0.0, 0
    clustered_lengths = np.clip(lengths, lower, upper)
    centres = np.linspace(lower, upper, clusters, dtype=np.float64)
    assignments = np.zeros(lengths.size, dtype=np.int32)
    for _round in range(20):
        assignments = np.argmin(
            np.abs(clustered_lengths[:, None] - centres[None, :]), axis=1
        ).astype(np.int32)
        updated = centres.copy()
        for cluster in range(clusters):
            selected = clustered_lengths[assignments == cluster]
            if selected.size:
                updated[cluster] = float(np.median(selected))
        updated = np.maximum.accumulate(updated)
        if np.allclose(updated, centres, atol=1e-4, rtol=0.0):
            break
        centres = updated

    # Empty clusters mean the image does not demonstrate the claimed printed
    # hierarchy. Retain visible lengths, but make the consistency gate fail.
    occupied = tuple(
        cluster for cluster in range(clusters) if np.any(assignments == cluster)
    )
    ordered = sorted(occupied, key=lambda item: float(centres[item]))
    rank = {cluster: order for order, cluster in enumerate(ordered)}
    observed = np.asarray(
        [rank[int(cluster)] for cluster in assignments], dtype=np.int32
    )
    occupied_count = len(ordered)
    if occupied_count != clusters:
        # Map the observed ranks across the full semantic range for a useful
        # display while preserving the low hierarchy score below.
        observed = np.rint(
            observed * (clusters - 1) / max(occupied_count - 1, 1)
        ).astype(np.int32)
    if two_level:
        observed = observed * 2

    strength_scale = max(float(np.median(strengths[strengths > 0])) if np.any(strengths > 0) else 1.0, 1e-6)
    weights = np.clip(np.sqrt(strengths / strength_scale), 0.35, 2.5)
    best_phase = 0
    best_score = -float("inf")
    best_parts = (0.0, 0.0, 0.0, 0.0)
    for phase in range(period):
        expected = np.asarray(
            [expected_level(int(index), phase) for index in indices],
            dtype=np.int32,
        )
        exact = float(np.average(observed == expected, weights=weights))
        rank_agreement = float(
            np.average(
                1.0 - np.abs(observed - expected) / max(level_count - 1, 1),
                weights=weights,
            )
        )
        observed_major = observed == level_count - 1
        expected_major = expected == level_count - 1
        intersection = observed_major & expected_major
        precision = float(
            weights[intersection].sum()
            / max(float(weights[observed_major].sum()), 1e-6)
        )
        recall = float(
            weights[intersection].sum()
            / max(float(weights[expected_major].sum()), 1e-6)
        )
        score = 0.48 * exact + 0.24 * rank_agreement + 0.14 * precision + 0.14 * recall
        if score > best_score:
            best_score = score
            best_phase = phase
            best_parts = (exact, rank_agreement, precision, recall)

    exact, rank_agreement, precision, recall = best_parts
    centre_values = np.asarray(
        [
            float(np.median(clustered_lengths[observed == level]))
            if np.any(observed == level)
            else float(default_lengths[level])
            for level in range(level_count)
        ],
        dtype=np.float64,
    )
    minimum_step = max(1.0, ruler_height * 0.018)
    for level in range(1, level_count):
        centre_values[level] = max(
            centre_values[level], centre_values[level - 1] + minimum_step
        )
    centre_values = np.minimum(centre_values, ruler_height * 0.52)
    within_spreads = []
    for level in range(level_count):
        selected = clustered_lengths[observed == level]
        if selected.size:
            within_spreads.append(
                float(np.median(np.abs(selected - np.median(selected))))
            )
    separated = centre_values[[0, level_count - 1]] if two_level else centre_values
    separation = float(np.min(np.diff(separated))) if level_count > 1 else 0.0
    spread = max(float(np.median(within_spreads)) if within_spreads else 1.0, 1.0)
    separation_quality = float(np.clip(separation / (3.0 * spread), 0.0, 1.0))
    occupancy_quality = occupied_count / clusters
    consistency = float(
        np.clip(
            (0.48 * exact + 0.22 * rank_agreement + 0.15 * precision + 0.15 * recall)
            * separation_quality
            * occupancy_quality,
            0.0,
            1.0,
        )
    )

    measured_classes = {
        int(index): int(level) for index, level in zip(indices, observed, strict=True)
    }
    hierarchy_valid = bool(consistency >= (0.55 if family == "metric" else 0.62))
    if hierarchy_valid:
        expected_levels = np.asarray(
            [
                expected_level(int(index), best_phase)
                for index in indices
            ],
            dtype=np.int32,
        )
        # Measured ranks have now validated the phase. Estimate each semantic
        # class from measurements at those increments so a few glare-shortened
        # strokes cannot become the normal minor-tick class.
        centre_values = np.asarray(
            [
                float(np.median(lengths[expected_levels == level]))
                if np.any(expected_levels == level)
                else float(centre_values[level])
                for level in range(level_count)
            ],
            dtype=np.float64,
        )
        for level in range(1, level_count):
            centre_values[level] = max(
                centre_values[level], centre_values[level - 1] + minimum_step
            )
    inferred = tuple(
        expected_level(index, best_phase)
        if hierarchy_valid
        else measured_classes.get(
            index, expected_level(index, best_phase)
        )
        for index in range(tick_count)
    )
    dividers = tuple(
        index
        for index in range(tick_count)
        if expected_level(index, best_phase) == level_count - 1
    )
    return (
        measured_classes,
        inferred,
        centre_values,
        dividers,
        consistency,
        int(best_phase),
    )


def _map_complete_tick_segments(
    sampled_ruler: np.ndarray,
    *,
    start: float,
    pitch: float,
    intervals: int,
    row_fraction: float,
    family: str,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    float,
]:
    """Map every tick in a reliable periodic family, including weak ticks.

    The lattice fit establishes the positions.  This routine then measures the
    transverse dark run at every lattice position and robustly pools the
    repeated printed length hierarchy (millimetre/5 mm/centimetre for metric;
    sixteenth/eighth/quarter/half/inch for imperial).  A glare-weakened or
    locally occluded dash therefore inherits the measured length of its peers
    rather than disappearing from the evidence overlay.

    Returned triples are ``(strip_u, outer_row, inner_row)``.  The first point
    is always the ruler-edge end of the dash.
    """

    import cv2

    gray = np.asarray(sampled_ruler, dtype=np.float32)
    if (
        gray.ndim != 2
        or min(gray.shape) < 6
        or pitch <= 0.0
        or intervals < 1
        or family not in {"metric", "imperial"}
    ):
        return (), (), (), (), 0.0
    height, width = gray.shape
    positions = start + np.arange(intervals + 1, dtype=np.float64) * pitch
    if positions[0] < -1.0 or positions[-1] > width:
        return (), (), (), (), 0.0

    # A horizontal black-hat isolates narrow transverse dashes without making
    # the long plastic edge itself dark evidence at every column.
    background_width = max(3, min(31, int(round(pitch * 1.8)) | 1))
    horizontal_background = cv2.blur(
        gray,
        (background_width, 1),
        borderType=cv2.BORDER_REPLICATE,
    )
    darkness = np.maximum(horizontal_background - gray, 0.0)
    outer_low = round(height * (0.02 if row_fraction <= 0.5 else 0.82))
    outer_high = round(height * (0.18 if row_fraction <= 0.5 else 0.98))
    outer_low = max(0, min(height - 1, outer_low))
    outer_high = max(outer_low + 1, min(height, outer_high))
    outer_profile = np.mean(darkness[outer_low:outer_high], axis=0)
    deep_low = round(height * (0.20 if row_fraction <= 0.5 else 0.58))
    deep_high = round(height * (0.42 if row_fraction <= 0.5 else 0.80))
    deep_low = max(0, min(height - 1, deep_low))
    deep_high = max(deep_low + 1, min(height, deep_high))
    deep_profile = np.mean(darkness[deep_low:deep_high], axis=0)

    # Follow the actual local maxima rather than forcing perfectly uniform
    # image-space spacing.  Mild residual perspective makes pitch drift by a
    # fraction of a pixel per dash; over 150 intervals that otherwise moves the
    # terminal lattice one or two whole ticks away from the print.
    profile_baseline = float(np.quantile(outer_profile, 0.45))
    profile_high = float(np.quantile(outer_profile, 0.98))
    peaks = _profile_peaks(outer_profile, pitch)

    lattice_observed_indices: tuple[int, ...] = ()
    if peaks.size:
        # Phase is owned by the repeated dashes, not by the coarse strip fit.
        # Search every plausible terminal peak; a bad broad-strip consensus can
        # otherwise strand refinement several centimetres from the true row.
        start_candidates = peaks[
            peaks <= width - max(pitch, intervals * pitch * 0.72)
        ]
        if start_candidates.size == 0:
            start_candidates = np.asarray((positions[0],), dtype=np.float64)
        best_follow: tuple[
            int,
            float,
            float,
            float,
            np.ndarray,
            tuple[int, ...],
        ] | None = None
        evidence_scale = max(profile_high - profile_baseline, 1e-6)
        for candidate_start in start_candidates:
            followed = np.empty(intervals + 1, dtype=np.float64)
            followed[0] = float(candidate_start)
            observed = [0]
            strength_sum = float(
                np.interp(candidate_start, np.arange(width), outer_profile)
            )
            local_pitch = float(pitch)
            for index in range(1, intervals + 1):
                expected = followed[index - 1] + local_pitch
                # The first neighbour establishes that the terminal candidate
                # belongs to the repeated train rather than being the nearby
                # plastic edge.  Subsequent ticks may use a slightly wider
                # window to accommodate gradual perspective drift.
                radius = max(
                    1.25,
                    pitch * (0.24 if index == 1 else 0.38),
                )
                nearby = peaks[
                    (peaks >= expected - radius)
                    & (peaks <= expected + radius)
                ]
                if nearby.size:
                    strengths = np.interp(nearby, np.arange(width), outer_profile)
                    choices = (
                        (strengths - profile_baseline) / evidence_scale
                        - 0.16 * np.abs(nearby - expected) / max(pitch, 1e-6)
                    )
                    selected = float(nearby[int(np.argmax(choices))])
                    followed[index] = selected
                    observed.append(index)
                    strength_sum += float(
                        np.interp(selected, np.arange(width), outer_profile)
                    )
                    step = selected - followed[index - 1]
                    if pitch * 0.78 <= step <= pitch * 1.22:
                        local_pitch = float(
                            np.clip(
                                local_pitch * 0.78 + step * 0.22,
                                pitch * 0.82,
                                pitch * 1.18,
                            )
                        )
                else:
                    followed[index] = expected
            steps = np.diff(followed)
            relative_errors = np.abs(steps / max(pitch, 1e-6) - 1.0)
            terminal_error = float(relative_errors[0] + relative_errors[-1])
            regularity = -float(np.mean(relative_errors) + terminal_error * 0.55)
            hierarchy_stride = 10 if family == "metric" else 16
            major_values = np.interp(
                followed[::hierarchy_stride],
                np.arange(width),
                deep_profile,
            )
            minor_values = np.interp(
                followed[1::hierarchy_stride],
                np.arange(width),
                deep_profile,
            )
            deep_scale = max(
                float(np.quantile(deep_profile, 0.95) - np.quantile(deep_profile, 0.45)),
                1e-6,
            )
            hierarchy = float(
                (np.mean(major_values) - np.mean(minor_values)) / deep_scale
            )
            score = (
                len(observed),
                hierarchy,
                regularity,
                strength_sum / max(len(observed), 1),
            )
            candidate_result = (
                int(score[0]),
                float(score[1]),
                float(score[2]),
                float(score[3]),
                followed,
                tuple(observed),
            )
            if best_follow is None or candidate_result[:4] > best_follow[:4]:
                best_follow = candidate_result
        if best_follow is not None and best_follow[0] >= max(12, (intervals + 1) * 0.55):
            positions = best_follow[4]
            lattice_observed_indices = best_follow[5]

    sample_radius = max(1, int(round(pitch * 0.18)))
    top_family = row_fraction <= 0.5

    def close_short_gaps(mask: np.ndarray, maximum_gap: int = 2) -> np.ndarray:
        closed = np.asarray(mask, dtype=bool).copy()
        indices = np.flatnonzero(closed)
        for first, second in zip(indices[:-1], indices[1:], strict=True):
            if 1 < second - first <= maximum_gap + 1:
                closed[first : second + 1] = True
        return closed

    outer_limit = max(3, round(height * 0.23))
    minimum_length = max(2, round(height * 0.035))
    def measure_ticks(
        current_positions: np.ndarray,
    ) -> dict[int, tuple[float, float, float]]:
        result: dict[int, tuple[float, float, float]] = {}
        for index, position in enumerate(current_positions):
            center = int(round(float(position)))
            low = max(0, center - sample_radius)
            high = min(width, center + sample_radius + 1)
            if high <= low:
                continue
            trace = np.max(darkness[:, low:high], axis=1)
            oriented = trace if top_family else trace[::-1]
            usable_stop = max(outer_limit + minimum_length, round(height * 0.58))
            usable = oriented[: min(height, usable_stop)]
            # Major ticks occupy most of this outward-to-inward trace.  A 40th
            # percentile baseline therefore sat *on the tick itself* and
            # truncated the longest runs, reversing some hierarchy classes.
            baseline = float(np.quantile(usable, 0.15))
            peak = float(np.quantile(usable, 0.97))
            if peak < baseline + 3.0:
                continue
            # Printed ticks remain geometrically connected even where glare
            # weakens their inward end. A low continuation threshold recovers
            # that length; the required outer-rooted run and hierarchy check
            # reject unrelated text strokes.
            threshold = baseline + max(1.5, (peak - baseline) * 0.10)
            support = close_short_gaps(usable >= threshold, maximum_gap=3)
            changes = np.diff(np.pad(support.astype(np.int8), (1, 1)))
            run_starts = np.flatnonzero(changes == 1)
            run_stops = np.flatnonzero(changes == -1)
            candidates: list[tuple[float, int, int]] = []
            for run_start, run_stop in zip(run_starts, run_stops, strict=True):
                run_length = int(run_stop - run_start)
                if run_start > outer_limit or run_length < minimum_length:
                    continue
                run_values = oriented[run_start:run_stop]
                strength = float(
                    np.mean(np.maximum(run_values - baseline, 0.0))
                )
                score = (
                    strength
                    * np.sqrt(float(run_length))
                    / (1.0 + run_start * 0.06)
                )
                candidates.append((score, int(run_start), int(run_stop - 1)))
            if not candidates:
                continue
            score, run_start, run_stop = max(candidates)
            if top_family:
                outer_row = float(run_start)
                inner_row = float(run_stop)
            else:
                outer_row = float(height - 1 - run_start)
                inner_row = float(height - 1 - run_stop)
            result[index] = (outer_row, inner_row, float(score))
        return result

    def classify(current: dict[int, tuple[float, float, float]]):
        hierarchy = _measured_tick_hierarchy(
            current, tick_count=intervals + 1, family=family, ruler_height=height
        )
        if family == "metric" and hierarchy[4] < 0.55:
            # Some scales are printed without 5 mm marks.
            two_level = _measured_tick_hierarchy(
                current,
                tick_count=intervals + 1,
                family=family,
                ruler_height=height,
                minor_and_major_only=True,
            )
            if two_level[4] > hierarchy[4]:
                hierarchy = two_level
        return hierarchy

    measured = measure_ticks(positions)

    # Length is evidence, not a consequence of lattice index.  The former code
    # assigned level(index) first and discarded any measured run that disagreed;
    # a later classifier therefore only rediscovered the synthetic modulo
    # pattern.  Infer classes from the measured runs, then ask whether those
    # independently observed classes agree with the expected printed hierarchy.
    (
        measured_classes,
        inferred_classes,
        representative_lengths,
        unit_dividers,
        hierarchy_consistency,
        hierarchy_phase,
    ) = classify(measured)
    if hierarchy_phase:
        period = 10 if family == "metric" else 16
        index_shift = int((-hierarchy_phase + period // 2) % period - period // 2)
        shifted_axis = np.arange(intervals + 1, dtype=np.float64) + index_shift
        base_axis = np.arange(intervals + 1, dtype=np.float64)
        shifted_positions = np.interp(
            np.clip(shifted_axis, 0.0, float(intervals)), base_axis, positions
        )
        below = shifted_axis < 0.0
        above = shifted_axis > float(intervals)
        if np.any(below):
            shifted_positions[below] = positions[0] + shifted_axis[below] * (
                positions[1] - positions[0]
            )
        if np.any(above):
            shifted_positions[above] = positions[-1] + (
                shifted_axis[above] - intervals
            ) * (positions[-1] - positions[-2])
        # Re-phasing extrapolates the lattice past the dashes it followed.
        # Accept it only where those positions land on printed dashes: a train
        # whose longest marks are not the major increments must not slide half
        # a major interval onto the plastic end or the background beyond it.
        extrapolated = shifted_positions[below | above]
        support_radius = max(1.25, pitch * 0.24)
        unsupported = sum(
            not np.any(np.abs(peaks - position) <= support_radius)
            for position in extrapolated
        )
        if (
            unsupported <= extrapolated.size // 4
            and shifted_positions[0] >= -1.0
            and shifted_positions[-1] <= width
        ):
            positions = shifted_positions
            measured = measure_ticks(positions)
            (
                measured_classes,
                inferred_classes,
                representative_lengths,
                unit_dividers,
                hierarchy_consistency,
                hierarchy_phase,
            ) = classify(measured)
            lattice_observed_indices = tuple(
                sorted(
                    index - index_shift
                    for index in lattice_observed_indices
                    if 0 <= index - index_shift <= intervals
                )
            )

    roots = np.asarray(
        [values[0] for values in measured.values()], dtype=np.float64
    )
    root_positions = np.asarray(
        [positions[index] for index in measured], dtype=np.float64
    )
    fallback_root = height * (0.055 if top_family else 0.945)
    if roots.size:
        root_median = float(np.median(roots))
        root_inliers = np.abs(roots - root_median) <= max(2.0, height * 0.055)
        if (
            int(np.count_nonzero(root_inliers)) >= 3
            and float(np.ptp(root_positions[root_inliers])) > 0.0
        ):
            centered_u = (
                root_positions[root_inliers] - float(np.mean(root_positions[root_inliers]))
            ) / max(float(width - 1), 1.0)
            coefficients = np.polyfit(centered_u, roots[root_inliers], 1)
            slope = float(
                np.clip(coefficients[0], -height * 0.08, height * 0.08)
            )
            intercept = float(coefficients[1])
            root_center_u = float(np.mean(root_positions[root_inliers]))
        else:
            slope = 0.0
            intercept = root_median
            root_center_u = float(np.mean(root_positions))
    else:
        slope = 0.0
        intercept = float(fallback_root)
        root_center_u = float(np.mean(positions))

    def fitted_root(position: float) -> float:
        return float(
            intercept
            + slope * (float(position) - root_center_u) / max(float(width - 1), 1.0)
        )

    segments: list[tuple[float, float, float]] = []
    for index, position in enumerate(positions):
        tick_class = int(inferred_classes[index])
        expected_length = float(representative_lengths[tick_class])
        measured_item = measured.get(index)
        root = fitted_root(float(position))
        if measured_item is not None:
            measured_root, measured_inner, _score = measured_item
            measured_length = abs(float(measured_inner - measured_root))
            class_length = float(representative_lengths[tick_class])
            if (
                abs(measured_root - root) <= max(3.0, height * 0.060)
                and class_length * 0.62 <= measured_length <= class_length * 1.45
            ):
                tick_length = measured_length
            else:
                tick_length = class_length
        else:
            tick_length = expected_length
        inner = root + tick_length if top_family else root - tick_length
        segments.append(
            (
                float(position),
                float(np.clip(root, 0.0, height - 1.0)),
                float(np.clip(inner, 0.0, height - 1.0)),
            )
        )
    supported_indices = tuple(
        sorted(set(lattice_observed_indices).union(measured))
    )
    return (
        tuple(segments),
        supported_indices,
        tuple(int(value) for value in inferred_classes),
        unit_dividers,
        float(hierarchy_consistency),
    )


def _profile_peaks(profile: np.ndarray, pitch: float) -> np.ndarray:
    """Sub-pixel dash centres: clear local maxima at least half a pitch apart."""

    import cv2

    values = np.asarray(profile, np.float32)
    peak_radius = max(1, int(round(pitch * 0.24)))
    dilated = cv2.dilate(
        values.reshape(1, -1),
        np.ones((1, peak_radius * 2 + 1), dtype=np.uint8),
    )[0]
    baseline = float(np.quantile(values, 0.45))
    high = float(np.quantile(values, 0.98))
    threshold = baseline + max(0.8, (high - baseline) * 0.10)
    raw_peaks = np.flatnonzero((values >= dilated - 1e-6) & (values >= threshold))
    accepted: list[int] = []
    minimum_gap = max(1.0, pitch * 0.48)
    for candidate in sorted(
        raw_peaks.tolist(), key=lambda value: float(values[value]), reverse=True
    ):
        if all(abs(candidate - existing) >= minimum_gap for existing in accepted):
            accepted.append(int(candidate))
    return _subpixel_peaks(values, np.asarray(sorted(accepted), dtype=np.int64))


def _subpixel_peaks(profile: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Place whole-pixel maxima at the vertex of a parabola through neighbours."""

    refined = peaks.astype(np.float64)
    inner = (peaks > 0) & (peaks < len(profile) - 1)
    index = peaks[inner]
    left, centre, right = profile[index - 1], profile[index], profile[index + 1]
    curvature = left - 2.0 * centre + right
    peaked = curvature < 0.0
    offset = np.zeros(index.shape, np.float64)
    offset[peaked] = 0.5 * (left[peaked] - right[peaked]) / curvature[peaked]
    refined[inner] += np.clip(offset, -0.5, 0.5)
    return refined


def _lattice_pitch(positions: np.ndarray) -> float:
    """Tick pitch as the least-squares slope of position against tick index.

    A median of neighbouring differences quantises the pitch to the position
    precision; the slope over a whole train does not. Ticks more than a
    quarter pitch from the first fit are dropped before refitting.
    """

    index = np.arange(len(positions), dtype=np.float64)
    slope, intercept = np.polyfit(index, positions, 1)
    keep = np.abs(positions - (slope * index + intercept)) <= 0.25 * abs(slope)
    if int(np.count_nonzero(keep)) >= max(3, len(positions) // 2):
        slope, _ = np.polyfit(index[keep], positions[keep], 1)
    return float(slope)


def _tick_constrained_outline_span(
    sampled_ruler: np.ndarray,
    families: tuple[tuple[float, float, float], ...],
    *,
    long_edge_support: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit transverse ruler ends immediately outside fitted tick families.

    A candidate is eligible only outside the terminal ticks and within a small
    number of tick pitches.  Its score rewards a straight vertical gradient
    supported through much of the ruler width, so nearby text and a remote
    colour-card boundary cannot define the ruler body.
    """

    import cv2

    gray = np.asarray(sampled_ruler, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 6 or not families:
        return 0.0, float(max(0, gray.shape[1] - 1))
    height, width = gray.shape
    tick_start = min(float(family[0]) for family in families)
    tick_end = max(float(family[1]) for family in families)
    pitch = float(np.median([family[2] for family in families]))
    if pitch <= 0.0 or tick_end <= tick_start:
        return 0.0, float(width - 1)

    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    global_threshold = max(3.0, float(np.quantile(gradient, 0.86)))
    coherence = np.mean(gradient >= global_threshold, axis=0)
    retained = max(2, round(height * 0.24))
    strongest = np.partition(gradient, -retained, axis=0)[-retained:]
    strength = np.mean(strongest, axis=0)
    score = strength * (0.35 + 2.4 * coherence)
    score = cv2.GaussianBlur(score.reshape(1, -1), (5, 1), 0.8)[0]
    typical = float(np.median(score))
    spread = max(float(np.quantile(score, 0.90) - typical), 1e-6)
    boundary_support = None
    boundary_spread = 1.0
    if long_edge_support is not None:
        candidate_support = np.asarray(long_edge_support, dtype=np.float64)
        if candidate_support.shape == (width,) and np.all(np.isfinite(candidate_support)):
            smoothing = max(3, int(round(pitch * 0.65)) | 1)
            boundary_support = cv2.GaussianBlur(
                candidate_support.reshape(1, -1),
                (smoothing, 1),
                max(0.8, pitch * 0.18),
            )[0]
            boundary_spread = max(
                float(
                    np.quantile(boundary_support, 0.90)
                    - np.quantile(boundary_support, 0.25)
                ),
                1e-6,
            )

    # A ruler end is close to the terminal printed dash.  Keeping this window
    # below five pitches is intentional: the supplied failure has a very strong
    # colour-card boundary 6.3 pitches to the left, which is not eligible no
    # matter how dark or straight it is.
    maximum_distance = max(pitch * 4.75, height * 0.15)
    minimum_distance = max(1.0, pitch * 0.35)

    def select(boundary: float, direction: int) -> float:
        if direction < 0:
            low = max(0, int(np.floor(boundary - maximum_distance)))
            high = min(width - 1, int(np.floor(boundary - minimum_distance)))
        else:
            low = max(0, int(np.ceil(boundary + minimum_distance)))
            high = min(width - 1, int(np.ceil(boundary + maximum_distance)))
        fallback = boundary + direction * pitch * 2.5
        if high < low:
            return float(np.clip(fallback, 0.0, width - 1.0))
        indices = np.arange(low, high + 1, dtype=np.int32)
        evidence = (score[indices] - typical) / spread
        distance = np.abs(indices.astype(np.float64) - boundary) / pitch
        combined = evidence * 0.15 - 0.08 * distance
        onset = None
        if boundary_support is not None:
            inner_near = max(1, round(pitch * 0.45))
            inner_far = max(inner_near + 1, round(pitch * 2.0))
            onset_values = []
            for index in indices:
                if direction < 0:
                    outside = boundary_support[
                        max(0, index - inner_far) : max(0, index - inner_near)
                    ]
                    inside = boundary_support[
                        min(width, index + inner_near) : min(width, index + inner_far)
                    ]
                else:
                    inside = boundary_support[
                        max(0, index - inner_far) : max(0, index - inner_near)
                    ]
                    outside = boundary_support[
                        min(width, index + inner_near) : min(width, index + inner_far)
                    ]
                onset_values.append(
                    (
                        float(np.mean(inside)) if inside.size else 0.0
                    )
                    - (
                        float(np.mean(outside)) if outside.size else 0.0
                    )
                )
            onset = np.asarray(onset_values, dtype=np.float64) / boundary_spread
            combined += 2.0 * onset
        selected = int(indices[int(np.argmax(combined))])
        if float(np.max(evidence)) < 0.35 and (
            onset is None or float(np.max(onset)) < 0.18
        ):
            return float(np.clip(fallback, 0.0, width - 1.0))
        return float(selected)

    left = select(tick_start, -1)
    right = select(tick_end, 1)
    left = min(left, tick_start - minimum_distance)
    right = max(right, tick_end + minimum_distance)
    if right - left < (tick_end - tick_start) + pitch:
        left = max(0.0, tick_start - pitch * 2.5)
        right = min(float(width - 1), tick_end + pitch * 2.5)
    return float(left), float(right)


def _fit_ruler_outline_quadrilateral(
    sampled_outline: np.ndarray,
    families: tuple[tuple[float, float, float], ...],
    *,
    coarse_top_row: float,
    coarse_bottom_row: float,
    initial_left: float,
    initial_right: float,
) -> tuple[tuple[float, float], ...]:
    """Fit four independent straight ruler sides around the detected ticks.

    The coarse parallel pair normally lands on the two tick-root rows, not the
    outer transparent-plastic edges.  Search outward for each long side and fit
    its slope independently.  The two end edges are then fitted independently
    just beyond the terminal tick families.  Their four intersections form a
    mildly distorted quadrilateral rather than a forced rectangle.
    """

    import cv2

    gray = np.asarray(sampled_outline, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 8 or not families:
        return (
            (float(initial_left), float(coarse_top_row)),
            (float(initial_right), float(coarse_top_row)),
            (float(initial_right), float(coarse_bottom_row)),
            (float(initial_left), float(coarse_bottom_row)),
        )
    height, width = gray.shape
    body_height = max(8.0, float(coarse_bottom_row - coarse_top_row))
    tick_start = min(float(item[0]) for item in families)
    tick_end = max(float(item[1]) for item in families)
    pitch = max(1.0, float(np.median([item[2] for item in families])))
    longitudinal_low = max(0, int(np.floor(min(initial_left, tick_start) - pitch)))
    longitudinal_high = min(
        width - 1, int(np.ceil(max(initial_right, tick_end) + pitch))
    )
    if longitudinal_high - longitudinal_low < max(20, round(width * 0.2)):
        longitudinal_low, longitudinal_high = 0, width - 1
    sample_step = max(1, round((longitudinal_high - longitudinal_low + 1) / 900))
    sample_columns = np.arange(
        longitudinal_low, longitudinal_high + 1, sample_step, dtype=np.int32
    )
    center_u = float((longitudinal_low + longitudinal_high) * 0.5)
    span_u = max(float(longitudinal_high - longitudinal_low), 1.0)
    gradient_v = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    def fit_long_side(base_row: float, direction: int) -> tuple[float, float]:
        minimum_offset = max(2.0, body_height * 0.018)
        maximum_offset = max(minimum_offset + 2.0, body_height * 0.24)
        offsets = np.linspace(minimum_offset, maximum_offset, 64)
        total_drifts = np.linspace(-body_height * 0.04, body_height * 0.04, 25)
        region_low = max(
            0,
            int(np.floor(base_row + direction * maximum_offset - 3.0)),
        )
        region_high = min(
            height,
            int(np.ceil(base_row + direction * minimum_offset + 4.0)),
        )
        if region_low > region_high:
            region_low, region_high = region_high, region_low
        reference = gradient_v[
            max(0, region_low) : min(height, max(region_low + 1, region_high)),
            sample_columns,
        ]
        support_threshold = max(2.0, float(np.quantile(reference, 0.78)))
        clip_high = max(support_threshold, float(np.quantile(reference, 0.97)))
        best: tuple[float, float, float] | None = None
        normalized_u = (sample_columns.astype(np.float64) - center_u) / span_u
        for drift in total_drifts:
            raise_if_analysis_cancelled()
            row_drift = drift * normalized_u
            for offset in offsets:
                rows = np.rint(base_row + direction * offset + row_drift).astype(
                    np.int32
                )
                if direction < 0 and np.any(rows > base_row - minimum_offset * 0.65):
                    continue
                if direction > 0 and np.any(rows < base_row + minimum_offset * 0.65):
                    continue
                valid = (rows >= 0) & (rows < height)
                if int(np.count_nonzero(valid)) < sample_columns.size * 0.75:
                    continue
                values = gradient_v[rows[valid], sample_columns[valid]]
                coverage = float(np.mean(values >= support_threshold))
                strength = float(np.mean(np.minimum(values, clip_high)))
                # Continuous weak plastic edges beat a locally strong tick-root
                # or shadow fragment; a small distance prior favours the first
                # genuine boundary outside the printed scale.
                score = (
                    strength / max(support_threshold, 1e-6)
                    + coverage * 2.4
                    - offset / max(maximum_offset, 1.0) * 0.08
                )
                candidate = (score, float(base_row + direction * offset), float(drift))
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return float(base_row + direction * minimum_offset), 0.0
        return best[1], best[2]

    top_intercept, top_drift = fit_long_side(coarse_top_row, -1)
    bottom_intercept, bottom_drift = fit_long_side(coarse_bottom_row, 1)

    def top_at(column: float) -> float:
        return float(top_intercept + top_drift * (column - center_u) / span_u)

    def bottom_at(column: float) -> float:
        return float(bottom_intercept + bottom_drift * (column - center_u) / span_u)

    gradient_u = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    def fit_end_side(boundary: float, direction: int) -> tuple[float, float]:
        minimum_distance = max(1.5, pitch * 0.28)
        maximum_distance = max(minimum_distance + 2.0, pitch * 2.25)
        intercepts = boundary + direction * np.linspace(
            minimum_distance, maximum_distance, 64
        )
        drifts = np.linspace(-pitch * 0.55, pitch * 0.55, 31)
        y_low = max(0, int(np.floor(min(top_at(boundary), bottom_at(boundary)) + 3)))
        y_high = min(
            height - 1,
            int(np.ceil(max(top_at(boundary), bottom_at(boundary)) - 3)),
        )
        if y_high <= y_low:
            return float(boundary + direction * pitch * 0.75), 0.0
        sample_rows = np.arange(
            y_low,
            y_high + 1,
            max(1, round((y_high - y_low + 1) / 420)),
            dtype=np.int32,
        )
        center_v = float((y_low + y_high) * 0.5)
        span_v = max(float(y_high - y_low), 1.0)
        search_low = max(0, int(np.floor(np.min(intercepts) - abs(drifts[0]) - 2)))
        search_high = min(
            width,
            int(np.ceil(np.max(intercepts) + abs(drifts[-1]) + 3)),
        )
        reference = gradient_u[sample_rows, search_low:search_high]
        support_threshold = max(2.0, float(np.quantile(reference, 0.76)))
        clip_high = max(support_threshold, float(np.quantile(reference, 0.97)))
        normalized_v = (sample_rows.astype(np.float64) - center_v) / span_v
        best: tuple[float, float, float] | None = None
        for drift in drifts:
            raise_if_analysis_cancelled()
            column_drift = drift * normalized_v
            for intercept in intercepts:
                columns = np.rint(intercept + column_drift).astype(np.int32)
                valid = (columns >= 0) & (columns < width)
                if int(np.count_nonzero(valid)) < sample_rows.size * 0.72:
                    continue
                values = gradient_u[sample_rows[valid], columns[valid]]
                coverage = float(np.mean(values >= support_threshold))
                strength = float(np.mean(np.minimum(values, clip_high)))
                distance = abs(float(intercept) - boundary) / pitch
                score = (
                    strength / max(support_threshold, 1e-6)
                    + coverage * 2.6
                    - distance * 0.10
                )
                candidate = (score, float(intercept), float(drift))
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return float(boundary + direction * pitch * 0.75), 0.0
        return best[1], best[2]

    left_intercept, left_drift = fit_end_side(tick_start, -1)
    right_intercept, right_drift = fit_end_side(tick_end, 1)
    center_v = float((top_intercept + bottom_intercept) * 0.5)
    span_v = max(float(bottom_intercept - top_intercept), 1.0)

    def left_at(row: float) -> float:
        return float(left_intercept + left_drift * (row - center_v) / span_v)

    def right_at(row: float) -> float:
        return float(right_intercept + right_drift * (row - center_v) / span_v)

    def intersection(long_side, end_side, initial_column: float) -> tuple[float, float]:
        column = float(initial_column)
        row = float(long_side(column))
        for _round in range(4):
            column = float(end_side(row))
            row = float(long_side(column))
        return (
            float(np.clip(column, 0.0, width - 1.0)),
            float(np.clip(row, 0.0, height - 1.0)),
        )

    return (
        intersection(top_at, left_at, left_intercept),
        intersection(top_at, right_at, right_intercept),
        intersection(bottom_at, right_at, right_intercept),
        intersection(bottom_at, left_at, left_intercept),
    )


def _detect_regular_tick_train(
    sampled_ruler,
    intervals: int,
    *,
    fixed_pitch: float = 0.0,
    excluded_row_fraction: float | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    int,
    float,
    tuple[int, ...],
]:
    """Find the best high-contrast tick band before fitting the ruler span.

    The two printed scales and the plastic outline must not be averaged into one
    signal.  Candidate bands are evaluated independently along both long edges;
    only a band containing a coherent, terminally bounded tick family is used
    to assign the ruler axis.
    """

    import torch
    import torch.nn.functional as functional

    if sampled_ruler.ndim != 2:
        raise ValueError("sampled_ruler must be a two-dimensional strip.")
    height, length = sampled_ruler.shape
    if height < 4 or length < intervals + 8:
        return (
            0.0,
            float(max(0, length - 1)),
            0.0,
            0.0,
            0,
            0.35,
            (),
        )

    kernel = min(31, max(7, (int(length) // 45) | 1))
    if kernel % 2 == 0:
        kernel += 1
    rows = sampled_ruler[:, None, :]
    local_light = functional.avg_pool1d(
        functional.pad(rows, (kernel // 2, kernel // 2), mode="replicate"),
        kernel,
        stride=1,
    )[:, 0]
    row_darkness = (local_light - sampled_ruler).clamp_min(0.0)

    band_fractions = (
        (0.02, 0.15),
        (0.02, 0.22),
        (0.02, 0.32),
        (0.08, 0.25),
        (0.68, 0.98),
        (0.78, 0.98),
        (0.85, 0.98),
        (0.75, 0.92),
    )
    profiles = []
    centres = []
    for low_fraction, high_fraction in band_fractions:
        low = min(int(height) - 1, max(0, round(float(height) * low_fraction)))
        high = min(int(height), max(low + 2, round(float(height) * high_fraction)))
        if high - low < 2:
            continue
        profile = row_darkness[low:high].mean(dim=0)[None, None]
        broad = functional.avg_pool1d(
            functional.pad(profile, (kernel // 2, kernel // 2), mode="replicate"),
            kernel,
            stride=1,
        )
        profiles.append((profile - broad * 0.30).clamp_min(0.0)[0, 0])
        centres.append((low + high - 1) * 0.5 / max(int(height) - 1, 1))

    compact_profiles = torch.stack(profiles).detach().cpu().numpy()
    def major_profile(low_fraction: float, high_fraction: float) -> np.ndarray:
        low = min(int(height) - 1, max(0, round(float(height) * low_fraction)))
        high = min(int(height), max(low + 2, round(float(height) * high_fraction)))
        profile = row_darkness[low:high].mean(dim=0)[None, None]
        broad = functional.avg_pool1d(
            functional.pad(profile, (kernel // 2, kernel // 2), mode="replicate"),
            kernel,
            stride=1,
        )
        return (
            (profile - broad * 0.30)
            .clamp_min(0.0)[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

    top_major_profile = major_profile(0.20, 0.42)
    bottom_major_profile = major_profile(0.58, 0.80)
    best: tuple[
        float,
        float,
        float,
        float,
        int,
        float,
        tuple[int, ...],
    ] | None = None
    for profile, centre in zip(compact_profiles, centres, strict=True):
        if (
            excluded_row_fraction is not None
            and abs(float(centre) - float(excluded_row_fraction)) < 0.25
        ):
            continue
        profile_pitch = (
            float(fixed_pitch)
            if fixed_pitch > 0.0
            else _regular_tick_pitch_consensus(profile, intervals)
        )
        start, end, pitch, confidence, matched, matched_indices = _regular_tick_span(
            profile,
            intervals,
            device=sampled_ruler.device,
            fixed_pitch=profile_pitch,
            major_profile=(
                top_major_profile if centre < 0.5 else bottom_major_profile
            ),
        )
        candidate = (
            start,
            end,
            pitch,
            confidence,
            matched,
            float(centre),
            matched_indices,
        )
        if best is None or (confidence, matched) > (best[3], best[4]):
            best = candidate
    if best is None:
        return (
            0.0,
            float(max(0, length - 1)),
            0.0,
            0.0,
            0,
            0.75 if excluded_row_fraction is not None and excluded_row_fraction < 0.5 else 0.25,
            (),
        )
    return best


def _whole_strip_regular_tick_pitch(sampled_ruler, intervals: int) -> float:
    """Estimate pitch from all rows without allowing them to choose the span.

    The localized tick band owns the axis and phase. Aggregate periodic support
    only stabilizes the millimetre spacing against aliasing of very dense ticks.
    """

    import torch
    import torch.nn.functional as functional

    if sampled_ruler.ndim != 2 or sampled_ruler.shape[1] < intervals + 8:
        return 0.0
    local_median = torch.median(sampled_ruler)
    darkness = (local_median - sampled_ruler).clamp_min(0.0)
    profile = darkness.mean(dim=0)[None, None]
    broad = functional.avg_pool1d(profile, 25, stride=1, padding=12)
    compact = (profile - broad * 0.45).clamp_min(0.0)[0, 0]
    return _regular_tick_pitch_consensus(
        compact.detach().cpu().numpy(), intervals
    )


def _regular_tick_pitch_consensus(profile: np.ndarray, intervals: int) -> float:
    """Return the strongest whole-strip periodic pitch without assigning phase."""

    values = np.asarray(profile, np.float64)
    count = values.size
    if count < intervals + 8 or float(values.max()) <= 1e-6:
        return 0.0
    body_span = count - 1
    nominal = body_span / intervals
    minimum_lag = max(2.0, nominal * 0.78)
    maximum_lag = min(count / 4.0, nominal * 1.04)
    if maximum_lag <= minimum_lag:
        return 0.0
    return _spectral_period(values, minimum_lag, maximum_lag)


def _spectral_period(values: np.ndarray, shortest: float, longest: float) -> float:
    """The period of the strongest spectral peak between two periods.

    Zero padding samples the spectrum finely enough for a sub-pixel period:
    whole-pixel lags quantise a 4.3-pixel pitch to 4 or 5, outside the span
    fit's narrow window around this prior. An autocorrelation, even at
    fractional lags, is pulled towards shorter lags by the strong
    low-frequency 5 and 10 mm hierarchy; the band excludes it.
    """

    values = np.asarray(values, np.float64)
    size = 16 * values.size
    magnitude = np.abs(np.fft.rfft(values - float(np.mean(values)), n=size))
    frequencies = np.fft.rfftfreq(size)
    band = np.flatnonzero(
        (frequencies >= 1.0 / longest) & (frequencies <= 1.0 / shortest)
    )
    if band.size < 3:
        return 0.0
    best = int(band[np.argmax(magnitude[band])])
    frequency = float(frequencies[best])
    if 0 < best < magnitude.size - 1:
        left, centre, right = magnitude[best - 1 : best + 2]
        curvature = left - 2.0 * centre + right
        if curvature < 0.0:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
            frequency += offset * float(frequencies[1])
    return 1.0 / frequency


def _visible_tick_run(sampled_ruler: np.ndarray) -> tuple[float, int] | None:
    """Pitch and interval count of the longest regular tick run in a strip.

    This serves rulers whose configured span is not in view: cut by the frame
    or the chart, or shorter than the setting. Each long edge of the strip is
    searched with no span prior. The strongest spectral period gives the
    pitch, and dashes are chained on that lattice, bridging up to ten missing
    dashes where a seed covers them. A run needs 30 intervals with nearly
    every dash present: barcodes and textures are periodic too, but leave
    much of the lattice empty.
    """

    import cv2

    gray = np.asarray(sampled_ruler, np.float32)
    if gray.ndim != 2 or gray.shape[0] < 6:
        return None
    height, length = gray.shape
    longest_pitch = length / _VISIBLE_RUN_MINIMUM_INTERVALS
    if longest_pitch <= _SHORTEST_TICK_PITCH:
        return None
    kernel = min(31, max(7, (int(length) // 45) | 1))
    background = cv2.blur(gray, (kernel, 1), borderType=cv2.BORDER_REPLICATE)
    darkness = np.maximum(background - gray, 0.0)
    best: tuple[int, float, int] | None = None
    for low_fraction, high_fraction in ((0.02, 0.18), (0.82, 0.98)):
        low = min(height - 1, max(0, round(height * low_fraction)))
        high = min(height, max(low + 2, round(height * high_fraction)))
        profile = darkness[low:high].mean(axis=0)
        pitch = _spectral_period(profile, _SHORTEST_TICK_PITCH, longest_pitch)
        if pitch <= 0.0:
            continue
        peaks = _profile_peaks(profile, pitch)
        tolerance = max(1.0, pitch * 0.22)
        unused = np.ones(peaks.size, dtype=bool)
        for seed in range(peaks.size):
            if not unused[seed]:
                continue
            unused[seed] = False
            indices, members = [0], [float(peaks[seed])]
            local_pitch = pitch
            while True:
                step_found = None
                for step in range(1, 11):
                    expected = members[-1] + step * local_pitch
                    nearby = np.flatnonzero(np.abs(peaks - expected) <= tolerance)
                    if nearby.size:
                        step_found = (
                            step,
                            int(nearby[np.argmin(np.abs(peaks[nearby] - expected))]),
                        )
                        break
                if step_found is None:
                    break
                step, member = step_found
                unused[member] = False
                indices.append(indices[-1] + step)
                members.append(float(peaks[member]))
                if len(members) >= 6:
                    local_pitch = float(np.polyfit(indices[-12:], members[-12:], 1)[0])
            # Bridged gaps may pick up stray texture lines beyond the scale,
            # so each end of a run must be four consecutive dashes.
            while len(indices) > 4 and indices[3] - indices[0] != 3:
                indices.pop(0)
                members.pop(0)
            while len(indices) > 4 and indices[-1] - indices[-4] != 3:
                indices.pop()
                members.pop()
            indices = [index - indices[0] for index in indices]
            intervals = indices[-1]
            if (
                intervals >= _VISIBLE_RUN_MINIMUM_INTERVALS
                and len(members) >= 0.85 * (intervals + 1)
                and (best is None or len(members) > best[0])
            ):
                fitted_pitch = float(np.polyfit(indices, members, 1)[0])
                best = (len(members), fitted_pitch, int(intervals))
    return None if best is None else (best[1], best[2])


def _run_end_is_blank_ruler(
    sampled_ruler: np.ndarray, end: float, pitch: float, direction: int
) -> bool:
    """Whether the ruler goes on unmarked beyond the end of its tick run.

    A printed scale ends on a whole unit where plastic or card continues
    blank. An end cut by the frame, the chart or an object covering the
    ruler, or less than three dashes from the strip's end, may fall anywhere.
    The brightest columns either side compare ruler surface, not dashes.
    """

    gray = np.asarray(sampled_ruler, np.float32)
    length = gray.shape[1]
    beyond = sorted((end + direction * 1.25 * pitch, end + direction * 3.0 * pitch))
    if beyond[0] < 0.0 or beyond[1] > length - 1:
        return False
    inside = sorted((end - direction * 8.0 * pitch, end - direction * 1.0 * pitch))
    inside = (max(0.0, inside[0]), min(float(length - 1), inside[1]))

    def surface(span):
        columns = gray[:, int(np.ceil(span[0])) : int(np.floor(span[1])) + 1]
        return np.quantile(columns, 0.80, axis=1)

    ruler_surface = surface(inside)
    difference = float(np.median(np.abs(surface(beyond) - ruler_surface)))
    return difference <= max(8.0, 0.06 * float(np.median(ruler_surface)))


def _regular_tick_span(
    profile: np.ndarray,
    intervals: int,
    *,
    fixed_pitch: float = 0.0,
    major_profile: np.ndarray | None = None,
    device=None,
) -> tuple[float, float, float, float, int, tuple[int, ...]]:
    """Fit one terminally bounded, regularly spaced tick train to a 1-D profile.

    Every pitch and phase hypothesis is scored at once; the million-sample
    gathers run on ``device`` (the strip's tensor device).
    """

    import torch

    values = np.asarray(profile, np.float64)
    count = values.size
    if count < intervals + 8 or float(values.max()) <= 1e-6:
        return 0.0, float(max(0, count - 1)), 0.0, 0.0, 0, ()
    body_span = count - 1
    nominal = body_span / intervals
    # The whole-strip consensus is only a prior.  Treating it as an exact pitch
    # caused a 1.4% error on the supplied ruler and shifted the entire lattice
    # several ticks left.  The localized row contains the actual dash centres,
    # so refine pitch jointly with phase over a narrow interval around that
    # prior (or the broader geometric range when no prior exists).
    pitches = (
        np.linspace(float(fixed_pitch) * 0.955, float(fixed_pitch) * 1.025, 81)
        if fixed_pitch > 0.0
        else np.linspace(nominal * 0.72, nominal * 1.01, 220)
    )
    best: tuple[float, float, float, float, np.ndarray, np.ndarray] | None = None
    baseline = float(np.quantile(values, 0.45))
    scale = max(float(np.quantile(values, 0.97) - baseline), 1e-6)
    normalized = np.clip((values - baseline) / scale, 0.0, 3.0)
    major_normalized = None
    if major_profile is not None:
        major_values = np.asarray(major_profile, np.float64)
        major_baseline = float(np.quantile(major_values, 0.45))
        major_scale = max(
            float(np.quantile(major_values, 0.97) - major_baseline),
            1e-6,
        )
        major_normalized = np.clip(
            (major_values - major_baseline) / major_scale,
            0.0,
            3.0,
        )
    maximum_margin = np.maximum(1.0, body_span - intervals * pitches)
    # A terminal printed dash can sit within one pitch of a plastic edge (as on
    # the supplied 15 cm ruler).  The former 1.25-pitch symmetric inset made the
    # true 151-dash train geometrically impossible and promoted the opposite
    # outline to tick zero.  Permit near-edge phases; the repeated lattice,
    # halfway rejection and outside-support terms below reject isolated outline
    # peaks.
    inset = np.minimum(
        maximum_margin * 0.20,
        np.maximum(pitches * 0.25, body_span * 0.001),
    )
    start_low = np.minimum(inset, maximum_margin * 0.48)
    start_high = np.maximum(start_low, maximum_margin - inset)
    phase = np.linspace(0.0, 1.0, 80)
    starts = start_low[:, None] + (start_high - start_low)[:, None] * phase
    positions = (
        starts[:, :, None]
        + np.arange(intervals + 1)[None, None, :] * pitches[:, None, None]
    )
    # Phase must resolve the actual one-pixel dash centre.  A search radius of
    # 28% pitch let a lattice shifted by almost half a minor dash spacing score
    # identically to the true one after the 1,200-pixel calibration downsample.
    radii = np.minimum(0.70, np.maximum(0.35, pitches * 0.14))
    device = torch.device("cpu") if device is None else torch.device(device)

    def on_device(array):
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64, device=device)

    radii_tensor = on_device(radii)[:, None, None]
    pitch_tensor = on_device(pitches)[:, None, None]
    position_tensor = on_device(positions)

    def interpolate(source, sample_positions):
        # Linear interpolation on the integer sample grid, clamped at the
        # ends as np.interp clamps.
        last = source.numel() - 1
        clamped = sample_positions.clamp(0.0, float(last))
        low = clamped.floor().clamp(max=float(max(last - 1, 0)))
        fraction = clamped - low
        index = low.long()
        upper = (index + 1).clamp(max=last)
        return source[index] * (1.0 - fraction) + source[upper] * fraction

    def sampled_max(sample_positions, offset_scales, source_values):
        sampled = None
        for offset_scale in offset_scales:
            raise_if_analysis_cancelled()
            values_at_offset = interpolate(
                source_values, sample_positions + radii_tensor * float(offset_scale)
            )
            sampled = (
                values_at_offset if sampled is None
                else torch.maximum(sampled, values_at_offset)
            )
        return sampled

    normalized_tensor = on_device(normalized)
    offsets = np.linspace(-1.0, 1.0, 5)
    tick_values = sampled_max(position_tensor, offsets, normalized_tensor)
    halfway = sampled_max(
        position_tensor[:, :, :-1] + pitch_tensor * 0.5,
        offsets * 0.55,
        normalized_tensor,
    )
    outside_positions = torch.stack(
        (
            position_tensor[:, :, 0] - pitch_tensor[:, :, 0],
            position_tensor[:, :, -1] + pitch_tensor[:, :, 0],
        ),
        dim=-1,
    )
    outside = interpolate(normalized_tensor, outside_positions)
    outside[(outside_positions < 0.0) | (outside_positions > body_span)] = 0.0
    coverage = (tick_values >= 0.18).to(torch.float64).mean(dim=2)
    scores = (
        tick_values.mean(dim=2)
        - 0.62 * halfway.mean(dim=2)
        - 0.48 * outside.mean(dim=2)
        + 0.06 * tick_values[:, :, (0, -1)].mean(dim=2)
        - 0.18 * (1.0 - coverage)
    )
    if major_normalized is not None and intervals >= 10:
        major_tensor = on_device(major_normalized)
        major_ticks = sampled_max(position_tensor[:, :, ::10], offsets, major_tensor)
        half_major_ticks = sampled_max(position_tensor[:, :, 5::10], offsets, major_tensor)
        scores += 0.38 * (
            major_ticks.mean(dim=2) - 0.50 * half_major_ticks.mean(dim=2)
        )
    pitch_index, phase_index = np.unravel_index(int(torch.argmax(scores)), tuple(scores.shape))
    best = (
        float(scores[pitch_index, phase_index]),
        float(starts[pitch_index, phase_index]),
        float(positions[pitch_index, phase_index, -1]),
        float(pitches[pitch_index]),
        tick_values[pitch_index, phase_index].cpu().numpy(),
        halfway[pitch_index, phase_index].cpu().numpy(),
    )
    matched_mask = best[4] >= max(0.18, float(np.median(best[5]) + 0.08))
    matched_indices = tuple(int(index) for index in np.flatnonzero(matched_mask))
    matched = len(matched_indices)
    coverage = matched / max(intervals + 1, 1)
    contrast = float(
        np.mean(best[4])
        - 0.62 * np.mean(best[5])
    )
    confidence = float(np.clip(0.68 * contrast + 0.32 * coverage, 0.0, 1.0))
    return best[1], best[2], best[3], confidence, matched, matched_indices


def _chart_colour_correction(card, settings, corrected_tensor, transform):
    """Fit the chart's reference colours, or return why grey balance is used."""

    if not settings.apply_colour_balance:
        return "Colour correction is disabled."
    if settings.colour_correction_model != CHART_REFERENCE_FIT:
        return "Grey balance only was selected."
    if card is None or not card.swatches:
        return "No colour chart was detected."
    patches = []
    for swatch in card.swatches:
        if settings.neutral_sample_frame == "corrected":
            corners = _transform_points(np.asarray(swatch.corners, np.float64), transform)
            points = _quadrilateral_sample_points(corners, corrected_tensor.device)
            height, width = corrected_tensor.shape[-2:]
            if bool((~points.isfinite().all() | (points[..., 0] < 0).any()
                     | (points[..., 0] > width - 1).any() | (points[..., 1] < 0).any()
                     | (points[..., 1] > height - 1).any()).item()):
                continue
            mean = _sample_quadrilateral_mean(corrected_tensor, corners)
        else:
            mean = swatch.mean_bgr
        patches.append((swatch.number, swatch.row, swatch.column, tuple(float(v) for v in mean)))
    anchors = (
        ()
        if settings.neutral_patch_selection == "automatic"
        else tuple(int(value) for value in settings.neutral_patch_selection.split(","))
    )
    return fit_chart_correction(
        patches,
        chart_name=card.chart,
        rotation_degrees=card.rotation_degrees,
        anchor_patch_numbers=anchors,
    )


def _neutral_balance(card, settings, corrected_tensor, transform):
    """Sample selected grid positions in the requested unbalanced frame."""
    swatches = () if card is None else card.swatches
    by_id = {swatch.number: swatch for swatch in swatches}
    available = tuple(sorted(by_id))
    detection = None if card is None else sha256(json.dumps(
        asdict(card), sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    identity = np.ones(3, np.float32)

    def facts(state, mode='none', selected=(), witnesses=(), gains=identity, reason=''):
        return NeutralBalanceFacts(settings.apply_colour_balance, settings.neutral_patch_selection,
            settings.neutral_sample_frame, state, mode, available, tuple(selected), tuple(witnesses),
            tuple(float(value) for value in gains), detection, reason)

    if not settings.apply_colour_balance:
        return identity, facts('disabled', reason='Neutral balance is disabled.')
    manual = settings.neutral_patch_selection != 'automatic'
    requested = tuple(map(int, settings.neutral_patch_selection.split(','))) if manual else ()
    if not swatches or any(identifier not in by_id for identifier in requested):
        return identity, facts('unavailable', reason='Requested neutral grid patches are unavailable; identity gains were used.')

    candidates = [by_id[identifier] for identifier in requested] if manual else list(swatches)
    witnesses = []
    for swatch in candidates:
        corners = np.asarray(swatch.corners, dtype=np.float64)
        if settings.neutral_sample_frame == 'corrected':
            homogeneous = np.column_stack((corners, np.ones(4))) @ np.asarray(transform, np.float64).T
            if not np.isfinite(homogeneous).all() or (homogeneous[:, 2] <= 0).any():
                return identity, facts('unavailable', reason='Neutral sampling has an invalid projective denominator; identity gains were used.')
        sample_corners = (_transform_points(corners, transform)
                          if settings.neutral_sample_frame == 'corrected' else corners)
        if settings.neutral_sample_frame == 'corrected':
            # Check the actual 11x11 sampling domain, not an edge-clamped proxy.
            import torch
            points = _quadrilateral_sample_points(sample_corners, corrected_tensor.device)
            height, width = corrected_tensor.shape[-2:]
            # The warp uses inverse homogeneous coordinates and clamps its
            # denominator. Do not certify samples that would use that clamp.
            inverse = torch.linalg.inv(torch.as_tensor(transform, device=points.device, dtype=torch.float32))
            mapped = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1) @ inverse.T
            if bool((~torch.isfinite(points).all() | (points[..., 0] < 0).any()
                    | (points[..., 0] > width-1).any() | (points[..., 1] < 0).any()
                    | (points[..., 1] > height-1).any()
                    | ~torch.isfinite(mapped).all() | (mapped[..., 2] <= 1e-8).any()).item()):
                return identity, facts('unavailable', reason='Neutral sampling leaves the corrected image; identity gains were used.')
            mean = _sample_quadrilateral_mean(corrected_tensor, sample_corners)
        else:
            mean = swatch.mean_bgr
        if len(mean) != 3 or not np.isfinite(mean).all():
            return identity, facts('unavailable', reason='Neutral samples are nonfinite; identity gains were used.')
        witnesses.append(NeutralPatchWitness(swatch.number, swatch.corners,
            tuple((float(x), float(y)) for x, y in sample_corners), swatch.mean_bgr, tuple(mean)))
    mode = 'manual'
    if not manual:
        chosen = [w for w in witnesses if max(w.sampled_mean_bgr)-min(w.sampled_mean_bgr) < 18
                  and 35 < float(np.mean(w.sampled_mean_bgr)) < 240]
        mode = 'automatic_threshold'
        if len(chosen) < 2:
            chosen = sorted(witnesses, key=lambda w: max(w.sampled_mean_bgr)-min(w.sampled_mean_bgr))[:6]
            mode = 'automatic_fallback'
    else:
        chosen = witnesses
    channels = np.median(np.asarray([w.sampled_mean_bgr for w in chosen], np.float32), axis=0)
    target = float(np.mean(channels))
    gains = np.clip(target / np.maximum(channels, 1.0), .72, 1.35).astype(np.float32)
    return gains, facts('applied', mode, [w.patch_id for w in chosen], chosen, gains)


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


def _chart_exclusion_region(card: ColourCardDetection, transform: np.ndarray) -> np.ndarray:
    """The chart's swatch outline in corrected coordinates, widened by a margin."""

    unit = np.float64(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    unit_to_image = _homography(unit, np.asarray(card.bounds, np.float64))
    margin_x = _CHART_EXCLUSION_MARGIN / max(card.columns, 1)
    margin_y = _CHART_EXCLUSION_MARGIN / max(card.rows, 1)
    widened = np.float64((
        (-margin_x, -margin_y), (1.0 + margin_x, -margin_y),
        (1.0 + margin_x, 1.0 + margin_y), (-margin_x, 1.0 + margin_y),
    ))
    source = _transform_points(widened, unit_to_image)
    return _transform_points(source, transform).astype(np.float64)


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
