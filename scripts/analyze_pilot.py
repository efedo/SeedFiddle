"""Batch the transparent baseline over a directory of pilot images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seedvision.segmentation.baseline import analyze_path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def proportion_fields(prefix: str, names, proportions) -> dict[str, str]:
    return {
        f"{prefix}_{name}_seed_fraction": f"{proportion:.5f}"
        for name, proportion in zip(names, proportions, strict=True)
    }


def write_overlay(image_path: Path, output_path: Path, result) -> None:
    del image_path
    image = result.calibration.corrected_bgr.copy()
    dish = result.dish
    cv2.circle(
        image,
        (dish.center_x, dish.center_y),
        dish.radius,
        (255, 220, 60),
        max(2, round(dish.radius * 0.004)),
    )
    for proposal in result.proposals:
        center = (round(proposal.center_x), round(proposal.center_y))
        cv2.circle(image, center, round(proposal.radius), (0, 220, 255), 2)
        cv2.circle(
            image, center, max(3, round(proposal.radius * 0.10)), (40, 40, 255), -1
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path, nargs="?", default=Path("images"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "pilot" / "baseline"
    )
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        result = analyze_path(path)
        rows.append(
            {
                "file": path.name,
                "width_px": image.shape[1],
                "height_px": image.shape[0],
                "proposal_count": result.count,
                "approximate": result.approximate,
                "crowding": result.crowding,
                "dish_center_x": result.dish.center_x,
                "dish_center_y": result.dish.center_y,
                "dish_radius_px": result.dish.radius,
                "dish_confidence": f"{result.dish.confidence:.4f}",
                "reference_seed_count": result.reference_seed_count,
                "estimated_seed_diameter_px": f"{result.estimated_seed_diameter_px:.2f}",
                "foreground_threshold": f"{result.foreground_threshold:.3f}",
                "foreground_pixel_count": result.foreground_pixel_count,
                "analysis_region_pixel_count": result.analysis_region_pixel_count,
                "distance_candidate_count": result.distance_candidate_count,
                "circle_candidate_count": result.circle_candidate_count,
                "colour_swatches_detected": (
                    result.calibration.colour_card.detected_swatch_count
                    if result.calibration.colour_card is not None
                    else 0
                ),
                "ruler_detected": result.calibration.ruler is not None,
                "deskew_degrees": f"{result.calibration.deskew_degrees:.4f}",
                "pixels_per_mm": (
                    f"{result.calibration.pixels_per_mm:.5f}"
                    if result.calibration.pixels_per_mm is not None
                    else ""
                ),
                "scale_confidence": f"{result.calibration.scale_confidence:.4f}",
                "advanced_compute_backend": result.advanced.backend.used,
                "advanced_compute_device": result.advanced.backend.device_name,
                "advanced_elapsed_seconds": f"{result.advanced.backend.elapsed_seconds:.4f}",
                "mean_wrinkling_likelihood": f"{result.advanced.mean_wrinkling_likelihood:.5f}",
                "mean_coat_damage_likelihood": f"{result.advanced.mean_coat_damage_likelihood:.5f}",
                **proportion_fields(
                    "colour",
                    result.advanced.colour_class_names,
                    result.advanced.seed_colour_proportions,
                ),
                **proportion_fields(
                    "pattern",
                    result.advanced.pattern_class_names,
                    result.advanced.seed_pattern_proportions,
                ),
                "method": result.method,
            }
        )
        if not args.no_overlays:
            write_overlay(path, args.output_dir / f"{path.stem}_overlay.jpg", result)
        print(
            f"{path.name}: approximately {result.count} proposals "
            f"({result.crowding} crowding)"
        )

    if rows:
        report_path = args.output_dir / "baseline_results.csv"
        with report_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
