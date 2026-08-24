"""Report material-evidence behavior over saved reference-region fixtures.

This is a read-only forensic tool. It runs the evidence-producing portion of
the desktop graph and summarizes reviewed regions in corrected-image space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seedvision.persistence.reference_regions import ReferenceRegionStore
from seedvision.segmentation import analyze_path


EVIDENCE_NODES = {
    "raw_images",
    "metadata",
    "reference_layers",
    "colour_reference",
    "ruler_detection",
    "deskew_colour",
    "layout_detection",
    "scale_calibration",
    "seed_scale_estimation",
    "perimeter_background_reference",
    "foreground_segmentation",
    "background_likelihood",
    "refined_background_likelihood",
    "foreground_noise_likelihood",
    "edge_gradients",
    "frequency_noise_masks",
    "edge_ridges",
    "reference_texture_prototypes",
    "material_evidence_decision",
}


def _region_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float32)[mask] / 255.0
    if not selected.size:
        return {"pixels": 0}
    return {
        "pixels": int(selected.size),
        "mean": round(float(selected.mean()), 4),
        "p10": round(float(np.quantile(selected, 0.10)), 4),
        "median": round(float(np.median(selected)), 4),
        "p90": round(float(np.quantile(selected, 0.90)), 4),
    }


def audit_image(root: Path, image: Path) -> dict[str, object]:
    store = ReferenceRegionStore(root)
    bundle = store.load_if_present(image, expected_shape=None)
    if bundle is None:
        raise RuntimeError(f"No saved reference regions exist for {image.name}.")
    result = analyze_path(
        image,
        background_reference_mask=bundle.background,
        foreground_reference_mask=bundle.foreground,
        background_exclusion_mask=bundle.other,
        foreground_exclusion_mask=bundle.other,
        seed_instance_annotations=bundle.annotated_seeds,
        enabled_nodes=EVIDENCE_NODES,
    )
    offset_x, offset_y = result.crop_offset
    crop_height, crop_width = np.asarray(result.layers.valid_mask).shape

    def local(values) -> np.ndarray:
        if values is None:
            return np.zeros((crop_height, crop_width), dtype=bool)
        return np.asarray(values)[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]

    regions = {
        "painted_foreground": local(bundle.foreground) > 0,
        "annotated_seed": local(bundle.annotated_seeds) > 0,
        "painted_background": local(bundle.background) > 0,
        "painted_other": local(bundle.other) > 0,
    }
    rasters = {
        "automatic_foreground": result.layers.automatic_foreground_colour_probability,
        "reviewed_foreground": result.layers.reviewed_foreground_colour_probability,
        "foreground_noise": result.layers.foreground_noise_likelihood,
        "background_colour": result.layers.background_likelihood,
        "background_noise": result.layers.refined_background_likelihood,
        "other_colour": result.layers.other_colour_probability,
        "other_noise": result.layers.other_noise_probability,
        "prototype_foreground": result.layers.reference_seed_surface_probability,
        "prototype_background": result.layers.reference_background_texture_probability,
        "prototype_other": result.layers.reference_other_texture_probability,
        "seed_support": result.layers.seed_evidence_support,
        "nonseed_support": result.layers.nonseed_evidence_support,
        "resolved_seed": result.layers.seed_material_probability,
        "resolved_nonseed": result.layers.nonseed_material_probability,
        "ambiguity": result.layers.material_ambiguity_probability,
        "unknown": result.layers.material_unknown_probability,
    }
    masses = sum(
        np.asarray(raster, dtype=np.int16)
        for raster in (
            result.layers.seed_material_probability,
            result.layers.nonseed_material_probability,
            result.layers.material_ambiguity_probability,
            result.layers.material_unknown_probability,
        )
    )
    decision_valid = masses > 0
    region_results: dict[str, object] = {}
    for region_name, region in regions.items():
        region = region & decision_valid
        region_results[region_name] = {
            raster_name: _region_summary(np.asarray(raster), region)
            for raster_name, raster in rasters.items()
            if raster is not None
        }
    return {
        "image": image.name,
        "corrected_shape": list(bundle.shape),
        "crop_shape": [crop_height, crop_width],
        "foreground_automatic_authority": round(
            float(result.layers.foreground_automatic_evidence_authority), 5
        ),
        "background_automatic_authority": round(
            float(result.layers.background_automatic_evidence_authority), 5
        ),
        "mass_sum_max_error_u8": int(
            np.max(np.abs(masses[decision_valid] - 255))
        ),
        "source_reliabilities": {
            name: round(float(value), 5)
            for name, value in result.layers.material_source_reliabilities
        },
        "regions": region_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image basenames or paths")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = []
    for value in args.images:
        candidate = Path(value)
        if not candidate.is_file():
            candidate = root / "images" / value
        reports.append(audit_image(root, candidate.resolve()))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
