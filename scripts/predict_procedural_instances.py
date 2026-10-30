"""Run the procedural seed-instance pipeline headlessly and export source-frame masks.

For every entry in ``seed-instance-references/manifest.json`` this script runs the
default pipeline graph up to the ``procedural_instances`` node (the same scope the
desktop uses), warps the resulting labels back into the original photograph's
coordinate frame, and writes one ``<stem>.instances.png`` (``uint16``) per image.
It also exports the calibrated RGB dish crop plus its frame geometry so other
methods (``predict_cellpose_instances.py``) can be scored on identical inputs.

The procedural pipeline produces no foreground without seed-colour references.
``--foreground-source reference-sample`` (default) paints a few eroded reference
instances as the foreground sample, imitating the operator brush strokes the
desktop needs; ``painted`` uses saved project reference-region sidecars where
they exist and falls back to the sample otherwise; ``none`` passes nothing.
The chosen source is recorded in every ``<stem>.run.json``.

Example::

    .venv\\Scripts\\python.exe scripts\\predict_procedural_instances.py \\
        --output artifacts/instance-benchmark/predictions/procedural
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

CALIBRATION_NODES = frozenset(
    {"project", "metadata", "image_selector", "deskew_colour", "ruler_detection"}
)


def procedural_scope(graph) -> frozenset[str]:
    """Return the enabled upstream closure of the procedural node."""
    scope, pending = {"procedural_instances"}, ["procedural_instances"]
    while pending:
        for parent in graph.upstream(pending.pop()):
            if parent not in scope:
                scope.add(parent)
                pending.append(parent)
    return frozenset(name for name in scope if graph.node(name).enabled)


def labels_to_source(labels, crop_shape, crop_offset, affine, source_shape):
    """Warp crop-frame labels (any working scale) into source coordinates."""
    labels = np.asarray(labels)
    ch, cw = crop_shape
    placement = np.array(
        [[cw / labels.shape[1], 0.0, crop_offset[0]],
         [0.0, ch / labels.shape[0], crop_offset[1]],
         [0.0, 0.0, 1.0]], np.float64)
    transform = np.linalg.inv(np.asarray(affine, np.float64)) @ placement
    return cv2.warpPerspective(
        labels.astype(np.uint16), transform, (source_shape[1], source_shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def dish_polygon_source(dish, affine, points=72):
    """Dish outer circle (corrected frame) as a polygon in source coordinates."""
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    circle = np.stack([dish.center_x + dish.outer_radius * np.cos(angles),
                       dish.center_y + dish.outer_radius * np.sin(angles)], axis=1)
    inverse = np.linalg.inv(np.asarray(affine, np.float64))
    return cv2.perspectiveTransform(circle[None].astype(np.float64), inverse)[0]


def sampled_foreground(reference, count, seed, erosion):
    """Erode a random subset of reference instances into a foreground sample."""
    ids = np.unique(reference)
    ids = ids[ids > 0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=min(count, len(ids)), replace=False)
    mask = np.isin(reference, chosen).astype(np.uint8)
    if erosion > 0:
        mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8))
    return mask, [int(value) for value in chosen]


def painted_bundle(project_root, project_file, image_path):
    """Load the saved reference-region sidecar for ``image_path`` if any."""
    from seedvision.persistence.reference_regions import ReferenceRegionStore
    document = json.loads(Path(project_file).read_text(encoding="utf-8"))
    for record in document.get("images", ()):
        source = record.get("source", {}).get("path")
        sidecar = record.get("sidecars", {}).get("reference_regions")
        if source is None or sidecar is None:
            continue
        if Path(source).name.casefold() != Path(image_path).name.casefold():
            continue
        archive = Path(project_root) / sidecar["path"]
        candidate = Path(project_root) / source
        if not archive.is_file() or not candidate.is_file():
            return None
        bundle = ReferenceRegionStore(Path(project_root)).load_project_archive(candidate, archive, None)
        if bundle is None or bundle.source_to_corrected is None or bundle.foreground is None:
            return None
        return bundle
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, default=ROOT / "seed-instance-references")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "instance-benchmark" / "predictions" / "procedural")
    parser.add_argument("--images", nargs="*", default=None, help="Image stems to run (default: every manifest entry)")
    parser.add_argument("--foreground-source", choices=("reference-sample", "painted", "none"), default="reference-sample")
    parser.add_argument("--sample-seeds", type=int, default=8, help="Reference instances painted as the foreground sample")
    parser.add_argument("--sample-erosion", type=int, default=7, help="Erosion kernel (px) applied to sampled instances")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Workspace holding projects/ sidecars for --foreground-source painted")
    parser.add_argument("--project", type=Path, default=None, help="Saved project master naming reference-region sidecars")
    args = parser.parse_args(argv)

    from seedvision.pipeline.model import build_default_pipeline
    from seedvision.pipeline.settings import production_settings
    from seedvision.segmentation.baseline import PipelineAnalysisCache, analyze_path

    manifest = json.loads((args.references / "manifest.json").read_text(encoding="utf-8"))
    graph = build_default_pipeline()
    settings = production_settings(graph)
    enabled = procedural_scope(graph)
    args.output.mkdir(parents=True, exist_ok=True)

    for entry in manifest["entries"]:
        image_path = ROOT / entry["image_path"]
        stem = image_path.stem
        if args.images and stem not in args.images:
            continue
        reference = cv2.imread(str(args.references / entry["mask_path"]), cv2.IMREAD_UNCHANGED)
        if reference is None or reference.shape != (entry["height"], entry["width"]):
            raise SystemExit(f"Reference mask for {stem} is missing or has the wrong shape.")
        started = time.perf_counter()
        cache = PipelineAnalysisCache()
        calibration_pass = analyze_path(image_path, **settings, enabled_nodes=CALIBRATION_NODES, node_cache=cache)
        affine = np.asarray(calibration_pass.calibration.affine_matrix, np.float64)
        corrected_shape = calibration_pass.calibration.corrected_bgr.shape[:2]

        kwargs = {}
        foreground_record: dict[str, object] = {"source": args.foreground_source}
        bundle = None
        if args.foreground_source == "painted" and args.project is not None:
            bundle = painted_bundle(args.project_root, args.project, image_path)
        if bundle is not None:
            kwargs.update(foreground_reference_mask=bundle.foreground, background_reference_mask=bundle.background,
                          background_exclusion_mask=bundle.other, foreground_exclusion_mask=bundle.other,
                          reference_transform=bundle.source_to_corrected)
            foreground_record.update(source="painted", foreground_pixels=int(np.count_nonzero(bundle.foreground)),
                                     background_pixels=int(np.count_nonzero(bundle.background)) if bundle.background is not None else 0)
        elif args.foreground_source != "none":
            sample, chosen = sampled_foreground(reference, args.sample_seeds, args.seed, args.sample_erosion)
            corrected_sample = cv2.warpPerspective(sample, affine, (corrected_shape[1], corrected_shape[0]), flags=cv2.INTER_NEAREST)
            kwargs.update(foreground_reference_mask=corrected_sample.astype(bool), reference_transform=affine)
            foreground_record.update(source="reference-sample", sampled_reference_ids=chosen,
                                     foreground_pixels=int(np.count_nonzero(corrected_sample)), seed=args.seed,
                                     erosion_px=args.sample_erosion)
        result = analyze_path(image_path, **settings, enabled_nodes=enabled, node_cache=cache, **kwargs)
        elapsed = time.perf_counter() - started

        instances = result.procedural_instances
        crop_shape = tuple(int(v) for v in np.asarray(result.layers.valid_mask).shape[-2:])
        offset = tuple(int(v) for v in result.crop_offset)
        source_shape = (entry["height"], entry["width"])
        labels = np.zeros(crop_shape, np.uint16) if instances is None else np.asarray(instances.labels)
        source_labels = labels_to_source(labels, crop_shape, offset, affine, source_shape)
        lost = len(set(np.unique(labels).tolist()) - {0}) - len(set(np.unique(source_labels).tolist()) - {0})

        corrected = np.asarray(result.calibration.corrected_bgr)
        crop = corrected[offset[1]:offset[1] + crop_shape[0], offset[0]:offset[0] + crop_shape[1]]
        cv2.imwrite(str(args.output / f"{stem}.instances.png"), source_labels)
        cv2.imwrite(str(args.output / f"{stem}.dish.png"), np.ascontiguousarray(crop))
        dish = result.dish
        frame = {
            "stem": stem, "image_path": entry["image_path"], "image_sha256": entry["image_sha256"],
            "source_shape_hw": list(source_shape), "corrected_shape_hw": [int(v) for v in corrected_shape],
            "source_to_corrected": affine.tolist(), "crop_offset_xy": list(offset), "crop_shape_hw": list(crop_shape),
            "dish_corrected": {"center_x": float(dish.center_x), "center_y": float(dish.center_y),
                               "outer_radius": float(dish.outer_radius), "inner_radius": float(dish.inner_radius)},
            "estimated_seed_diameter_px": float(result.estimated_seed_diameter_px),
            "pixels_per_mm": None if result.pixels_per_mm is None else float(result.pixels_per_mm),
        }
        (args.output / f"{stem}.frame.json").write_text(json.dumps(frame, indent=2), encoding="utf-8")
        polygon = dish_polygon_source(dish, affine)
        (args.output / f"{stem}.region.json").write_text(json.dumps(
            {"stem": stem, "coordinate_space": "source", "polygon_xy": polygon.round(2).tolist(),
             "description": "Detected petri-dish outer circle; instances with centroids outside are external to the dish."},
            indent=2), encoding="utf-8")
        run = {
            "stem": stem, "method": "procedural", "elapsed_seconds": round(elapsed, 3),
            "instance_count": int(len(set(np.unique(source_labels).tolist()) - {0})),
            "instances_lost_in_warp": int(lost), "working_scale": None if instances is None else float(instances.working_scale),
            "foreground": foreground_record, "foreground_pixel_count": int(result.foreground_pixel_count),
            "warnings": list(result.warnings), "enabled_nodes": sorted(enabled),
            "node_timings_seconds": {k: round(float(v), 3) for k, v in result.node_timings_seconds.items()},
        }
        (args.output / f"{stem}.run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"{stem}: {run['instance_count']} instances in {elapsed:.1f}s (foreground {foreground_record['source']})")
        del result, calibration_pass, cache
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
