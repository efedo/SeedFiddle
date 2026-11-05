"""A run to one node recomputes that node alone and leaves other results standing."""
from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from tests.functional_units import functional_units


ROOT = Path(__file__).resolve().parents[1]

# Everything the layer builder derives from edge gradients and their ridges.
AFTER_RIDGES = frozenset({
    "reference_texture_prototypes", "material_evidence_decision", "reference_seed_traits",
    "reference_edge_probability", "reference_edge_ridges", "edge_traces", "seed_edge_curves",
})


def seed_arguments():
    image = np.full((96, 96, 3), 225, np.uint8)
    cv2.ellipse(image, (48, 48), (28, 18), 0, 0, 360, (30, 75, 170), -1)
    valid = np.full((96, 96), 255, np.uint8)
    return image, valid, np.asarray(((48, 48),), np.float32), np.asarray((23,), np.float32), 46.0


@functional_units
class RunToNodeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def check_retained_layers_keep_their_results_when_their_inputs_change(self) -> None:
        from seedvision.visualization import build_analysis_layers

        cache = {}
        build_analysis_layers(*seed_arguments(), offset_x=0, offset_y=0, cache_values=cache)
        kept = {
            name: cache[name]
            for name in ("layer.reference_texture_prototypes", "layer.material_evidence",
                         "layer.reference_seed_traits", "layer.reference_edge_probability",
                         "layer.edge_traces")
        }
        gradients = cache["layer.edge_gradients"]
        ridges = cache["layer.edge_ridges"]
        fit = cache["layer.seed_edge_curves"]

        layers = build_analysis_layers(
            *seed_arguments(), offset_x=0, offset_y=0, cache_values=cache,
            dirty_nodes={"edge_gradients"}, retained_nodes=AFTER_RIDGES,
        )

        self.assertIsNot(cache["layer.edge_gradients"], gradients)
        self.assertIsNot(cache["layer.edge_ridges"], ridges)
        for name, product in kept.items():
            self.assertIs(cache[name], product, name)
        # The fit stands, but it also carries the ridges this run redid.
        held = cache["layer.seed_edge_curves"]
        self.assertIs(held.final_likelihood, fit.final_likelihood)
        self.assertIs(held.geometry, fit.geometry)
        self.assertIs(held.ridges, cache["layer.edge_ridges"])
        self.assertIs(layers.edge_ridges, held.ridges)

    def check_a_later_curve_fit_starts_from_traces_that_ran_under_a_kept_fit(self) -> None:
        from seedvision.visualization import build_analysis_layers

        cache = {}
        arguments = dict(offset_x=0, offset_y=0, cache_values=cache)
        build_analysis_layers(*seed_arguments(), **arguments)
        fit = cache["layer.seed_edge_curves"]

        build_analysis_layers(
            *seed_arguments(), **arguments,
            dirty_nodes={"edge_traces"}, retained_nodes=frozenset({"seed_edge_curves"}),
        )
        traces = cache["layer.seed_edge_curves"].trace_state
        self.assertIsNot(traces, fit.trace_state)
        self.assertIs(cache["layer.seed_edge_curves"].final_likelihood, fit.final_likelihood)

        build_analysis_layers(*seed_arguments(), **arguments, dirty_nodes={"seed_edge_curves"})
        refitted = cache["layer.seed_edge_curves"]
        self.assertIsNot(refitted.final_likelihood, fit.final_likelihood)
        self.assertIs(refitted.trace_state, traces)

    def check_a_retained_node_with_no_result_still_gets_one(self) -> None:
        from seedvision.visualization import build_analysis_layers

        # The caller retains only what the cache holds, and a missing curve fit
        # also holds the ridges and traces, so they cannot be skipped either.
        cache = {}
        build_analysis_layers(
            *seed_arguments(), offset_x=0, offset_y=0, cache_values=cache,
            retained_nodes=frozenset({"edge_traces", "edge_ridges"}),
        )
        self.assertIn("layer.seed_edge_curves", cache)

    def check_seed_traits_follow_annotation_content_rather_than_the_array(self) -> None:
        from seedvision.visualization import build_analysis_layers

        labels = np.zeros((96, 96), np.uint16)
        cv2.ellipse(labels, (48, 48), (28, 18), 0, 0, 360, 1, -1)
        cache = {}

        def build(annotations):
            build_analysis_layers(
                *seed_arguments(), offset_x=0, offset_y=0, cache_values=cache,
                seed_instance_annotations=annotations,
            )
            return cache["layer.reference_seed_traits"]

        first = build(labels)
        self.assertIs(build(labels.copy()), first)
        changed = labels.copy()
        changed[48, 48:60] = 0
        self.assertIsNot(build(changed), first)

    def check_analysis_runs_only_the_target_and_keeps_every_other_result(self) -> None:
        from seedvision.export.results import result_report
        from seedvision.pipeline import build_default_pipeline
        from seedvision.pipeline.node_aliases import owning_node
        from seedvision.persistence.image_decoding import read_image
        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.segmentation.baseline import analyze_image

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        image = cv2.resize(read_image(path), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        graph = build_default_pipeline()
        enabled = frozenset(key for key, node in graph.nodes.items() if node.enabled)
        target = "frequency_noise_masks"
        scope = {target}
        pending = [target]
        while pending:
            parents = set(graph.upstream(pending.pop())) & enabled - scope
            scope |= parents
            pending.extend(parents)
        scope = frozenset(scope)
        left_out = enabled - scope
        self.assertIn("reference_texture_prototypes", left_out)
        self.assertIn("procedural_instances", left_out)
        cache = PipelineAnalysisCache()

        def run(**options):
            cache.node_timings_seconds.clear()
            result = analyze_image(image, image_path=path, node_cache=cache, **options)
            return result, {owning_node(node) for node, seconds in cache.node_timings_seconds.items()
                            if seconds > 0}

        full, _ = run(enabled_nodes=enabled)
        before = dict(cache.values)
        self.assertIsNotNone(full.layers.reference_texture_profile)
        self.assertIsNotNone(full.procedural_instances)

        repeat, executed = run(enabled_nodes=scope, retained_nodes=left_out, dirty_nodes={target})
        self.assertEqual(executed, {target})
        self.assertIsNot(cache.values["layer.frequency_noise_masks"], before["layer.frequency_noise_masks"])
        for key in ("layer.reference_texture_prototypes", "layer.material_evidence",
                    "layer.seed_edge_curves", "segmentation.procedural_instances",
                    "advanced.layers", "foreground", "dish"):
            self.assertIs(cache.values[key], before[key], key)
        # With nothing stale, the seed count and overlays stay available.
        self.assertIs(repeat.procedural_instances, full.procedural_instances)
        self.assertIs(repeat.layers.reference_texture_profile, full.layers.reference_texture_profile)
        self.assertEqual(
            [row["seed_id"] for row in result_report(repeat, enabled=enabled)["instances"]],
            [row["seed_id"] for row in result_report(full, enabled=enabled)["instances"]],
        )

        # The deskew and ruler stages share one calibration call, but the ruler stands.
        deskew_scope = frozenset({"image_selector", "deskew_colour"})
        ruler = cache.values["calibration"].ruler
        self.assertIsNotNone(ruler)
        _, executed = run(enabled_nodes=deskew_scope, retained_nodes=enabled - deskew_scope,
                          dirty_nodes={"deskew_colour"})
        self.assertIn("deskew_colour", executed)
        self.assertNotIn("ruler_detection", executed)
        self.assertIs(cache.values["calibration"].ruler, ruler)
        self.assertEqual(cache.values["calibration"].pixels_per_mm, full.calibration.pixels_per_mm)

        cache.stale_nodes = {"procedural_instances"}
        outdated, _ = run(enabled_nodes=scope, retained_nodes=left_out, dirty_nodes={target})
        self.assertIsNone(outdated.procedural_instances)
        self.assertIs(cache.values["segmentation.procedural_instances"], before["segmentation.procedural_instances"])

        cache.stale_nodes = set()
        again, executed = run(enabled_nodes=enabled)
        self.assertEqual(executed, set())
        self.assertIs(again.procedural_instances, full.procedural_instances)

    def check_a_changed_crop_gives_up_what_no_longer_lines_up(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.persistence.image_decoding import read_image
        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.segmentation.baseline import analyze_image

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        image = cv2.resize(read_image(path), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        graph = build_default_pipeline()
        enabled = frozenset(key for key, node in graph.nodes.items() if node.enabled)
        cache = PipelineAnalysisCache()
        full = analyze_image(image, image_path=path, node_cache=cache, enabled_nodes=enabled)
        crop, offset_x, offset_y = cache.values["crop"]
        # Stand in for a dish that moved: this run's crop will differ from the cached one.
        cache.values["crop"] = (crop[1:, 1:], offset_x + 1, offset_y + 1)
        scope = frozenset({"project", "deskew_colour", "ruler_detection", "layout_detection"})
        left_out = enabled - scope

        narrow = analyze_image(
            image, image_path=path, node_cache=cache, enabled_nodes=scope,
            retained_nodes=left_out, dirty_nodes={"layout_detection"},
        )

        self.assertEqual(cache.stale_nodes, set(left_out))
        self.assertIsNone(narrow.procedural_instances)
        self.assertNotIn("segmentation.procedural_instances", cache.values)
        self.assertEqual(narrow.layers.valid_mask.shape, cache.values["crop"][0].shape[:2])
        self.assertEqual(full.layers.valid_mask.shape, narrow.layers.valid_mask.shape)

    def test_layer_builder(self):
        """How a run to one node treats the layer products it leaves out."""
        self.scenario(self.check_retained_layers_keep_their_results_when_their_inputs_change)
        self.scenario(self.check_a_later_curve_fit_starts_from_traces_that_ran_under_a_kept_fit)
        self.scenario(self.check_a_retained_node_with_no_result_still_gets_one)
        self.scenario(self.check_seed_traits_follow_annotation_content_rather_than_the_array)

    def test_analysis(self):
        """What a run to one node executes and keeps across the whole analysis."""
        self.scenario(self.check_analysis_runs_only_the_target_and_keeps_every_other_result)
        self.scenario(self.check_a_changed_crop_gives_up_what_no_longer_lines_up)


if __name__ == "__main__":
    unittest.main()
