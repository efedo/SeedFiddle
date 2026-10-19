"""One actual small CUDA witness for independent added-node execution/cache."""

from tests.catalogue_fixtures import CatalogueTestCase
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.catalogue_nodes import (catalogue_entries, instance_nodes,
    node_instances_from_payload, node_instances_to_payload)
from seedvision.ui.catalogue_nodes import CatalogueNodesController
from tests.test_catalogue_nodes_ui import _Window
from tests.test_native_recipe import fixture


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CatalogueNodesCUDAIsolationTests(CatalogueTestCase):
    def test_duplicate_thresholds_keep_independent_pixels_cache_and_saved_controls(self):
        app = QApplication.instance() or QApplication([])
        window = _Window()
        controller = CatalogueNodesController(window)
        primary = build_default_pipeline()
        controller.install_primary(primary)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'seeds.png'
                gray, _ = fixture()
                self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2)))
                window.image_view.image_path = path
                entry = next(entry for entry in catalogue_entries() if entry.backend == 'native'
                    and entry.recipe.stage(entry.target_node).operation_id == 'threshold.binary')
                a_id, b_id = controller.add_entry(entry.key), controller.add_entry(entry.key)
                a, b = controller._component(a_id), controller._component(b_id)
                controller.parameter_changed(a_id, 'threshold', 100)
                controller.parameter_changed(b_id, 'threshold', 100)
                self.assertIsNot(a.cache, b.cache)
                self.assertNotEqual(a.instance.instance_id, b.instance.instance_id)
                self.assertEqual(len(instance_nodes(a.instance)), 3)
                source = next(stage.stage_id for stage in a.instance.workflow.recipe.stages
                    if stage.operation_id == 'external.image_inputs')
                median = next(stage.stage_id for stage in a.instance.workflow.recipe.stages
                    if stage.operation_id == 'filter.median_blur')
                target = a.instance.target_node
                scope = {source, median, target}

                def run(component, identifier):
                    # Use the desktop coordinator's actual preparation/dispatch
                    # adapter; only thread scheduling is omitted in this witness.
                    result = controller._prepare_run(component, identifier, path)(lambda: False)
                    component.result = result
                    return result

                cold_a = run(a, a_id)
                self.assertEqual(set(a.cache.last_computed_nodes), scope)
                cold_b = run(b, b_id)
                self.assertEqual(set(b.cache.last_computed_nodes), scope)
                mask_a = cold_a.node(target).outputs['mask']
                mask_b = cold_b.node(target).outputs['mask']
                self.assertEqual(mask_a.device.type, 'cuda')
                self.assertEqual(mask_a.dtype, torch.uint8)
                torch.testing.assert_close(mask_a, mask_b, rtol=0, atol=0)
                self.assertNotEqual(mask_a.data_ptr(), mask_b.data_ptr())
                self.assertNotEqual(cold_a.node(source).outputs['gray_image'].data_ptr(),
                                    cold_b.node(source).outputs['gray_image'].data_ptr())
                self.assertTrue(bool(mask_b.any()))
                expected = torch.where(cold_a.node(median).outputs['image'] > 100, 255, 0).to(torch.uint8)
                torch.testing.assert_close(mask_a, expected, rtol=0, atol=0)
                held_b = mask_b.clone()
                b_identity = b.instance.workflow.recipe.identity

                warm_a = run(a, a_id)
                self.assertEqual(set(a.cache.last_reused_nodes), scope)
                self.assertEqual(a.cache.last_computed_nodes, ())
                self.assertIs(warm_a.node(target).outputs['mask'], mask_a)

                controller.parameter_changed(a_id, 'threshold', 250)
                changed_a = run(a, a_id)
                self.assertEqual(set(a.cache.last_reused_nodes), {source, median})
                self.assertEqual(a.cache.last_computed_nodes, (target,))
                self.assertIs(changed_a.node(median).outputs['image'], cold_a.node(median).outputs['image'])
                changed_mask = changed_a.node(target).outputs['mask']
                self.assertFalse(bool(changed_mask.any()))
                self.assertFalse(torch.equal(changed_mask, mask_a))

                warm_b = run(b, b_id)
                self.assertEqual(set(b.cache.last_reused_nodes), scope)
                self.assertEqual(b.cache.last_computed_nodes, ())
                self.assertIs(warm_b.node(target).outputs['mask'], mask_b)
                torch.testing.assert_close(mask_b, held_b, rtol=0, atol=0)
                self.assertEqual(b.instance.workflow.recipe.identity, b_identity)
                self.assertEqual(b.instance.workflow.recipe.parameters_for_runner(target)['threshold'], 100)

                instances = controller.snapshot_instances()
                reopened = node_instances_from_payload(json.loads(json.dumps(node_instances_to_payload(instances))))
                self.assertEqual(reopened, instances)
                self.assertEqual([instance.workflow.recipe.parameters_for_runner(instance.target_node)['threshold']
                                  for instance in reopened], [250, 100])
                controller.set_instances(reopened)
                self.assertTrue(all(not component.cache.values for component in controller._components.values()))
                self.assertIs(controller.primary, primary)
                self.assertIsNone(window.pipeline_workflow.workflow)
        finally:
            controller.stop()
            window.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
