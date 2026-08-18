from __future__ import annotations

import os
import unittest


class PipelineInspectorActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def test_procedural_fit_action_is_contextual_and_emits_loss_weights(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector

        graph = build_default_pipeline()
        inspector = PipelineInspector()
        inspector.set_node(graph.node("edge_gradients"))
        self.assertTrue(inspector.procedural_fit_container.isHidden())

        inspector.set_procedural_fit_eligibility(True)
        inspector.set_node(graph.node("procedural_instances"))
        self.assertFalse(inspector.procedural_fit_container.isHidden())
        self.assertTrue(inspector.procedural_fit_button.isEnabled())
        self.assertAlmostEqual(
            inspector.procedural_fit_overreach_penalty_spin.value(), 2.0
        )
        self.assertAlmostEqual(
            inspector.procedural_fit_overreach_distance_scale_spin.value(), 0.50
        )
        self.assertIn(
            "exponential",
            inspector.procedural_fit_overreach_distance_scale_spin.toolTip(),
        )
        self.assertEqual(
            inspector.procedural_fit_missing_penalty_label.text(), "Missing: 1.0×"
        )
        self.assertFalse(
            inspector.procedural_fit_complete_annotations_checkbox.isChecked()
        )
        self.assertIn(
            "disjoint predictions",
            inspector.procedural_fit_complete_annotations_checkbox.toolTip(),
        )

        requests: list[tuple[str, str, object]] = []
        inspector.node_action_requested.connect(
            lambda node_id, action_id, payload: requests.append(
                (node_id, action_id, payload)
            )
        )
        inspector.procedural_fit_overreach_penalty_spin.setValue(2.7)
        inspector.procedural_fit_overreach_distance_scale_spin.setValue(0.35)
        inspector.procedural_fit_complete_annotations_checkbox.setChecked(True)
        inspector.procedural_fit_button.click()
        self.assertEqual(len(requests), 1)
        node_id, action_id, payload = requests[0]
        self.assertEqual(node_id, "procedural_instances")
        self.assertEqual(action_id, inspector.PROCEDURAL_FIT_ACTION)
        self.assertEqual(payload["false_positive_weight"], 2.7)
        self.assertEqual(payload["overreach_distance_scale_fraction"], 0.35)
        self.assertEqual(payload["false_negative_weight"], 1.0)
        self.assertTrue(payload["annotations_are_complete"])
        inspector.close()

    def test_procedural_fit_state_api_disables_busy_or_ineligible_requests(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector

        inspector = PipelineInspector()
        inspector.set_node(build_default_pipeline().node("procedural_instances"))
        inspector.set_procedural_fit_eligibility(
            False, "Apply complete annotations first."
        )
        self.assertFalse(inspector.procedural_fit_button.isEnabled())
        self.assertEqual(
            inspector.procedural_fit_status_label.text(),
            "Apply complete annotations first.",
        )

        inspector.set_procedural_fit_eligibility(True)
        inspector.set_procedural_fit_busy(True, "Evaluation 4 of 33")
        self.assertFalse(inspector.procedural_fit_button.isEnabled())
        self.assertFalse(
            inspector.procedural_fit_overreach_penalty_spin.isEnabled()
        )
        self.assertFalse(
            inspector.procedural_fit_overreach_distance_scale_spin.isEnabled()
        )
        self.assertFalse(
            inspector.procedural_fit_complete_annotations_checkbox.isEnabled()
        )
        self.assertEqual(inspector.procedural_fit_button.text(), "Fitting…")
        self.assertEqual(
            inspector.procedural_fit_status_label.text(), "Evaluation 4 of 33"
        )

        inspector.set_procedural_fit_result_summary("Loss improved 0.42 → 0.19.")
        inspector.set_procedural_fit_busy(False)
        self.assertTrue(inspector.procedural_fit_button.isEnabled())
        self.assertEqual(
            inspector.procedural_fit_status_label.text(),
            "Loss improved 0.42 → 0.19.",
        )
        inspector.close()


if __name__ == "__main__":
    unittest.main()
