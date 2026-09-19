"""Explicit review declarations for synthetic complete-outline test fixtures."""
import numpy as np
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


def reviewed_outlines(labels):
    return tuple(SeedInstanceAnnotation(int(identifier),shape_reviewed=True,
        outline_visibility='complete',pose='flat') for identifier in np.unique(labels) if identifier)


def dispose_window(window):
    """Explicitly discard test-only drafts; close protection is tested separately."""
    from unittest.mock import patch
    from PySide6.QtWidgets import QMessageBox
    window._reference_masks_dirty.clear()
    window._instance_annotations_dirty.clear()
    window._unsaved_reference_sidecars.clear()
    window._unsaved_manual_centre_sidecars.clear()
    with patch.object(QMessageBox,'warning',return_value=QMessageBox.StandardButton.Discard), \
         patch.object(QMessageBox,'critical'):
        window.close()


def install_test_frame(window):
    """Simulate calibration completing for a saved identity-frame UI fixture."""
    from types import SimpleNamespace
    key = window._current_image_key()
    bundle = window._pending_unbound_reference_bundles[key]
    result = SimpleNamespace(image_path=window.image_view.image_path,
        calibration=SimpleNamespace(affine_matrix=np.eye(3),corrected_bgr=np.zeros((*bundle.shape,3),np.uint8)))
    window._resolve_pending_project_reference_bundle(key,result)
    window._sync_reference_masks_to_view(key,render=False)
    window.image_view.set_instance_annotations(window._applied_instance_annotations.get(key),copy=False,render=False)
