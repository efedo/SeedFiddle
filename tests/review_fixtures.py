"""Explicit review declarations for synthetic complete-outline test fixtures."""
import numpy as np
from functools import lru_cache
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


def open_workspace_images(window):
    """Add the photographs in ``<root>/images`` to the window's untitled project.

    Seed Fiddle starts with an empty project. Fixtures that work on the
    committed photographs add them as Open images… would, open the first one
    and leave the fixture project unmodified.
    """
    from seedvision.ui.main_window import SUPPORTED_SUFFIXES
    folder = window._root / 'images'
    paths = sorted(path for path in folder.iterdir()
                   if path.suffix.lower() in SUPPORTED_SUFFIXES) if folder.is_dir() else []
    for path in paths:
        window._add_image(path, open_now=False, mark_project_dirty=False)
    if paths:
        window._open_path(paths[0], mark_project_dirty=False)
    window._set_project_dirty(False)
    return window


def editable_copy(window, name='Test pipeline'):
    """Put the window on a named copy, as accepting the default-pipeline offer does.

    The project's unsaved state is kept: a fixture is not a user's edit.
    """
    documents = window.pipeline_documents
    if documents.locked:
        dirty = window._project_dirty
        documents.create_copy(documents.unique_name(name))
        window._set_project_dirty(dirty)


def isolated_window(root, *, settings_path=None, editable=True, images=True):
    """Fresh desktop state with local settings.

    Edit fixtures work on a named copy; startup-lock witnesses pass editable=False.
    ``images`` adds the root's ``images/`` photographs (see open_workspace_images).
    """
    from unittest.mock import patch
    from PySide6.QtCore import QSettings
    from seedvision.ui.main_window import MainWindow
    settings = QSettings(str(settings_path if settings_path is not None
                             else root / 'test-application-settings.ini'), QSettings.Format.IniFormat)
    settings.clear()
    with patch('seedvision.ui.main_window.QSettings', return_value=settings):
        window = MainWindow(root)
        if images:
            open_workspace_images(window)
    window._test_application_settings = settings
    if editable and hasattr(window, 'pipeline_documents'):
        editable_copy(window)
    return window


def workspace_window(root):
    """``open_workspace_images(MainWindow(root))``.

    Unlike isolated_window, it keeps the real application settings and the locked default pipeline.
    """
    from seedvision.ui.main_window import MainWindow
    return open_workspace_images(MainWindow(root))


def controls_window(testcase):
    """Independent tiny-image workspace for tests of controls, not image content."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PySide6.QtGui import QColor, QImage
    temporary = TemporaryDirectory(prefix='seedfiddle-controls-')
    testcase.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    (root / 'images').mkdir()
    image = QImage(48, 36, QImage.Format.Format_RGB32)
    image.fill(QColor('#6f624a'))
    if not image.save(str(root / 'images' / 'controls.png')):
        raise AssertionError('Could not create control-test image')
    window = isolated_window(root)
    testcase.addCleanup(dispose_window, window)
    return window


@lru_cache(maxsize=1)
def parity_trait_model_json():
    """Train once for integration fixtures, leaving classifier witnesses real."""
    from tests.test_trait_classifier import TraitClassifierTests
    from seedvision.learning.trait_classifier import train_trait_model, TraitDefinition, TraitVocabulary
    vocabulary = TraitVocabulary('synthetic-v1', 'synthetic definition fixture', 'a' * 64,
        (TraitDefinition('dark', 'Synthetic low first feature', 'synthetic-only'),
         TraitDefinition('light', 'Synthetic high first feature', 'synthetic-only')))
    training = TraitClassifierTests.evidence('train', [.1, .15, .2, .25, .75, .8, .85, .9])
    calibration = TraitClassifierTests.evidence('calibrate', [.12, .22, .78, .88])
    return train_trait_model(training, calibration, vocabulary,
                             iterations=120, calibration_iterations=31).to_json()


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
