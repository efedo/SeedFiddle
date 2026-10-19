"""Explicit review declarations for synthetic complete-outline test fixtures."""
import numpy as np
from functools import lru_cache
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


def _workflow_template_key():
    """Small source/config digest plus identities of the loaded factory helpers.

    No production validation is memoized: this key governs UI fixture setup only.
    Reading bytes also detects edits which preserve timestamps or file sizes.
    """
    from hashlib import sha256
    from importlib import import_module
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    module_names = ['seedvision.pipeline.workflows', 'seedvision.benchmarking.recipes',
                    'seedvision.benchmarking.recipe_catalogue']
    module_names.extend('seedvision.benchmarking.' + path.stem
        for path in sorted((root / 'seedvision' / 'benchmarking').glob('native*recipe*.py')))
    modules = tuple((name, import_module(name)) for name in module_names)
    paths = [root / 'config' / 'peer_reference_sources.json',
             root / 'config' / 'plantcv-reference-requirements.txt']
    for directory in ('benchmarking', 'pipeline', 'cuda'):
        paths.extend((root / 'seedvision' / directory).glob('*.py'))
        paths.extend((root / 'seedvision' / directory).glob('*.cu'))
    digest = sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256(path.read_bytes()).digest())
    factories = tuple((name, member, id(value))
        for name, module in modules
        for member, value in sorted(vars(module).items()) if callable(value))
    return digest.digest(), factories


@lru_cache(maxsize=1)
def _workflow_templates(key, builder):
    """Share frozen recipes and retain their original serialized identities."""
    templates = tuple(builder())
    return templates, tuple(recipe.to_json() for recipe in templates)


def isolated_window(root, *, reuse_templates=True, settings_path=None, editable=True):
    """Fresh desktop state with local settings and optional immutable templates.

    Factory/source-validation witnesses can opt out. An explicitly mocked
    catalogue remains in control rather than being replaced by this fixture.
    Legacy edit fixtures explicitly unfreeze; startup-lock witnesses pass editable=False.
    """
    from contextlib import ExitStack
    from unittest.mock import Mock, patch
    from PySide6.QtCore import QSettings
    from seedvision.pipeline import workflows
    from seedvision.ui.main_window import MainWindow
    settings = QSettings(str(settings_path if settings_path is not None
                             else root / 'test-application-settings.ini'), QSettings.Format.IniFormat)
    settings.clear()
    templates = None
    with ExitStack() as stack:
        stack.enter_context(patch('seedvision.ui.main_window.QSettings', return_value=settings))
        if reuse_templates and not isinstance(workflows.workflow_catalogue, Mock):
            templates, identities = _workflow_templates(_workflow_template_key(), workflows.workflow_catalogue)
            if tuple(recipe.to_json() for recipe in templates) != identities:
                raise AssertionError('Shared workflow templates were changed before construction')
            stack.enter_context(patch.object(workflows, 'workflow_catalogue', return_value=templates))
        window = MainWindow(root)
        if templates is not None:
            if tuple(recipe.to_json() for recipe in templates) != identities:
                raise AssertionError('Window construction changed shared workflow templates')
    window._test_application_settings = settings
    if editable and hasattr(window, 'pipeline_documents'):
        window.pipeline_documents.set_frozen(False)
    return window


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
