"""Controller support for calibrated reference frames."""
import numpy as np
from PySide6.QtWidgets import QMessageBox

from seedvision.annotation.coordinates import reproject_bundle, validated_transform


class ReferenceFrameController:
    def _init_reference_frames(self):
        self._source_fingerprints = {}

    def _check_source_revision(self, path, *, opening=False):
        from seedvision.persistence.reference_regions import file_sha256
        from seedvision.ui.main_window import _path_identity
        from seedvision.export.preflight_outcomes import begin_preflight
        from seedvision.export.run_outcomes import AnalysisInputIntegrityError
        attempt = begin_preflight(path, 'image', 'desktop_source')
        revision = self.pipeline.revision
        key = _path_identity(path)
        try:
            digest = file_sha256(path)
        except OSError as error:
            if key not in self._active_tasks:
                self._discard_analysis_cache(key)
            diagnostic = self._desktop_rejection(attempt, error, pipeline_revision=revision, image_install=True)
            QMessageBox.warning(self, 'Source image unavailable', diagnostic)
            return False
        previous = self._source_fingerprints.get(key)
        if previous is not None and previous != digest:
            self._discard_analysis_cache(key)
            has_references = key in self._unapplied_project_draft_keys() or any(key in getattr(self,name) for name in (
                '_applied_instance_annotations','_applied_background_reference_masks','_applied_foreground_reference_masks',
                '_applied_background_exclusion_masks','_manual_seed_centre_states'))
            if has_references or not opening:
                diagnostic = self._desktop_rejection(attempt, AnalysisInputIntegrityError(
                    'The image bytes changed outside Seed Fiddle. Cached results were invalidated. References and drafts remain in memory, but cannot be applied to the changed source. Restore the original file or open a fresh workspace to revalidate saved references.'),
                    pipeline_revision=revision, image_install=True)
                QMessageBox.warning(self,'Source image changed',
                    diagnostic)
                return False
            self.statusBar().showMessage('Source image changed; cached calculations were invalidated.')
        self._source_fingerprints[key] = digest
        return True

    def _prepare_reference_frame(self, key, result):
        from seedvision.measurement.lineage import reference_source_identity_state
        archive, snapshot = self._reference_source_identities.get(key, (None, None))
        try:
            for known, name in ((archive, 'reference archive source SHA-256'),
                                (snapshot, 'reference annotation snapshot source SHA-256')):
                reference_source_identity_state(getattr(result, 'source_sha256', None), known, name)
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return False
        current = validated_transform(result.calibration.affine_matrix)
        previous = self._reference_transforms.get(key)
        if previous is not None and (not np.allclose(previous,current,rtol=0,atol=1e-8)
                or self._reference_layer_shapes.get(key) != tuple(result.calibration.corrected_bgr.shape[:2])):
            if key in self._unapplied_project_draft_keys():
                if not self._resolve_unapplied_project_drafts(recompute=False):
                    self.statusBar().showMessage('New calibration display withheld until reference edits are saved or discarded.')
                    return False
            bundle = self._reference_bundle_for_project_key(key,result.image_path)
            aligned = reproject_bundle(bundle,current,result.calibration.corrected_bgr.shape[:2])
            self._install_reference_region_bundle(key,aligned,sync_view=False)
            self._reference_layer_shapes[key] = aligned.shape
            self.statusBar().showMessage('References were reprojected from their saved calibration frame; old-coordinate undo history was cleared.')
        self._reference_transforms[key] = current.copy()
        return True
