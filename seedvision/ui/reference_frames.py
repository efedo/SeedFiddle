"""Controller support for calibrated reference frames and explicit migration."""
from dataclasses import replace
import numpy as np
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QMessageBox, QVBoxLayout
from PySide6.QtCore import Qt

from seedvision.annotation.coordinates import reproject_bundle, validated_transform


class ReferenceFrameController:
    def _init_reference_frames(self):
        self._source_fingerprints = {}
        action = QAction('Review legacy reference alignment…',self)
        action.triggered.connect(self._review_legacy_reference_alignment)
        for menu_action in self.menuBar().actions():
            if menu_action.text() == '&Analysis':
                menu_action.menu().addAction(action)
                break
        self.reference_alignment_action = action

    def _check_source_revision(self, path, *, opening=False):
        from seedvision.persistence.reference_regions import file_sha256
        from seedvision.ui.main_window import _path_identity
        key = _path_identity(path)
        digest = file_sha256(path)
        previous = self._source_fingerprints.get(key)
        if previous is not None and previous != digest:
            self._discard_analysis_cache(key)
            has_references = key in self._unapplied_project_draft_keys() or any(key in getattr(self,name) for name in (
                '_applied_instance_annotations','_applied_background_reference_masks','_applied_foreground_reference_masks',
                '_applied_background_exclusion_masks','_manual_seed_centre_states'))
            if has_references or not opening:
                QMessageBox.warning(self,'Source image changed',
                    'The image bytes changed outside Seed Fiddle. Cached results were invalidated. References and drafts remain in memory, but cannot be applied to the changed source. Restore the original file or open a fresh workspace to revalidate saved references.')
                return False
            self.statusBar().showMessage('Source image changed; cached calculations were invalidated.')
        self._source_fingerprints[key] = digest
        return True

    def _prepare_reference_frame(self, key, result):
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

    def _review_legacy_reference_alignment(self):
        if self._background_work_is_active():
            return
        key = self._current_image_key()
        bundle = self._pending_unbound_reference_bundles.get(key)
        result = self._analyses.get(key)
        if bundle is None or bundle.source_to_corrected is not None or result is None:
            QMessageBox.information(self,'Reference alignment','Analyze an image with a withheld legacy reference archive first.')
            return
        image = result.calibration.corrected_bgr
        if tuple(bundle.shape) != tuple(image.shape[:2]):
            QMessageBox.warning(self,'Original transform required','The legacy raster has a different canvas size. Supply its original source-to-corrected transform or recreate the annotations; resizing would not establish alignment.')
            return
        import cv2
        scale = min(1.,900/max(bundle.shape))
        size = (max(1,round(bundle.shape[1]*scale)),max(1,round(bundle.shape[0]*scale)))
        preview = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        for raster,colour in ((bundle.background,(255,120,0)),(bundle.foreground,(0,220,0)),
                              (bundle.other,(220,0,220)),(bundle.annotated_seeds,(0,220,255))):
            if raster is not None:
                mask = cv2.resize(np.uint8(raster>0),size,interpolation=cv2.INTER_NEAREST)>0
                preview[mask] = (preview[mask]*.55 + np.asarray(colour)*.45).astype(np.uint8)
        rgb = cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)
        dialog = QDialog(self)
        dialog.setWindowTitle('Verify legacy reference alignment')
        layout = QVBoxLayout(dialog)
        explanation = QLabel('This archive has no recorded calibration transform. Inspect the overlay carefully.\n'
            'Confirm only if the saved regions already align with the current image.\n'
            'This binds coordinates; it does not approve biological or shape annotations.')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        view = QLabel()
        view.setPixmap(QPixmap.fromImage(QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()))
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('I verified alignment — bind and save')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        bound = replace(bundle,source_to_corrected=validated_transform(result.calibration.affine_matrix))
        try:
            from pathlib import Path
            from uuid import uuid4
            import shutil
            original = Path(self._reference_region_store.path_for(result.image_path))
            if original.is_file():
                shutil.copyfile(original,original.with_name(original.stem+'.unbound-'+uuid4().hex+original.suffix))
            path = self._reference_region_store.save(result.image_path,bound)
        except (ValueError,OSError) as error:
            QMessageBox.warning(self,'Could not bind references',str(error))
            return
        self._pending_unbound_reference_bundles[key] = bound
        self._project_reference_sidecar_paths[key] = path
        affected = self._resolve_pending_project_reference_bundle(key,result)
        self._set_project_dirty()
        self._analyze_current_image(dirty_nodes=affected)
