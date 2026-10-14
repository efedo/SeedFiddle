"""Frozen-centre picker and cluster-mask previews on the native recipe canvas."""
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton

from seedvision.benchmarking import native_kmeans_recipe as _kmeans_recipe
from seedvision.ui.native_naive_bayes_dialog import NativeNaiveBayesDialog
from seedvision.ui.native_kmeans_training_dialog import NativeKMeansTrainingDialog


class NativeKMeansDialog(NativeNaiveBayesDialog):
    model_stage_id = _kmeans_recipe.MODEL_STAGE_ID
    model_parameter_id = _kmeans_recipe.MODEL_PARAMETER_ID
    classifier_stage_id = _kmeans_recipe.CLASSIFIER_STAGE_ID
    model_button_caption = 'Choose centres model…'
    model_button_object_name = 'nativeKMeansChooseModel'
    model_dialog_title = 'Choose frozen SeedVision K-means centres'
    model_file_filter = 'SeedVision K-means centre models (*.json);;All files (*)'
    class_selector_object_name = 'nativeKMeansClusterSelector'
    class_selector_accessible_name = 'Cluster mask'
    no_model_text = 'No centre model selected. Choose a SeedVision JSON centre model.'
    model_required_text = 'Choose a frozen K-means centre model before running this recipe.'
    model_selected_text = 'Centre model selected and byte-bound. Run to compute its cluster masks.'

    def __init__(self, recipe, parent=None, *, model_asset_paths=None, keep_in_project=None):
        super().__init__(recipe, parent, model_asset_paths=model_asset_paths, keep_in_project=keep_in_project)
        self.train_model_button = QPushButton('Train centres…', self)
        self.train_model_button.setObjectName('nativeKMeansTrainModel')
        self.choose_model_button.parentWidget().layout().insertWidget(1, self.train_model_button)
        self.train_model_button.clicked.connect(self._train_model)

    def _train_model(self):
        if self._worker is not None:
            return
        dialog = NativeKMeansTrainingDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved_model_path:
            try:
                selected = self._model_source_for_path(dialog.saved_model_path)
            except (ValueError, OSError, TypeError) as error:
                QMessageBox.warning(self, 'Choose trained centres', str(error))
                return
            self._set_model_source(selected)

    def start_run(self):
        super().start_run()
        self.train_model_button.setEnabled(self._worker is None)

    def _completed(self, result):
        super()._completed(result)
        self.train_model_button.setEnabled(not self._close_requested)

    def _model_source_for_path(self, path):
        return _kmeans_recipe.model_source_for_path(path)

    def _format_model_label(self, selected):
        channels = int(selected['channels'])
        channel_text = 'grayscale' if channels == 1 else f'{channels} channels (BGR features)'
        return (
            f"{selected['path']} · SHA-256 {selected['source_sha256'][:12]}… · "
            f"{selected['cluster_count']} clusters · "
            f"{selected['patch_size']}×{selected['patch_size']} patches · "
            f"{channel_text} · {selected['preprocessing']} · saved recipes reference this local file")
