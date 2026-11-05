"""Build the dataset Train uses: analyse each chosen image, then save it as a sample."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from seedvision.ui.image_list import UNKNOWN_SPECIES
from seedvision.ui.learning_workflow import LearningTrainingRequest, TrainingImage
from seedvision.ui.persistence_jobs import run_persistence

# A first run can install saved outlines, which requires one more run with them.
_MAX_ANALYSIS_RUNS_PER_IMAGE = 3


@dataclass(slots=True)
class _DatasetBuild:
    request: LearningTrainingRequest
    remaining: list
    return_path: Path | None
    progress: QProgressDialog
    analysis_runs: int = 0
    advancing: bool = False
    failed: bool = False
    stopped: bool = False


def _outline_summary(labels, annotations):
    """Count seeds, name those export refuses, and report whether all were reviewed."""
    from seedvision.annotation.eligibility import connected_instances
    seed_ids = set(np.unique(labels).tolist()) - {0}
    allowed = set(np.unique(connected_instances(labels, annotations)).tolist())
    reviewed = {item.seed_id for item in annotations if item.shape_reviewed}
    return len(seed_ids), tuple(sorted(seed_ids - allowed)), bool(seed_ids) and seed_ids <= reviewed


def _read_outline_summaries(window, sources):
    """Summarise in-memory labels and read saved ones, which is slow enough for a worker."""
    from seedvision.persistence.reference_regions import (
        ImageFingerprintMismatch, InvalidReferenceArchive, ReferenceRegionError,
    )
    summaries = {}
    for key, (path, labels, annotations, species, archive_path, read_saved) in sources.items():
        if read_saved:
            store = window._reference_region_store
            try:
                bundle = (store.load_project_archive(path, archive_path, None) if archive_path is not None
                          else store.load_if_present(path, None))
            except (ImageFingerprintMismatch, InvalidReferenceArchive, ReferenceRegionError, OSError) as error:
                summaries[key] = (0, (), False, species, f'Saved outlines could not be read: {error}')
                continue
            if bundle is None:
                continue
            labels, annotations, species = bundle.annotated_seeds, bundle.seed_annotations, bundle.annotation_species
        if labels is None or not np.any(labels):
            continue
        summaries[key] = (*_outline_summary(labels, tuple(annotations)), species, None)
    return summaries


def _seed_list(identifiers, limit=8):
    shown = ', '.join(map(str, identifiers[:limit]))
    return shown + (f' and {len(identifiers) - limit} more' if len(identifiers) > limit else '')


class TrainingDatasetController:
    _training_dataset_build = None

    def _training_images(self) -> tuple[TrainingImage, ...]:
        """List project images that have seed outlines, reading any not yet opened."""
        sources = {}
        for key, path in self._image_paths.items():
            pending = self._pending_unbound_reference_bundles.get(key)
            labels = self._draft_instance_annotations.get(key, self._applied_instance_annotations.get(key))
            annotations = self._draft_seed_annotations.get(key, self._applied_seed_annotations.get(key, {}))
            annotations = tuple(annotations.values())
            species = self._applied_seed_annotation_species.get(key, '')
            if labels is None and pending is not None:
                labels, annotations, species = pending.annotated_seeds, pending.seed_annotations, pending.annotation_species
            read_saved = (
                labels is None
                and key not in self._reference_region_autoload_attempted
                and (not self._project_manifest_sidecar_policy_active or key in self._project_unbound_reference_loads)
            )
            sources[key] = (Path(path), labels, annotations, species,
                            self._project_reference_sidecar_paths.get(key), read_saved)
        summaries = run_persistence(self, partial(_read_outline_summaries, self, sources),
                                    'Reading saved seed outlines…')
        images = []
        for key, (count, blocked, reviewed, species, problem) in summaries.items():
            if blocked:
                problem = f'Seeds {_seed_list(blocked)} are disconnected or excluded.'
            elif key in self._instance_annotations_dirty or key in self._reference_masks_dirty:
                problem = 'Apply or discard the draft on this image first.'
            context = self._project_image_biological_contexts.get(key)
            images.append(TrainingImage(key=key, path=sources[key][0], seed_count=count,
                                        species=self._training_species_name(key, species),
                                        reviewed=reviewed, problem=problem,
                                        group=(context.seed_lot_id if context is not None else None) or ''))
        return tuple(images)

    def _training_species_name(self, key, saved_species):
        """Prefer the image's recorded species, then its outlines', then the project's."""
        context = self._project_image_biological_contexts.get(key)
        species = context.species_id if context is not None else saved_species
        if species not in {'', 'unknown'}:
            return self._seed_trait_catalogue.for_species(species).display_name
        name = self._image_species_name(key)
        return '' if name == UNKNOWN_SPECIES else name

    def _training_analysis_scope(self) -> frozenset[str]:
        """Name the prerequisites of a learned node, which compute the model's input features."""
        scope = set()
        pending = list(self.pipeline.upstream('unet_instances'))
        while pending:
            current = pending.pop()
            if current in scope or current not in self.pipeline.nodes or not self.pipeline.node(current).implemented:
                continue
            scope.add(current)
            pending.extend(self.pipeline.upstream(current))
        return frozenset(scope)

    def _start_training_dataset_build(self, request: LearningTrainingRequest) -> None:
        progress = QProgressDialog('Preparing training dataset…', 'Cancel', 0, len(request.images), self)
        progress.setWindowTitle('Building training dataset')
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.canceled.connect(self._stop_training_dataset_build)
        self._training_dataset_build = _DatasetBuild(
            request=request, remaining=list(request.images),
            return_path=self.image_view.image_path, progress=progress)
        progress.show()
        self._update_analysis_availability()
        self._advance_training_dataset_build()

    def _stop_training_dataset_build(self) -> None:
        build = self._training_dataset_build
        if build is None:
            return
        build.stopped = True
        build.progress.setLabelText('Stopping…')
        self._stop_current_work()
        self._advance_training_dataset_build()

    def _advance_training_dataset_build(self) -> None:
        """Take the next step once the single analysis worker has nothing running."""
        build = self._training_dataset_build
        if build is None or build.advancing or self._active_tasks:
            return
        build.advancing = True
        try:
            self._run_training_dataset_steps(build)
        finally:
            build.advancing = False

    def _run_training_dataset_steps(self, build) -> None:
        scope = self._training_analysis_scope()
        total = len(build.request.images)
        while build.remaining:
            if build.stopped or build.failed:
                self._end_training_dataset_build()
                if build.stopped:
                    self.statusBar().showMessage('Training stopped; no model was trained.')
                return
            image = build.remaining[0]
            position = total - len(build.remaining) + 1
            if self._current_image_key() != image.key:
                self._open_path(image.path, mark_project_dirty=False)
                build.analysis_runs = 0
                if self._current_image_key() != image.key:
                    self._fail_training_dataset_build(image, 'The image could not be opened.')
                    return
            ran = self._analysis_scopes.get(image.key)
            if (self._analyses.get(image.key) is None or self._cache_dirty_nodes.get(image.key)
                    or (ran is not None and not scope <= ran)):
                if build.analysis_runs >= _MAX_ANALYSIS_RUNS_PER_IMAGE:
                    self._fail_training_dataset_build(image, 'Analysis did not produce a usable result.')
                    return
                build.analysis_runs += 1
                build.progress.setLabelText(f'Analysing {image.path.name} ({position} of {total})…')
                self._analyze_current_image(enabled_node_scope=scope)
                if not self._active_tasks:
                    self._fail_training_dataset_build(image, 'Analysis could not start.')
                return
            build.progress.setLabelText(f'Saving {image.path.name} ({position} of {total})…')
            try:
                self._save_training_sample(build.request, image)
            except Exception as error:  # noqa: BLE001 - user-facing export boundary
                self._fail_training_dataset_build(image, str(error))
                return
            build.remaining.pop(0)
            build.analysis_runs = 0
            build.progress.setValue(position)
        self._end_training_dataset_build()
        if build.stopped:
            self.statusBar().showMessage('Training stopped; no model was trained.')
        else:
            self._train_on_built_dataset(build.request)

    def _save_training_sample(self, request, image) -> None:
        from seedvision.learning.export import export_analysis_sample
        labels = self._applied_instance_annotations.get(image.key)
        if labels is None or not np.any(labels):
            raise ValueError('It has no applied seed outlines after analysis.')
        origin = self._applied_instance_annotation_origins.get(image.key, 'manual')
        run_persistence(self, partial(
            export_analysis_sample,
            self._analyses[image.key],
            labels,
            request.manifest_path,
            dataset_id=request.manifest_path.parent.name,
            identifier=image.identifier,
            species=image.species,
            group=image.group,
            species_reviewed=True,
            split=image.split,
            reviewed=image.reviewed,
            seed_annotations=tuple(self._applied_seed_annotations.get(image.key, {}).values()),
            annotation_author=request.reviewer if image.reviewed else None,
            notes=f'Built by Train from applied outlines; draft origin={origin}.',
        ), f'Saving {image.path.name}…')

    def _fail_training_dataset_build(self, image, reason) -> None:
        self._end_training_dataset_build()
        QMessageBox.critical(self, 'Could not build the training dataset', f'{image.path.name}\n\n{reason}')

    def _end_training_dataset_build(self) -> None:
        build = self._training_dataset_build
        # Closing a progress dialog emits canceled, so forget the build first.
        self._training_dataset_build = None
        build.progress.close()
        from seedvision.ui.main_window import _path_identity
        if build.return_path is not None and self._current_image_key() != _path_identity(build.return_path):
            self._open_path(build.return_path, mark_project_dirty=False)
        self._update_analysis_availability()
