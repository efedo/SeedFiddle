"""Turn the seeds a project has labelled into evidence for trait training.

Reads each image's saved annotations, reports which seeds can be used and why
the others cannot, and writes the manifest ``read_trait_manifest`` checks. Which
seeds qualify is decided by ``seed_exclusion_reason``, the rule training enforces.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from seedvision.annotation.coordinates import validated_transform
from seedvision.annotation.seed_traits import trait_display_name
from seedvision.learning.trait_features import TRAIT_FEATURE_SCHEMA
from seedvision.learning.trait_training import seed_exclusion_reason
from seedvision.persistence.reference_regions import (
    ImageFingerprintMismatch, InvalidReferenceArchive, ReferenceRegionError, file_sha256,
)

COAT_PATTERN = 'coat_pattern'
_CONDITION = 'condition:'
TRAIN, CALIBRATION = 'train', 'calibration'
_HELD_OUT_SHARE = .2


@dataclass(frozen=True, slots=True)
class TraitImageSource:
    """Where one project image's annotations are, and what the window knows of it."""
    key: str
    name: str
    path: Path
    archive_path: Path | None
    species: str = ''
    group: str = ''
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class TraitSeed:
    seed_id: int
    coat_pattern: str | None
    conditions: tuple[str, ...]
    conditions_reviewed: bool
    physical_id: str | None
    exclusion: str | None


@dataclass(frozen=True, slots=True)
class TraitImage:
    source: TraitImageSource
    seeds: tuple[TraitSeed, ...] = ()
    image_sha256: str = ''
    archive_sha256: str = ''
    problem: str | None = None


def _seeds_outside_source(labels, matrix, image_path):
    """Name the seeds with a pixel beyond the decoded image, as training would find."""
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ReferenceRegionError(f'Cannot decode {Path(image_path).name}.')
    height, width = image.shape[:2]
    rows, columns = np.nonzero(labels)
    points = np.stack((columns, rows, np.ones_like(rows)), 1).astype(np.float64) @ np.linalg.inv(matrix).T
    with np.errstate(divide='ignore', invalid='ignore'):
        x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
    inside = (points[:, 2] != 0) & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    return set(np.unique(labels[rows[~inside], columns[~inside]]).tolist())


def read_trait_image(source, store):
    """Read one image's labelled seeds. A bad archive is that image's problem, not a failure."""
    if source.problem is not None or source.archive_path is None:
        return TraitImage(source, problem=source.problem)
    try:
        bundle = store.load_project_archive(source.path, source.archive_path, None)
        if bundle is None or bundle.annotated_seeds is None:
            return TraitImage(source)
        labelled = [item for item in bundle.seed_annotations
                    if item.coat_pattern is not None or item.conditions or item.conditions_reviewed]
        if not labelled:
            return TraitImage(source)
        labels = bundle.annotated_seeds
        if bundle.source_to_corrected is None:
            outside, unplaced = set(), 'no recorded image correction'
        else:
            matrix = validated_transform(bundle.source_to_corrected)
            outside, unplaced = _seeds_outside_source(labels, matrix, source.path), None
        edge = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
        seeds = tuple(
            TraitSeed(
                item.seed_id, item.coat_pattern, tuple(item.conditions), bool(item.conditions_reviewed),
                item.physical_seed_id,
                seed_exclusion_reason(item, touches_edge=item.seed_id in edge) or unplaced
                or ('outside the decoded image' if item.seed_id in outside else None))
            for item in labelled)
        return TraitImage(source, seeds, file_sha256(source.path), file_sha256(source.archive_path))
    except (ImageFingerprintMismatch, InvalidReferenceArchive, ReferenceRegionError, OSError, ValueError) as error:
        return TraitImage(source, problem=f'Saved outlines could not be read: {error}')


def read_trait_images(sources, store):
    return tuple(read_trait_image(source, store) for source in sources)


def seed_label(target, seed):
    """Return the class one seed is evidence of, or None when it says nothing about the target."""
    if target == COAT_PATTERN:
        return seed.coat_pattern
    condition = target.removeprefix(_CONDITION)
    if condition in seed.conditions:
        return condition
    return f'not_{condition}' if seed.conditions_reviewed else None


def trait_targets(images):
    """Offer what a model could classify: coat pattern, and each condition a seed has."""
    seeds = [seed for image in images for seed in image.seeds]
    targets = [COAT_PATTERN] if any(seed.coat_pattern is not None for seed in seeds) else []
    return (*targets, *(_CONDITION + name for name in sorted({name for seed in seeds for name in seed.conditions})))


def target_title(target):
    if target == COAT_PATTERN:
        return 'Coat pattern'
    return f'Condition: {trait_display_name(target.removeprefix(_CONDITION)).lower()}'


def target_file_stem(target):
    return target.replace(':', '-').replace('_', '-')


def target_seeds(target, image):
    """Count one image's usable seeds by class, and its unusable ones by reason."""
    usable, skipped = Counter(), Counter()
    for seed in image.seeds:
        label = seed_label(target, seed)
        if label is None:
            continue
        if seed.exclusion is None:
            usable[label] += 1
        else:
            skipped[seed.exclusion] += 1
    return usable, skipped


def default_trait_splits(target, images):
    """Hold out about a fifth of the seeds by whole group, keeping every class in training."""
    groups, usable = {}, []
    for image in images:
        counts = target_seeds(target, image)[0]
        if image.problem is None and counts:
            usable.append(image)
            groups.setdefault(image.source.group, Counter()).update(counts)
    classes = set().union(*groups.values()) if groups else set()
    total = sum(counts.total() for counts in groups.values())
    training, held_out = dict(groups), 0
    for group, counts in sorted(groups.items(), key=lambda item: (item[1].total(), item[0])):
        if held_out >= total * _HELD_OUT_SHARE:
            break
        remaining = [other for name, other in training.items() if name != group]
        if remaining and set().union(*remaining) >= classes:
            del training[group]
            held_out += counts.total()
    return {image.source.key: TRAIN if image.source.group in training else CALIBRATION for image in usable}


def split_counts(target, images, splits):
    """Add up the usable seeds of each class the chosen splits send to each side."""
    counts = {TRAIN: Counter(), CALIBRATION: Counter()}
    for image in images:
        split = splits.get(image.source.key)
        if split in counts and image.problem is None:
            counts[split].update(target_seeds(target, image)[0])
    return counts


def _class_list(labels):
    return ', '.join(trait_display_name(label) for label in sorted(labels))


def evidence_problem(target, images, splits):
    """Say what stops training on these splits, or None when it can start."""
    counts = split_counts(target, images, splits)
    classes = set(counts[TRAIN]) | set(counts[CALIBRATION])
    if not classes:
        return 'No image in use has a seed this trait can use.'
    if len(classes) < 2:
        return f'Only {_class_list(classes)} has usable seeds. A model needs at least two classes.'
    missing = classes - set(counts[TRAIN])
    if missing:
        return f'{_class_list(missing)} has no seeds in a Training image.'
    if not counts[CALIBRATION]:
        return 'Set an image from a different seed lot to Calibration. It tunes the confidence of the model on seeds it was not trained on.'
    return None


def _definition(target, label):
    if target == COAT_PATTERN:
        return f'Seed coat pattern annotated as {trait_display_name(label).lower()}.'
    condition = trait_display_name(target.removeprefix(_CONDITION)).lower()
    if label.startswith('not_'):
        return f'Seed condition reviewed and not annotated as {condition}.'
    return f'Seed annotated with the {condition} condition.'


def build_trait_manifest(target, images, splits, *, definition_source):
    """Write what these annotations say in the form a hand-written evidence file takes."""
    digest = file_sha256(definition_source)
    examples, classes = [], set()
    for image in images:
        split = splits.get(image.source.key)
        if split not in (TRAIN, CALIBRATION) or image.problem is not None:
            continue
        for seed in image.seeds:
            label = seed_label(target, seed)
            if label is None or seed.exclusion is not None:
                continue
            classes.add(label)
            identifier = f'{image.source.name}-seed-{seed.seed_id}'
            examples.append(dict(
                example_id=identifier, physical_id=seed.physical_id or identifier,
                group_id=image.source.group, trait_id=label,
                truth_provenance=f'Project annotation of seed {seed.seed_id} in {image.source.path.name}',
                permitted_use=True, split=split,
                input=dict(kind='reference_seed', image_path=str(image.source.path), image_sha256=image.image_sha256,
                           archive_path=str(image.source.archive_path), archive_sha256=image.archive_sha256,
                           seed_id=seed.seed_id)))
    name = Path(definition_source).name
    return dict(
        format='seedfiddle-trait-training', version=1, feature_schema=TRAIT_FEATURE_SCHEMA,
        vocabulary=dict(
            revision=f'{name}:{digest[:12]}', definition_source=name, definition_source_sha256=digest,
            traits=[dict(identifier=label, definition=_definition(target, label),
                         evidence_reference=f'Project annotations; vocabulary in {name}')
                    for label in sorted(classes)]),
        examples=examples)


def new_trait_model_path(directory, name):
    """Name a new model after what it classifies and when, never reusing a file."""
    stem = f'{name}-{datetime.now():%Y%m%d-%H%M%S}'
    path, suffix = Path(directory) / f'{stem}.json', 2
    while path.exists() or trait_evidence_path(path).exists():
        path, suffix = Path(directory) / f'{stem}-{suffix}.json', suffix + 1
    return path


def trait_evidence_path(model_path):
    """Keep the evidence a model was trained on beside it."""
    return Path(model_path).with_suffix('.evidence.json')
