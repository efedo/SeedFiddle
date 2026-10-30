"""The annotation folder a saved project owns.

A project ``name.seedfiddle-project.json`` keeps its per-image annotations in
``name.seedfiddle-annotations/`` beside it:

- ``manual-seed-centres/`` and ``reference-regions/`` hold what the last Save
  recorded in the project file;
- ``unsaved/`` holds edits made since that Save, so a crash does not lose them.
  Save copies them into the folders above; Discard deletes them.

A loose workspace and an untitled project have no such folder. Their edits go
to the workspace's shared ``projects/`` folders, and the first Save copies them
into the new project's folder.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from seedvision.persistence.project_analysis import (
    MANUAL_SEED_CENTRES_SIDECAR,
    PROJECT_ANALYSIS_EXTENSION,
    REFERENCE_REGIONS_SIDECAR,
)


PROJECT_ANNOTATIONS_SUFFIX = ".seedfiddle-annotations"
UNSAVED_ANNOTATIONS_FOLDER = "unsaved"
ANNOTATION_FOLDERS = {
    REFERENCE_REGIONS_SIDECAR: "reference-regions",
    MANUAL_SEED_CENTRES_SIDECAR: "manual-seed-centres",
}


def project_annotation_directory(project_path: Path | str) -> Path:
    """Return ``name.seedfiddle-annotations`` beside ``name.seedfiddle-project.json``."""

    path = Path(project_path)
    name = path.name
    if name.casefold().endswith(PROJECT_ANALYSIS_EXTENSION.casefold()):
        stem = name[: -len(PROJECT_ANALYSIS_EXTENSION)]
    else:
        stem = path.stem
    return path.with_name(f"{stem or 'project'}{PROJECT_ANNOTATIONS_SUFFIX}")


def annotation_folder(
    directory: Path | str, kind: str, *, unsaved: bool = False
) -> Path:
    """Return the saved or unsaved folder for one annotation kind."""

    base = Path(directory)
    if unsaved:
        base = base / UNSAVED_ANNOTATIONS_FOLDER
    return base / ANNOTATION_FOLDERS[kind]


def copy_annotation_file(source: Path | str, destination: Path | str) -> Path:
    """Copy one archive so the destination is never left half-written."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def unsaved_annotation_files(directory: Path | str) -> tuple[Path, ...]:
    """List the archives a previous session left unsaved, oldest first."""

    files = [
        path
        for kind in ANNOTATION_FOLDERS
        if (folder := annotation_folder(directory, kind, unsaved=True)).is_dir()
        for path in folder.iterdir()
        if path.is_file() and path.suffix == ".npz" and not path.name.startswith(".")
    ]
    return tuple(sorted(files, key=lambda path: path.stat().st_mtime))


def discard_unsaved_annotations(directory: Path | str) -> None:
    """Delete every unsaved edit kept for one project."""

    unsaved = Path(directory) / UNSAVED_ANNOTATIONS_FOLDER
    if unsaved.is_dir():
        shutil.rmtree(unsaved)
