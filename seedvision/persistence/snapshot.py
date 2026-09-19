"""Verified immutable project packages, separate from mutable working sidecars."""
from dataclasses import replace
from pathlib import Path, PurePosixPath
from uuid import uuid4
import json
import shutil
import zipfile

from seedvision.persistence.project_analysis import (ProjectAnalysisStore, ProjectFileStatus,
    ProjectFileReference, ProjectSidecarReference, PROJECT_RELATIVE, _image_identifier)
from seedvision.persistence.reference_regions import file_sha256


def create_snapshot(root, project_path, output_directory, *, library_store=None):
    root = Path(root)
    loaded = ProjectAnalysisStore(root).load(project_path)
    if loaded.issues:
        raise ValueError('Resolve missing, changed or invalid project files before creating a snapshot.')
    destination = Path(output_directory)/('seedfiddle-snapshot-'+uuid4().hex)
    destination.mkdir(parents=True,exist_ok=False)
    inventory = {}
    supplementary = {}
    def copy(source, expected=None):
        source = Path(source)
        digest = file_sha256(source)
        if expected is not None and digest != expected:
            raise ValueError(f'Source changed while snapshotting: {source}')
        relative = 'objects/'+digest+source.suffix
        target = destination/relative
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(source,target)
        if file_sha256(target) != digest or file_sha256(source) != digest:
            raise ValueError(f'Copy verification failed: {source}')
        inventory[relative] = digest
        return relative
    images = []
    for image in loaded.images:
        source = replace(image.record.source,location=PROJECT_RELATIVE,path=copy(image.path,image.record.source.sha256))
        sidecars = tuple(replace(item.reference,path=copy(item.path,item.reference.sha256)) for item in image.sidecars)
        images.append(replace(image.record,source=source,sidecars=sidecars,identifier=_image_identifier(source.location,source.path)))
    nodes = []
    for node in loaded.document.analysis_settings.nodes:
        parameters = dict(node.parameters)
        if node.enabled and 'checkpoint_path' in parameters:
            path = Path(parameters['checkpoint_path'])
            parameters['checkpoint_path'] = copy(path if path.is_absolute() else root/path)
        nodes.append(replace(node,parameters=tuple(parameters.items())))
    if loaded.document.species_library is not None:
        if library_store is None:
            raise ValueError('The pinned library store is required to make a complete portable snapshot.')
        bundle = library_store.export_library(loaded.document.species_library,destination/'pinned-library.zip')
        inventory[bundle.relative_to(destination).as_posix()] = file_sha256(bundle)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    document = replace(loaded.document,images=tuple(images),selected_image_id=selected,
        analysis_settings=replace(loaded.document.analysis_settings,nodes=tuple(nodes)))
    master = ProjectAnalysisStore(destination).save(document,destination/'snapshot.seedfiddle-project.json')
    inventory[master.name] = file_sha256(master)
    # Keep scientific context and review history as verified evidence. These are
    # not automatically applied to relocated images with different identities.
    for folder in ('config','result-reviews'):
        if (root/folder).is_dir():
            for source in sorted((root/folder).rglob('*')):
                if source.is_file():
                    supplementary[source.relative_to(root).as_posix()] = copy(source)
    project = Path(project_path)
    for source in sorted(project.parent.glob(project.stem+'.optimization-*.json')):
        supplementary[str(source)] = copy(source)
        if source.name.endswith('.optimization-targets.json'):
            for target in json.loads(source.read_text(encoding='utf-8')).values():
                path = Path(target)
                path = path if path.is_absolute() else project.parent/path
                supplementary[str(path)] = copy(path)
    # The completion marker is published last. A failed partial folder cannot be restored.
    (destination/'snapshot-index.json').write_text(json.dumps({'version':1,'files':inventory,
        'original_project_sha256':file_sha256(project_path),'master':master.name,
        'supplementary_evidence':supplementary},indent=2)+'\n',encoding='utf-8')
    return destination


def restore_snapshot(snapshot, workspace, *, library_store=None):
    source = Path(snapshot)
    index = json.loads((source/'snapshot-index.json').read_text(encoding='utf-8'))
    if index.get('version') != 1:
        raise ValueError('Unsupported snapshot schema.')
    files = index['files']
    for name,digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe snapshot path.')
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or file_sha256(path) != digest:
            raise ValueError(f'Snapshot verification failed: {name}')
    if index['master'] not in files:
        raise ValueError('Snapshot master is not verified.')
    loaded = ProjectAnalysisStore(source).load(source/index['master'])
    if loaded.issues:
        raise ValueError('Snapshot dependencies are incomplete.')
    for item in loaded.document.images:
        if item.source.location != PROJECT_RELATIVE or item.source.path not in files:
            raise ValueError('Portable snapshot source is not in its verified inventory.')
        if any(sidecar.path not in files for sidecar in item.sidecars):
            raise ValueError('Portable snapshot sidecar is not in its verified inventory.')
    for node in loaded.document.analysis_settings.nodes:
        if node.enabled and (checkpoint := dict(node.parameters).get('checkpoint_path')) and checkpoint not in files:
            raise ValueError('Portable snapshot checkpoint is not in its verified inventory.')
    destination = Path(workspace)/'restored-projects'/uuid4().hex
    destination.mkdir(parents=True,exist_ok=False)
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
        if file_sha256(target) != files[name]:
            raise ValueError(f'Restored copy failed verification: {name}')
    (destination/'snapshot-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    prefix = destination.relative_to(Path(workspace)).as_posix()+'/'
    images = tuple(replace(image,source=replace(image.source,path=prefix+image.source.path),
        identifier=_image_identifier(PROJECT_RELATIVE,prefix+image.source.path),
        sidecars=tuple(replace(sidecar,path=prefix+sidecar.path) for sidecar in image.sidecars)) for image in loaded.document.images)
    nodes = []
    for node in loaded.document.analysis_settings.nodes:
        parameters = dict(node.parameters)
        if node.enabled and 'checkpoint_path' in parameters:
            parameters['checkpoint_path'] = prefix+parameters['checkpoint_path']
        nodes.append(replace(node,parameters=tuple(parameters.items())))
    if loaded.document.species_library:
        if library_store is None:
            raise ValueError('Provide a library store to restore the pinned library.')
        library_store.import_bundle(destination/'pinned-library.zip')
        library_store.load(loaded.document.species_library)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    document = replace(loaded.document,images=images,selected_image_id=selected,analysis_settings=replace(loaded.document.analysis_settings,nodes=tuple(nodes)))
    return ProjectAnalysisStore(workspace).save(document,destination/'working.seedfiddle-project.json')
