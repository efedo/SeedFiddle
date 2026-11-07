"""Verified immutable project packages, separate from mutable working sidecars."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import uuid4
import json
import shutil
import zipfile
from seedvision.export.review import MAX_REVIEW_BYTES

from seedvision.persistence.project_analysis import (ProjectAnalysisStore, ProjectFileStatus,
    ProjectFileReference, ProjectSidecarReference, PROJECT_RELATIVE, _image_identifier)
from seedvision.persistence.reference_regions import file_sha256
from seedvision.persistence.trait_model_assets import (
    capture_trait_model, snapshot_trait_assets, trait_model_bytes, copy_trait_model,
)
from seedvision.persistence.terminal_history import (
    capture_terminal_evidence, evidence_bytes, verify_terminal_evidence,
)


def _review_bytes(path, expected=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_REVIEW_BYTES + 1)
    if len(raw) > MAX_REVIEW_BYTES or expected is not None and sha256(raw).hexdigest() != expected:
        raise ValueError('Portable review evidence exceeds its limit or changed bytes.')
    return raw


def _review_evidence(index):
    """Map only the known review namespace into a new private project folder."""
    supplementary = index.get('supplementary_evidence', {})
    if not isinstance(supplementary, dict):
        raise ValueError('Invalid supplementary evidence inventory.')
    rows, seen = [], set()
    for name, object_path in supplementary.items():
        if not isinstance(name, str):
            raise ValueError('Invalid supplementary evidence name.')
        if not name.casefold().startswith('result-reviews'):
            continue
        relative = PurePosixPath(name)
        if (len(relative.parts) < 2 or relative.parts[0] != 'result-reviews'
                or relative.as_posix() != name or '..' in relative.parts or '\\' in name or ':' in name
                or any(part.endswith((' ', '.')) for part in relative.parts)
                or name.casefold() in seen or not isinstance(object_path, str)
                or object_path not in index['files']
                or PurePosixPath(object_path).is_absolute() or '..' in PurePosixPath(object_path).parts
                or '\\' in object_path or ':' in object_path):
            raise ValueError('Portable review evidence has an unsafe or unverified mapping.')
        seen.add(name.casefold())
        rows.append((name, object_path, index['files'][object_path]))
    return rows


def create_snapshot(root, project_path, output_directory, *, library_store=None):
    root = Path(root)
    loaded = ProjectAnalysisStore(root).load(project_path)
    if loaded.issues:
        raise ValueError('Resolve missing, changed or invalid project files before creating a snapshot.')
    trait_model = capture_trait_model(root, loaded.document.analysis_settings)
    instance_traits = {instance.instance_id: capture_trait_model(root, instance.default_settings)
                       for instance in loaded.document.node_instances}
    instance_traits = {key: value for key, value in instance_traits.items() if value is not None}
    terminal_evidence = capture_terminal_evidence(root, project_path)
    destination = Path(output_directory)/('seedfiddle-snapshot-'+uuid4().hex)
    destination.mkdir(parents=True,exist_ok=False)
    inventory = {}
    supplementary = {}
    for name, raw in terminal_evidence:
        digest = sha256(raw).hexdigest()
        namespace = name.split('/')[0]
        suffix = {'terminal-history': '.terminal.json', 'terminal-acknowledgments': '.terminal-ack.json'}[namespace]
        object_path = 'objects/' + digest + suffix
        target = destination / object_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        evidence_bytes(target, digest, len(raw))
        inventory[object_path] = digest
        supplementary[name] = object_path
    if trait_model is not None:
        trait_source, trait_asset, trait_raw = trait_model
        trait_target = destination / trait_asset.path
        trait_target.parent.mkdir(parents=True, exist_ok=True)
        copy_trait_model(trait_source, trait_target, trait_asset, trait_raw)
        inventory[trait_asset.path] = trait_asset.source_sha256
    for trait_source_item, trait_asset_item, trait_raw_item in instance_traits.values():
        target = destination / trait_asset_item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_trait_model(trait_source_item, target, trait_asset_item, trait_raw_item)
        inventory[trait_asset_item.path] = trait_asset_item.source_sha256
    def copy(source, expected=None, image_id=None):
        source = Path(source)
        digest = file_sha256(source)
        if expected is not None and digest != expected:
            raise ValueError(f'Source changed while snapshotting: {source}')
        # Preserve distinct logical images even when their bytes are identical.
        relative = (('images/'+image_id+'/') if image_id is not None else 'objects/')+digest+source.suffix
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source,target)
        if file_sha256(target) != digest or file_sha256(source) != digest:
            raise ValueError(f'Copy verification failed: {source}')
        inventory[relative] = digest
        return relative
    images = []
    for image in loaded.images:
        source = replace(image.record.source,location=PROJECT_RELATIVE,
                         path=copy(image.path,image.record.source.sha256,image_id=image.record.identifier))
        sidecars = tuple(replace(item.reference,path=copy(item.path,item.reference.sha256)) for item in image.sidecars)
        images.append(replace(image.record,source=source,sidecars=sidecars,identifier=_image_identifier(source.location,source.path)))
    def portable_settings(profile, trait=None):
        nodes = []
        for node in profile.nodes:
            parameters = dict(node.parameters)
            if 'checkpoint_path' in parameters and (node.enabled or
                    (root / Path(parameters['checkpoint_path']).expanduser()).is_file()):
                path = Path(parameters['checkpoint_path'])
                parameters['checkpoint_path'] = copy(path if path.is_absolute() else root/path)
            if node.identifier == 'trait_classifier' and trait is not None:
                parameters['trait_model_path'] = trait[1].path
            nodes.append(replace(node,parameters=tuple(parameters.items())))
        return replace(profile, nodes=tuple(nodes))
    portable_profile = portable_settings(loaded.document.analysis_settings, trait_model)
    portable_instances = tuple(replace(instance,
        default_settings=portable_settings(instance.default_settings, instance_traits.get(instance.instance_id)))
        for instance in loaded.document.node_instances)
    if loaded.document.species_library is not None:
        if library_store is None:
            raise ValueError('The pinned library store is required to make a complete portable snapshot.')
        bundle = library_store.export_library(loaded.document.species_library,destination/'pinned-library.zip')
        inventory[bundle.relative_to(destination).as_posix()] = file_sha256(bundle)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    image_ids = {old.identifier: new.identifier for old, new in zip(loaded.document.images, images, strict=True)}
    if len(set(image_ids.values())) != len(image_ids):
        raise ValueError('Snapshot image relocation collapsed distinct project identities.')
    document = replace(loaded.document,images=tuple(images),selected_image_id=selected,
        analysis_settings=portable_profile, node_instances=portable_instances)
    master = ProjectAnalysisStore(destination).save(document,destination/'snapshot.seedfiddle-project.json')
    inventory[master.name] = file_sha256(master)
    # Context stays supplementary; exact-revision review history is restored
    # privately and can only apply when the current semantic revision matches.
    for folder in ('config','result-reviews'):
        if (root/folder).is_dir():
            for source in sorted((root/folder).rglob('*')):
                if source.is_file():
                    raw = _review_bytes(source) if folder == 'result-reviews' else None
                    supplementary[source.relative_to(root).as_posix()] = copy(source,
                        None if raw is None else sha256(raw).hexdigest())
    project = Path(project_path)
    for source in sorted(project.parent.glob(project.stem+'.optimization-*.json')):
        supplementary[str(source)] = copy(source)
        if source.name.endswith('.optimization-targets.json'):
            for target in json.loads(source.read_text(encoding='utf-8')).values():
                path = Path(target)
                path = path if path.is_absolute() else project.parent/path
                supplementary[str(path)] = copy(path)
    # The completion marker is published last. A failed partial folder cannot be restored.
    index = {'version':2,'files':inventory,
        'original_project_sha256':file_sha256(project_path),'master':master.name,
        'supplementary_evidence':supplementary}
    if trait_model is not None:
        trait_model_bytes(trait_source, trait_asset)
        trait_model_bytes(trait_target, trait_asset)
        index['trait_model_assets'] = [trait_asset.payload()]
    if instance_traits:
        index['node_instance_trait_model_assets'] = {key: value[1].payload() for key, value in instance_traits.items()}
        for selected_source, selected_asset, _ in instance_traits.values():
            trait_model_bytes(selected_source, selected_asset)
            trait_model_bytes(destination / selected_asset.path, selected_asset)
    verify_terminal_evidence(destination, index)
    (destination/'snapshot-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return destination


def restore_snapshot(snapshot, workspace, *, library_store=None):
    source = Path(snapshot)
    index = json.loads((source/'snapshot-index.json').read_text(encoding='utf-8'))
    if index.get('version') != 2:
        raise ValueError('This snapshot uses an earlier, unsupported snapshot format.')
    files = index['files']
    review_evidence = _review_evidence(index)
    terminal_evidence = verify_terminal_evidence(source, index)
    terminal_objects = {object_path: (digest, limit)
        for _, object_path, digest, limit, _ in terminal_evidence}
    if index['master'] not in files:
        raise ValueError('Snapshot master is not verified.')
    master = PurePosixPath(index['master'])
    if (master.is_absolute() or '..' in master.parts or '\\' in index['master'] or ':' in index['master']
            or not (source/index['master']).resolve().is_relative_to(source.resolve())):
        raise ValueError('Unsafe snapshot master path.')
    # Decode bounded metadata before inventory reads to obtain model byte caps.
    # Files and dependencies are still verified before any restore publication.
    preliminary = ProjectAnalysisStore(source).load(source/index['master'], verify_files=False)
    trait_assets_by_path = snapshot_trait_assets(preliminary.document.analysis_settings, index)
    instance_trait_rows = index.get('node_instance_trait_model_assets', {})
    defaults = {instance.instance_id: instance for instance in preliminary.document.node_instances}
    if type(instance_trait_rows) is not dict or set(instance_trait_rows) - set(defaults):
        raise ValueError('Portable node instance trait-model selection is invalid.')
    for identifier, instance in defaults.items():
        trait_assets_by_path.update(snapshot_trait_assets(instance.default_settings,
            {'files': files, 'trait_model_assets': ([instance_trait_rows[identifier]] if identifier in instance_trait_rows else [])}))
    for name,digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe snapshot path.')
        path = (source/name).resolve()
        try:
            if not path.is_relative_to(source.resolve()):
                actual = None
            elif name in trait_assets_by_path:
                actual = sha256(trait_model_bytes(path, trait_assets_by_path[name])[0]).hexdigest()
            elif name in terminal_objects:
                digest_expected, limit = terminal_objects[name]
                actual = sha256(evidence_bytes(path, digest_expected, limit)).hexdigest()
            else:
                actual = file_sha256(path)
        except OSError as error:
            raise ValueError(f'Snapshot verification failed: {name}') from error
        if actual != digest:
            raise ValueError(f'Snapshot verification failed: {name}')
    for _, object_path, digest in review_evidence:
        _review_bytes(source / object_path, digest)
    loaded = ProjectAnalysisStore(source).load(source/index['master'])
    if loaded.document != preliminary.document:
        raise ValueError('Snapshot master changed during dependency verification.')
    if loaded.issues:
        raise ValueError('Snapshot dependencies are incomplete.')
    for item in loaded.document.images:
        if item.source.location != PROJECT_RELATIVE or item.source.path not in files:
            raise ValueError('Portable snapshot source is not in its verified inventory.')
        if any(sidecar.path not in files for sidecar in item.sidecars):
            raise ValueError('Portable snapshot sidecar is not in its verified inventory.')
    profiles = (loaded.document.analysis_settings,) + tuple(instance.default_settings
        for instance in loaded.document.node_instances)
    for node in (node for profile in profiles for node in profile.nodes):
        if ((checkpoint := dict(node.parameters).get('checkpoint_path')) and
                (node.enabled or checkpoint.startswith('objects/')) and checkpoint not in files):
            raise ValueError('Portable snapshot checkpoint is not in its verified inventory.')
    destination = Path(workspace)/'restored-projects'/uuid4().hex
    destination.mkdir(parents=True,exist_ok=False)
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if name in trait_assets_by_path:
            copy_trait_model(source/name, target, trait_assets_by_path[name])
        elif name in terminal_objects:
            digest, limit = terminal_objects[name]
            target.write_bytes(evidence_bytes(source/name, digest, limit))
            evidence_bytes(target, digest, limit)
        else:
            shutil.copyfile(source/name,target)
            if file_sha256(target) != files[name]:
                raise ValueError(f'Restored copy failed verification: {name}')
    for name, object_path, digest, limit, _ in terminal_evidence:
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe restored terminal evidence path.')
        raw = evidence_bytes(destination / object_path, digest, limit)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        evidence_bytes(target, digest, limit)
    verify_terminal_evidence(destination, index)
    for name, object_path, digest in review_evidence:
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe restored review path.')
        raw = _review_bytes(destination / object_path, digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Conflicting restored review evidence.')
        if not target.exists():
            target.write_bytes(raw)
        _review_bytes(target, digest)
    prefix = destination.relative_to(Path(workspace)).as_posix()+'/'
    images = tuple(replace(image,source=replace(image.source,path=prefix+image.source.path),
        identifier=_image_identifier(PROJECT_RELATIVE,prefix+image.source.path),
        sidecars=tuple(replace(sidecar,path=prefix+sidecar.path) for sidecar in image.sidecars)) for image in loaded.document.images)
    def rebased_settings(profile):
        nodes = []
        for node in profile.nodes:
            parameters = dict(node.parameters)
            if 'checkpoint_path' in parameters and parameters['checkpoint_path'] in files:
                parameters['checkpoint_path'] = prefix+parameters['checkpoint_path']
            if node.identifier == 'trait_classifier' and parameters.get('trait_model_path') in trait_assets_by_path:
                parameters['trait_model_path'] = prefix+parameters['trait_model_path']
            nodes.append(replace(node,parameters=tuple(parameters.items())))
        return replace(profile, nodes=tuple(nodes))
    rebased_profile = rebased_settings(loaded.document.analysis_settings)
    rebased_instances = tuple(replace(instance, default_settings=rebased_settings(instance.default_settings))
        for instance in loaded.document.node_instances)
    if loaded.document.species_library:
        if library_store is None:
            raise ValueError('Provide a library store to restore the pinned library.')
        library_store.import_bundle(destination/'pinned-library.zip')
        library_store.load(loaded.document.species_library)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    image_ids = {old.identifier: new.identifier for old, new in zip(loaded.document.images, images, strict=True)}
    if len(set(image_ids.values())) != len(image_ids):
        raise ValueError('Restored image relocation collapsed distinct project identities.')
    document = replace(loaded.document,images=images,selected_image_id=selected,
        analysis_settings=rebased_profile, node_instances=rebased_instances)
    working = ProjectAnalysisStore(workspace).save(document,destination/'working.seedfiddle-project.json')
    (destination/'snapshot-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return working
