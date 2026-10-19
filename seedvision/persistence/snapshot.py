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
from seedvision.persistence.reference_models import (
    capture_reference_model_assets, verify_reference_model_assets,
)
from seedvision.persistence.correction_inputs import (
    verify_correction_bindings, resolve_correction_asset_path, copy_verified_input,
)
from seedvision.persistence.trait_model_assets import (
    capture_trait_model, snapshot_trait_assets, trait_model_bytes, copy_trait_model,
)
from seedvision.persistence.terminal_history import (
    capture_terminal_evidence, evidence_bytes, verify_terminal_evidence,
)
from seedvision.persistence.method_qc_history import (
    capture_method_qc_evidence, history_root, verify_method_qc_evidence,
)


def _checkpoint_roots(document):
    return tuple(sorted({digest for attachment in document.reference_graphs
        for record in attachment.stage_runs
        for digest in (record.checkpoint_sha256, record.upstream_checkpoint_sha256)
        if digest is not None}))


def _model_bytes(path, asset):
    """Keep model transport bounded even if the file changes after preflight."""
    path = Path(path)
    if path.stat().st_size != asset.source_bytes:
        raise ValueError(f'Copy verification failed: {path}')
    with path.open('rb') as stream:
        raw = stream.read(asset.source_bytes+1)
    if len(raw) != asset.source_bytes or sha256(raw).hexdigest() != asset.source_sha256:
        raise ValueError(f'Copy verification failed: {path}')
    return raw


def _copy_model(source, target, asset):
    raw = _model_bytes(source, asset)
    if not target.exists():
        target.write_bytes(raw)
    _model_bytes(target, asset)
    _model_bytes(source, asset)


def _copy_correction(source, target, asset):
    if not target.exists():
        with target.open('wb') as stream:
            copy_verified_input(source, asset, stream)
    copy_verified_input(target, asset)
    copy_verified_input(source, asset)


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


def create_snapshot(root, project_path, output_directory, *, library_store=None,
                    reference_checkpoint_store=None):
    root = Path(root)
    loaded = ProjectAnalysisStore(root).load(project_path)
    if loaded.issues:
        raise ValueError('Resolve missing, changed or invalid project files before creating a snapshot.')
    trait_model = capture_trait_model(root, loaded.document.analysis_settings)
    instance_traits = {instance.instance_id: capture_trait_model(root, instance.default_settings)
                       for instance in loaded.document.node_instances if instance.default_settings is not None}
    instance_traits = {key: value for key, value in instance_traits.items() if value is not None}
    model_assets = capture_reference_model_assets(root, loaded.document.reference_graphs,
                                                 loaded.document.reference_model_assets,
                                                 native_recipes=loaded.document.native_recipes, pipeline_workflow=loaded.document.pipeline_workflow, node_instances=loaded.document.node_instances)
    verify_reference_model_assets(root, model_assets, loaded.document.reference_graphs, require_complete=True,
                                  native_recipes=loaded.document.native_recipes, pipeline_workflow=loaded.document.pipeline_workflow, node_instances=loaded.document.node_instances)
    verify_correction_bindings(root, loaded.document.correction_bindings, loaded.document.images,
                               loaded.document.reference_graphs, loaded.document.native_recipes, pipeline_workflow=loaded.document.pipeline_workflow, node_instances=loaded.document.node_instances)
    terminal_evidence = capture_terminal_evidence(root, project_path)
    method_qc_evidence = capture_method_qc_evidence(root, project_path)
    destination = Path(output_directory)/('seedfiddle-snapshot-'+uuid4().hex)
    destination.mkdir(parents=True,exist_ok=False)
    inventory = {}
    supplementary = {}
    for name, raw in (*terminal_evidence, *method_qc_evidence):
        digest = sha256(raw).hexdigest()
        namespace = name.split('/')[0]
        suffix = {'terminal-history': '.terminal.json', 'terminal-acknowledgments': '.terminal-ack.json',
            'method-qc': '.method-qc.json', 'method-qc-acknowledgments': '.method-qc-ack.json'}[namespace]
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
    def copy(source, expected=None, model_asset=None, correction_asset=None, image_id=None):
        source = Path(source)
        if correction_asset is not None:
            relative = 'objects/'+correction_asset.source_sha256+'.image'
            target = destination/relative
            target.parent.mkdir(exist_ok=True)
            _copy_correction(source, target, correction_asset)
            inventory[relative] = correction_asset.source_sha256
            return relative
        if model_asset is not None:
            relative = 'objects/'+model_asset.source_sha256+source.suffix
            target = destination/relative
            target.parent.mkdir(exist_ok=True)
            _copy_model(source, target, model_asset)
            inventory[relative] = model_asset.source_sha256
            return relative
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
        if instance.default_settings is not None else instance for instance in loaded.document.node_instances)
    if loaded.document.species_library is not None:
        if library_store is None:
            raise ValueError('The pinned library store is required to make a complete portable snapshot.')
        bundle = library_store.export_library(loaded.document.species_library,destination/'pinned-library.zip')
        inventory[bundle.relative_to(destination).as_posix()] = file_sha256(bundle)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    model_assets = tuple(replace(asset, path=copy(root/asset.path, model_asset=asset)) for asset in model_assets)
    image_ids = {old.identifier: new.identifier for old, new in zip(loaded.document.images, images, strict=True)}
    if len(set(image_ids.values())) != len(image_ids):
        raise ValueError('Snapshot image relocation collapsed distinct project identities.')
    correction_bindings = tuple(replace(binding, image_id=image_ids[binding.image_id],
        inputs=tuple(replace(asset, path=copy(resolve_correction_asset_path(root, asset.path),
                                             correction_asset=asset)) for asset in binding.inputs))
        for binding in loaded.document.correction_bindings)
    document = replace(loaded.document,images=tuple(images),selected_image_id=selected,
        reference_model_assets=model_assets,
        correction_bindings=correction_bindings,
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
    reference_bundle = None
    if reference_checkpoint_store is not None:
        from seedvision.benchmarking.checkpoint_bundle import export_checkpoint_bundle
        roots = _checkpoint_roots(loaded.document)
        if roots:
            bundle = export_checkpoint_bundle(reference_checkpoint_store, roots,
                destination/'reference-checkpoints')
            for name, digest in bundle.files:
                inventory['reference-checkpoints/'+name] = digest
            reference_bundle = 'reference-checkpoints'
    # The completion marker is published last. A failed partial folder cannot be restored.
    index = {'version':1,'files':inventory,
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
    if reference_bundle is not None:
        index['reference_checkpoint_bundle'] = reference_bundle
    verify_terminal_evidence(destination, index)
    verify_method_qc_evidence(destination, index)
    (destination/'snapshot-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return destination


def restore_snapshot(snapshot, workspace, *, library_store=None):
    source = Path(snapshot)
    index = json.loads((source/'snapshot-index.json').read_text(encoding='utf-8'))
    if index.get('version') != 1:
        raise ValueError('Unsupported snapshot schema.')
    files = index['files']
    review_evidence = _review_evidence(index)
    terminal_evidence = verify_terminal_evidence(source, index)
    method_qc_evidence = verify_method_qc_evidence(source, index)
    terminal_objects = {object_path: (digest, limit)
        for _, object_path, digest, limit, _ in (*terminal_evidence, *method_qc_evidence)}
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
    defaults = {instance.instance_id: instance for instance in preliminary.document.node_instances
                if instance.default_settings is not None}
    if type(instance_trait_rows) is not dict or set(instance_trait_rows) - set(defaults):
        raise ValueError('Portable node instance trait-model selection is invalid.')
    for identifier, instance in defaults.items():
        trait_assets_by_path.update(snapshot_trait_assets(instance.default_settings,
            {'files': files, 'trait_model_assets': ([instance_trait_rows[identifier]] if identifier in instance_trait_rows else [])}))
    model_assets_by_path = {asset.path: asset for asset in preliminary.document.reference_model_assets}
    correction_assets_by_path = {asset.path: asset for binding in preliminary.document.correction_bindings
                                for asset in binding.inputs}
    for name,digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe snapshot path.')
        path = (source/name).resolve()
        try:
            if not path.is_relative_to(source.resolve()):
                actual = None
            elif name in model_assets_by_path:
                actual = sha256(_model_bytes(path, model_assets_by_path[name])).hexdigest()
            elif name in trait_assets_by_path:
                actual = sha256(trait_model_bytes(path, trait_assets_by_path[name])[0]).hexdigest()
            elif name in correction_assets_by_path:
                actual = copy_verified_input(path, correction_assets_by_path[name])
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
    verify_reference_model_assets(source, loaded.document.reference_model_assets,
                                  loaded.document.reference_graphs, require_complete=True,
                                  native_recipes=loaded.document.native_recipes, pipeline_workflow=loaded.document.pipeline_workflow, node_instances=loaded.document.node_instances)
    if any(asset.path not in files or files[asset.path] != asset.source_sha256
           for asset in loaded.document.reference_model_assets):
        raise ValueError('Portable snapshot model asset is not in its verified inventory.')
    verify_correction_bindings(source, loaded.document.correction_bindings, loaded.document.images,
                               loaded.document.reference_graphs, loaded.document.native_recipes, pipeline_workflow=loaded.document.pipeline_workflow, node_instances=loaded.document.node_instances)
    if any(asset.path not in files or files[asset.path] != asset.source_sha256
           for binding in loaded.document.correction_bindings for asset in binding.inputs):
        raise ValueError('Portable snapshot correction asset is not in its verified inventory.')
    profiles = (loaded.document.analysis_settings,) + tuple(instance.default_settings
        for instance in loaded.document.node_instances if instance.default_settings is not None)
    for node in (node for profile in profiles for node in profile.nodes):
        if ((checkpoint := dict(node.parameters).get('checkpoint_path')) and
                (node.enabled or checkpoint.startswith('objects/')) and checkpoint not in files):
            raise ValueError('Portable snapshot checkpoint is not in its verified inventory.')
    bundle_path = index.get('reference_checkpoint_bundle')
    if bundle_path is not None:
        from seedvision.benchmarking.checkpoint_bundle import verify_checkpoint_bundle
        if bundle_path != 'reference-checkpoints':
            raise ValueError('Unsupported portable reference checkpoint location.')
        bundle = verify_checkpoint_bundle(source/bundle_path)
        if bundle.roots != _checkpoint_roots(loaded.document):
            raise ValueError('Portable reference checkpoint selection differs from project history.')
        expected_files = {bundle_path+'/'+name: digest for name, digest in bundle.files}
        actual_files = {name: digest for name, digest in files.items() if name.startswith(bundle_path+'/')}
        if expected_files != actual_files:
            raise ValueError('Portable reference checkpoint bundle is not in its verified inventory.')
    destination = Path(workspace)/'restored-projects'/uuid4().hex
    destination.mkdir(parents=True,exist_ok=False)
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if name in model_assets_by_path:
            _copy_model(source/name, target, model_assets_by_path[name])
        elif name in trait_assets_by_path:
            copy_trait_model(source/name, target, trait_assets_by_path[name])
        elif name in correction_assets_by_path:
            _copy_correction(source/name, target, correction_assets_by_path[name])
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
    qc_private = history_root(workspace, destination / 'working.seedfiddle-project.json')
    for name, object_path, digest, limit, _ in method_qc_evidence:
        target = qc_private / name
        if not target.resolve().is_relative_to(qc_private.resolve()):
            raise ValueError('Unsafe restored recipe QC evidence path.')
        raw = evidence_bytes(destination / object_path, digest, limit)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        evidence_bytes(target, digest, limit)
    verify_method_qc_evidence(destination, index)
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
    if bundle_path is not None:
        verify_checkpoint_bundle(destination/bundle_path)
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
        if instance.default_settings is not None else instance for instance in loaded.document.node_instances)
    if loaded.document.species_library:
        if library_store is None:
            raise ValueError('Provide a library store to restore the pinned library.')
        library_store.import_bundle(destination/'pinned-library.zip')
        library_store.load(loaded.document.species_library)
    selected = next((new.identifier for old,new in zip(loaded.document.images,images) if old.identifier==loaded.document.selected_image_id),None)
    model_assets = tuple(replace(asset, path=prefix+asset.path) for asset in loaded.document.reference_model_assets)
    image_ids = {old.identifier: new.identifier for old, new in zip(loaded.document.images, images, strict=True)}
    if len(set(image_ids.values())) != len(image_ids):
        raise ValueError('Restored image relocation collapsed distinct project identities.')
    correction_bindings = tuple(replace(binding, image_id=image_ids[binding.image_id],
        inputs=tuple(replace(asset, path=prefix+asset.path) for asset in binding.inputs))
        for binding in loaded.document.correction_bindings)
    document = replace(loaded.document,images=images,selected_image_id=selected,
        reference_model_assets=model_assets,correction_bindings=correction_bindings,
        analysis_settings=rebased_profile, node_instances=rebased_instances)
    working = ProjectAnalysisStore(workspace).save(document,destination/'working.seedfiddle-project.json')
    (destination/'snapshot-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return working
