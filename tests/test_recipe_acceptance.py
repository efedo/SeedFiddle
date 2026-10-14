"""Finite recipe lifecycle acceptance, separate from public/source conformance.

Expected evidence lives only in ignored artifacts and is never refreshed by a
test. Cache keys contain allocation identities, so their exact preservation is
checked within a run; they are not treated as portable numerical fingerprints.
"""
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_median_roi_recipe as median
from seedvision.benchmarking import native_reader_recipe as reader
from seedvision.benchmarking import native_correction_recipe as correction
from seedvision.benchmarking import native_correction_inputs as correction_files
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CheckpointLimits
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.acceptance_diff import assert_json_equal, json_difference_summary


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'artifacts'/'recipe-acceptance-baseline-v2.json'
# Filled only after a completed, reviewed candidate capture. Never auto-update.
BASELINE_SHA256 = '82861ec3480b7ac9bcaa65be1d27bbc9dc15d928d3e813210e0fad29c5bf6eb1'
HISTORICAL_BASELINE = ROOT/'artifacts'/'recipe-acceptance-baseline-v1.json'
HISTORICAL_BASELINE_SHA256 = 'b8fefc1b5d3bf0d595bfa243303bcc194f181d49f36cf05502d69654ace4c8f2'
SCHEMA = 'seedfiddle.recipe-acceptance.v1'
CODE_TRANSITION = ROOT/'artifacts'/'complete-software-recipe-provenance-transition-v2.json'
CODE_TRANSITION_SHA256 = 'd5b42dda845b4aed842ef6807403af874e5dfc55c81b936221ff107359ba087d'
EXPORTER_SOURCE_PIN = ('seedvision/export/native_results.py',
    'db660bbab977b508e0dd7cd9f37ec7c04f7a0aced0a7e92a067f1f4a4602c3d4',
    '5dd37e122604328f55295a13fa7aebdb47fcfd66d9480f0e0c410eee37d2aa28')


def _reviewed_capture_pins(pins):
    """Migrate one exact exporter source identity, preserving all other pins."""
    name, previous, current = EXPORTER_SOURCE_PIN
    if pins['source_files'][name] != previous:
        raise AssertionError('Historical exporter identity does not match the reviewed pin.')
    result = dict(pins)
    result['source_files'] = dict(pins['source_files'])
    result['source_files'][name] = current
    return result


def _reviewed_code_provenance(expected):
    """Apply prespecified code identities; every numerical baseline field stays exact."""
    raw = CODE_TRANSITION.read_bytes()
    if sha256(raw).hexdigest() != CODE_TRANSITION_SHA256:
        raise AssertionError('Reviewed code-provenance transition bytes changed.')
    record = json.loads(raw)
    if (record['schema'] != 'seedfiddle.recipe-code-provenance-transition-v2'
            or record['baseline_sha256'] != BASELINE_SHA256
            or record['status'] != 'CODE_IDENTITIES_APPROVED_NUMERICAL_ACCEPTANCE_PENDING'):
        raise AssertionError('Code transition does not bind the immutable recipe baseline.')
    changes = record['transitions']
    if len(changes) != 495 or len({change['path'] for change in changes}) != 495:
        raise AssertionError('Expected exactly 495 unique reviewed code-identity changes.')
    allowed = re.compile(r'^/cases/(?:0|[1-9][0-9]*)/(?:runtime_calculation_pins/(?:executor|correction_tensor_executor)|preparation/(?:rgb|gray))$')
    tokens = re.compile(r'((?:input helper|preparation|CUDA converter) )[0-9a-f]{64}')
    for change in changes:
        path = change['path']
        if path == '/capture_pins/source_files/seedvision~1export~1native_results.py':
            if (change['old'], change['new']) != EXPORTER_SOURCE_PIN[1:]:
                raise AssertionError('Unreviewed exporter source identity.')
            expected['capture_pins'] = _reviewed_capture_pins(expected['capture_pins'])
            continue
        if not allowed.fullmatch(path):
            raise AssertionError('Unreviewed transition path: ' + path)
        _, _, index, section, key = path.split('/')
        target = expected['cases'][int(index)][section]
        if target[key] != change['old']:
            raise AssertionError('Historical code identity does not match: ' + path)
        if section == 'preparation':
            if tokens.sub(r'\1<CODE>', change['old']) != tokens.sub(r'\1<CODE>', change['new']):
                raise AssertionError('Preparation facts changed beyond code tokens: ' + path)
        elif not all(re.fullmatch('[0-9a-f]{64}', change[value]) for value in ('old', 'new')):
            raise AssertionError('Invalid reviewed code identity: ' + path)
        target[key] = change['new']
    return expected


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _change(recipe, operation, **values):
    return replace(recipe, stages=tuple(replace(stage, parameters=tuple(
        replace(parameter, value=values.get(parameter.parameter_id, parameter.value))
        for parameter in stage.parameters)) if stage.operation_id == operation else stage
        for stage in recipe.stages))


def canonical_profiles(*, validate=True):
    """Cover every current admission branch, with historical versions retained."""
    catalogue = build_recipe_catalogue()
    bases = [catalogue.recipe(name) for name in runner._SEED_RECIPES]
    bases += [runner.build_native_seed_recipe(catalogue.recipe(name))
              for name in runner._WORKING_SEED_RECIPES.values()]
    bases += [runner._calibrated_seed.build_native_calibrated_seed_recipe(name.rsplit('_', 1)[1])
              for name in runner._calibrated_seed.RECIPE_IDS]
    known = {recipe.recipe_id: recipe for recipe in bases}
    for factory, builder in (
            (runner._diagnostic_recipe, runner._diagnostic_recipe.build_native_watershed_diagnostic_recipe),
            (runner._morphology_roi_recipe, runner._morphology_roi_recipe.build_native_morphology_roi_recipe),
            (runner._fill_roi_recipe, runner._fill_roi_recipe.build_native_fill_roi_recipe)):
        additions = [builder(known[factory.base_recipe_id(name)]) for name in factory.RECIPE_IDS]
        bases += additions
        known.update((recipe.recipe_id, recipe) for recipe in additions)
    bases += [runner._naive_bayes_recipe.build_native_naive_bayes_recipe(),
              runner._kmeans_recipe.build_native_kmeans_recipe(),
              runner._card_recipe.build_native_card_recipe()]
    bases += [runner._recovery_recipe.build_native_core_recovery_recipe(),
              runner._paper_recipe.build_native_smpts_recipe(),
              runner._range_recipe.build_native_range_recipe()]
    bases += list(runner._extended_recipe_specs.build_native_extended_threshold_recipes().values())
    profiles = [('canonical', recipe) for recipe in bases]
    by_id = {recipe.recipe_id: recipe for recipe in bases}
    if len(by_id) != len(bases):
        raise AssertionError('Duplicate canonical admission entry.')
    for name in median.BASE_IDS:
        profiles.append(('median-roi', median._upgrade(by_id[name])))
    for name in reader.BASE_IDS:
        for add_median in (False, True):
            base = by_id[name]
            if add_median:
                base = median._upgrade(base)
            upgraded = reader._upgrade(base)
            for mode in reader.MODES:
                profiles.append(('reader-median' if add_median else 'reader',
                    _change(upgraded, reader.OPERATION, gray_source_mode=mode)))
    if validate:
        for _, recipe in profiles:
            runner._validate_recipe(recipe)
    historical = catalogue.recipe(correction.RECIPE_ID)
    profiles += [('four-input-historical', historical),
                 ('four-input-native', correction.build_native_correction_recipe(historical))]
    for _, recipe in profiles[-2:]:
        correction._validate_recipe(recipe)
    keys = [(r.recipe_id, r.recipe_version, reader.selected_mode(r),
             r.provenance.mode.value) for _, r in profiles]
    if len(keys) != len(set(keys)):
        raise AssertionError('Duplicate recipe version/mode/provenance profile.')
    return tuple(profiles)


def _fixtures(root):
    # Reuse committed fixture definitions, not earlier generated output files.
    from tests import test_native_recipe as seeds
    from tests import test_native_calibrated_seed_recipe as calibrated
    from tests import test_native_extended_threshold_recipe as extended
    from tests import test_native_kmeans_recipe as kmeans
    from tests import test_native_naive_bayes_recipe as bayes
    from tests import test_native_core_recovery_recipe as recovery
    from tests import test_native_smpts_recipe as paper
    from tests import test_plantcv_card_detection as card
    from tests import test_native_correction_dialog as four
    gray, _ = seeds.fixture()
    mask, _ = paper.contact_fixture()
    images = {
        # Colored bytes keep hue observations defined while retaining the
        # synthetic intensity shapes (the isolated saturated speck is filtered).
        'seeds': np.clip(gray[..., None].astype(np.int16)+[-5, -3, 7], 0, 255).astype(np.uint8),
        'calibrated': calibrated.fixture(), 'card': card.fixture(varied=True),
        'extended': extended._fixture()[1], 'kmeans': kmeans._image(),
        'bayes': bayes._image(), 'recovery': recovery.fixture()[1],
        'paper': np.repeat(np.where(mask, 220, 10).astype(np.uint8)[..., None], 3, axis=2),
    }
    paths = {}
    for name, image in images.items():
        path = root/(name+'.png')
        if not cv2.imwrite(str(path), image):
            raise AssertionError('Could not encode acceptance fixture '+name)
        paths[name] = path
    bayes_path, kmeans_path = root/'bayes.tsv', root/'kmeans.json'
    bayes_path.write_bytes(bayes._pdf())
    kmeans._write_model(kmeans_path)
    models = {'bayes': bayes_path, 'kmeans': kmeans_path}
    four_paths = four._write_inputs(root)
    sources = {**{'image:'+name: path for name, path in paths.items()},
               **{'model:'+name: path for name, path in models.items()},
               **{'correction:'+name: path for name, path in four_paths.items()}}
    inventory = [{'role': name, 'bytes': len(path.read_bytes()),
                  'sha256': sha256(path.read_bytes()).hexdigest()}
                 for name, path in sorted(sources.items())]
    return paths, models, four_paths, inventory


def _select(recipe, paths, models):
    from tests import test_native_recipe as seeds
    from tests import test_native_calibrated_seed_recipe as calibrated
    from tests import test_native_calibrated_range_recipe as calibrated_range
    from tests import test_native_core_recovery_recipe as recovery
    from tests import test_native_smpts_recipe as paper
    kind = next((kind for kind in ('binary', 'mean', 'otsu', 'range')
                 if 'seed_'+kind in recipe.recipe_id), None)
    is_calibrated = 'plantcv_calibrated_seed_' in recipe.recipe_id
    assets = {}
    if kind:
        selected = (calibrated_range.selected_recipe() if kind == 'range' else
                    calibrated.selected_recipe(kind) if is_calibrated else seeds.selected_recipe(kind))
        values = {stage.operation_id: {p.parameter_id: p.value for p in stage.parameters}
                  for stage in selected.stages if stage.operation_id != 'external.image_inputs'}
        for operation, controls in values.items():
            recipe = _change(recipe, operation, **controls)
        if kind == 'mean':
            recipe = _change(recipe, 'threshold.mean', ksize=21 if is_calibrated else 9,
                             offset=-1 if is_calibrated else -2)
        family = 'calibrated' if is_calibrated else 'seeds'
        for operation in (median.OPERATION, 'morphology.fill', 'morphology.erode', 'morphology.dilate'):
            stage = next((s for s in recipe.stages if s.operation_id == operation), None)
            if stage and any(p.parameter_id == median.PARAMETER for p in stage.parameters):
                recipe = _change(recipe, operation, roi_bounds_xyxy=[4, 4, 124, 92])
    elif recipe.recipe_id in runner._NAIVE_BAYES_RECIPES + runner._KMEANS_RECIPES:
        family = 'bayes' if recipe.recipe_id in runner._NAIVE_BAYES_RECIPES else 'kmeans'
        factory = runner._naive_bayes_recipe if family == 'bayes' else runner._kmeans_recipe
        source = factory.model_source_for_path(models[family])
        # Canonical absolute synthetic origins plus verified local transports
        # preserve the application's portable model-provenance contract.
        source['path'] = '/seedfiddle-acceptance/models/'+models[family].name
        recipe = _change(recipe, recipe.stage(factory.MODEL_STAGE_ID).operation_id,
                         model_source=source)
        assets[factory.MODEL_STAGE_ID] = str(models[family])
    elif recipe.recipe_id == runner._card_recipe.RECIPE_ID:
        family = 'card'
    elif recipe.recipe_id == runner._recovery_recipe.RECIPE_ID:
        family = 'recovery'
        for stage in recovery.selected_recipe().stages:
            recipe = _change(recipe, stage.operation_id,
                             **{p.parameter_id: p.value for p in stage.parameters})
    elif recipe.recipe_id == runner._paper_recipe.RECIPE_ID:
        family = 'paper'
        for stage in paper.selected_recipe().stages:
            recipe = _change(recipe, stage.operation_id,
                             **{p.parameter_id: p.value for p in stage.parameters})
    else:
        family = 'extended'
    return recipe, paths[family], assets, family


def _array(value):
    array = value.detach().contiguous().cpu().numpy() if torch.is_tensor(value) else value
    array = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'sha256': sha256(array.tobytes(order='C')).hexdigest()}


def _tree(value, arrays):
    if isinstance(value, dict):
        if value.get('kind') == 'array':
            return {'kind': 'array', 'name': value['name'], **_array(arrays[value['name']])}
        return {key: _tree(item, arrays) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree(item, arrays) for item in value]
    return value


def _result_nodes(recipe, result):
    if result.status != 'executed' or tuple(n.stage_id for n in result.nodes) != tuple(s.stage_id for s in recipe.stages):
        raise AssertionError([(n.stage_id, n.status, n.reason) for n in result.nodes])
    for stage, node in zip(recipe.stages, result.nodes):
        declared = {p.name for p in stage.ports if p.direction == 'output'}
        if node.status not in ('provided', 'executed', 'empty') or set(node.outputs) != declared:
            raise AssertionError((node.stage_id, node.status, declared, set(node.outputs)))
    return _encoded_nodes(recipe, result)


def _encoded_nodes(recipe, result):
    """Project available products; this helper alone makes no completeness claim."""
    collector = archives._Collector(result.coordinate_frame, CheckpointLimits(), None)
    nodes = []
    for node in result.nodes:
        outputs = []
        for name, value in node.outputs.items():
            if name == 'model':
                encoded = archives._model_asset(recipe, node, collector, result.model_asset_paths or {})
            elif recipe.recipe_id == correction.RECIPE_ID and name != 'corrected_image':
                encoded = collector.array(value, cpu_matrix=True)
            else:
                encoded = collector.encode(value)
            outputs.append([name, encoded])
        nodes.append({'stage_id': node.stage_id, 'operation_id': node.operation_id,
                      'status': node.status, 'reason': node.reason,
                      'parameters': collector.encode(node.parameters),
                      'outputs': outputs, 'calculation': collector.encode(node.calculation)})
    return _tree(nodes, collector.arrays)


def _archived_nodes(report, arrays):
    return _tree([{key: node[key] for key in ('stage_id', 'operation_id', 'status', 'reason',
                  'parameters', 'outputs', 'calculation')} for node in report['nodes']], arrays)


def _logical_paths(value, paths):
    """Normalize only known temporary origins; retain all other decode facts."""
    if isinstance(value, str):
        for role, path in paths.items():
            value = value.replace(str(path.resolve()), '<temporary-origin:'+role+'>')
        return value
    if hasattr(value, 'items'):
        return {key: _logical_paths(item, paths) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_logical_paths(item, paths) for item in value]
    return value


def _capture_pins():
    helpers = (_json, _change, canonical_profiles, _fixtures, _select, _array,
               _tree, _result_nodes, _encoded_nodes, _archived_nodes, _logical_paths, _mean_boundary,
               _capture_pins, capture_recipe_acceptance)
    files = ('seedvision/export/native_results.py', 'tests/test_native_recipe.py',
             'tests/test_native_calibrated_seed_recipe.py', 'tests/test_native_calibrated_range_recipe.py',
             'tests/test_native_extended_threshold_recipe.py', 'tests/test_native_kmeans_recipe.py',
             'tests/test_native_naive_bayes_recipe.py', 'tests/test_native_core_recovery_recipe.py',
             'tests/test_native_smpts_recipe.py', 'tests/test_plantcv_card_detection.py',
             'tests/test_smpts_paper.py', 'tests/test_plantcv_naive_bayes.py',
             'tests/test_native_correction_dialog.py')
    return {'helpers': {fn.__name__: sha256(inspect.getsource(fn).encode()).hexdigest() for fn in helpers},
            'source_files': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}


def _mean_boundary(root, paths, models):
    """Keep the known excluded ellipse case; never include it among complete runs."""
    canonical = build_recipe_catalogue().recipe('plantcv_seed_mean')
    recipe, path, assets, _ = _select(canonical, paths, models)
    recipe = _change(recipe, 'threshold.mean', ksize=9, offset=2)
    result = runner.run_native_recipe_path(recipe, path, model_asset_paths=assets)
    size_id = 'pcv_seed_mean_analyze_size'
    size = result.node(size_id)
    if (result.status != 'partial' or size.status != 'unavailable' or size.outputs
            or 'orientation numerically unstable' not in size.reason
            or any(n.status not in ('provided', 'executed', 'empty') for n in result.nodes if n.stage_id != size_id)):
        raise AssertionError('The declared ellipse boundary outcome changed.')
    try:
        _result_nodes(recipe, result)
    except AssertionError:
        pass
    else:
        raise AssertionError('A partial boundary satisfied complete acceptance.')
    destination = root/'mean-boundary-archive'
    archives.export_native_recipe_result(destination, recipe, result)
    report, arrays = archives.load_native_recipe_result(destination)
    projected = _encoded_nodes(recipe, result)
    if report['status'] != 'partial' or _archived_nodes(report, arrays) != projected:
        raise AssertionError('Partial boundary archive changed its availability or outputs.')
    return {'recipe': recipe.to_dict(), 'source_sha256': result.source_sha256,
            'expected_status': 'partial', 'excluded_from_complete_profile_count': True,
            'nodes': projected, 'verified_partial_archive_equal': True}


def capture_recipe_acceptance(progress=None, *, include_recipe_ids=None):
    """Capture all finite profiles; missing/partial outcomes never satisfy a case."""
    if not torch.cuda.is_available():
        raise AssertionError('CUDA is a required acceptance prerequisite.')
    cases = []
    replayed = set()
    with tempfile.TemporaryDirectory(prefix='seedfiddle-recipe-acceptance-') as temporary:
        root = Path(temporary)
        paths, models, four_paths, inventory = _fixtures(root)
        boundary = _mean_boundary(root, paths, models)
        # Execution validates each final profile through its actual driver.
        # The separate inventory gate validates all profiles before any GPU run.
        profiles = canonical_profiles(validate=False)
        if include_recipe_ids is not None:
            wanted = frozenset(include_recipe_ids)
            if not wanted or wanted - {r.recipe_id for _, r in profiles}:
                raise AssertionError('Smoke selection contains a missing recipe identity.')
            profiles = tuple((label, recipe) for label, recipe in profiles if recipe.recipe_id in wanted)
        for index, (profile, canonical) in enumerate(profiles):
            if progress:
                progress(index, len(profiles), canonical.recipe_id, {'phase': 'starting', 'profile': profile})
            cache = PipelineAnalysisCache()
            if canonical.recipe_id == correction.RECIPE_ID:
                recipe, family = canonical, 'four-input'
                cold, prepared = correction_files.run_native_correction_files(recipe, four_paths, cache=cache)
                warm, _ = correction_files.run_native_correction_files(recipe, four_paths, previous=prepared, cache=cache)
                origins = list(four_paths.values())
            else:
                recipe, path, assets, family = _select(canonical, paths, models)
                cold = runner.run_native_recipe_path(recipe, path, cache=cache, model_asset_paths=assets)
                warm = runner.run_native_recipe_path(recipe, path, cache=cache, model_asset_paths=assets)
                origins = [path, *(Path(p) for p in assets.values())]
            cold_nodes, warm_nodes = _result_nodes(recipe, cold), _result_nodes(recipe, warm)
            if cold_nodes != warm_nodes or not all(n.reused for n in warm.nodes):
                raise AssertionError('Warm outputs/cache differ: '+recipe.recipe_id)
            if any(a.product_key != b.product_key or dict(a.input_keys) != dict(b.input_keys)
                   for a, b in zip(cold.nodes, warm.nodes)):
                raise AssertionError('Warm producer/input keys changed.')
            destination = root/('archive-'+str(index))
            archives.export_native_recipe_result(destination, recipe, cold)
            report, arrays = archives.load_native_recipe_result(destination)
            if report['version'] != 5 or _archived_nodes(report, arrays) != cold_nodes:
                raise AssertionError('Verified archive outputs differ.')
            if any(node['product_key'] != actual.product_key or node['input_keys'] != dict(actual.input_keys)
                   for node, actual in zip(report['nodes'], cold.nodes)):
                raise AssertionError('Archive producer/input keys differ.')
            # Hide each actual origin; replay must consume archived direct products.
            moved = []
            checks = []
            try:
                for path in origins:
                    hidden = path.with_suffix(path.suffix+'.hidden')
                    path.rename(hidden)
                    moved.append((path, hidden))
                for stage in recipe.stages:
                    contract = (stage.operation_id, _json([p.to_dict() for p in stage.ports]))
                    if stage.operation_id not in replay.SUPPORTED_REPLAY_OPERATIONS or contract in replayed:
                        continue
                    result = replay.replay_native_stage(destination, stage.stage_id,
                        expected_revision=report['result_revision'], expected_recipe_identity=recipe.identity)
                    if result.status != 'matched' or result.metadata['comparison']['mismatched_elements'] != 0:
                        raise AssertionError((recipe.recipe_id, stage.stage_id, result.status, result.reason))
                    checks.append(stage.operation_id)
                    replayed.add(contract)
            finally:
                for path, hidden in moved:
                    hidden.rename(path)
            row = {'profile': profile, 'family': family, 'recipe': recipe.to_dict(),
                   'source_sha256': cold.source_sha256, 'frame': cold.coordinate_frame.to_dict(),
                   'preparation': _logical_paths(cold.preparation, four_paths),
                   'runtime_calculation_pins': dict(cold.runtime),
                   'correction_sources': _logical_paths(cold.input_sources, four_paths)
                       if canonical.recipe_id == correction.RECIPE_ID else None,
                   'nodes': cold_nodes, 'cold_warm_outputs_equal': True,
                   'all_warm_nodes_reused': True, 'warm_keys_equal': True,
                   'verified_v5_archive_equal': True, 'archive_keys_equal': True,
                   'source_absent_replayed_operations': checks,
                   'replay_scope': 'first representative per operation and serialized port contract',
                   'replay_unavailable_operations': sorted({s.operation_id for s in recipe.stages
                       if s.operation_id not in replay.SUPPORTED_REPLAY_OPERATIONS})}
            cases.append(row)
            if progress:
                progress(index+1, len(profiles), recipe.recipe_id, row)
            del cold, warm, arrays, cache
    return {'schema': SCHEMA, 'scope': 'finite selected controls and fixtures; native lifecycle agreement',
            'profile_selection': None if include_recipe_ids is None else sorted(include_recipe_ids),
            'reference_conformance': 'not_established_by_this_capture', 'scientifically_validated': False,
            'fixture_inventory': inventory, 'cases': cases,
            'expected_partial_boundaries': [boundary],
            'source_absent_replay_contract_count': len(replayed),
            'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                            'device': torch.cuda.get_device_name(), 'opencv': cv2.__version__,
                            'numpy': np.__version__},
            'capture_pins': _capture_pins()}


class RecipeAcceptanceTests(unittest.TestCase):
    def test_reviewed_revision_preserves_historical_evidence_and_exact_dependency_changes(self):
        old_raw, new_raw = HISTORICAL_BASELINE.read_bytes(), BASELINE.read_bytes()
        self.assertEqual(sha256(old_raw).hexdigest(), HISTORICAL_BASELINE_SHA256)
        self.assertEqual(sha256(new_raw).hexdigest(), BASELINE_SHA256)
        old, new = json.loads(old_raw), json.loads(new_raw)
        self.assertEqual(_reviewed_capture_pins(old['capture_pins']), _capture_pins())
        self.assertEqual(new['capture_pins'], old['capture_pins'])
        pairs = {
            '550f69bf132094fb2d2a033c7cc482e2590da783b31f55580ceb3b30ed4c1f84':
                'f8f29a7da0b73f5b350c080ec9bdb439abf273f9c06f9088c50e2a8752942aa4',
            'bf3660048d5cedfce257c6d8f27cbe431adc4a4e5990506e99666403f37fe384':
                '551ce9dcedf817316d3f1e033925e05112d9eb1f8bcf2a126e183414c9049a3c',
            '1480127602d358b53b89d67edbceeff3d80a2b5827f2dc4ed26ce157dc9fbedc':
                '2823cea16bcc223ba9155dbc65db58ffea7cadd51d5a5f9be05703d9a3fa3bc8',
            '15f062bfa32eb9d9d1b156b952e3016d0503729bd89f6ef463aa043dd9edf898':
                '2695e5b8949702979ca43dfc7c5fd26af77cc1396d9231d5b223628ccf619a7b',
        }
        expected = {}
        for index, row in enumerate(old['cases']):
            operations = {stage['operation_id'] for stage in row['recipe']['stages']}
            pin = ('correction_tensor_executor' if row['family'] == 'four-input' else
                   'calculation_modules_sha256' if 'color_card.affine' in operations else None)
            if pin is not None:
                previous = row['runtime_calculation_pins'][pin]
                expected[f'/cases/{index}/runtime_calculation_pins/{pin}'] = (previous, pairs[previous])
        self.assertEqual(len(expected), 67)
        self.assertEqual(sum(path.endswith('/correction_tensor_executor') for path in expected), 2)

        def differences(left, right, path=''):
            self.assertIs(type(left), type(right), path)
            if isinstance(left, dict):
                self.assertEqual(set(left), set(right), path)
                return {key: value for name in left
                        for key, value in differences(left[name], right[name], path+'/'+name).items()}
            if isinstance(left, list):
                self.assertEqual(len(left), len(right), path)
                return {key: value for index, (a, b) in enumerate(zip(left, right))
                        for key, value in differences(a, b, path+'/'+str(index)).items()}
            return {} if left == right else {path: (left, right)}

        self.assertEqual(differences(old, new), expected)

    def test_complete_admitted_version_profile_inventory(self):
        profiles = canonical_profiles()
        self.assertEqual(len(profiles), 223)
        self.assertEqual(sum(label == 'canonical' for label, _ in profiles), 69)
        self.assertEqual(sum(label == 'median-roi' for label, _ in profiles), 56)
        self.assertEqual(sum(label in ('reader', 'reader-median') for label, _ in profiles), 96)
        self.assertEqual(sum(label.startswith('four-input') for label, _ in profiles), 2)

    def test_mean_orientation_boundary_stays_partial_and_archives_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, models, _, _ = _fixtures(root)
            boundary = _mean_boundary(root, paths, models)
            self.assertTrue(boundary['excluded_from_complete_profile_count'])
            self.assertEqual(boundary['expected_status'], 'partial')

    def test_selected_complete_mean_inputs_match_pinned_public_chains(self):
        from tests import test_native_recipe as seeds
        from tests import test_native_calibrated_seed_recipe as calibrated
        self.assertTrue(seeds.REFERENCE.is_file(), 'Pinned public PlantCV interpreter is required.')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, models, _, _ = _fixtures(root)
            for is_calibrated in (False, True):
                with self.subTest(calibrated=is_calibrated):
                    canonical = (runner._calibrated_seed.build_native_calibrated_seed_recipe('mean')
                                 if is_calibrated else build_recipe_catalogue().recipe('plantcv_seed_mean'))
                    recipe, path, assets, _ = _select(canonical, paths, models)
                    actual = runner.run_native_recipe_path(recipe, path, model_asset_paths=assets)
                    _result_nodes(recipe, actual)
                    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    source = root/'mean-source.npz'
                    if is_calibrated:
                        np.savez(source, bgr=bgr)
                        script = calibrated.reference_script()
                        replacement = 'ksize=21, offset=-1,'
                    else:
                        np.savez(source, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                 rgb=np.ascontiguousarray(bgr[..., ::-1]))
                        script = seeds.REFERENCE_SCRIPT
                        replacement = 'ksize=9, offset=-2,'
                    self.assertEqual(script.count('ksize=9, offset=2,'), 1)
                    script = script.replace('ksize=9, offset=2,', replacement)
                    output, observations_path = root/'mean-public.npz', root/'mean-public.json'
                    completed = subprocess.run([str(seeds.REFERENCE), '-c', script, str(source),
                        str(output), str(observations_path), 'mean', 'rgb'],
                        capture_output=True, text=True, timeout=120)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    with np.load(output, allow_pickle=False) as loaded:
                        expected = {name: loaded[name].copy() for name in loaded.files}
                    observations = json.loads(observations_path.read_text(encoding='utf-8'))
                    for suffix, port, name in (
                            ('median_blur','image','median_blur'), ('mean_threshold','mask','binary_threshold'),
                            ('small_object_filter','mask','small_object_filter'), ('erode','mask','erode'),
                            ('dilate','mask','dilate'), ('ws_distance_map','distance','distance'),
                            ('ws_local_maxima','coordinates','coordinates'), ('ws_local_maxima','peaks','peaks'),
                            ('ws_markers','markers','markers'), ('ws_invert_distance','negative_distance','negative_distance'),
                            ('ws_masked_flood','labels','labels'), ('analyze_size','diagnostic','size_diagnostic_rgb')):
                        np.testing.assert_array_equal(actual.node('pcv_seed_mean_'+suffix).outputs[port].cpu(), expected[name])
                    size = actual.node(runner._calibrated_seed.SCALED_SIZE).outputs['observations'] if is_calibrated else actual.node('pcv_seed_mean_analyze_size').outputs['observations']
                    calibrated.CalibratedSeedTests.assert_observations(self, size, observations['__size_observations__'])
                    color = actual.node('pcv_seed_mean_analyze_color').outputs
                    self.assertEqual(color['histogram'].data_records(), observations['__returned_hue_chart__'])
                    for label in color['observations'].labels:
                        for histogram in label.histograms:
                            np.testing.assert_array_equal(histogram.values, observations[label.sample][histogram.variable]['value'])

    def test_all_selected_recipes_cold_warm_verified_archive_and_source_absent_replay(self):
        self.assertTrue(BASELINE.is_file(), 'The reviewed ignored recipe baseline is required; tests never generate it.')
        raw = BASELINE.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), BASELINE_SHA256,
                         'Recipe baseline is missing, changed or not yet reviewed/frozen.')
        expected = _reviewed_code_provenance(json.loads(raw))
        def progress(count, total, recipe_id, row):
            if row.get('phase') != 'starting':
                print(f'\nAccepted {count}/{total}: {recipe_id}', flush=True)

        assert_json_equal(self, capture_recipe_acceptance(progress=progress), expected)


class AcceptanceDifferenceTests(unittest.TestCase):
    def test_complete_comparison_preserves_array_order_keys_lengths_and_types(self):
        expected = {'a/b~c': [1, 2], 'removed': {'value': 3}, 'flag': True}
        actual = {'a/b~c': [2, 1, 7], 'added': None, 'flag': 1}
        summary = json_difference_summary(actual, expected)
        self.assertEqual(summary['total_changes'], 7)
        self.assertEqual({sample['path'] for sample in summary['samples']},
                         {'/a~1b~0c', '/a~1b~0c/0', '/a~1b~0c/1', '/a~1b~0c/2',
                          '/removed', '/added', '/flag'})
        with self.assertRaisesRegex(AssertionError, '7 changes'):
            assert_json_equal(self, actual, expected)

    def test_sampling_cap_does_not_hide_late_changes_and_values_are_bounded(self):
        summary = json_difference_summary(['x' * 10000] * 80, ['y' * 10000] * 80,
                                          maximum_samples=3, maximum_value_characters=32)
        self.assertEqual(summary['total_changes'], 80)
        self.assertEqual(len(summary['samples']), 3)
        self.assertTrue(all(len(item[key]) <= 32 for item in summary['samples']
                            for key in ('actual', 'expected')))
        late = list(range(10000))
        expected = late.copy()
        late[-1] = -1
        with self.assertRaisesRegex(AssertionError, '/9999'):
            assert_json_equal(self, late, expected)

    def test_json_string_subclasses_and_object_order_compare_without_false_changes(self):
        class VersionString(str):
            pass
        assert_json_equal(self, {'b': [VersionString('2.13.0'), None], 'a': 1.5},
                          {'a': 1.5, 'b': ['2.13.0', None]})
        self.assertEqual(json_difference_summary(1, 1.0)['total_changes'], 1)
        self.assertEqual(json_difference_summary(float('nan'), float('nan'))['total_changes'], 1)

    def test_invalid_bounds_and_non_json_values_are_rejected_or_mismatched(self):
        for kwargs in ({'maximum_samples': 0}, {'maximum_samples': True},
                       {'maximum_value_characters': 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                json_difference_summary({}, {}, **kwargs)
        for actual in ((1, 2), {1: 'value'}, float('inf')):
            with self.subTest(kind=type(actual).__name__), self.assertRaises(AssertionError):
                assert_json_equal(self, actual, actual)
