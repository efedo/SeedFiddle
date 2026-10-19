"""Representative calculations plus exhaustive cheap recipe-contract coverage.

The former matrix reran the same numerical operators for every permutation of
reader, median, ROI and diagnostic adapters. All such recipes are still built,
admitted and compared to their frozen selected specification. Calculations are
performed for every distinct operation/port domain and input family instead.
"""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


LAST_SELECTION = {}


def domains(profile):
    label,recipe=profile
    values={('operation',stage.operation_id,tuple((p.direction,p.name,p.data_type) for p in stage.ports))
            for stage in recipe.stages}
    values.add(('family',label))
    from seedvision.benchmarking import native_reader_recipe as reader
    mode=reader.selected_mode(recipe)
    if mode is not None:values.add(('reader',label,mode))
    return values


def representative_indexes(profiles):
    required=set().union(*(domains(profile) for profile in profiles))
    selected=set()
    # Whole seed arms and a calibrated arm retain distinct complete outcomes.
    for name in ('plantcv_seed_binary','plantcv_seed_mean','plantcv_seed_otsu','plantcv_calibrated_seed_binary'):
        selected.add(next(i for i,(label,recipe) in enumerate(profiles)
                          if label=='canonical' and recipe.recipe_id==name))
    covered=set().union(*(domains(profiles[index]) for index in selected))
    while required-covered:
        index=max((i for i in range(len(profiles)) if i not in selected),
                  key=lambda i:len(domains(profiles[i])-covered))
        assert domains(profiles[index])-covered
        selected.add(index);covered|=domains(profiles[index])
    assert required==covered
    return sorted(selected)


def compare_functional_profiles(testcase):
    from tests import test_recipe_acceptance as capture
    from tests.acceptance_provenance import apply
    from tests.acceptance_diff import assert_json_equal
    raw=capture.BASELINE.read_bytes()
    testcase.assertEqual(sha256(raw).hexdigest(),capture.BASELINE_SHA256)
    expected=apply(capture._workspace_environment_provenance(capture._workspace_code_provenance(
        capture._reviewed_code_provenance(json.loads(raw)))),'recipe')
    profiles=capture.canonical_profiles(validate=True)
    testcase.assertEqual(len(profiles),len(expected['cases']))
    # Every supported recipe still exercises the real construction/admission
    # path. Exact edges, controls, versions and typed ports are compared cheaply.
    with tempfile.TemporaryDirectory(prefix='recipe-specifications-') as directory:
        paths,models,_,_=capture._fixtures(Path(directory))
        for (label,recipe),baseline in zip(profiles,expected['cases']):
            selected=recipe if label.startswith('four-input') else capture._select(recipe,paths,models)[0]
            with testcase.subTest(specification=recipe.recipe_id,profile=label):
                testcase.assertEqual(label,baseline['profile'])
                assert_json_equal(testcase,selected.to_dict(),baseline['recipe'])
    indexes=representative_indexes(profiles)
    selected=tuple(profiles[i] for i in indexes)
    def selected_profiles(*,validate=True):
        return selected
    def progress(count,total,recipe_id,row):
        if row.get('phase')!='starting':print(f'\nFunctional recipe {count}/{total}: {recipe_id}',flush=True)
    with patch.object(capture,'canonical_profiles',selected_profiles):
        actual=capture.capture_recipe_acceptance(progress=progress)
    testcase.assertEqual(len(actual['cases']),len(indexes))
    for index,row in zip(indexes,actual['cases']):
        baseline=expected['cases'][index]
        # The first representative owning a replay changes with the reduced
        # execution order. Actual replay equality is checked inside capture;
        # every numerical, frame, runtime, control and archive field stays exact.
        omit={'source_absent_replayed_operations'}
        with testcase.subTest(numerical=row['recipe']['recipe_id'],profile=row['profile']):
            assert_json_equal(testcase,{k:v for k,v in row.items() if k not in omit},
                              {k:v for k,v in baseline.items() if k not in omit})
    assert_json_equal(testcase,actual['expected_partial_boundaries'],expected['expected_partial_boundaries'])
    testcase.assertEqual(actual['fixture_inventory'],expected['fixture_inventory'])
    # Reorganization changes whole-test-file hashes, never numerical capture
    # helpers. Check their original source pins outside the profile selection.
    testcase.assertEqual(capture._capture_pins()['helpers'],expected['capture_pins']['helpers'])
    all_operations={stage.operation_id for _,recipe in profiles for stage in recipe.stages}
    selected_operations={stage.operation_id for _,recipe in selected for stage in recipe.stages}
    testcase.assertEqual(all_operations,selected_operations)
    LAST_SELECTION.update(specifications=len(profiles),executed_profiles=len(indexes),
        operations=sorted(selected_operations),selected_baseline_indexes=indexes,
        scope='Exhaustive recipe specifications; numerical operation/port/input-family representatives')


def verify_frozen_baselines(testcase):
    from tests import test_recipe_acceptance as capture
    root=Path(__file__).resolve().parents[1]
    hashes={
        'recipe-acceptance-baseline-v1.json':'b8fefc1b5d3bf0d595bfa243303bcc194f181d49f36cf05502d69654ace4c8f2',
        'recipe-acceptance-baseline-v2.json':'82861ec3480b7ac9bcaa65be1d27bbc9dc15d928d3e813210e0fad29c5bf6eb1',
        'public-seed-acceptance-baseline-v1.json':'5f84f7f8e8d9b71e6f0704a262c6002e4ed382a16cc13511566596b0cf87eb6a',
        'reader-acceptance-baseline-v1.json':'3f06d1f4a9ed2ba4568ad3c14797bbb72c81ca9a422be294df98ba0503c4cd68',
    }
    for name,digest in hashes.items():
        with testcase.subTest(baseline=name):
            testcase.assertEqual(sha256((root/'artifacts'/name).read_bytes()).hexdigest(),digest)
