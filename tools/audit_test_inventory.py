"""Inventory unittest source without running tests or their fixtures.

Use --collect to additionally reconcile AST methods with unittest discovery.
Descriptions are static evidence, not measured statement/branch coverage.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'testing'
RECEIPT = ROOT / 'artifacts' / 'test-suite-cleanup-full-v2.json'
TIMINGS = ROOT / 'artifacts' / 'test-suite-cleanup-full-v2-timings.json'

# Module purposes for older files without module docstrings. Newer files use
# their own explicit scope declarations. Method scenarios and check expressions
# are always extracted separately; a module purpose is not a coverage claim.
PURPOSES = dict(line.split(' | ', 1) for line in '''advanced_analysis | Advanced image layers and CUDA analysis products
analysis_settings_persistence | Analysis settings serialization, migration, normalization and private workspace state
annotation_edge_sources | Annotation tools select the intended edge evidence and clear obsolete sources
annotation_shape_revision | Shape review revisions, visibility and reference-library updates
annotation_tools | Annotation drawing, brush geometry, region operations and instance tools
assignment_budget | Assignment metrics and segmentation matching respect explicit work budgets
benchmark_checkpoints | Checkpoint identities, array limits, manifests, storage and corruption rejection
benchmark_conformance | Comparison metrics, tolerances and checkpoint conformance reports
benchmark_manifest | Benchmark manifests, source/environment identity and evidence contracts
bootstrap | Python/runtime dependency admission and single-entry startup
checkpoint_bundle | Checkpoint bundle serialization, import/export and project references
color_histogram | Color histogram calculation, comparison and desktop display
core_recovery | Seed core extraction, recovery, label ownership and CUDA bounds
critical_review_fixes | Earlier regression fixes spanning annotation coordinates, learning and export
diagnostics | Runtime diagnostic facts
edge_trace_commit | Edge trace tool commits and desktop annotation updates
entrypoint | Native desktop application launch and command routing
icon_asset | Application icon assets
image_calibration | Image calibration, geometry, chart/ruler detection and real-image fixtures
instance_annotation_tiling | Annotation tile rendering, invalidation and image-view state
instance_continuity | Instance continuity during annotation updates
instance_continuity_ui | Desktop selection follows instance continuity
instance_mask_import | Instance mask loading, source identity and validation
instance_mask_import_ui | Instance mask import dialogs and desktop application
instance_reference_evidence | Annotated-instance reference evidence and visualization
instance_reference_repository | Committed reference fixture repository integrity
learned_segmentation | Learned segmentation features, targets, model, loss, decoding and metrics
learning_workflow | Training workflow, evidence export and desktop controls
manual_seed_centre_persistence | Manual seed centre sidecar persistence and source binding
manual_seed_centre_ui | Manual centre drawing, selection and desktop persistence
material_evidence | CUDA material likelihoods, references and evidence decisions
measurements | Seed measurement geometry
method_qc | Method quality-control facts, provenance and exported summaries
native_checkpoint_inspection | Native checkpoint inspection, evidence limits and retained outputs
native_core_recovery_recipe | Native core/recovery graph contracts, CUDA execution and cache behavior
native_correction_dialog | Color correction dialog controls, workers and imported inputs
native_correction_recipe | Native color correction recipe composition and execution
native_extended_threshold_recipe | Extended threshold recipe variants, contracts and native execution
native_mask_stage | Native mask-stage checkpoints and output identity
native_preprocessing | Native preprocessing chains, caching and public PlantCV parity
native_recipe | Native recipe execution, cache behavior, cancellation and results
native_recipe_dialog | Native recipe dialog selection, execution and result state
native_seed_working_recipe | Complete native seed working workflows
native_smpts_recipe | Native soybean paper workflow composition and outputs
native_threshold_stage | Native threshold stage inputs, outputs, resources and provenance
native_watershed_diagnostic_recipe | Watershed diagnostic recipe variants and CUDA products
native_watershed_diagnostic_replay | Saved watershed diagnostic stage replay and guards
native_watershed_stage | Native watershed stage execution and checkpoints
node_optimization | Node parameter search, optimizer state and cache reuse
node_optimization_ui | Node optimizer controls and desktop result application
oriented_edge_components | Oriented edge components, connectivity and CUDA outputs
pilot_analysis | Committed pilot-image segmentation, calibration and end-to-end analysis
pipeline | Pipeline graph topology, controls, settings, state and invalidation
pipeline_cable_routing | Pipeline cable geometry and routing
pipeline_inspector_actions | Pipeline inspector parameter editing and actions
pipeline_ui | Pipeline desktop controls, graph state, navigation, workers and status
plantcv_card_detection | CUDA color card detection geometry and public-source parity
plantcv_card_preprocess | CUDA color card image preparation and source semantics
plantcv_color | CUDA color-space conversion and PlantCV-compatible image analysis
plantcv_color_card | CUDA color card chip sampling and correction facts
plantcv_contours | CUDA contour extraction, topology and public geometry
plantcv_correction | CUDA polynomial color correction and numerical compatibility
plantcv_distance | CUDA distance transform and public-source numerical parity
plantcv_extended_thresholds | CUDA extended threshold algorithms and input boundaries
plantcv_fill | CUDA small-object and hole filling with public PlantCV parity
plantcv_kmeans | CUDA k-means analysis and public-model compatibility
plantcv_kmeans_classifier | CUDA k-means class assignment and model boundaries
plantcv_kmeans_features | CUDA k-means features and input conversion
plantcv_kmeans_model | K-means model structure, identity and validation
plantcv_kmeans_predict | CUDA k-means predictions and source-model agreement
plantcv_kmeans_preprocess | CUDA k-means image preparation and feature order
plantcv_median | CUDA median filtering and public PlantCV pixel parity
plantcv_minrect | CUDA minimum-area rectangle geometry and degeneracies
plantcv_morphology | CUDA erosion/dilation and public PlantCV pixel parity
plantcv_peaks | CUDA peak detection, spacing and source semantics
plantcv_range | CUDA range thresholding and public pixel parity
plantcv_raster_spans | CUDA contour rasterization, spans and output masks
plantcv_size | CUDA PlantCV-compatible shape/size measurements
plantcv_size_diagnostic | CUDA size diagnostic products and public compatibility
plantcv_thresholds | CUDA binary/mean/Otsu thresholds and isolated public-source comparison
plantcv_watershed | CUDA watershed labeling and public-source numerical parity
plantcv_watershed_diagnostic | CUDA watershed diagnostic labels, plots and bounds
procedural_fit | Procedural seed fitting, search behavior and geometric evidence
procedural_fit_ui | Procedural fit controls and result application
procedural_instances | CUDA procedural instance splitting, ownership and analysis integration
project_analysis_persistence | Project analysis settings, image locators, migration and atomic persistence
project_analysis_ui | Project analysis save/reopen, dirty state, selections and desktop guards
reference_capture | Public reference capture sessions, manifests and publication failures
reference_checkpoint_inspection | Reference checkpoint inspection and graph-stage evidence
reference_edge_fit | Reference-driven edge fitting and rendered evidence
reference_edge_strip_descriptor | Reference edge strip extraction, descriptors and overlays
reference_edge_toolbar_revision | Edge reference toolbar controls, graph revisions and image-view state
reference_graph_persistence | Reference graph project serialization and historical identity
reference_graph_readonly | Read-only public reference graph presentation and edit prevention
reference_graph_view | Reference graph desktop viewing, inspection and replay
reference_history | Reference history revision selection and UI presentation
reference_library | Reference library aggregation, extraction, contracts and persistence
reference_mask_inspection | Saved public mask stage inspection
reference_mask_stage | Public mask-stage capture and checkpoint identity
reference_persistence | Reference region sidecars, migration, source binding and desktop lifecycle
reference_preprocessing | Public preprocessing chains, capture, controls and cache identity
reference_runner | Isolated PlantCV worker requests, limits, transport and failure outcomes
reference_seed_chain | Actual public seed-chain composition, controls and cancellation
reference_setup | Pinned reference environment and source installation checks
reference_stage | Public stage capture, graph adaptation and checkpoint identity
reference_texture_collage | Reference texture collage layout and image rendering
reference_undo | Reference annotation undo/redo and independent history
resources | CUDA resource estimation and runtime admission
review_workflows | Review export, portable snapshots and desktop workflow
seed_crop_artifacts | Seed crop artifacts, export identity and portable review
seed_shape_model | Seed shape model fitting, reference aggregation and scoring
selected_instance_view | Selected instance overlay and desktop view state
shape_guided_fill | Shape-guided annotation filling and selection behavior
shape_guided_ui | Shape-guided annotation desktop controls
smpts_paper | Soybean paper method calculations and explicitly supported boundaries
species_library_dialog | Species reference library desktop editing
threshold_conformance | Public/native threshold checkpoint comparison and conformance
visualization_layers | Analysis visualization layers, overlays, crops and display contracts'''.splitlines())


def category(stem: str) -> str:
    """Exclusive navigation buckets; mixed suites retain per-method evidence."""
    name = stem.removeprefix('test_')
    if name in {'reference_library', 'species_library_dialog'}:
        return 'Measurements and shape models'
    if name in {'reference_persistence', 'reference_source_identity'}:
        return 'Persistence, export, identity and outcomes'
    if name in {'reference_undo', 'reference_history', 'reference_texture_collage'} or name.startswith('reference_edge_'):
        return 'Annotation and interactive image review'
    if name in {'reader_acceptance', 'recipe_acceptance', 'public_seed_acceptance'}:
        return 'Frozen acceptance matrices'
    if name.startswith('plantcv_') or name == 'smpts_paper':
        return 'Numerical primitives and public-source parity'
    if (name.startswith(('native_', 'reference_', 'benchmark_', 'peer_', 'catalogue_'))
            or name.endswith('_roi_recipe') or name in {
                'calibrated_public_capture_methods', 'calibrated_public_capture_dialog',
                'checkpoint_bundle', 'kmeans_model_artifact', 'kmeans_training_directory',
                'naive_bayes_plots', 'naive_bayes_training_directory', 'public_capture_layout',
                'public_reference_attempt_qc', 'threshold_conformance', 'color_histogram',
                'classifier_evaluation', 'grouped_comparison', 'adaptive_kernel_normalization'}):
        return 'Recipe, replay and reference infrastructure'
    if name.startswith(('project_', 'method_qc', 'recipe_', 'correction_')) or name in {
            'terminal_snapshot', 'trait_snapshot', 'portable_result_identity',
            'result_prediction_revision', 'calibration_result_revision',
            'reviewed_measurement_lineage', 'seed_crop_artifacts', 'review_workflows',
            'run_outcomes', 'backend_outcomes', 'preflight_outcomes', 'desktop_terminal_history'}:
        return 'Persistence, export, identity and outcomes'
    if name.startswith('pipeline') or name.startswith('node_optimization') or name == 'analysis_settings_persistence':
        return 'Pipeline workspace and controls'
    if name.startswith(('annotation_', 'instance_', 'manual_', 'shape_guided', 'selected_instance', 'edge_trace')):
        return 'Annotation and interactive image review'
    if name.startswith(('trait_', 'learned_', 'learning_', 'assignment_')):
        return 'Learning and trait analysis'
    if name.startswith(('shape_', 'seed_shape', 'measurement', 'conditioned_', 'local_shape', 'physical_shape')):
        return 'Measurements and shape models'
    if name in {'bootstrap', 'entrypoint', 'diagnostics', 'icon_asset', 'resources', 'performance_report', 'test_runner'}:
        return 'Startup, resources and diagnostics'
    return 'Image analysis, calibration and rendering'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expr(node: ast.AST) -> str:
    return ' '.join(ast.unparse(node).split())


def calls(node: ast.AST) -> list[ast.Call]:
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))


def check(call: ast.Call) -> bool:
    tail = expr(call.func).split('.')[-1]
    return tail.startswith('assert') or tail == 'fail'


def unique(items):
    return list(dict.fromkeys(items))


def read_trees():
    result = {}
    for path in sorted((ROOT / 'tests').glob('*.py')):
        source = path.read_text(encoding='utf-8-sig')
        result['tests.' + path.stem] = (path, source, ast.parse(source))
    return result


def collect_ids() -> list[str]:
    # Import-only collection. No TestCase.run, setUp, setUpClass or numerical
    # methods are invoked. Modules may perform their normal import-time checks.
    import os
    import sys
    import tempfile
    import unittest
    from uuid import uuid4
    sys.path.insert(0, str(ROOT))
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    temporary = ROOT / 'artifacts' / 'test-temp' / ('inventory-' + uuid4().hex)
    temporary.mkdir(parents=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temporary)
    tempfile.tempdir = str(temporary)
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
    if loader.errors:
        raise RuntimeError('\n'.join(loader.errors))

    def flatten(value):
        for item in value:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item.id()
    return list(flatten(suite))


def build_inventory(trees, receipt, timings):
    functions, imports = {}, {}

    def import_aliases(nodes):
        aliases = {}
        for node in nodes:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = (node.module or '') + '.' + alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        return aliases

    for module, (_, _, tree) in trees.items():
        imports[module] = import_aliases(tree.body)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[module + '.' + node.name] = node
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions[module + '.' + node.name + '.' + member.name] = member

    def resolve(name, module, cls, node):
        first, dot, rest = name.partition('.')
        if first in {'self', 'cls'}:
            return module + '.' + cls + dot + rest
        aliases = imports[module] | import_aliases(ast.walk(node))
        candidate = aliases.get(first, module + '.' + first)
        return candidate + (dot + rest if dot else '')

    def helper_closure(node, module, cls):
        found = {}
        pending = [(node, module, cls)]
        while pending:
            current, mod, class_name = pending.pop()
            for call in calls(current):
                targets = [resolve(expr(call.func), mod, class_name, current)]
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'scenario' and call.args:
                    targets.append(resolve(expr(call.args[0]), mod, class_name, current))
                for target in targets:
                    if target in functions and target not in found:
                        found[target] = functions[target]
                        parts = target.split('.')
                        pending.append((functions[target], '.'.join(parts[:2]), parts[2] if len(parts) == 4 else ''))
        return found

    timing_map = {}
    for label, seconds in timings.get('test_durations', []):
        match = re.search(r'\((tests\..*)\)$', label)
        if match:
            if match[1] in timing_map:
                raise ValueError('Duplicate timing ID: ' + match[1])
            timing_map[match[1]] = seconds
    skips = {row['id']: row['reason'] for row in timings.get('skips', [])}
    failures = {row['id'] for row in timings.get('failures', []) + timings.get('errors', [])}
    modules, tests = [], []
    for module, (path, source, tree) in trees.items():
        if not path.name.startswith('test'):
            continue
        stem = path.stem.removeprefix('test_')
        doc = ast.get_docstring(tree)
        if not doc and stem not in PURPOSES:
            raise ValueError('Add a reviewed module purpose: ' + stem)
        purpose = ' '.join(doc.split('\n\n')[0].split()) if doc else PURPOSES[stem]
        sha = digest(path)
        pinned = receipt.get('source_hashes', {}).get(path.relative_to(ROOT).as_posix()) == sha
        module_rows = []
        fixtures = {}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            setup = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'setUp', 'setUpClass', 'tearDown', 'tearDownClass'}]
            fixtures[cls.name] = [{'name':n.name,'line':n.lineno,'calls':unique(expr(c.func) for c in calls(n))} for n in setup]
            for node in (n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test')):
                ident = module + '.' + cls.name + '.' + node.name
                direct_calls = calls(node)
                closure = helper_closure(node, module, cls.name)
                direct_checks = [expr(c) for c in direct_calls if check(c)] + [expr(n) for n in ast.walk(node) if isinstance(n, ast.Assert)]
                targets = [resolve(expr(c.func), module, cls.name, node) for c in direct_calls]
                subjects = unique(target for target in targets if target.startswith('seedvision.'))
                indirect_subjects = []
                helper_checks = []
                for helper_id, helper in closure.items():
                    parts = helper_id.split('.')
                    mod = '.'.join(parts[:2])
                    class_name = parts[2] if len(parts) == 4 else ''
                    for call in calls(helper):
                        target = resolve(expr(call.func), mod, class_name, helper)
                        if target.startswith('seedvision.'):
                            indirect_subjects.append(target)
                        if check(call):
                            helper_checks.append(helper_id + ': ' + expr(call))
                body_doc = ast.get_docstring(node)
                scenario = ' '.join(body_doc.split()) if body_doc else node.name.removeprefix('test_').replace('_', ' ')
                row = {
                    'id': ident, 'file': path.relative_to(ROOT).as_posix(),
                    'class': cls.name, 'method': node.name, 'line': node.lineno,
                    'end_line': node.end_lineno, 'module_purpose': purpose,
                    'functionality': scenario, 'description_basis': 'method docstring' if body_doc else 'method scenario name; see assertion evidence',
                    'category': category(path.stem), 'direct_subjects': subjects,
                    'helper_subjects': unique(indirect_subjects),
                    'calls': unique(expr(c.func) for c in direct_calls),
                    'direct_checks': direct_checks, 'helper_checks': unique(helper_checks),
                    'helpers': list(closure), 'fixtures': [n.name for n in setup],
                    'decorators': [expr(n) for n in node.decorator_list],
                    'class_decorators': [expr(n) for n in cls.decorator_list],
                    'subtest_sites': sum(expr(c.func).endswith('.subTest') for c in direct_calls),
                    'loop_iterables': [expr(n.iter) for n in ast.walk(node) if isinstance(n, (ast.For, ast.comprehension))],
                    'skip_calls': [expr(c) for c in direct_calls if expr(c.func).endswith('.skipTest')],
                    'source_sha256': sha, 'historical_source_matches': pinned,
                    'last_full_status': 'unrecorded' if not receipt else (('skipped' if ident in skips else 'failed' if ident in failures else 'passed' if ident in timing_map and timings.get('passed') else 'unrecorded') if pinned else 'source changed; historical only'),
                    'last_full_seconds': timing_map.get(ident), 'skip_reason': skips.get(ident, ''),
                }
                module_rows.append(row)
                tests.append(row)
        modules.append({'module':module,'file':path.relative_to(ROOT).as_posix(),'purpose':purpose,
                        'category':category(path.stem),'lines':len(source.splitlines()),
                        'source_sha256':sha,'historical_source_matches':pinned,
                        'test_count':len(module_rows),'class_fixtures':fixtures,
                        'production_imports':sorted(set(v for v in imports[module].values() if v.startswith('seedvision.'))),
                        'method_seconds':sum(r['last_full_seconds'] or 0 for r in module_rows)})
    if len({r['id'] for r in tests}) != len(tests):
        raise ValueError('Duplicate AST test ID')
    unknown = set(timing_map) - {r['id'] for r in tests}
    helpers = {}
    for target in unique(h for row in tests for h in row['helpers']):
        node = functions[target]
        helpers[target] = {'file':trees['.'.join(target.split('.')[:2])][0].relative_to(ROOT).as_posix(),
                           'line':node.lineno,'checks':[expr(c) for c in calls(node) if check(c)] + [expr(n) for n in ast.walk(node) if isinstance(n, ast.Assert)]}
    return modules, tests, sorted(unknown), helpers


def md(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write_outputs(record):
    OUT.mkdir(parents=True, exist_ok=True)
    # Module and shared-helper facts are stored once, not copied into all 2,811 rows.
    compact = dict(record)
    compact['tests'] = [{k:v for k,v in row.items() if k not in {'module_purpose','source_sha256','helper_checks'}} for row in record['tests']]
    (OUT / 'test_inventory.json').write_text(json.dumps(compact, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Complete test inventory', '',
        f"This is a source-derived inventory of every discovered test method, audited on {record['audit_date']}. "
        'Functionality descriptions use method docstrings or scenario names, with assertion and call evidence in '
        '[the JSON inventory](test_inventory.json). '
        'These are intended coverage claims, **not measured statement or branch coverage**.', '',
        'Markdown shows at most the first two assertion expressions, shortened to 260 characters. '
        'JSON retains every direct assertion expression, lexical production-call candidates, interface calls, '
        'helper references and their shared assertion evidence, loops, subtest sites, decorators and fixture calls. '
        'Dynamic dispatch and inherited fixture behavior require source inspection; this is not a manually proven '
        'call graph or an exhaustive enumeration of runtime subtest cases.', '',
        f"Total: **{len(record['tests']):,} methods in {len(record['modules'])} test files**. "
        'A method can contain many subtests, input cases or expensive executions. Those are not separate unittest methods.', '',
        'Discovery reconciliation: ' + record['collection_status'] + '.', '',
        f"Historical status and method seconds come from `{record['historical_receipt']}`. "
        'Class setup/teardown, discovery and interpreter shutdown are not allocated to methods. '
        'A dash means no method duration was recorded, not zero cost. '
        'Runtime skips can differ by environment; their exact historical reasons are in JSON.', '',
        'Regenerate: `.venv\\Scripts\\python.exe tools/audit_test_inventory.py --collect`. '
        'This imports modules for discovery but never runs tests or setup fixtures. '
        'Without the ignored historical receipts, use `--without-history`; historical status/timing fields are then unrecorded.', '',
        'See [the multi-pass audit](../TEST_INFRASTRUCTURE_AUDIT.md) for overlap, cost and infrastructure findings.', '',
        '| Module | Tests | Historical method seconds | Scope |', '| --- | ---: | ---: | --- |']
    for module in record['modules']:
        anchor = module['module'].replace('.', '')
        lines.append(f"| [{module['module']}](#{anchor}) | {module['test_count']} | {module['method_seconds']:.3f} | {md(module['purpose'])} |")
    for module in record['modules']:
        lines += ['', f"## {module['module']}", '', module['purpose'], '',
                  f"Source: [{module['file']}](../../{module['file']}). Category: {module['category']}.", '']
        for class_name, fixtures in module['class_fixtures'].items():
            rows = [r for r in record['tests'] if r['file'] == module['file'] and r['class'] == class_name]
            if not rows:
                continue
            lines += [f'### {class_name}', '', 'Fixtures: ' + (', '.join(f"`{r['name']}` at line {r['line']}" for r in fixtures) or 'no class-local setup/teardown methods') + '.', '',
                      '| Test method / source | Functionality exercised | Assertion evidence | Last full run |', '| --- | --- | --- | --- |']
            for row in rows:
                evidence = row['direct_checks'][:2]
                if not evidence and row['helper_checks']:
                    evidence = ['helper: ' + row['helper_checks'][0]]
                brief = '; '.join(evidence) or 'No direct assertion call; inspect helper/exception evidence in JSON'
                if len(brief) > 260:
                    brief = brief[:257] + '…'
                seconds = '—' if row['last_full_seconds'] is None else f"{row['last_full_seconds']:.3f}s"
                result = row['last_full_status'] + ' / ' + seconds
                lines.append(f"| [`{row['method']}`](../../{row['file']}#L{row['line']}) | {md(row['functionality'])} | `{md(brief).replace('`', chr(39))}` | {result} |")
    (OUT / 'TEST_INVENTORY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', help='Import-only unittest discovery reconciliation')
    parser.add_argument('--without-history', action='store_true', help='Do not require local ignored full-run artifacts')
    parser.add_argument('--receipt', type=Path, default=RECEIPT, help='Completed source-pinned run receipt')
    parser.add_argument('--timings', type=Path, help='Override receipt timing report location')
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = {} if args.without_history else json.loads(receipt_path.read_text(encoding='utf-8'))
    timing_path = (args.timings or ROOT / receipt.get('timing_report', str(TIMINGS))).resolve()
    timings = {} if args.without_history else json.loads(timing_path.read_text(encoding='utf-8'))
    if not args.without_history and digest(timing_path) != receipt['timing_report_sha256']:
        raise ValueError('Historical timings do not match their recorded SHA256')
    trees = read_trees()
    modules, tests, unknown, helpers = build_inventory(trees, receipt, timings)
    core_targets = next(ast.literal_eval(n.value) for n in trees['tests.selection'][2].body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CORE_TARGETS' for t in n.targets))
    for row in tests:
        row['runs_in_default_core'] = any(row['id'] == target or row['id'].startswith(target + '.') for target in core_targets)
    collected = collect_ids() if args.collect else None
    if collected is not None and (len(collected) != len(tests) or set(collected) != {r['id'] for r in tests}):
        raise ValueError('AST methods and unittest discovery differ; investigate inherited/dynamic tests before publishing')
    changed = [] if args.without_history else [p for p,h in receipt['source_hashes'].items() if not (ROOT/p).is_file() or digest(ROOT/p) != h]
    record = {
        'schema':'seedfiddle.test-inventory.v1', 'audit_date':'2026-10-20',
        'generator':'tools/audit_test_inventory.py', 'generator_sha256':digest(Path(__file__)),
        'description_limit':'Static scenario, assertion and call evidence; no measured line/branch coverage. Import aliases are lexical candidates, not runtime type resolution.',
        'helper_limit':'Module/class/imported test helpers traced statically; dynamic dispatch, assigned object types and inherited methods are not resolved. Fixture calls are separate from method helper closure.',
        'historical_receipt':None if args.without_history else receipt_path.relative_to(ROOT).as_posix(),
        'historical_receipt_sha256':None if args.without_history else digest(receipt_path),
        'historical_timings_sha256':None if args.without_history else digest(timing_path),
        'historical_source_changes':changed, 'unmatched_historical_timing_ids':unknown,
        'collection_status':f'exact match, {len(collected):,} canonical unittest IDs; zero discovery errors' if collected is not None else 'not performed; AST inventory only',
        'test_count':len(tests), 'test_file_count':len(modules),
        'core_targets':core_targets, 'core_test_count':sum(t['runs_in_default_core'] for t in tests),
        'test_source_lines':sum(m['lines'] for m in modules),
        'historical_method_seconds':sum(r['last_full_seconds'] or 0 for r in tests),
        'category_totals':{cat:{'methods':sum(t['category']==cat for t in tests),
                                'files':sum(m['category']==cat for m in modules),
                                'method_seconds':sum(t['last_full_seconds'] or 0 for t in tests if t['category']==cat)}
                           for cat in sorted({t['category'] for t in tests})},
        'modules':modules, 'tests':tests, 'shared_helpers':helpers,
        'support_files':[{'file':p.relative_to(ROOT).as_posix(),'source_sha256':digest(p),'lines':len(s.splitlines()),'docstring':ast.get_docstring(t)} for p,s,t in trees.values() if not p.name.startswith('test')],
    }
    write_outputs(record)
    print(json.dumps({k:v for k,v in record.items() if k not in {'modules','tests','support_files','shared_helpers'}}, indent=2))


if __name__ == '__main__':
    main()
