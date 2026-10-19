"""Distinguish numerical fixture definitions from reorganized test methods."""
import ast
from hashlib import sha256
import json
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'docs/testing/consolidation-2026-10-18/fixture_definitions.json'


@lru_cache(maxsize=24)
def definition_digest(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (isinstance(node,ast.ImportFrom)
                 and node.module in ('tests.functional_units','tests.catalogue_fixtures'))]
    for cls in (node for node in tree.body if isinstance(node,ast.ClassDef)):
        # Base classes govern test fixture ownership, not numerical inputs.
        # Normalize only the explicit immutable-catalogue test fixture base.
        cls.bases = [ast.Attribute(value=ast.Name(id='unittest',ctx=ast.Load()),
            attr='TestCase',ctx=ast.Load()) if isinstance(node,ast.Name) and
            node.id=='CatalogueTestCase' else node for node in cls.bases]
        cls.decorator_list = [node for node in cls.decorator_list if not
            (isinstance(node,ast.Name) and node.id=='functional_units')]
        cls.body = [node for node in cls.body if not
            (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and
             (node.name.startswith('test_') or node.name.startswith('check_test_')))
            and not isinstance(node,ast.Pass)]
    return sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def verify_fixture_definitions(name, historical_source_digest):
    records=json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
    expected=records.get(name)
    if (expected is None or expected['historical_source_sha256']!=historical_source_digest
            or definition_digest((ROOT/name).read_text(encoding='utf-8'))!=expected['definition_sha256']):
        raise AssertionError('Numerical fixture definitions changed: '+name)
