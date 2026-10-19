"""Reuse the immutable factory catalogue in calculation/composition fixtures.

Admission of each selected recipe remains real. Only rebuilding and auditing
the identical five-template catalogue is shared. Catalogue/configuration and
loaded-code mutation witnesses deliberately do not use this fixture.
"""
from contextlib import contextmanager, ExitStack
from functools import lru_cache
import sys
import unittest
from unittest.mock import patch

from tests.review_fixtures import _workflow_template_key


@lru_cache(maxsize=1)
def _catalogue(key, builder):
    value = builder()

    def build_fixture_catalogue():
        return value

    return value, value.to_json(), build_fixture_catalogue


@contextmanager
def catalogue_fixture():
    from seedvision.benchmarking import recipe_catalogue
    builder = recipe_catalogue.build_recipe_catalogue
    value, snapshot, build_fixture_catalogue = _catalogue(_workflow_template_key(), builder)
    if value.to_json() != snapshot:
        raise AssertionError('Shared factory catalogue changed before reuse')

    # Keep imported aliases coherent, as the real runtime verifier requires.
    # Patches exist only during this test, never in the application/runner.
    with ExitStack() as stack:
        for name, module in tuple(sys.modules.items()):
            if (module is not None and name.startswith(('seedvision.', 'tests.'))
                    and vars(module).get('build_recipe_catalogue') is builder):
                stack.enter_context(patch.object(module, 'build_recipe_catalogue', build_fixture_catalogue))
        try:
            yield value
        finally:
            # A test can lazily import another module while the fixture is in
            # scope. Restore its newly bound alias too; it was not in ExitStack.
            for name, module in tuple(sys.modules.items()):
                if (module is not None and name.startswith(('seedvision.', 'tests.'))
                        and vars(module).get('build_recipe_catalogue') is build_fixture_catalogue):
                    module.build_recipe_catalogue = builder
            if value.to_json() != snapshot:
                raise AssertionError('Test mutated the shared factory catalogue')


class CatalogueTestCase(unittest.TestCase):
    """Explicit opt-in; per-test workspaces and numerical results stay private."""

    def run(self, result=None):
        with catalogue_fixture():
            return super().run(result)
