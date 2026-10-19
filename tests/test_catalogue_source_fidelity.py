"""Canonical source fidelity, independent of a supplied graph's self-consistency."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest

from seedvision.benchmarking.recipe_catalogue import (
    RecipeCatalogue, RecipeCatalogueError, build_recipe_catalogue, validate_recipe_catalogue,
)
from seedvision.benchmarking.recipes import RecipeSpec, initial_stage_run, to_pipeline_graph
from seedvision.persistence.reference_graphs import (
    ReferenceGraphAttachment, validate_reference_graphs,
    reference_graphs_from_payload, reference_graphs_to_payload,
)
from seedvision.benchmarking.recipe_matching import _matches_catalogue_with_selected_values


@functional_units
class CatalogueSourceFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = build_recipe_catalogue()
        cls.recipe = cls.catalogue.recipe('plantcv_seed_binary')

    def catalogue_with(self, recipe):
        return replace(self.catalogue, recipes=tuple(
            recipe if value.recipe_id == recipe.recipe_id else value for value in self.catalogue.recipes))

    def changed_stage(self, operation, transform, recipe=None):
        recipe = recipe or self.recipe
        return replace(recipe, stages=tuple(transform(stage) if stage.operation_id == operation else stage
                                             for stage in recipe.stages))

    def rejects(self, recipe):
        # The forged graph is internally coherent. That is insufficient to
        # establish fidelity to the independently authored source contract.
        to_pipeline_graph(recipe)
        changed = self.catalogue_with(recipe)
        with self.assertRaisesRegex(RecipeCatalogueError, 'canonical source-mapped contract'):
            validate_recipe_catalogue(changed)
        with self.assertRaisesRegex(RecipeCatalogueError, 'canonical source-mapped contract'):
            RecipeCatalogue.from_json(changed.to_json())

    def check_test_changed_operation_source_symbol_and_port_semantics_reject_despite_valid_pins(self):
        changes = (
            lambda s: replace(s, operation_id='threshold.claimed_but_not_pinned'),
            lambda s: replace(s, source_mappings=tuple(
                replace(mapping, source_symbol='plantcv.plantcv.threshold.unmapped_function')
                for mapping in s.source_mappings)),
        )
        for transform in changes:
            with self.subTest(transform=transform):
                forged = self.changed_stage('threshold.binary', transform)
                self.assertEqual(forged.provenance, self.recipe.provenance)
                self.assertEqual(forged.recipe_id, self.recipe.recipe_id)
                self.rejects(forged)
        # Change every matching mask endpoint so ordinary topology validation
        # does not catch the inconsistency before the source-fidelity guard.
        forged = replace(self.recipe, stages=tuple(replace(stage, ports=tuple(
            replace(port, semantics={**port.semantics, 'foreground':1})
            if port.data_type == 'BinaryMask' else port for port in stage.ports))
            for stage in self.recipe.stages))
        self.rejects(forged)

    def check_test_coherent_missing_extra_reordered_and_rewired_stages_reject(self):
        terminal = next(s for s in self.recipe.stages if s.operation_id == 'analysis.color')
        omitted = replace(self.recipe,
            stages=tuple(s for s in self.recipe.stages if s.stage_id != terminal.stage_id),
            dependencies=tuple(e for e in self.recipe.dependencies if e.target_stage != terminal.stage_id),
            layout=tuple(p for p in self.recipe.layout if p.stage_id != terminal.stage_id))
        extra = replace(self.recipe, stages=(*self.recipe.stages,
            replace(self.recipe.stages[0], stage_id='extra_prepared_input')))
        reordered = replace(self.recipe, stages=tuple(reversed(self.recipe.stages)))
        threshold = next(s for s in self.recipe.stages if s.operation_id == 'threshold.binary')
        rewired = replace(self.recipe, dependencies=tuple(
            replace(e, source_stage=self.recipe.stages[0].stage_id, source_port='gray_image')
            if e.target_stage == threshold.stage_id else e for e in self.recipe.dependencies))
        for name, forged in (('omitted',omitted),('extra',extra),('reordered',reordered),('rewired',rewired)):
            with self.subTest(name=name): self.rejects(forged)

    def check_test_fixed_parameter_declarations_and_values_cannot_be_replaced_by_selected_recipes(self):
        for name, transform in (
            ('declaration', lambda p: replace(p, editable=not p.editable)),
            ('selected-value', lambda p: replace(p, value=3)),
            ('description', lambda p: replace(p, description='A different declared calculation.')),
        ):
            forged = self.changed_stage('filter.median_blur', lambda s: replace(s,
                parameters=tuple(transform(p) for p in s.parameters)))
            with self.subTest(name=name): self.rejects(forged)

    def check_test_json_numeric_and_boolean_semantics_do_not_compare_as_equal(self):
        for key, value in (('background',False),('foreground',255.0)):
            forged = self.changed_stage('threshold.binary', lambda s: replace(s,
                ports=tuple(replace(p, semantics={**p.semantics,key:value})
                    if p.direction == 'output' else p for p in s.ports)))
            with self.subTest(key=key): self.rejects(forged)

    def check_test_paper_branch_condition_and_unavailable_original_mappings_remain_exact(self):
        paper = self.catalogue.recipe('smpts_paper_reconstruction')
        for transform in (
            lambda s: replace(s, condition=None),
            lambda s: replace(s, condition=replace(s.condition, expression='invented resolved rule')),
            lambda s: replace(s, source_mappings=tuple(
                replace(m, source_symbol='invented_original_implementation') if m.reference_id == 'smpts_original'
                else m for m in s.source_mappings)),
        ):
            forged = self.changed_stage('paper_reconstruction.targeted_erosion', transform, paper)
            with self.subTest(transform=transform): self.rejects(forged)

    def check_test_canonical_bytes_roundtrip_and_layout_only_changes_remain_valid(self):
        original = self.catalogue.to_json()
        self.assertEqual(RecipeCatalogue.from_json(original).to_json(),original)
        moved = replace(self.recipe, layout=tuple(replace(p,x=p.x+50,y=p.y-25) for p in self.recipe.layout))
        changed = self.catalogue_with(moved)
        validate_recipe_catalogue(changed)
        self.assertEqual(RecipeCatalogue.from_json(changed.to_json()),changed)
        self.assertEqual(moved.identity.analytical_dict(),self.recipe.identity.analytical_dict())
        self.assertNotEqual(moved.identity.layout_sha256,self.recipe.identity.layout_sha256)


    def check_test_selected_and_historical_attachment_apis_preserve_exact_metadata(self):
        selected = self.changed_stage('filter.median_blur',lambda s:replace(s,
            parameters=tuple(replace(p,value=3) for p in s.parameters)))
        historical = replace(selected,recipe_version='99.4')
        for recipe in (selected,historical):
            restored = RecipeSpec.from_json(recipe.to_json())
            self.assertEqual(restored,recipe)
            attachment = ReferenceGraphAttachment(restored,tuple(
                initial_stage_run(restored,s.stage_id) for s in restored.stages),
                selected_stage_id=restored.stages[0].stage_id)
            self.assertEqual(validate_reference_graphs((attachment,)),(attachment,))
            self.assertEqual(reference_graphs_from_payload(reference_graphs_to_payload((attachment,))), (attachment,))
            self.assertEqual(_matches_catalogue_with_selected_values(recipe, self.recipe), recipe is selected)

    def test_validation_and_recovery(self):
        """3 related CatalogueSourceFidelityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_changed_operation_source_symbol_and_port_semantics_reject_despite_valid_pins)
        self.scenario(self.check_test_coherent_missing_extra_reordered_and_rewired_stages_reject)
        self.scenario(self.check_test_paper_branch_condition_and_unavailable_original_mappings_remain_exact)

    def test_workflow_lifecycle(self):
        """4 related CatalogueSourceFidelityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_fixed_parameter_declarations_and_values_cannot_be_replaced_by_selected_recipes)
        self.scenario(self.check_test_json_numeric_and_boolean_semantics_do_not_compare_as_equal)
        self.scenario(self.check_test_canonical_bytes_roundtrip_and_layout_only_changes_remain_valid)
        self.scenario(self.check_test_selected_and_historical_attachment_apis_preserve_exact_metadata)


if __name__ == '__main__':
    unittest.main()
