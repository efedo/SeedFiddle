"""The engine's node vocabulary must stay bound to the pipeline graph."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.node_aliases import ENGINE_NODE_ALIASES, INPUT_DIRTY_MARKERS

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCE = ROOT / "seedvision" / "segmentation" / "baseline.py"
DIRTY_NAMES = {"dirty", "dirty_nodes"}
NODE_CALLS = {"node_enabled", "measure", "start", "add_seconds"}
LIST_CALLS = {"computed", "reused"}


def _graph_node_ids() -> frozenset[str]:
    graph = build_default_pipeline()
    return frozenset(graph.nodes) | frozenset(graph.unused_nodes)


def _string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def engine_node_literals(source: Path = ENGINE_SOURCE) -> dict[str, set[int]]:
    """Collect every string the engine compares with the dirty set or names as a stage."""

    literals: dict[str, set[int]] = {}

    def record(node: ast.AST) -> None:
        value = _string(node)
        if value is not None:
            literals.setdefault(value, set()).add(node.lineno)

    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Compare):
            # "name" in dirty / "name" in dirty_nodes
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(
                isinstance(target, ast.Name) and target.id in DIRTY_NAMES
                for target in node.comparators
            ):
                record(node.left)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            # dirty & {"name", ...}
            for side in (node.left, node.right):
                if isinstance(side, ast.Set):
                    for element in side.elts:
                        record(element)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            if isinstance(function, ast.Name) and function.id in NODE_CALLS:
                record(node.args[0])
            elif isinstance(function, ast.Attribute) and (
                function.attr in NODE_CALLS
                or (
                    function.attr == "append"
                    and isinstance(function.value, ast.Name)
                    and function.value.id in LIST_CALLS
                )
            ):
                record(node.args[0])
    return literals


class NodeVocabularyTests(unittest.TestCase):
    def test_engine_literals_are_graph_nodes_or_declared_aliases(self) -> None:
        literals = engine_node_literals()
        self.assertGreater(len(literals), 20, "the engine scan found suspiciously few names")
        known = _graph_node_ids() | set(ENGINE_NODE_ALIASES) | INPUT_DIRTY_MARKERS
        unknown = {
            name: sorted(lines) for name, lines in literals.items() if name not in known
        }
        self.assertEqual(
            unknown,
            {},
            "baseline.py names stages that are neither graph nodes nor declared in "
            "seedvision.pipeline.node_aliases",
        )

    def test_alias_targets_and_markers_are_consistent(self) -> None:
        node_ids = _graph_node_ids()
        self.assertTrue(set(ENGINE_NODE_ALIASES.values()) <= node_ids)
        self.assertFalse(set(ENGINE_NODE_ALIASES) & node_ids, "an alias shadows a node id")
        self.assertFalse(INPUT_DIRTY_MARKERS & node_ids)
        self.assertFalse(INPUT_DIRTY_MARKERS & set(ENGINE_NODE_ALIASES))

    def test_window_overlay_owners_are_graph_nodes(self) -> None:
        from seedvision.ui.main_window import OVERLAY_NODE_OWNERS

        node_ids = _graph_node_ids()
        unknown = sorted(set(OVERLAY_NODE_OWNERS.values()) - node_ids)
        self.assertEqual(unknown, [])


if __name__ == "__main__":
    unittest.main()
