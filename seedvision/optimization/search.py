"""Deterministic sequential search, independent of Qt and image calculations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from statistics import mean

from .registry import affected_nodes, capability, candidate_values


class OptimizationCancelled(Exception):
    pass


class Ineligible(ValueError):
    """Missing supervision/input; never silently counted as zero loss."""


@dataclass
class NodeResult:
    node_id: str
    status: str
    reason: str = ''
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)
    trials: list = field(default_factory=list)


@dataclass
class OptimizationReport:
    started: str
    mode: str
    original: dict
    nodes: list[NodeResult] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    cancelled: bool = False
    final_before: dict = field(default_factory=dict)
    final_after: dict = field(default_factory=dict)
    previews: dict = field(default_factory=dict)

    @property
    def changes(self):
        return {row.node_id: row.changes for row in self.nodes if row.changes}


def aggregate(scores):
    if not scores or any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError('An objective must return finite per-image losses.')
    return mean(scores.values())


def optimize(graph, evaluator, *, selected=None, passes=1, maximum_evaluations=0,
             cancelled=lambda: False, progress=lambda *args: None):
    """Search an isolated graph. All images contribute to each shared proposal.

    Evaluator scores are keyed by frozen image/group IDs. A node cannot win by
    dropping an image. Zero budget means one complete sweep of every control;
    explicit budgets record exhaustion, rather than claiming complete coverage.
    """
    if passes < 1 or maximum_evaluations < 0:
        raise ValueError('Invalid optimization budget.')
    working = deepcopy(graph)
    selected = set(graph.nodes if selected is None else selected)
    report = OptimizationReport(datetime.now(timezone.utc).isoformat(), evaluator.mode,
        {key: dict(node.parameters) for key, node in graph.nodes.items()},
        provenance=evaluator.provenance)
    report.provenance['graph_revision'] = graph.revision
    for key, node in graph.unused_nodes.items():
        report.nodes.append(NodeResult(key, 'skipped', 'Inactive toolbox node.'))
    try:
        evaluator.check_snapshot()
        report.final_before = evaluator.final_scores(working)
        for identifier in working.topological_order():
            if identifier not in selected:
                report.nodes.append(NodeResult(identifier, 'skipped', 'Not selected for this run.'))
                continue
            if cancelled():
                raise OptimizationCancelled()
            node = working.node(identifier)
            contract = capability(node)
            row = NodeResult(identifier, 'skipped')
            report.nodes.append(row)
            if not node.enabled or not node.implemented or not contract.searchable:
                row.reason = contract.reason or 'Node disabled or unimplemented.'
                continue
            try:
                eligible, exclusions = evaluator.parameters(working, identifier, contract.searchable)
                if not eligible:
                    raise Ineligible('; '.join(exclusions.values()) or 'No active searchable inputs.')
                scores = evaluator.score(working, identifier)
            except Ineligible as error:
                row.reason = str(error)
                continue
            row.before = dict(scores)
            row.trials.append({'parameters': {}, 'scores': dict(scores), 'excluded': exclusions})
            specs = {spec.key: spec for spec in node.parameter_specs}
            exhausted = False
            for pass_number in range(passes):
                for key in eligible:
                    origin = node.parameters[key]
                    best_value, best_scores = origin, scores
                    for value in candidate_values(specs[key], origin, pass_number):
                        if cancelled():
                            raise OptimizationCancelled()
                        if maximum_evaluations and len(row.trials) >= maximum_evaluations:
                            exhausted = True
                            break
                        node.set_parameter(key, value)
                        try:
                            candidate = evaluator.score(working, identifier)
                            if set(candidate) != set(row.before):
                                raise ValueError('Candidate changed the evaluation image set.')
                            loss = aggregate(candidate)
                            row.trials.append({'parameter': key, 'value': value, 'scores': candidate})
                            if loss < aggregate(best_scores) - 1e-9:
                                best_value, best_scores = value, candidate
                        except (ValueError, Ineligible) as error:
                            row.trials.append({'parameter': key, 'value': value, 'invalid': str(error)})
                        finally:
                            node.set_parameter(key, origin)
                        progress(identifier, len(row.trials), aggregate(best_scores))
                    node.set_parameter(key, best_value)
                    scores = best_scores
                    if exhausted:
                        break
                if exhausted:
                    break
            row.after = dict(scores)
            row.changes = {key: value for key, value in node.parameters.items()
                           if value != report.original[identifier][key]}
            row.status = 'improved' if row.changes else 'unchanged'
            row.reason = 'Budget exhausted; some candidates were not evaluated.' if exhausted else 'Completed requested parameter sweeps.'
            evaluator.accept(working, identifier)
        if cancelled():
            raise OptimizationCancelled()
        evaluator.check_snapshot()
        report.final_after = evaluator.final_scores(working)
        if cancelled():
            raise OptimizationCancelled()
        report.previews = getattr(evaluator, 'previews', {})
    except OptimizationCancelled:
        report.cancelled = True
        for row in report.nodes:
            row.changes.clear()
    return report


def apply_report(graph, report):
    """Validate everything on a copy before touching the live project graph."""
    if report.cancelled:
        raise ValueError('Cancelled optimization cannot be applied.')
    if set(graph.nodes) != set(report.original) or graph.revision != report.provenance.get('graph_revision', graph.revision) or any(key not in graph.nodes or graph.node(key).parameters != original
           for key, original in report.original.items()):
        raise ValueError('The optimization proposal is stale.')
    proposed = deepcopy(graph)
    affected = set()
    for identifier, changes in report.changes.items():
        proposed.set_parameters(identifier, changes)
        affected.update(affected_nodes(proposed, identifier))
    for identifier, changes in report.changes.items():
        graph.set_parameters(identifier, changes)
    graph.invalidate(affected)
    return affected
