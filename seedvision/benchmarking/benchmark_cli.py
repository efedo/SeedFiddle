"""Lazy single-launch-point benchmark adapter; no desktop or GPU imports here."""


def run_comparison(input_path, output_path):
    """Evaluate a prespecified grouped comparison and publish its report.

    Plans whose methods bind a trained classifier need evidence that ties the
    model to its training data; no such adapter exists, so they are refused.
    """
    from seedvision.benchmarking import grouped_comparison as grouped
    manifest, records, plan = grouped.load_comparison_inputs(input_path)
    if any(method.model_sha256 is not None or method.training_evidence_sha256 is not None
           for method in (plan.reference, plan.candidate)):
        raise ValueError('Comparison plans whose methods bind a trained classifier are not supported.')
    return grouped.export_grouped_comparison(output_path, manifest, records, plan)
