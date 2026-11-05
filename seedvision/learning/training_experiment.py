"""Train U-Net alone, StarDist alone and one shared network on the same budget, then score them.

Every arm of a seed starts from the same backbone weights and sees the same tile
sequence, runs for the same number of optimiser steps with early stopping off, and
has its decoder chosen on the validation split only. The report states whether the
scored masks were reviewed and held out, and applies the adoption rule only if so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
from statistics import mean, stdev

from seedvision.learning.contracts import ModelFamily
from seedvision.learning.data import (
    LearningManifest,
    audit_manifest,
    file_sha256,
    read_label_image,
    resolve_sample_path,
)
from seedvision.learning.evaluation import evaluate_checkpoint
from seedvision.learning.metrics import evaluate_instances
from seedvision.learning.training import TrainingConfiguration, train_from_manifest


ARMS = {
    "unet": (ModelFamily.UNET_WATERSHED, (ModelFamily.UNET_WATERSHED,)),
    "stardist": (ModelFamily.STARDIST, (ModelFamily.STARDIST,)),
    "shared": (
        ModelFamily.SHARED_UNET_STARDIST,
        (ModelFamily.UNET_WATERSHED, ModelFamily.STARDIST),
    ),
}
MINIMUM_SEEDS_FOR_VERDICT = 3
ADOPTION_METRICS = ("f1_iou50", "f1_iou75")
# Metrics reported for each image and averaged. Unavailable values are null.
INSTANCE_METRICS = (
    "panoptic_quality",
    "mean_matched_iou",
    "absolute_count_error",
    "relative_count_error",
    "split_instances",
    "merged_instances",
    "boundary_f1",
)
NOT_VALIDATION = (
    "NOT VALIDATION: {reason} These numbers show that the experiment runs. They say "
    "nothing about accuracy and cannot decide between the arms."
)


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    steps: int = 8000
    validate_every: int = 200
    seeds: tuple[int, ...] = (20260811, 20260812, 20260813)
    arms: tuple[str, ...] = ("unet", "stardist", "shared")
    batch_size: int = 4
    tile_size: int = 256
    validation_tiles_per_sample: int = 32
    ray_count: int = 32
    base_channels: int = 24
    depth: int = 4
    learning_rate: float = 2e-4
    evaluation_tile_size: int = 512
    evaluation_overlap: int = 96
    device: str = "cuda"
    allow_unreviewed: bool = False

    def __post_init__(self) -> None:
        if self.steps < 1 or self.validate_every < 1:
            raise ValueError("The step budget and validation interval must be positive.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Give at least one seed, and no seed twice.")
        unknown = set(self.arms) - set(ARMS)
        if unknown or not self.arms:
            raise ValueError(f"Arms must come from {sorted(ARMS)}.")


def training_configuration(
    configuration: ExperimentConfiguration, arm: str, seed: int, training_images: int
) -> TrainingConfiguration:
    """Express the step budget as validation rounds, with early stopping off."""

    rounds = ceil(configuration.steps / configuration.validate_every)
    tiles = ceil(configuration.validate_every * configuration.batch_size / training_images)
    return TrainingConfiguration(
        family=ARMS[arm][0].value,
        epochs=rounds,
        batch_size=configuration.batch_size,
        tile_size=configuration.tile_size,
        tiles_per_sample=tiles,
        validation_tiles_per_sample=configuration.validation_tiles_per_sample,
        ray_count=configuration.ray_count,
        base_channels=configuration.base_channels,
        depth=configuration.depth,
        learning_rate=configuration.learning_rate,
        random_seed=seed,
        patience=rounds + 1,
        device=configuration.device,
        allow_unreviewed=configuration.allow_unreviewed,
    )


def describe_evidence(manifest: LearningManifest) -> dict:
    """Say what the scores can support, given which masks they are measured against."""

    samples = manifest.samples
    test = [item for item in samples if item.split == "test"]
    training_reviewed = all(item.reviewed for item in samples if item.split == "train")
    validation_reviewed = all(item.reviewed for item in samples if item.split == "validation")
    if test and all(item.reviewed for item in test):
        state = "reviewed_test"
        note = (
            "Scored on reviewed images held out of training and decoder selection. This "
            "compares the arms; it is not a scientific validation of accuracy."
        )
    elif test:
        state = "unreviewed_test"
        note = NOT_VALIDATION.format(reason="the test masks have not been reviewed.")
    else:
        state = "no_test_split"
        note = NOT_VALIDATION.format(
            reason="the dataset has no test split, so the scores come from the validation "
            "images that also chose the decoder and checkpoint."
        )
    return {
        "state": state,
        "note": note,
        "scored_split": "test" if test else "validation",
        "training_masks_reviewed": training_reviewed,
        "validation_masks_reviewed": validation_reviewed,
        "scored_images": [item.identifier for item in (test or [
            item for item in samples if item.split == "validation"
        ])],
    }


def _train_arm(manifest_path: Path, directory: Path, configuration: TrainingConfiguration, log) -> dict:
    """Train one arm, or reuse its finished report when an earlier run matches exactly."""

    checkpoint = directory / "model.pt"
    report_path = directory / "training.json"
    if checkpoint.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("configuration") == asdict(configuration)
            and report.get("manifest_sha256") == file_sha256(manifest_path)
        ):
            log(f"{directory.name}: reusing the finished training run")
            return report
    directory.mkdir(parents=True, exist_ok=True)
    return train_from_manifest(manifest_path, checkpoint, configuration, report_path=report_path)


def _rescore_at_iou_thresholds(manifest_path: Path, report: dict, predictions: Path, samples) -> list[dict]:
    """Add F1 at both IoU thresholds to each image, from the saved prediction."""

    records = {item["sample_id"]: item for item in report["samples"]}
    rows = []
    for sample in samples:
        truth = read_label_image(resolve_sample_path(manifest_path, sample.instances))
        prediction = read_label_image(predictions / f"{sample.identifier}.prediction.png")
        strict = evaluate_instances(truth, prediction, iou_threshold=0.75).to_dict()
        record = records[sample.identifier]
        rows.append(
            {
                "sample_id": sample.identifier,
                "reviewed": sample.reviewed,
                "true_instances": record["true_instances"],
                "predicted_instances": record["predicted_instances"],
                "f1_iou50": record["f1"],
                "f1_iou75": strict["f1"],
                **{name: record[name] for name in INSTANCE_METRICS},
            }
        )
    return rows


def _average_images(images: list[dict]) -> dict:
    averages = {}
    for name in (*ADOPTION_METRICS, *INSTANCE_METRICS):
        values = [item[name] for item in images if item[name] is not None]
        averages[name] = mean(values) if values else None
    return averages


def _score_output(
    manifest_path: Path,
    directory: Path,
    output: ModelFamily,
    evidence: dict,
    configuration: ExperimentConfiguration,
    log,
) -> dict:
    """Choose the decoder on validation, freeze it, then score the scored split."""

    checkpoint = directory / "model.pt"
    record_path = directory / f"score-{output.value}.json"
    checkpoint_sha256 = file_sha256(checkpoint)
    split = evidence["scored_split"]
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            record.get("checkpoint_sha256") == checkpoint_sha256
            and record.get("scored_split") == split
            and record.get("manifest_sha256") == file_sha256(manifest_path)
        ):
            log(f"{directory.name}: reusing the finished {output.value} score")
            return record
    settings = {
        "output": output,
        "device": configuration.device,
        "tile_size": configuration.evaluation_tile_size,
        "overlap": configuration.evaluation_overlap,
    }
    log(f"{directory.name}: choosing the {output.value} decoder on validation")
    selection_directory = directory / f"{output.value}-validation"
    selection = evaluate_checkpoint(
        checkpoint, manifest_path, selection_directory,
        split="validation", optimize_decoder=True, **settings,
    )
    if split == "validation":
        scored, predictions = selection, selection_directory
    else:
        log(f"{directory.name}: scoring {output.value} on the test split")
        predictions = directory / f"{output.value}-test"
        scored = evaluate_checkpoint(
            checkpoint, manifest_path, predictions, split="test",
            decoder_source_path=selection_directory / "evaluation.json", **settings,
        )
    samples = [item for item in LearningManifest.load(manifest_path).samples if item.split == split]
    images = _rescore_at_iou_thresholds(manifest_path, scored, predictions, samples)
    record = {
        "output": output.value,
        "scored_split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": file_sha256(manifest_path),
        "decoder_settings": selection["decoder_settings"],
        "images": images,
        "mean": _average_images(images),
        "membership_audit": scored["checkpoint_membership_audit"],
    }
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def _best_round_for_output(history: list[dict], arm: str, output: ModelFamily) -> int:
    """Find when this output's own validation loss was lowest."""

    name = "total"
    if arm == "shared":
        name = "unet_total" if output is ModelFamily.UNET_WATERSHED else "stardist_total"
    return min(history, key=lambda item: item["validation"][name])["epoch"]


def summarise(runs: list[dict]) -> list[dict]:
    """Average each arm and output across its seeds, with run-to-run spread."""

    rows = []
    for arm, output in sorted({(item["arm"], item["output"]) for item in runs}):
        matching = [item for item in runs if item["arm"] == arm and item["output"] == output]
        metrics = {}
        for name in (*ADOPTION_METRICS, *INSTANCE_METRICS):
            values = [item["mean"][name] for item in matching if item["mean"][name] is not None]
            metrics[name] = {
                "mean": mean(values) if values else None,
                "spread": stdev(values) if len(values) > 1 else None,
            }
        rows.append({"arm": arm, "output": output, "seeds": len(matching), "metrics": metrics})
    return rows


def adoption_verdict(summary: list[dict], evidence: dict, seeds: tuple[int, ...]) -> dict:
    """Apply the adoption rule: each shared output matches its counterpart within spread."""

    if evidence["state"] != "reviewed_test":
        return {"decided": False, "reason": "The scored masks are not reviewed, held-out images."}
    if len(seeds) < MINIMUM_SEEDS_FOR_VERDICT:
        return {"decided": False, "reason": f"Needs at least {MINIMUM_SEEDS_FOR_VERDICT} seeds for a spread."}
    arms = {(row["arm"], row["output"]): row for row in summary}
    comparisons = []
    for arm, output in (("unet", ModelFamily.UNET_WATERSHED.value), ("stardist", ModelFamily.STARDIST.value)):
        alone = arms.get((arm, output))
        shared = arms.get(("shared", output))
        if alone is None or shared is None:
            return {"decided": False, "reason": "Needs the shared arm and both separate arms."}
        for metric in ADOPTION_METRICS:
            separate, joint = alone["metrics"][metric], shared["metrics"][metric]
            if separate["mean"] is None or joint["mean"] is None:
                return {"decided": False, "reason": f"{metric} is unavailable for {output}."}
            spread = max(separate["spread"] or 0.0, joint["spread"] or 0.0)
            comparisons.append(
                {
                    "output": output,
                    "metric": metric,
                    "alone": separate["mean"],
                    "shared": joint["mean"],
                    "spread": spread,
                    "shared_holds_up": joint["mean"] >= separate["mean"] - spread,
                }
            )
    holds = all(item["shared_holds_up"] for item in comparisons)
    return {
        "decided": True,
        "shared_network_holds_up": holds,
        "comparisons": comparisons,
        "reason": (
            "Each shared output matches its separately trained counterpart within spread."
            if holds
            else "Joint training fell short on at least one output; try separate final "
            "decoder stages before deciding to keep the networks apart."
        ),
    }


def _value(summary_metric: dict) -> str:
    average, spread = summary_metric["mean"], summary_metric["spread"]
    if average is None:
        return "n/a"
    return f"{average:.3f}" + (f" ± {spread:.3f}" if spread is not None else "")


def render_report(report: dict) -> str:
    """Write the result as a short table that opens with what it can support."""

    evidence = report["evidence"]
    steps = ", ".join(f"{arm} {count}" for arm, count in report["optimizer_steps_by_arm"].items())
    labels = ", ".join(
        f"{name} {'reviewed' if evidence[f'{name}_masks_reviewed'] else 'unreviewed'}"
        for name in ("training", "validation")
    )
    lines = [
        "# Training experiment",
        "",
        f"**{evidence['note']}**",
        "",
        f"Scored split: {evidence['scored_split']} ({', '.join(evidence['scored_images'])}). "
        f"Other masks: {labels}. Optimiser steps: {steps}. "
        f"Seeds: {', '.join(map(str, report['seeds']))}.",
        "",
        "| Arm | Output | Seeds | F1 at IoU 0.5 | F1 at IoU 0.75 | Matched IoU | Count error "
        "| Splits | Merges | Boundary F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["summary"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['arm']} | {row['output']} | {row['seeds']} | "
            + " | ".join(
                _value(metrics[name])
                for name in (
                    "f1_iou50", "f1_iou75", "mean_matched_iou", "absolute_count_error",
                    "split_instances", "merged_instances", "boundary_f1",
                )
            )
            + " |"
        )
    verdict = report["verdict"]
    lines += ["", f"Adoption rule: {verdict['reason']}", ""]
    return "\n".join(lines)


def _write_reports(output_directory: Path, report: dict) -> None:
    (output_directory / "experiment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_directory / "experiment.md").write_text(render_report(report), encoding="utf-8")


def run_training_experiment(
    manifest_path: Path | str,
    output_directory: Path | str,
    configuration: ExperimentConfiguration = ExperimentConfiguration(),
    *,
    log=print,
) -> dict:
    """Run every arm for every seed; rerunning the same command resumes where it stopped."""

    manifest_path = Path(manifest_path).resolve()
    output_directory = Path(output_directory)
    audit = audit_manifest(manifest_path)
    if not audit["valid"]:
        raise ValueError("Learning dataset audit failed: " + "; ".join(audit["errors"]))
    manifest = LearningManifest.load(manifest_path)
    training_images = sum(item.split == "train" for item in manifest.samples)
    if not training_images or not any(item.split == "validation" for item in manifest.samples):
        raise ValueError("The experiment needs at least one Training and one Validation image.")
    evidence = describe_evidence(manifest)
    log(evidence["note"])
    output_directory.mkdir(parents=True, exist_ok=True)
    runs = []
    report = {}
    steps_by_arm = {}
    for seed in configuration.seeds:
        for arm in configuration.arms:
            directory = output_directory / f"{arm}-seed{seed}"
            log(f"{directory.name}: training")
            training = _train_arm(
                manifest_path,
                directory,
                training_configuration(configuration, arm, seed, training_images),
                log,
            )
            steps_by_arm[arm] = training["optimizer_steps"]
            for output in ARMS[arm][1]:
                score = _score_output(manifest_path, directory, output, evidence, configuration, log)
                runs.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "training_seconds": training["elapsed_seconds"],
                        "best_round": training["best_epoch"],
                        "best_round_for_output": _best_round_for_output(training["history"], arm, output),
                        "last_validation": training["history"][-1]["validation"],
                        **score,
                    }
                )
            summary = summarise(runs)
            report = {
                "scientifically_validated": False,
                "evidence": evidence,
                "configuration": asdict(configuration),
                "dataset_id": manifest.dataset_id,
                "manifest_sha256": file_sha256(manifest_path),
                "seeds": list(configuration.seeds),
                "optimizer_steps_by_arm": steps_by_arm,
                "summary": summary,
                "verdict": adoption_verdict(summary, evidence, configuration.seeds),
                "runs": runs,
            }
            _write_reports(output_directory, report)
    log(render_report(report))
    return report
