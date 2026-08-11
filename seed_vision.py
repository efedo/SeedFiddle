"""Single launch point for the Seed Fiddle desktop application."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seedvision.bootstrap import (  # noqa: E402 - intentionally after ROOT setup
    BootstrapError,
    create_virtual_environment,
    format_report,
    inspect_runtime,
    install_dependencies,
    load_configuration,
    probe_torch,
    project_venv_python,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local Seed Fiddle desktop application."
    )
    parser.add_argument(
        "--environment",
        choices=("auto", "current", "venv"),
        default="auto",
        help="dependency target when bootstrapping is required",
    )
    parser.add_argument(
        "--bootstrap",
        choices=("ask", "auto", "never"),
        default="ask",
        help="whether missing dependencies may be installed",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="alias for --bootstrap never",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="install only from compatible wheels in the local wheels directory",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="print runtime and package status without installing or launching",
    )
    learning = parser.add_mutually_exclusive_group()
    learning.add_argument(
        "--learning-audit",
        metavar="MANIFEST",
        help="audit a learned-segmentation dataset manifest and exit",
    )
    learning.add_argument(
        "--create-synthetic-learning-data",
        metavar="DIRECTORY",
        help="create a labelled simulator dataset for software verification only",
    )
    learning.add_argument(
        "--train-learned-model",
        choices=("unet_watershed", "stardist"),
        help="train one native PyTorch learned instance model and exit",
    )
    learning.add_argument(
        "--evaluate-learned-model",
        metavar="CHECKPOINT",
        help="evaluate a learned checkpoint on complete manifest images and exit",
    )
    learning.add_argument(
        "--review-learned-fixtures",
        metavar="CHECKPOINT",
        help="render qualitative learned-instance outputs for an unlabelled image directory",
    )
    learning.add_argument(
        "--evaluate-hybrid-models",
        metavar="UNET_CHECKPOINT",
        help="evaluate U-Net boundaries with interior-gated StarDist markers",
    )
    parser.add_argument("--learning-manifest", help="manifest used for learned-model training")
    parser.add_argument("--model-output", help="destination .pt checkpoint")
    parser.add_argument("--training-report", help="optional training-report JSON destination")
    parser.add_argument("--training-epochs", type=int, default=40)
    parser.add_argument("--training-batch-size", type=int, default=4)
    parser.add_argument("--training-tile-size", type=int, default=256)
    parser.add_argument("--training-tiles-per-sample", type=int, default=16)
    parser.add_argument("--training-rays", type=int, default=32)
    parser.add_argument("--training-base-channels", type=int, default=24)
    parser.add_argument("--training-depth", type=int, default=4)
    parser.add_argument("--training-learning-rate", type=float, default=2e-4)
    parser.add_argument("--training-seed", type=int, default=20260811)
    parser.add_argument("--pattern-boundary-loss-weight", type=float, default=0.8)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--synthetic-train-count", type=int, default=48)
    parser.add_argument("--synthetic-validation-count", type=int, default=12)
    parser.add_argument("--synthetic-test-count", type=int, default=12)
    parser.add_argument("--synthetic-image-size", type=int, default=256)
    parser.add_argument(
        "--learning-split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--evaluation-output", help="directory for metrics and visual comparisons")
    parser.add_argument("--fixture-directory", default="images")
    parser.add_argument("--fixture-output")
    parser.add_argument("--fixture-species", default="unknown")
    parser.add_argument("--hybrid-stardist-checkpoint")
    parser.add_argument("--unet-decoder-settings")
    parser.add_argument("--stardist-decoder-settings")
    parser.add_argument("--hybrid-gate-threshold", type=float, default=0.50)
    parser.add_argument("--hybrid-distance-threshold", type=float, default=0.35)
    parser.add_argument("--optimize-hybrid-decoder", action="store_true")
    parser.add_argument("--evaluation-tile-size", type=int, default=512)
    parser.add_argument("--evaluation-overlap", type=int, default=96)
    parser.add_argument(
        "--decoder-settings",
        help="frozen decoder-settings JSON or prior evaluation report",
    )
    parser.add_argument(
        "--optimize-decoder",
        action="store_true",
        help="search decoder settings; accepted only for the validation split",
    )
    parser.add_argument(
        "--allow-unreviewed-learning-data",
        action="store_true",
        help="permit an explicitly non-scientific training experiment on unreviewed labels",
    )
    return parser


def _ask_environment() -> str | None:
    if not sys.stdin.isatty():
        return None
    print("\nRequired packages are missing or incompatible.")
    print("  1. Create or use .venv beside seed_vision.py (recommended)")
    print("  2. Install into the current Python interpreter")
    print("  q. Exit without making changes")
    while True:
        answer = input("Select an environment [1/2/q]: ").strip().lower()
        if answer in {"1", "venv"}:
            return "venv"
        if answer in {"2", "current"}:
            return "current"
        if answer in {"q", "quit", "exit"}:
            return None
        print("Please enter 1, 2, or q.")


def _confirm_install(environment: str, specs: Sequence[str]) -> bool:
    if not sys.stdin.isatty():
        return False
    target = ".venv" if environment == "venv" else sys.executable
    print(f"\nTarget: {target}")
    print("Packages:")
    for spec in specs:
        print(f"  - {spec}")
    answer = input("Proceed with package installation? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _run_with_interpreter(
    python_executable: Path | str,
    *,
    offline: bool,
    bootstrap: str,
    forwarded_arguments: Sequence[str] = (),
) -> int:
    command = [
        str(python_executable),
        str(ROOT / "seed_vision.py"),
        "--environment",
        "current",
        "--bootstrap",
        bootstrap,
    ]
    if offline:
        command.append("--offline")
    command.extend(forwarded_arguments)
    try:
        return subprocess.call(command)
    except OSError as error:
        raise BootstrapError(
            f"Could not start Seed Fiddle with {python_executable}: {error}"
        ) from error


def _launch_application() -> int:
    # Third-party imports begin only inside this delayed entry point.
    try:
        from seedvision.application import run

        return run(ROOT)
    except (ImportError, OSError) as error:
        raise BootstrapError(
            "A required package was discovered but could not be imported. "
            f"Run --diagnostics and check binary compatibility. Details: {error}"
        ) from error


def _learning_arguments(args) -> list[str]:
    """Forward only non-bootstrap command arguments into a project venv."""

    values: list[str] = []
    if args.learning_audit:
        values.extend(("--learning-audit", args.learning_audit))
    if args.create_synthetic_learning_data:
        values.extend(
            ("--create-synthetic-learning-data", args.create_synthetic_learning_data)
        )
        values.extend(
            (
                "--synthetic-train-count", str(args.synthetic_train_count),
                "--synthetic-validation-count", str(args.synthetic_validation_count),
                "--synthetic-test-count", str(args.synthetic_test_count),
                "--synthetic-image-size", str(args.synthetic_image_size),
            )
        )
    if args.train_learned_model:
        values.extend(("--train-learned-model", args.train_learned_model))
        for option, value in (
            ("--learning-manifest", args.learning_manifest),
            ("--model-output", args.model_output),
            ("--training-report", args.training_report),
            ("--initial-checkpoint", args.initial_checkpoint),
        ):
            if value:
                values.extend((option, str(value)))
        values.extend(
            (
                "--training-epochs", str(args.training_epochs),
                "--training-batch-size", str(args.training_batch_size),
                "--training-tile-size", str(args.training_tile_size),
                "--training-tiles-per-sample", str(args.training_tiles_per_sample),
                "--training-rays", str(args.training_rays),
                "--training-base-channels", str(args.training_base_channels),
                "--training-depth", str(args.training_depth),
                "--training-learning-rate", str(args.training_learning_rate),
                "--training-seed", str(args.training_seed),
                "--pattern-boundary-loss-weight", str(args.pattern_boundary_loss_weight),
            )
        )
        if args.allow_unreviewed_learning_data:
            values.append("--allow-unreviewed-learning-data")
    if args.evaluate_learned_model:
        values.extend(("--evaluate-learned-model", args.evaluate_learned_model))
        for option, value in (
            ("--learning-manifest", args.learning_manifest),
            ("--evaluation-output", args.evaluation_output),
            ("--decoder-settings", args.decoder_settings),
        ):
            if value:
                values.extend((option, str(value)))
        values.extend(
            (
                "--learning-split", args.learning_split,
                "--evaluation-tile-size", str(args.evaluation_tile_size),
                "--evaluation-overlap", str(args.evaluation_overlap),
            )
        )
        if args.optimize_decoder:
            values.append("--optimize-decoder")
    if args.review_learned_fixtures:
        values.extend(("--review-learned-fixtures", args.review_learned_fixtures))
        values.extend(("--fixture-directory", args.fixture_directory))
        if args.fixture_output:
            values.extend(("--fixture-output", args.fixture_output))
        values.extend(("--fixture-species", args.fixture_species))
    if args.evaluate_hybrid_models:
        values.extend(("--evaluate-hybrid-models", args.evaluate_hybrid_models))
        for option, value in (
            ("--hybrid-stardist-checkpoint", args.hybrid_stardist_checkpoint),
            ("--learning-manifest", args.learning_manifest),
            ("--evaluation-output", args.evaluation_output),
            ("--unet-decoder-settings", args.unet_decoder_settings),
            ("--stardist-decoder-settings", args.stardist_decoder_settings),
        ):
            if value:
                values.extend((option, str(value)))
        values.extend(
            (
                "--learning-split", args.learning_split,
                "--evaluation-tile-size", str(args.evaluation_tile_size),
                "--evaluation-overlap", str(args.evaluation_overlap),
                "--hybrid-gate-threshold", str(args.hybrid_gate_threshold),
                "--hybrid-distance-threshold", str(args.hybrid_distance_threshold),
            )
        )
        if args.optimize_hybrid_decoder:
            values.append("--optimize-hybrid-decoder")
    return values


def _run_learning_action(args) -> int | None:
    if args.learning_audit:
        from seedvision.learning.data import audit_manifest

        audit = audit_manifest(args.learning_audit)
        print(json.dumps(audit, indent=2))
        return 0 if audit["valid"] else 2
    if args.create_synthetic_learning_data:
        from seedvision.learning.synthetic import create_synthetic_dataset

        manifest = create_synthetic_dataset(
            args.create_synthetic_learning_data,
            train_count=args.synthetic_train_count,
            validation_count=args.synthetic_validation_count,
            test_count=args.synthetic_test_count,
            image_size=args.synthetic_image_size,
            random_seed=args.training_seed,
        )
        print(manifest)
        return 0
    if args.train_learned_model:
        if not args.learning_manifest or not args.model_output:
            raise BootstrapError(
                "--train-learned-model requires --learning-manifest and --model-output."
            )
        from seedvision.learning.training import (
            TrainingConfiguration,
            train_from_manifest,
        )

        configuration = TrainingConfiguration(
            family=args.train_learned_model,
            epochs=args.training_epochs,
            batch_size=args.training_batch_size,
            tile_size=args.training_tile_size,
            tiles_per_sample=args.training_tiles_per_sample,
            ray_count=args.training_rays,
            base_channels=args.training_base_channels,
            depth=args.training_depth,
            learning_rate=args.training_learning_rate,
            random_seed=args.training_seed,
            allow_unreviewed=args.allow_unreviewed_learning_data,
            pattern_boundary_loss_weight=args.pattern_boundary_loss_weight,
            initial_checkpoint=args.initial_checkpoint,
        )
        report = train_from_manifest(
            args.learning_manifest,
            args.model_output,
            configuration,
            report_path=args.training_report,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
        return 0
    if args.evaluate_learned_model:
        if not args.learning_manifest or not args.evaluation_output:
            raise BootstrapError(
                "--evaluate-learned-model requires --learning-manifest and --evaluation-output."
            )
        from seedvision.learning.checkpoint import load_checkpoint
        from seedvision.learning.contracts import ModelFamily
        from seedvision.learning.evaluation import (
            evaluate_checkpoint,
            load_decoder_settings,
        )

        decoder_settings = None
        if args.decoder_settings:
            _model, _spec, checkpoint_payload = load_checkpoint(
                args.evaluate_learned_model, device="cpu"
            )
            decoder_settings = load_decoder_settings(
                args.decoder_settings, ModelFamily(checkpoint_payload["family"])
            )

        report = evaluate_checkpoint(
            args.evaluate_learned_model,
            args.learning_manifest,
            args.evaluation_output,
            split=args.learning_split,
            tile_size=args.evaluation_tile_size,
            overlap=args.evaluation_overlap,
            decoder_settings=decoder_settings,
            optimize_decoder=args.optimize_decoder,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
        return 0
    if args.review_learned_fixtures:
        if not args.fixture_output:
            raise BootstrapError("--review-learned-fixtures requires --fixture-output.")
        from seedvision.learning.fixture_review import review_fixtures

        report = review_fixtures(
            args.review_learned_fixtures,
            args.fixture_directory,
            args.fixture_output,
            learning_root=ROOT,
            species=args.fixture_species,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
        return 0
    if args.evaluate_hybrid_models:
        if not all(
            (
                args.hybrid_stardist_checkpoint,
                args.learning_manifest,
                args.evaluation_output,
            )
        ):
            raise BootstrapError(
                "--evaluate-hybrid-models requires --hybrid-stardist-checkpoint, "
                "--learning-manifest, and --evaluation-output."
            )
        from seedvision.learning.contracts import ModelFamily
        from seedvision.learning.decode import StarDistDecodeSettings, UNetWatershedSettings
        from seedvision.learning.evaluation import load_decoder_settings
        from seedvision.learning.hybrid_evaluation import evaluate_hybrid_checkpoints

        unet_decoder = (
            load_decoder_settings(
                args.unet_decoder_settings, ModelFamily.UNET_WATERSHED
            )
            if args.unet_decoder_settings
            else UNetWatershedSettings()
        )
        star_decoder = (
            load_decoder_settings(
                args.stardist_decoder_settings, ModelFamily.STARDIST
            )
            if args.stardist_decoder_settings
            else StarDistDecodeSettings()
        )
        report = evaluate_hybrid_checkpoints(
            args.evaluate_hybrid_models,
            args.hybrid_stardist_checkpoint,
            args.learning_manifest,
            args.evaluation_output,
            split=args.learning_split,
            unet_settings=unet_decoder,
            stardist_settings=star_decoder,
            gate_threshold=args.hybrid_gate_threshold,
            distance_threshold=args.hybrid_distance_threshold,
            optimize_decoder=args.optimize_hybrid_decoder,
            tile_size=args.evaluation_tile_size,
            overlap=args.evaluation_overlap,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
        return 0
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_mode = "never" if args.no_bootstrap else args.bootstrap

    try:
        configuration = load_configuration(ROOT)
        report = inspect_runtime(configuration)

        if args.diagnostics:
            print(format_report(report, configuration))
            torch_check = next(
                (
                    check
                    for check in report.dependency_checks
                    if check.dependency.import_name == "torch"
                ),
                None,
            )
            if torch_check and torch_check.compatible:
                print(f"Acceleration: {probe_torch(report.python_executable)}")
                print(
                    "Image pipeline: PyTorch CUDA for full-raster calibration, "
                    "colour/noise probabilities, image-quality products, and edge "
                    "diagnostics; GPU-resized evidence plus bounded OpenCV/NumPy CPU "
                    "topology for active procedural instances; former experimental "
                    "proposal/mask stages retained in the node toolbox; CPU also used "
                    "for decoding, Qt display, and compact metadata."
                )
            return 0 if report.ready else 2

        if report.ready:
            learning_result = _run_learning_action(args)
            if learning_result is not None:
                return learning_result
            return _launch_application()

        # If a local environment already exists, auto mode gives it first chance
        # before proposing changes to the current interpreter.
        venv_python = project_venv_python(ROOT)
        if (
            args.environment in {"auto", "venv"}
            and venv_python.is_file()
            and Path(sys.executable).resolve() != venv_python.resolve()
        ):
            return _run_with_interpreter(
                venv_python,
                offline=args.offline,
                bootstrap=bootstrap_mode,
                forwarded_arguments=_learning_arguments(args),
            )

        print(format_report(report, configuration))
        if not report.python_compatible or report.architecture_bits != 64:
            raise BootstrapError(
                "Dependency installation cannot repair this Python runtime. Use a "
                f"64-bit Python >= {configuration.python.minimum} and "
                f"< {configuration.python.maximum_exclusive}."
            )
        if bootstrap_mode == "never":
            print("\nBootstrap is disabled; no changes were made.", file=sys.stderr)
            return 2

        environment = args.environment
        if environment == "auto":
            environment = "venv" if bootstrap_mode == "auto" else _ask_environment()
        if environment is None:
            print("No changes were made.")
            return 2

        running_in_project_venv = (
            report.in_virtual_environment
            and Path(sys.executable).resolve() == project_venv_python(ROOT).resolve()
        )
        dependencies = (
            configuration.required_packages
            if environment == "venv" and not running_in_project_venv
            else report.packages_to_install
        )
        specs = tuple(dependency.install_spec for dependency in dependencies)
        if bootstrap_mode == "ask" and not _confirm_install(environment, specs):
            print("No changes were made.")
            return 2

        target_python: Path | str
        if environment == "venv":
            target_python = create_virtual_environment(ROOT)
        else:
            target_python = sys.executable

        install_dependencies(
            target_python,
            dependencies,
            root=ROOT,
            offline=args.offline,
        )
        return _run_with_interpreter(
            target_python,
            offline=args.offline,
            bootstrap="never",
            forwarded_arguments=_learning_arguments(args),
        )
    except BootstrapError as error:
        print(f"Seed Fiddle bootstrap error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
