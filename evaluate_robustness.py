from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PER_CASE_FIELDS = [
    "case_id",
    "degradation",
    "strength_name",
    "strength_value",
    "preprocessing",
    "dice_3d",
    "hd95_3d",
    "spacing",
    "empty_pred",
    "empty_label",
    "prediction_path",
]

SUMMARY_FIELDS = [
    "degradation",
    "strength_name",
    "strength_value",
    "preprocessing",
    "num_cases",
    "mean_dice_3d",
    "std_dice_3d",
    "mean_hd95_3d",
    "std_hd95_3d",
    "num_valid_hd95",
    "num_empty_pred",
    "num_empty_label",
]

DEGRADATION_STRENGTH_KEYS = {
    "gaussian_noise": "std",
    "salt_pepper_noise": "amount",
    "gaussian_blur": "sigma",
}


@dataclass(frozen=True)
class ExperimentCondition:
    """Single robustness condition evaluated across all test cases."""

    degradation: str
    strength_name: str
    strength_value: float | None
    preprocessing: str

    @property
    def output_name(self):
        """Create a stable directory-safe name for condition-specific outputs."""
        if self.degradation == "clean":
            return f"clean_{self.preprocessing}"
        value = _format_strength_value(self.strength_value)
        return f"{self.degradation}_{self.strength_name}_{value}_{self.preprocessing}"


def _format_strength_value(value):
    """Format numeric strengths consistently for condition names."""
    if value is None:
        return ""
    return f"{float(value):g}"


def _csv_value(value):
    """Normalize Python values before writing them to CSV files."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _validate_preprocessing(config, preprocessing):
    """Check that a requested preprocessing method is declared in the config."""
    methods = config.get("preprocessing", {}).get("methods", [])
    if preprocessing not in methods:
        raise ValueError(
            f"Unsupported preprocessing '{preprocessing}'. "
            f"Configured methods: {methods or 'none'}."
        )


def build_experiment_matrix(config):
    """Build the full robustness matrix from enabled config sections."""
    conditions = []
    preprocessing_methods = config.get("preprocessing", {}).get("methods", [])
    if not preprocessing_methods:
        raise ValueError("preprocessing.methods must contain at least one method.")

    evaluation_config = config.get("evaluation", {})
    if evaluation_config.get("include_clean_baseline", True):
        clean_preprocessing = evaluation_config.get("clean_preprocessing", ["none"])
        for preprocessing in clean_preprocessing:
            _validate_preprocessing(config, preprocessing)
            conditions.append(
                ExperimentCondition(
                    degradation="clean",
                    strength_name="none",
                    strength_value=None,
                    preprocessing=preprocessing,
                )
            )

    degradations_config = config.get("degradations", {})
    for degradation, strength_name in DEGRADATION_STRENGTH_KEYS.items():
        degradation_config = degradations_config.get(degradation, {})
        if not degradation_config.get("enabled", False):
            continue

        strengths = degradation_config.get(strength_name, [])
        if not strengths:
            raise ValueError(f"degradations.{degradation}.{strength_name} must not be empty.")

        for strength in strengths:
            for preprocessing in preprocessing_methods:
                _validate_preprocessing(config, preprocessing)
                conditions.append(
                    ExperimentCondition(
                        degradation=degradation,
                        strength_name=strength_name,
                        strength_value=float(strength),
                        preprocessing=preprocessing,
                    )
                )

    return conditions


def build_single_condition(config, degradation, strength, preprocessing):
    """Build and validate a one-condition evaluation request from CLI arguments."""
    if not degradation or not preprocessing:
        raise ValueError(
            "Single-condition mode requires --degradation and --preprocess. "
            "Use --run_all to run the full matrix."
        )

    _validate_preprocessing(config, preprocessing)
    if degradation == "clean":
        if strength is not None:
            raise ValueError("Clean condition does not accept --strength.")
        return [
            ExperimentCondition(
                degradation="clean",
                strength_name="none",
                strength_value=None,
                preprocessing=preprocessing,
            )
        ]

    if degradation not in DEGRADATION_STRENGTH_KEYS:
        choices = ["clean", *DEGRADATION_STRENGTH_KEYS.keys()]
        raise ValueError(f"Unsupported degradation '{degradation}'. Choices: {choices}.")
    if strength is None:
        raise ValueError(f"--strength is required for degradation '{degradation}'.")

    strength_name = DEGRADATION_STRENGTH_KEYS[degradation]
    return [
        ExperimentCondition(
            degradation=degradation,
            strength_name=strength_name,
            strength_value=float(strength),
            preprocessing=preprocessing,
        )
    ]


def select_conditions(config, args):
    """Select either the full matrix or a single condition based on CLI mode."""
    if args.run_all:
        if args.degradation or args.strength is not None or args.preprocess:
            raise ValueError("--run_all cannot be combined with single-condition arguments.")
        return build_experiment_matrix(config)

    return build_single_condition(
        config=config,
        degradation=args.degradation,
        strength=args.strength,
        preprocessing=args.preprocess,
    )


def summarize_condition(condition, rows):
    """Aggregate per-case metrics for one experimental condition."""
    import numpy as np

    dice_values = np.asarray([float(row["dice_3d"]) for row in rows], dtype=float)
    hd95_values = np.asarray([float(row["hd95_3d"]) for row in rows], dtype=float)
    valid_hd95 = hd95_values[~np.isnan(hd95_values)]

    return {
        "degradation": condition.degradation,
        "strength_name": condition.strength_name,
        "strength_value": _csv_value(condition.strength_value),
        "preprocessing": condition.preprocessing,
        "num_cases": len(rows),
        "mean_dice_3d": _csv_value(float(np.mean(dice_values)) if dice_values.size else float("nan")),
        "std_dice_3d": _csv_value(float(np.std(dice_values)) if dice_values.size else float("nan")),
        "mean_hd95_3d": _csv_value(float(np.nanmean(hd95_values)) if valid_hd95.size else float("nan")),
        "std_hd95_3d": _csv_value(float(np.nanstd(hd95_values)) if valid_hd95.size else float("nan")),
        "num_valid_hd95": int(valid_hd95.size),
        "num_empty_pred": sum(str(row["empty_pred"]).lower() == "true" for row in rows),
        "num_empty_label": sum(str(row["empty_label"]).lower() == "true" for row in rows),
    }


def _prediction_filename(case_id, prediction_format):
    """Return the configured prediction-mask filename for one case."""
    prediction_format = str(prediction_format or "nii.gz").lstrip(".")
    return f"{case_id}_pred.{prediction_format}"


def run_evaluation(config, config_path, conditions):
    """Run model loading, 3D case inference, metric calculation, and CSV output."""
    import numpy as np
    import torch

    from src.inference import build_unet, load_model_checkpoint, predict_volume_slicewise, resolve_device
    from src.io_utils import (
        ensure_output_dir,
        find_test_cases,
        format_spacing,
        load_case,
        normalize_image_volume,
        resolve_path,
        save_prediction_mask,
        setup_logger,
        should_save_prediction,
        write_csv,
    )
    from src.metrics import compute_3d_dice, compute_3d_hd95, empty_mask_flags

    output_dir = ensure_output_dir(resolve_path(config["output"]["output_dir"], PROJECT_ROOT))
    logger = setup_logger(output_dir / "logs" / "robustness_eval.log")
    logger.info("=== Robustness Evaluation Started ===")
    logger.info(f"Config: {Path(config_path).resolve()}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Experiment conditions: {len(conditions)}")

    # Seed both NumPy and PyTorch for deterministic degradation sampling and
    # reproducible deterministic model operators where the backend supports it.
    seed = int(config.get("random", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    case_files = find_test_cases(config["data"], PROJECT_ROOT)
    logger.info(f"Loaded test case list: {len(case_files)} cases")

    device = resolve_device(config.get("model", {}).get("device", "cuda"), logger=logger)
    checkpoint_path = resolve_path(config["paths"]["model_weights_path"], PROJECT_ROOT)
    model = build_unet(config.get("model", {}), device=device)
    model = load_model_checkpoint(model, checkpoint_path=checkpoint_path, device=device)
    logger.info(f"Model checkpoint loaded: {checkpoint_path}")

    all_per_case_rows = []
    summary_rows = []
    use_spacing_for_hd95 = bool(config.get("data", {}).get("use_spacing_for_hd95", True))
    save_predictions = bool(config.get("output", {}).get("save_predictions", False))
    save_prediction_cases = config.get("output", {}).get("save_prediction_cases", "all")
    prediction_format = config.get("output", {}).get("prediction_format", "nii.gz")

    start_time = time.time()
    for condition in conditions:
        logger.info("")
        logger.info(
            "Running condition: degradation=%s strength=%s=%s preprocessing=%s output=%s",
            condition.degradation,
            condition.strength_name,
            _csv_value(condition.strength_value),
            condition.preprocessing,
            condition.output_name,
        )

        condition_rows = []
        for case_index, case_file in enumerate(case_files, start=1):
            logger.info("[%d/%d] case_id=%s", case_index, len(case_files), case_file.case_id)
            loaded_case = load_case(case_file)
            normalized_image = normalize_image_volume(loaded_case.image, config["data"])

            spacing = loaded_case.spacing if use_spacing_for_hd95 else None
            if spacing is None:
                logger.warning(
                    "HD95 spacing unavailable or disabled for %s; using voxel spacing (1,1,1).",
                    loaded_case.case_id,
                )

            # Degradation and filtering are applied inside the slicewise
            # predictor so that each 2D input follows the configured test-time
            # corruption and preprocessing sequence before model inference.
            pred_volume = predict_volume_slicewise(
                model=model,
                image_volume=normalized_image,
                degradation_name=condition.degradation,
                strength_value=condition.strength_value,
                preprocessing_name=condition.preprocessing,
                config=config,
                device=device,
                case_id=loaded_case.case_id,
                condition_key=condition.output_name,
            )

            dice_3d = compute_3d_dice(pred_volume, loaded_case.label)
            hd95_3d = compute_3d_hd95(pred_volume, loaded_case.label, spacing=spacing)
            empty_pred, empty_label = empty_mask_flags(pred_volume, loaded_case.label)

            prediction_path = ""
            if save_predictions and should_save_prediction(loaded_case.case_id, save_prediction_cases):
                prediction_path = (
                    output_dir
                    / "predictions"
                    / condition.output_name
                    / _prediction_filename(loaded_case.case_id, prediction_format)
                )
                save_prediction_mask(
                    pred_volume=pred_volume,
                    output_path=prediction_path,
                    affine=loaded_case.affine,
                    header=loaded_case.header,
                )
                logger.info("Prediction saved: %s", prediction_path)

            row = {
                "case_id": loaded_case.case_id,
                "degradation": condition.degradation,
                "strength_name": condition.strength_name,
                "strength_value": _csv_value(condition.strength_value),
                "preprocessing": condition.preprocessing,
                "dice_3d": _csv_value(dice_3d),
                "hd95_3d": _csv_value(hd95_3d),
                "spacing": format_spacing(spacing if spacing is not None else (1.0, 1.0, 1.0)),
                "empty_pred": bool(empty_pred),
                "empty_label": bool(empty_label),
                "prediction_path": str(prediction_path),
            }
            condition_rows.append(row)
            all_per_case_rows.append(row)
            logger.info(
                "Metrics case_id=%s dice_3d=%s hd95_3d=%s empty_pred=%s empty_label=%s",
                loaded_case.case_id,
                row["dice_3d"],
                row["hd95_3d"],
                row["empty_pred"],
                row["empty_label"],
            )

        summary_rows.append(summarize_condition(condition, condition_rows))
        write_csv(all_per_case_rows, output_dir / "per_case_results.csv", PER_CASE_FIELDS)
        write_csv(summary_rows, output_dir / "summary_results.csv", SUMMARY_FIELDS)
        logger.info("CSV updated: %s", output_dir / "per_case_results.csv")
        logger.info("CSV updated: %s", output_dir / "summary_results.csv")

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=== Robustness Evaluation Finished in %.2f seconds ===", elapsed)
    return output_dir


def print_conditions(conditions):
    """Print selected conditions for dry-run validation."""
    for index, condition in enumerate(conditions, start=1):
        print(
            f"{index:02d}. degradation={condition.degradation} "
            f"strength={condition.strength_name}={_csv_value(condition.strength_value)} "
            f"preprocessing={condition.preprocessing} "
            f"output={condition.output_name}"
        )


def parse_args():
    """Define the robustness-evaluation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net spleen CT segmentation robustness."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to robustness YAML config.",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run the full robustness experiment matrix.",
    )
    parser.add_argument(
        "--degradation",
        type=str,
        help="Single condition degradation: clean, gaussian_noise, salt_pepper_noise, or gaussian_blur.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        help="Single condition degradation strength. Omit for clean.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        help="Single condition preprocessing method.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config and print selected experiment conditions without reading data or checkpoints.",
    )
    return parser.parse_args()


def main():
    """CLI entry point."""
    from src.io_utils import load_yaml_config

    args = parse_args()
    try:
        config = load_yaml_config(args.config)
        conditions = select_conditions(config, args)
        if args.dry_run:
            print(f"Selected {len(conditions)} experiment condition(s):")
            print_conditions(conditions)
            return 0
        run_evaluation(config, args.config, conditions)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
