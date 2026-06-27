import csv
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


@dataclass(frozen=True)
class CaseFile:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass
class LoadedCase:
    case_id: str
    image: np.ndarray
    label: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    spacing: tuple
    image_path: Path
    label_path: Path


def load_yaml_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def setup_logger(log_file):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("robustness_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _strip_suffix(filename, suffix):
    if not filename.endswith(suffix):
        raise ValueError(f"File does not end with expected suffix '{suffix}': {filename}")
    return filename[: -len(suffix)]


def find_test_cases(data_config, project_root):
    images_dir = resolve_path(data_config["test_images_dir"], project_root)
    labels_dir = resolve_path(data_config["test_labels_dir"], project_root)
    image_suffix = data_config.get("image_suffix", ".nii.gz")
    label_suffix = data_config.get("label_suffix", ".nii.gz")

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Test labels directory not found: {labels_dir}")

    image_paths = sorted(p for p in images_dir.iterdir() if p.name.endswith(image_suffix))
    label_paths = sorted(p for p in labels_dir.iterdir() if p.name.endswith(label_suffix))
    if not image_paths:
        raise FileNotFoundError(f"No test images ending with '{image_suffix}' found in {images_dir}")
    if not label_paths:
        raise FileNotFoundError(f"No test labels ending with '{label_suffix}' found in {labels_dir}")

    images_by_case = {_strip_suffix(p.name, image_suffix): p for p in image_paths}
    labels_by_case = {_strip_suffix(p.name, label_suffix): p for p in label_paths}

    missing_labels = sorted(set(images_by_case) - set(labels_by_case))
    missing_images = sorted(set(labels_by_case) - set(images_by_case))
    if missing_labels or missing_images:
        raise ValueError(
            "Image/label case mismatch. "
            f"Missing labels: {missing_labels or 'none'}; "
            f"missing images: {missing_images or 'none'}."
        )

    return [
        CaseFile(case_id=case_id, image_path=images_by_case[case_id], label_path=labels_by_case[case_id])
        for case_id in sorted(images_by_case)
    ]


def load_case(case_file):
    image_nii = nib.load(str(case_file.image_path))
    label_nii = nib.load(str(case_file.label_path))

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)
    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(
            f"Expected 3D image/label for {case_file.case_id}, got {image.shape} and {label.shape}."
        )
    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch for {case_file.case_id}: {image.shape} vs {label.shape}."
        )

    zooms = image_nii.header.get_zooms()
    spacing = tuple(float(v) for v in zooms[:3]) if len(zooms) >= 3 else None
    label = (label > 0).astype(np.uint8)

    return LoadedCase(
        case_id=case_file.case_id,
        image=image,
        label=label,
        affine=image_nii.affine,
        header=image_nii.header.copy(),
        spacing=spacing,
        image_path=case_file.image_path,
        label_path=case_file.label_path,
    )


def normalize_image_volume(image, data_config):
    window = data_config.get("intensity_window", {})
    a_min = float(window.get("a_min", -160.0))
    a_max = float(window.get("a_max", 240.0))
    value_range = data_config.get("normalized_range", [0.0, 1.0])
    b_min = float(window.get("b_min", value_range[0]))
    b_max = float(window.get("b_max", value_range[1]))

    if a_max <= a_min:
        raise ValueError("data.intensity_window.a_max must be greater than a_min.")
    if b_max <= b_min:
        raise ValueError("Normalized output range max must be greater than min.")

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, a_min, a_max)
    image = (image - a_min) / (a_max - a_min)
    image = image * (b_max - b_min) + b_min
    return np.clip(image, b_min, b_max).astype(np.float32, copy=False)


def save_prediction_mask(pred_volume, output_path, affine, header):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_header = header.copy()
    save_header.set_data_dtype(np.uint8)
    nii = nib.Nifti1Image(np.asarray(pred_volume, dtype=np.uint8), affine, save_header)
    nib.save(nii, str(output_path))
    return output_path


def write_csv(rows, output_path, fieldnames):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_spacing(spacing):
    if spacing is None:
        return ""
    return ",".join(f"{float(v):.6g}" for v in spacing)


def make_rng(seed, *parts):
    payload = "|".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_value = int.from_bytes(digest[:8], byteorder="little") % (2**32)
    return np.random.default_rng(seed_value)


def should_save_prediction(case_id, save_prediction_cases):
    if save_prediction_cases == "all":
        return True
    if isinstance(save_prediction_cases, list):
        return case_id in save_prediction_cases
    raise ValueError('output.save_prediction_cases must be "all" or a list of case IDs.')


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

