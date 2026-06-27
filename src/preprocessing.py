import numpy as np
from monai.transforms import GaussianSmooth, MedianSmooth


SUPPORTED_PREPROCESSING = {"none", "gaussian_filter", "median_filter"}


def _normalized_range(config):
    """Read the expected post-windowing intensity range from the config."""
    value_range = config.get("data", {}).get("normalized_range", [0.0, 1.0])
    if len(value_range) != 2:
        raise ValueError("data.normalized_range must contain exactly two values.")
    return float(value_range[0]), float(value_range[1])


def _clip(image, config):
    """Clamp filtered images to the model's normalized input range."""
    clip_min, clip_max = _normalized_range(config)
    return np.clip(image, clip_min, clip_max).astype(np.float32, copy=False)


def apply_preprocessing(image, preprocess_name, config):
    """Apply optional filtering after degradation and before model inference.

    Filtering parameters are intentionally read from YAML so robustness
    experiments can be changed without editing source code.
    """
    if preprocess_name not in SUPPORTED_PREPROCESSING:
        raise ValueError(f"Unsupported preprocessing method: {preprocess_name}")

    image = np.asarray(image, dtype=np.float32)
    if preprocess_name == "none":
        return _clip(image, config)

    preprocessing_config = config.get("preprocessing", {})

    if preprocess_name == "gaussian_filter":
        sigma = float(preprocessing_config.get("gaussian_filter", {}).get("sigma", 1.0))
        filtered = GaussianSmooth(sigma=sigma)(image[np.newaxis, ...])
        return _clip(np.asarray(filtered[0], dtype=np.float32), config)

    if preprocess_name == "median_filter":
        size = int(preprocessing_config.get("median_filter", {}).get("size", 3))
        if size <= 0 or size % 2 == 0:
            raise ValueError("preprocessing.median_filter.size must be a positive odd integer.")
        radius = size // 2
        filtered = MedianSmooth(radius=radius)(image[np.newaxis, ...])
        return _clip(np.asarray(filtered[0], dtype=np.float32), config)

    raise ValueError(f"Unsupported preprocessing method: {preprocess_name}")
