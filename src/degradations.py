import numpy as np
from monai.transforms import GaussianSmooth


SUPPORTED_DEGRADATIONS = {
    "clean",
    "gaussian_noise",
    "salt_pepper_noise",
    "gaussian_blur",
}


def _normalized_range(config):
    value_range = config.get("data", {}).get("normalized_range", [0.0, 1.0])
    if len(value_range) != 2:
        raise ValueError("data.normalized_range must contain exactly two values.")
    return float(value_range[0]), float(value_range[1])


def _clip(image, config):
    clip_min, clip_max = _normalized_range(config)
    return np.clip(image, clip_min, clip_max).astype(np.float32, copy=False)


def _gaussian_smooth_2d(image, sigma):
    smoothed = GaussianSmooth(sigma=float(sigma))(image[np.newaxis, ...])
    return np.asarray(smoothed[0], dtype=np.float32)


def apply_degradation(image, degradation_name, strength_value, config, rng=None):
    """Apply a configured image degradation to one normalized 2D slice.

    The current MONAI pipeline normalizes CT intensities to [0, 1] after
    windowing [-160, 240]. Degraded slices are clipped back to that configured
    normalized range before inference.
    """
    if degradation_name not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"Unsupported degradation: {degradation_name}")

    image = np.asarray(image, dtype=np.float32)
    if degradation_name == "clean":
        return _clip(image, config)

    if rng is None:
        rng = np.random.default_rng()

    clip_min, clip_max = _normalized_range(config)

    if degradation_name == "gaussian_noise":
        std = float(strength_value)
        noise = rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
        return _clip(image + noise, config)

    if degradation_name == "salt_pepper_noise":
        amount = float(strength_value)
        if not 0.0 <= amount <= 1.0:
            raise ValueError("salt_pepper_noise amount must be between 0 and 1.")
        degraded = image.copy()
        noisy_mask = rng.random(image.shape) < amount
        salt_mask = rng.random(image.shape) < 0.5
        degraded[noisy_mask & salt_mask] = clip_max
        degraded[noisy_mask & ~salt_mask] = clip_min
        return _clip(degraded, config)

    if degradation_name == "gaussian_blur":
        # Gaussian blur is an image degradation, not a noise process.
        return _clip(_gaussian_smooth_2d(image, float(strength_value)), config)

    raise ValueError(f"Unsupported degradation: {degradation_name}")

