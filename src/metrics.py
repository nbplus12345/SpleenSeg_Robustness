import numpy as np


def compute_3d_dice(pred_volume, label_volume):
    """Compute foreground-only Dice for binary spleen segmentation volumes."""
    pred = np.asarray(pred_volume) > 0
    label = np.asarray(label_volume) > 0

    pred_sum = int(pred.sum())
    label_sum = int(label.sum())
    denominator = pred_sum + label_sum
    if denominator == 0:
        # Empty/empty is treated as a perfect foreground agreement.
        return 1.0

    intersection = int(np.logical_and(pred, label).sum())
    return float((2.0 * intersection) / denominator)


def compute_3d_hd95(pred_volume, label_volume, spacing=None):
    """Compute 95th percentile symmetric surface distance for 3D masks.

    Empty handling is explicit:
    - pred empty and label empty: return 0.0.
    - exactly one side empty: return NaN so summaries do not hide failures.
    """
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "HD95 requires scipy. Install project requirements or add scipy to the active environment."
        ) from exc

    pred = np.asarray(pred_volume) > 0
    label = np.asarray(label_volume) > 0

    empty_pred = not bool(pred.any())
    empty_label = not bool(label.any())
    if empty_pred and empty_label:
        return 0.0
    if empty_pred or empty_label:
        return float("nan")

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing = tuple(float(v) for v in spacing)
    if len(spacing) != pred.ndim:
        raise ValueError(f"spacing must have {pred.ndim} values, got {spacing}.")

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = np.logical_xor(
        pred, binary_erosion(pred, structure=structure, border_value=0)
    )
    label_surface = np.logical_xor(
        label, binary_erosion(label, structure=structure, border_value=0)
    )

    distance_to_label = distance_transform_edt(~label_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_label = distance_to_label[pred_surface]
    label_to_pred = distance_to_pred[label_surface]
    distances = np.concatenate([pred_to_label, label_to_pred])
    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))


def empty_mask_flags(pred_volume, label_volume):
    """Return foreground-empty flags used for HD95 interpretation and summaries."""
    pred = np.asarray(pred_volume) > 0
    label = np.asarray(label_volume) > 0
    return not bool(pred.any()), not bool(label.any())
