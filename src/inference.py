from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNet

from src.degradations import apply_degradation
from src.io_utils import make_rng
from src.preprocessing import apply_preprocessing


def resolve_device(requested_device, logger=None):
    requested_device = str(requested_device or "cuda").lower()
    if requested_device in {"directml", "dml"}:
        try:
            import torch_directml

            return torch_directml.device()
        except ImportError:
            if logger:
                logger.warning("DirectML requested but torch_directml is not installed; falling back to CPU.")
            return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested_device)


def build_monai_unet(model_config, device):
    channels = tuple(int(v) for v in model_config.get("channels", [64, 128, 256, 512, 1024]))
    strides = tuple(int(v) for v in model_config.get("strides", [2, 2, 2, 2]))
    model = UNet(
        spatial_dims=2,
        in_channels=int(model_config.get("input_channels", 1)),
        out_channels=int(model_config.get("output_channels", 1)),
        channels=channels,
        strides=strides,
        num_res_units=int(model_config.get("num_res_units", 2)),
    )
    return model.to(device)


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned_state_dict = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def predict_volume_slicewise(
    model,
    image_volume,
    degradation_name,
    strength_value,
    preprocessing_name,
    config,
    device,
    case_id,
    condition_key,
):
    """Run 2D model inference slice by slice and rebuild a 3D prediction volume."""
    image_volume = np.asarray(image_volume, dtype=np.float32)
    if image_volume.ndim != 3:
        raise ValueError(f"Expected a 3D image volume, got shape {image_volume.shape}.")

    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    seed = int(config.get("random", {}).get("seed", 42))
    pred_volume = np.zeros(image_volume.shape, dtype=np.uint8)

    with torch.no_grad():
        for slice_index in range(image_volume.shape[2]):
            slice_2d = image_volume[:, :, slice_index]
            rng = make_rng(seed, condition_key, case_id, slice_index)
            degraded = apply_degradation(
                slice_2d,
                degradation_name=degradation_name,
                strength_value=strength_value,
                config=config,
                rng=rng,
            )
            preprocessed = apply_preprocessing(
                degraded,
                preprocess_name=preprocessing_name,
                config=config,
            )

            input_tensor = torch.from_numpy(preprocessed[np.newaxis, np.newaxis, ...]).to(
                device=device, dtype=torch.float32
            )
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).to(torch.uint8).squeeze().cpu().numpy()
            pred_volume[:, :, slice_index] = mask

    return pred_volume

