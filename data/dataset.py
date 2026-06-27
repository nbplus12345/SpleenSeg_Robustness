import glob
import os

from monai.data import DataLoader, PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToTensord,
)


def _paired_case_files(data_root, split_name):
    """Collect sorted image-label pairs for one dataset split."""
    images = sorted(glob.glob(os.path.join(data_root, split_name, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_root, split_name, "labels", "*.nii.gz")))
    return [{"image": image, "label": label} for image, label in zip(images, labels)]


def _slice_sampling_transforms(num_samples):
    """Build the 3D-to-2D training transform used by the 2D segmentation model."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-160.0,
                a_max=240.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Sample axial 2D slices from each volume while balancing foreground
            # and background examples around the spleen label.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(512, 512, 1),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            SqueezeDimd(keys=["image", "label"], dim=-1),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_dataloaders(data_root="./dataset", batch_size=2, num_workers=0):
    """Create training and validation dataloaders for sampled 2D slices.

    The source data remains 3D NIfTI. The transform samples 2D axial slices
    during loading so the network can be trained as a 2D model while preserving
    case-level data organization on disk.
    """
    train_files = _paired_case_files(data_root, "train")
    val_files = _paired_case_files(data_root, "val")

    train_transforms = _slice_sampling_transforms(num_samples=4)
    val_transforms = _slice_sampling_transforms(num_samples=10)

    cache_dir = os.path.join(data_root, "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, len(train_files), len(val_files)
