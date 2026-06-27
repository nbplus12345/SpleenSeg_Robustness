import os
import random
import shutil
import sys
from pathlib import Path


def split_medical_dataset(
    data_root="../dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    raw_images_dir = os.path.join(data_root, "imagesTr")
    raw_labels_dir = os.path.join(data_root, "labelsTr")

    if not os.path.exists(raw_images_dir) or not os.path.exists(raw_labels_dir):
        print("\n[ERROR] Source directories are missing.")
        print("[HINT] Expected dataset/imagesTr and dataset/labelsTr.")
        sys.exit(1)

    all_cases = [
        file_name
        for file_name in os.listdir(raw_images_dir)
        if file_name.endswith(".nii.gz") and not file_name.startswith("._")
    ]

    if len(all_cases) == 0:
        print("\n[ERROR] Found 0 NIfTI files in dataset/imagesTr.")
        sys.exit(1)
    if len(all_cases) != 41:
        print(
            f"\n[ERROR] Data integrity check failed. Expected 41 cases, found {len(all_cases)}."
        )
        sys.exit(1)

    print(f"\n[INFO] Total valid cases found: {len(all_cases)}.")

    all_cases.sort()
    random.seed(42)
    random.shuffle(all_cases)

    num_total = len(all_cases)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    splits = {
        "train": all_cases[:num_train],
        "val": all_cases[num_train : num_train + num_val],
        "test": all_cases[num_train + num_val :],
    }

    for split_name, file_list in splits.items():
        target_img_dir = Path(data_root) / split_name / "images"
        target_lab_dir = Path(data_root) / split_name / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lab_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PROCESS] Moving {len(file_list)} cases to the {split_name} split.")

        for file_name in file_list:
            src_img = os.path.join(raw_images_dir, file_name)
            dst_img = target_img_dir / file_name
            shutil.move(src_img, dst_img)

            src_lab = os.path.join(raw_labels_dir, file_name)
            dst_lab = target_lab_dir / file_name
            if os.path.exists(src_lab):
                shutil.move(src_lab, dst_lab)
            else:
                print(f"[WARN] Label missing for {file_name}.")

    print("\n=====================================")
    print(
        f"[SUMMARY] Train: {len(splits['train'])} | "
        f"Val: {len(splits['val'])} | Test: {len(splits['test'])}"
    )
    print("Original imagesTr and labelsTr folders are now empty or partially depleted.")
    print("=====================================")


if __name__ == "__main__":
    split_medical_dataset()

