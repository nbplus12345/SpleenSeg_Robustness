import os
import random
import shutil
import sys
from pathlib import Path


def split_medical_dataset(
    data_root="../dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    # 1. 指定原始路径
    raw_images_dir = os.path.join(data_root, "imagesTr")
    raw_labels_dir = os.path.join(data_root, "labelsTr")

    if not os.path.exists(raw_images_dir) or not os.path.exists(raw_labels_dir):
        print(
            "\n[ERROR] Source directories missing! Are you sure the 'dataset' folder is set up correctly?"
        )
        sys.exit(1)

    # 获取所有有效的 .nii.gz 文件名（自动过滤掉 ._ 垃圾文件）
    all_cases = [
        f
        for f in os.listdir(raw_images_dir)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ]

    if len(all_cases) == 0:
        print("\n[ERROR] Found 0 files! The source folder is completely empty.")
        print("[HINT] Please put the raw .nii.gz files into dataset/imagesTr/ first.")
        sys.exit(1)  # 强行终止程序

    elif len(all_cases) != 41:
        print(
            f"\n[ERROR] Data integrity failed! Expected 41 cases, but found {len(all_cases)}."
        )
        print(
            "[HINT] Please check if the dataset was downloaded and extracted completely."
        )
        sys.exit(1)  # 强行终止程序

    else:
        print(f"\n[INFO] Total valid cases found: {len(all_cases)}.")

    # 2. 排序并随机打乱 (固定种子 42 )
    all_cases.sort()
    random.seed(42)
    random.shuffle(all_cases)

    # 3. 计算切分点
    num_total = len(all_cases)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    splits = {
        "train": all_cases[:num_train],
        "val": all_cases[num_train : num_train + num_val],
        "test": all_cases[num_train + num_val :],
    }

    # 4. 物理分发文件
    for split_name, file_list in splits.items():
        # 创建 dataset/train/images 和 dataset/train/labels 这种结构
        target_img_dir = Path(data_root) / split_name / "images"
        target_lab_dir = Path(data_root) / split_name / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lab_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PROCESS] Moving {len(file_list)} cases to 【{split_name}】 set...")

        for file_name in file_list:
            # Move Image
            src_img = os.path.join(raw_images_dir, file_name)
            dst_img = target_img_dir / file_name
            shutil.move(src_img, dst_img)  # Physical cut and paste

            # Move corresponding Label
            src_lab = os.path.join(raw_labels_dir, file_name)
            dst_lab = target_lab_dir / file_name
            if os.path.exists(src_lab):
                shutil.move(src_lab, dst_lab)
            else:
                print(f"[WARN] Label missing for {file_name}!")
    print("\n=====================================✈")
    print(
        f"[SUMMARY] Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}"
    )
    print("Original imagesTr and labelsTr folders are now empty/cleared.")
    print("======================================")


if __name__ == "__main__":
    split_medical_dataset()
