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


def get_monai_dataloaders(data_root="./dataset", batch_size=2, num_workers=0):
    # ==========================================
    # 第一步：获取 3D 原始文件路径
    # ==========================================
    # glob.glob：这是一个文件搜索工具。它会自动去 dataset/train/images 目录下，把所有后缀是 .nii.gz 的文件路径全部抓出来，变成一个列表。
    # sorted：这一步极其关键！ 操作系统抓取文件有时是乱序的，如果我们不排序，可能取到了“病人A的CT”，对应的却是“病人B的标签”，那就彻底串号了。用 sorted 保证名字排得整整齐齐。
    train_images = sorted(
        glob.glob(os.path.join(data_root, "train", "images", "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(os.path.join(data_root, "train", "labels", "*.nii.gz"))
    )
    # 这是 MONAI 的“灵魂数据结构”。zip 把排好序的图片路径和标签路径两两配对。然后我们用一个 for 循环，把它们包进一个字典里。
    # train_files 变成了一个长长的列表，里面全是这种东西：
    # [ {"image": "路径/img_1.nii.gz", "label": "路径/lab_1.nii.gz"}, {"image": "...", "label": "..."} ]
    train_files = [
        {"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)
    ]

    val_images = sorted(glob.glob(os.path.join(data_root, "val", "images", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_root, "val", "labels", "*.nii.gz")))
    val_files = [
        {"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)
    ]

    # ==========================================
    # 第二步：定义 2D 字典变换流水线
    # ==========================================
    # Compose：像一条工厂流水线，把下面的工序串联起来。数据进去后，会按顺序经过每一道工序。
    train_transforms = Compose(
        [
            # 1. 加载 3D 图像
            # 在 MONAI 里，带 d（如 LoadImaged）全称是 Dictionary Transforms（字典变换）。
            # 它意味着：“请去字典里找到 image 和 label 这两把钥匙，把它们对应的路径打开，读取里面的内容”。
            LoadImaged(keys=["image", "label"]),
            # 2. 增加通道维度 [C, X, Y, Z]
            # 在最前面强行塞入一个维度 1
            EnsureChannelFirstd(keys=["image", "label"]),
            # 3. 窗宽窗位归一化（原 windowing 逻辑）
            # 把HU值160-240的转换为0-1，clip=True：截断。如果骨头亮度是 1000，超出 240 了，强行变成 240
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-160.0,
                a_max=240.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 4. 【核心降维魔法】从 3D 体数据中，随机抽取 2D 切片
            # pos=1, neg=1：它会在120层里上下扫描，保证找出来的切片，包含脾脏的（正）和纯背景的（负）比例是 1比1。
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(512, 512, 1),  # X和Y保留512，Z轴只切1层！
                pos=1,
                neg=1,
                num_samples=4,  # 一个 3D 病人随机抽 4 张切片
                image_key="image",
                image_threshold=0,
            ),
            # 5. 【清理维度】把多余的 Z 轴挤压掉
            # 形状完美变成了标准的 2D 张量：[1, 512, 512]
            # 等dataloader再给前面打一个batch 1，就变成pytorch需要的[ 1, 1, 512, 512]了
            SqueezeDimd(keys=["image", "label"], dim=-1),
            # 6. 转为 PyTorch Tensor
            # 前面的操作，数据有的是 Numpy 数组，有的是 MONAI 特有的 MetaTensor。这一步统一将它们转换为 PyTorch 认识的标准 torch.Tensor。
            ToTensord(keys=["image", "label"]),
        ]
    )

    # 验证集也需要抽切片，但我们去掉随机性，或者直接固定抽几层
    # 为了简单对齐原版，我们验证集也做一样的切片抽取测试
    val_transforms = Compose(
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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(512, 512, 1),
                pos=1,
                neg=1,
                num_samples=10,
                image_key="image",
                image_threshold=0,
            ),
            SqueezeDimd(keys=["image", "label"], dim=-1),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # ==========================================
    # 第三步：装载进缓存数据集
    # ==========================================

    cache_dir = os.path.join(data_root, "monai_persistent_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, val_loader, len(train_files), len(val_files)
