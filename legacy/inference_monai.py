import time

import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityRanged,
)

from utils.config_utils import get_args, load_config

args = get_args()
config = load_config(args.config)
print("=== Pre-flight Checklist ===")

try:
    import torch_directml

    device = torch_directml.device()
except ImportError as e:
    torch_directml = None
    print(f"[ERROR] Failed to load DirectML! Reason: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device set to: {device}")

# ==========================================
# 1. 召唤模型并加载权重
# ==========================================
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# 读取我们用 MONAI 跑出来的新权重
model.load_state_dict(
    torch.load(
        config.paths.weight_path.replace(".pth", "_monai.pth"), map_location=device
    )
)
model.eval()

# ==========================================
# 2. 定义【数据读取】流水线
# ==========================================
# 我们把待预测的文件包装成字典
test_data = [{"image": config.paths.CT_dir}]

pre_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # 窗宽窗位
        ScaleIntensityRanged(
            keys=["image"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
)

print(f"[INFO] Loading CT: {config.paths.CT_dir}")
# 执行预处理
data = pre_transforms(test_data[0])

c, w, h, z_slices = data["image"].shape
print(f"[INFO] Image Shape: [ {w} , {h} , {z_slices} ]")

# 创建一个全零的 Tensor 用来装预测结果
pred_3d = torch.zeros_like(data["image"])

# ==========================================
# 3. 逐层推理核心逻辑
# ==========================================
# (因为是2D网络且显存有限，我们保留Z轴循环，防止整个 3D 体素直接塞爆显存)
start_time = time.time()
with torch.no_grad():
    for z in range(z_slices):
        # 取出一层，加上 Batch 维度 -> [1, 1, 512, 512]
        slice_2d = data["image"][:, :, :, z].unsqueeze(0).to(device)

        # 模型预测 (出来的是 Raw Logits)
        output = model(slice_2d)

        # 直接把 Logits 塞回 3D 矩阵，去掉 Batch 维度
        pred_3d[:, :, :, z] = output.squeeze(0).cpu()

        if (z + 1) % 20 == 0:
            print(f"  -> Scanned {z + 1}/{z_slices} slices")

print(f"[INFO] Scan complete! Time: {time.time() - start_time:.2f} seconds")

# 把装满 Logits 的预测矩阵放回字典里
data["pred"] = pred_3d

# ==========================================
# 4. 定义【后处理与保存】流水线 (高能预警)
# ==========================================
post_transforms = Compose(
    [
        # 1. 过 Sigmoid 变成概率
        Activationsd(keys=["pred"], sigmoid=True),
        # 2. 0.5 阈值切分变硬标签
        AsDiscreted(keys=["pred"], threshold=0.5),
        # 3. 只保留最大连通域 (完美替代你的 scipy 代码！)
        KeepLargestConnectedComponentd(keys=["pred"], applied_labels=[1]),
        # 4. 自动保存为 .nii.gz，自动复制所有的仿射矩阵和物理信息！
        SaveImaged(
            keys=["pred"],
            meta_keys="image_meta_dict",  # 告诉它去 image_meta_dict 里找物理信息
            output_dir="./output/predictions",
            output_postfix="monai_seg",  # 输出文件名会自动加上这个后缀
            output_ext=".nii.gz",
            resample=False,
        ),
    ]
)

print("[INFO] Starting 3D post-processing and saving...")
# 执行后处理流水线，一切都在后台瞬间完成
post_transforms(data)
print("[INFO] Success! 3D spleen mask saved to ./output/predictions/")
