import glob
import os
import time

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    ScaleIntensityRanged,
)

from utils.config_utils import get_args, load_config
from utils.logger_utils import Logger

config = load_config(get_args().config)
current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(
    logger_name="SpleenSeg-Unet(MONAI)",
    log_file=f"output/logs/SpleenSeg_UNet_MONAI_evaluate_{current_time}.log",
)
logger = log_manager.get_logger()
logger.info("")
logger.info("=== Pre-flight Checklist ===")

try:
    import torch_directml

    device = torch_directml.device()
except ImportError as e:
    torch_directml = None
    logger.error(f"[ERROR] Failed to load DirectML! Reason: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

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
logger.info(
    f"[INFO] MONAI UNet Initialized. Total params: {sum(p.numel() for p in model.parameters()):,}"
)
model.load_state_dict(
    torch.load(
        config.paths.weight_path.replace(".pth", "_monai.pth"), map_location=device
    )
)
logger.info(f"[INFO] Model weight path is {config.paths.weight_path}")
model.eval()

# ==========================================
# 2. 定义【测试集】的读取流水线
# ==========================================
# 自动搜索测试集下的图片和标签
test_images = sorted(glob.glob(os.path.join(config.paths.test_image_dir, "*.nii.gz")))
test_labels = sorted(glob.glob(os.path.join(config.paths.test_label_dir, "*.nii.gz")))
test_files = [
    {"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)
]

# 测试集预处理：只做最基础的读取和归一化，绝对不能做 RandCrop！我们要完整的 3D 图！
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # 增加了channel维度，至此有四个维度
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
)

test_ds = Dataset(data=test_files, transform=test_transforms)
# batch_size=1，因为每个病人的 3D 层数不一样，只能一个一个测，至此增加到3D图像需要的五个维度
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
logger.info(f"[INFO] Dataset loaded. Test: {len(test_files)} files")
logger.info(f"[INFO] Dataloaders ready. Train Batch Size: {test_loader.batch_size}")

# ==========================================
# 3. 定义【后处理流水线】与【官方指标计算器】
# ==========================================
post_transforms = Compose(
    [
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
        # 消除孤立噪点，只保留最大的脾脏
        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
    ]
)

# 官方的 Dice 计算器 (include_background=False 代表不把大面积的黑背景算进准确率里)
dice_metric = DiceMetric(include_background=False, reduction="mean")
logger.info("============================")

# ==========================================
# 4. 正式进入考试模式
# ==========================================
logger.info("")
logger.info("==== Evaluation Started ===✈")
start_time = time.time()
with torch.no_grad():
    # 进入单个 3D 文件的循环
    for i, batch_data in enumerate(test_loader):
        # 此时的 test_inputs 是完整的 3D 矩阵！形状：[1, 1, 512, 512, 120]
        batch_data["image"] = batch_data["image"].to(device)
        batch_data["label"] = batch_data["label"].to(device)

        # 滑动窗口推理
        def predictor_wrapper(x):
            # 进去前，去掉最后多余的 Z 轴维度: [4, 1, 512, 512, 1] -> [4, 1, 512, 512]
            x_2d = x.squeeze(-1)
            # 2D 网络预测
            out_2d = model(x_2d)
            # 出来后，把 Z 轴维度补回来，还给 MONAI 去拼 3D: [4, 1, 512, 512] -> [4, 1, 512, 512, 1]
            return out_2d.unsqueeze(-1)

        # sw_batch_size=4 表示它每次会切 4 层一起送进显卡，极大加快速度！
        batch_data["pred"] = sliding_window_inference(
            inputs=batch_data["image"],
            roi_size=(512, 512, 1),
            sw_batch_size=4,
            predictor=predictor_wrapper,
        )

        # 固定用法，解包并执行后处理
        post_data = [post_transforms(i) for i in decollate_batch(batch_data)]

        # 把处理完的预测结果和真实标签，送给 Dice 计算器
        # 注意我们要取出来，变成 list of Tensors
        val_outputs = [d["pred"] for d in post_data]
        val_labels = [d["label"] for d in post_data]
        dice_metric(y_pred=val_outputs, y=val_labels)

        # 打印单个病人的进度
        # dice_metric.get_buffer() 会记录每次的得分
        current_patient_dice = dice_metric.get_buffer()[-1][0].item()
        print(
            f"[{i + 1}/{len(test_files)}] {os.path.basename(test_files[i]['image'])} | 3D Dice: {current_patient_dice * 100:.2f}%"
        )

# ==========================================
# 5. 提交成绩单
# ==========================================
# aggregate() 会自动把所有病人的得分求平均
final_dice = dice_metric.aggregate().item()
# 算完后必须清空计算器，准备下次使用 (虽然这里脚本结束了，但这是好习惯)
dice_metric.reset()

total_time = time.time() - start_time
logger.info("")
logger.info("======= Evaluation Summary (MONAI) ======")
logger.info(
    f"Average 3D Dice : {final_dice * 100:.2f}% | Time : {int(total_time // 60)}m {int(total_time % 60):02d}s"
)
logger.info("=========================================")
