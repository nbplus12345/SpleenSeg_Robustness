import os
import sys
import time

import torch
from monai.losses import DiceLoss

# ==========================================
# 导入 MONAI 官方组件与新的 Dataset
# ==========================================
from monai.networks.nets import UNet
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset_monai import get_monai_dataloaders
from utils.config_utils import get_args, load_config
from utils.logger_utils import Logger

config = load_config(get_args().config)
current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(
    logger_name="SpleenSeg_MONAI",
    log_file=f"output/logs/MONAI_train_{current_time}.log",
)
logger = log_manager.get_logger()
TQDM_BASE_CONFIG = {
    "file": sys.stdout,
    "colour": "white",
    "disable": not sys.stdout.isatty(),
    "leave": False,
    "dynamic_ncols": True,
}
logger.info("")
logger.info("=== Pre-flight Checklist (MONAI Pipeline) ===✈")

# 1. 硬件配置
try:
    import torch_directml

    device = torch_directml.device()
except ImportError:
    torch_directml = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

# ==========================================
# 2. MONAI 数据处理
# ==========================================
batch_size = config.train.batch_size
train_loader, val_loader, num_train_dataset, num_val_dataset = get_monai_dataloaders(
    data_root=config.paths.data_root_dir,
    batch_size=batch_size,
    num_workers=config.train.num_worker,
)
logger.info(
    f"[INFO] Dataset loaded. Train: {num_train_dataset} files | Val: {num_val_dataset} files"
)
logger.info(f"[INFO] Dataloaders ready. Train Batch Size: {batch_size}")

# ==========================================
# 3. 初始化 MONAI 官方 U-Net
# ==========================================
# 对标你原来手搓的网络：1层输入，1层输出，5层下采样 (64 -> 1024)
model = UNet(
    spatial_dims=2,  # 明确告诉它是 2D 网络
    in_channels=1,  # 单通道输入 (黑白CT)
    out_channels=1,  # 单通道输出 (二分类背景与脾脏)
    channels=(64, 128, 256, 512, 1024),  # 各层通道数，完美对齐你的手写版
    strides=(2, 2, 2, 2),  # 每次下采样的步长
    num_res_units=2,  # MONAI 特性：加入残差单元(ResNet思想)，比普通UNet更好收敛
).to(device)
logger.info(
    f"[INFO] MONAI UNet Initialized. Total params: {sum(p.numel() for p in model.parameters()):,}"
)

# ==========================================
# 4. 初始化 MONAI 官方 DiceLoss
# ==========================================
# 参数 sigmoid=True 非常关键！这代表你不需要在网络输出后手动写 torch.sigmoid() 了
# MONAI 会在底层极其安全地计算 Sigmoid，避免梯度爆炸
criterion = DiceLoss(sigmoid=True)
logger.info("[INFO] Criterion: MONAI DiceLoss(sigmoid=True)")
optimizer = optim.SGD(
    model.parameters(),
    lr=config.train.lr,
    momentum=config.train.momentum,
    weight_decay=1e-4,
)
logger.info(
    f"[INFO] Optimizer configured: SGD (lr={config.train.lr}, momentum={config.train.momentum})"
)

# 5. 其他项
# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=config.train.scheduler_patience,
    min_lr=0.002,
)
logger.info("[INFO] Learning Rate Scheduler configured: ReduceLROnPlateau")
# TensorBoard
tb_log_dir = os.path.join("./output", f"tensorboard/MONAI_Board_{current_time}")
board_writer = SummaryWriter(log_dir=tb_log_dir)
global_step = 0
# 早停机制
highest_val_dice = 0.0
patience = config.train.patience
max_epochs = config.train.epochs
logger.info(
    f"[INFO] Early Stopping configured. Patience: {patience} epochs, max_epochs: {max_epochs}"
)
counter = 0
# 断点续训
start_epoch = 0
if config.train.resume_training and os.path.exists(config.paths.checkpoint_path):
    # 1. 把包裹取回来
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    # 2. 依次把记忆注入到对应的身体里
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # 从断点的下一轮开始
    highest_val_dice = checkpoint["highest_val_dice"]
    counter = checkpoint["counter"]
    global_step = checkpoint.get("global_step", 0)
    logger.info(
        f"[INFO] Find Checkpoint {config.paths.checkpoint_path}, Continue training from epoch {start_epoch + 1} ."
    )

logger.info("============================")
logger.info("")
logger.info("\n===== Training Started ====✈\n")
logger.info("====================================================")

# 6. 开始正式循环
for epoch in range(start_epoch, max_epochs):
    # --- 训练阶段 ---
    model.train()
    epoch_start_time = time.time()
    epoch_total_loss = 0.0

    train_pbar = tqdm(train_loader, desc="[Train]", **TQDM_BASE_CONFIG)
    for batch_data in train_pbar:
        global_step += 1
        # 注意取数据的方式：由于送进来的是字典，所以要按键取值
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # 不需要 sigmoid 了！因为 DiceLoss 里设置了 sigmoid=True
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        board_writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        epoch_total_loss += loss.item()

    epoch_avg_train_loss = epoch_total_loss / len(train_loader)
    board_writer.add_scalar("Train/Epoch_Loss", epoch_avg_train_loss, epoch)

    # --- 验证阶段 ---
    if (epoch + 1) % config.train.val_interval == 0:
        model.eval()
        val_total_loss = 0.0
        val_total_intersection = 0.0
        val_total_union = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Val]", **TQDM_BASE_CONFIG)
            for batch_data in val_pbar:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                outputs = model(images)
                val_batch_loss = criterion(outputs, labels)
                val_total_loss += val_batch_loss.item()

                # 手动算 Dice 用于显示进度 (Metric计算时依然需要把预测变成 0 和 1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                intersection = (preds * labels).sum().item()
                union = preds.sum().item() + labels.sum().item()
                val_total_intersection += intersection
                val_total_union += union

        val_avg_loss = val_total_loss / len(val_loader)
        val_avg_dice = (2.0 * val_total_intersection + 1e-5) / (val_total_union + 1e-5)

        board_writer.add_scalar("Val/Epoch_Loss", val_avg_loss, epoch)
        board_writer.add_scalar("Val/Epoch_Dice", val_avg_dice, epoch)

        epoch_time = time.time() - epoch_start_time

        logger.info("--------------------------------------------------------")
        logger.info(
            f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] | Train Loss: {epoch_avg_train_loss:.4f}  | Val Loss: {val_avg_loss:.4f}"
        )
        logger.info(
            f"                | Val Dice  : {val_avg_dice:.4f}  | Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"
        )

        # 把当前 epoch 的验证集 loss 喂给调度器
        scheduler.step(val_avg_loss)
        # 获取当前最新的学习率，打印到日志里，方便你监控
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"                | Learning Rate: {current_lr}")
        logger.info("========================================================")
        # 早停与权重保存逻辑
        if val_avg_dice > highest_val_dice:
            highest_val_dice = val_avg_dice
            counter = 0
            # 我们给权重换个名字，避免覆盖手写版的权重
            torch.save(
                model.state_dict(),
                config.paths.weight_path.replace(".pth", "_monai.pth"),
            )
            logger.info("[SAVE] New best record! MONAI Model saved!")
            logger.info("")
            logger.info("====================================================")
        else:
            counter += 1
            logger.info(f"[WARN] No improvement. Patience: {counter}/{patience}")
            logger.info("")
            logger.info("====================================================")
            if counter >= patience:
                board_writer.close()
                logger.info("[STOP] Early stopping triggered! Training halted.")
                logger.info(
                    "       Final best Val Dice: {:.4f}".format(highest_val_dice)
                )
                logger.info("=================================================")
                break
    else:
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] | Train Loss: {epoch_avg_train_loss:.4f}  | Time: {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"
        )

    # 断点续训
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "highest_val_dice": highest_val_dice,
        "counter": counter,
        "global_step": global_step,
    }
    weight_dir = os.path.dirname(config.paths.weight_path)
    checkpoint_path = os.path.join(weight_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)  # 覆盖保存最新的快照
