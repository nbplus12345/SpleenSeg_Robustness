import os
import sys
import time

import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import get_dataloaders
from utils.config_utils import get_args, load_config
from utils.logger_utils import Logger


TQDM_BASE_CONFIG = {
    "file": sys.stdout,
    "colour": "white",
    "disable": not sys.stdout.isatty(),
    "leave": False,
    "dynamic_ncols": True,
}


def get_device():
    """Choose the best available compute backend for local training."""
    try:
        import torch_directml

        return torch_directml.device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config, device):
    """Build the 2D segmentation network from the shared YAML configuration."""
    model_config = config.model
    return UNet(
        spatial_dims=2,
        in_channels=model_config.input_channels,
        out_channels=model_config.output_channels,
        channels=tuple(model_config.channels),
        strides=tuple(model_config.strides),
        num_res_units=model_config.num_res_units,
    ).to(device)


def main():
    """Train the segmentation model and maintain best-weight/checkpoint files."""
    config = load_config(get_args().config)
    current_time = time.strftime("%Y%m%d_%H%M")
    log_manager = Logger(
        logger_name="SpleenSeg",
        log_file=f"output/logs/train_{current_time}.log",
    )
    logger = log_manager.get_logger()

    logger.info("")
    logger.info("=== Training pre-flight checks ===")
    device = get_device()
    logger.info(f"[INFO] Device set to: {device}")

    train_loader, val_loader, num_train_dataset, num_val_dataset = get_dataloaders(
        data_root=config.paths.data_root_dir,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_worker,
    )
    logger.info(
        f"[INFO] Dataset loaded. Train: {num_train_dataset} files | Val: {num_val_dataset} files"
    )
    logger.info(f"[INFO] Dataloader batch size: {config.train.batch_size}")

    model = build_model(config, device)
    logger.info(
        f"[INFO] U-Net initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # The loss applies sigmoid internally, so the training loop keeps model
    # outputs as raw logits until metric calculation.
    criterion = DiceLoss(sigmoid=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.train.lr,
        momentum=config.train.momentum,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=config.train.scheduler_patience,
        min_lr=0.002,
    )
    logger.info("[INFO] Criterion: DiceLoss(sigmoid=True)")
    logger.info(
        f"[INFO] Optimizer: SGD (lr={config.train.lr}, momentum={config.train.momentum})"
    )
    logger.info("[INFO] Scheduler: ReduceLROnPlateau")

    tb_log_dir = os.path.join("./output", f"tensorboard/board_{current_time}")
    board_writer = SummaryWriter(log_dir=tb_log_dir)

    global_step = 0
    highest_val_dice = 0.0
    counter = 0
    start_epoch = 0

    if config.train.resume_training and os.path.exists(config.paths.training_checkpoint_path):
        checkpoint = torch.load(config.paths.training_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        highest_val_dice = checkpoint["highest_val_dice"]
        counter = checkpoint["counter"]
        global_step = checkpoint.get("global_step", 0)
        logger.info(
            f"[INFO] Resuming from {config.paths.training_checkpoint_path} at epoch {start_epoch + 1}."
        )

    logger.info("")
    logger.info("===== Training started =====")

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_total_loss = 0.0

        train_pbar = tqdm(train_loader, desc="[Train]", **TQDM_BASE_CONFIG)
        for batch_data in train_pbar:
            global_step += 1
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            board_writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            epoch_total_loss += loss.item()

        epoch_avg_train_loss = epoch_total_loss / len(train_loader)
        board_writer.add_scalar("Train/Epoch_Loss", epoch_avg_train_loss, epoch)

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
                    val_total_loss += criterion(outputs, labels).item()

                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    val_total_intersection += (preds * labels).sum().item()
                    val_total_union += preds.sum().item() + labels.sum().item()

            val_avg_loss = val_total_loss / len(val_loader)
            val_avg_dice = (2.0 * val_total_intersection + 1e-5) / (
                val_total_union + 1e-5
            )
            board_writer.add_scalar("Val/Epoch_Loss", val_avg_loss, epoch)
            board_writer.add_scalar("Val/Epoch_Dice", val_avg_dice, epoch)

            epoch_time = time.time() - epoch_start_time
            logger.info("--------------------------------------------------------")
            logger.info(
                f"[Epoch {epoch + 1:03d}/{config.train.epochs:03d}] "
                f"Train Loss: {epoch_avg_train_loss:.4f} | Val Loss: {val_avg_loss:.4f}"
            )
            logger.info(
                f"Val Dice: {val_avg_dice:.4f} | "
                f"Time: {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"
            )

            scheduler.step(val_avg_loss)
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

            if val_avg_dice > highest_val_dice:
                highest_val_dice = val_avg_dice
                counter = 0
                torch.save(model.state_dict(), config.paths.model_weights_path)
                logger.info("[SAVE] New best validation Dice. Model weights saved.")
            else:
                counter += 1
                logger.info(
                    f"[WARN] No validation improvement. Patience: {counter}/{config.train.patience}"
                )
                if counter >= config.train.patience:
                    board_writer.close()
                    logger.info("[STOP] Early stopping triggered.")
                    logger.info(f"Final best Val Dice: {highest_val_dice:.4f}")
                    break
        else:
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"[Epoch {epoch + 1:03d}/{config.train.epochs:03d}] "
                f"Train Loss: {epoch_avg_train_loss:.4f} | "
                f"Time: {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "highest_val_dice": highest_val_dice,
            "counter": counter,
            "global_step": global_step,
        }
        checkpoint_path = config.paths.training_checkpoint_path
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
