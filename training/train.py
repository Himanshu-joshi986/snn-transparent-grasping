"""
training/train.py

Main Training Script for SNN-DTA
==================================
Supports CNN baseline, Spiking U-Net baseline, and DTA-SNN.
Includes mixed-precision training, TensorBoard logging, and
automatic checkpoint saving.

Usage:
    python training/train.py --model dta --epochs 50 --batch_size 8 --lr 5e-5
    python training/train.py --model cnn --epochs 30 --config configs/cnn.yaml
    python training/train.py --model dta --pretrained --checkpoint path/to/ckpt.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from models.pretrained_adapter import PretrainedAdapter
from training.dataset import build_dataloaders
from training.losses import build_loss
from training.scheduler import build_scheduler
from utils.metrics import compute_metrics, MetricsTracker

# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SNN-DTA Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str, default="dta",
                        choices=["cnn", "snn", "dta"],
                        help="Model to train.")
    parser.add_argument("--config",     type=str, default=None,
                        help="Path to YAML config (auto-selected if not given).")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--loss",       type=str, default=None,
                        choices=["dice", "bce", "focal", "combined"],
                        help="Override loss function.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load pretrained SNN backbone.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint.")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Resume training from this checkpoint.")
    parser.add_argument("--device",     type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'mps', 'cpu'.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--run_name",   type=str, default=None,
                        help="Experiment name for logging.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """Load YAML config and apply CLI overrides."""
    # Auto-select config based on model
    if args.config is None:
        config_map = {"cnn": "configs/cnn.yaml",
                      "snn": "configs/snn.yaml",
                      "dta": "configs/dta.yaml"}
        args.config = config_map[args.model]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load base config first, then overlay model-specific
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        _deep_update(base_cfg, cfg)
        cfg = base_cfg

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["epochs"]     = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.lr         is not None: cfg["optimizer"]["lr"]        = args.lr
    if args.loss       is not None: cfg["loss"]["type"]           = args.loss
    if args.pretrained:
        cfg.setdefault("pretrained", {})["use"] = True
    if args.checkpoint is not None:
        cfg.setdefault("pretrained", {})["checkpoint"] = args.checkpoint

    return cfg


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model_name: str, cfg: dict, device: torch.device, run_name: str):
        self.model_name = model_name
        self.cfg = cfg
        self.device = device
        self.run_name = run_name

        # Dirs
        self.ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / run_name
        self.log_dir  = Path(cfg["training"]["log_dir"]) / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logs: {self.log_dir}")

        # Model
        self.model = build_model(model_name, cfg["model"]).to(device)
        logger.info(
            f"Model: {model_name.upper()} | "
            f"Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        # Pretrained loading
        if cfg.get("pretrained", {}).get("use", False):
            adapter = PretrainedAdapter(
                pretrained_type=cfg["pretrained"].get("backbone", "spikingyolox")
            )
            self.model = adapter.load_and_adapt(
                self.model,
                cfg["pretrained"].get("checkpoint"),
                freeze_epochs=cfg["pretrained"].get("freeze_backbone_epochs", 5),
            )
            self._freeze_epoch = cfg["pretrained"].get("freeze_backbone_epochs", 5)
        else:
            self._freeze_epoch = 0

        # Loss, optimizer, scheduler
        self.criterion = build_loss(cfg["loss"])
        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.scaler    = GradScaler(enabled=cfg["training"].get("amp", True))

        # Data
        self.train_loader, self.val_loader, _ = build_dataloaders(cfg)

        self.best_iou  = 0.0
        self.start_epoch = 0

    # ------------------------------------------------------------------
    def _build_optimizer(self):
        opt_cfg = self.cfg["optimizer"]
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr           = opt_cfg.get("lr", 5e-5),
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
            betas        = tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

    # ------------------------------------------------------------------
    def resume(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_iou    = ckpt.get("best_iou", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {self.start_epoch - 1} | Best IoU: {self.best_iou:.4f}")

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, iou: float, tag: str = ""):
        ckpt = {
            "epoch":           epoch,
            "model_name":      self.model_name,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_iou":        self.best_iou,
            "cfg":             self.cfg,
        }
        fname = f"{self.model_name}_epoch{epoch:03d}{tag}.pth"
        torch.save(ckpt, self.ckpt_dir / fname)
        if tag == "_best":
            # Also save as canonical best checkpoint
            torch.save(ckpt, Path(self.cfg["training"]["checkpoint_dir"]) / f"{self.model_name}_best.pth")

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricsTracker()
        grad_clip = self.cfg["training"].get("gradient_clip", 1.0)
        log_interval = self.cfg["training"].get("log_interval", 10)
        amp_enabled = self.cfg["training"].get("amp", True)

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch:03d}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            events = batch["events"].to(self.device)  # (B, T, C, H, W)
            masks  = batch["mask"].to(self.device)    # (B, 1, H, W)

            # Rearrange to (T, B, C, H, W) for SNN models
            if events.dim() == 5:
                events = events.permute(1, 0, 2, 3, 4).contiguous()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                preds = self.model(events)             # (B, 1, H, W)
                loss  = self.criterion(preds, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics = compute_metrics(preds.detach(), masks)
            metrics["loss"] = loss.item()
            tracker.update(metrics)

            if batch_idx % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 iou=f"{metrics['iou']:.4f}")

        return tracker.mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        tracker = MetricsTracker()

        for batch in tqdm(self.val_loader, desc=f"Val   E{epoch:03d}", leave=False):
            events = batch["events"].to(self.device)
            masks  = batch["mask"].to(self.device)
            if events.dim() == 5:
                events = events.permute(1, 0, 2, 3, 4).contiguous()

            preds = self.model(events)
            loss  = self.criterion(preds, masks)
            metrics = compute_metrics(preds, masks)
            metrics["loss"] = loss.item()
            tracker.update(metrics)

        return tracker.mean()

    # ------------------------------------------------------------------
    def train(self, resume_path: str = None):
        if resume_path:
            self.resume(resume_path)

        epochs = self.cfg["training"]["epochs"]
        logger.info(f"Starting training for {epochs} epochs on {self.device}.")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()

            # Unfreeze encoder after freeze_epoch
            if epoch == self._freeze_epoch and self._freeze_epoch > 0:
                PretrainedAdapter.unfreeze_encoder(self.model)
                # Rebuild optimizer to include newly unfrozen params
                self.optimizer = self._build_optimizer()
                logger.info(f"Epoch {epoch}: Encoder unfrozen. Optimizer rebuilt.")

            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.val_epoch(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            val_iou = val_metrics["iou"]

            # Log to TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.save_checkpoint(epoch, val_iou, tag="_best")
                logger.info(f"  ✓ New best IoU: {self.best_iou:.4f}")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_iou)

        self.writer.close()
        logger.info(f"\nTraining complete. Best Val IoU: {self.best_iou:.4f}")
        return self.best_iou


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args)

    # Seed
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Run name
    run_name = args.run_name or (
        f"{args.model}_{cfg['loss']['type']}_"
        f"lr{cfg['optimizer']['lr']:.0e}"
    )

    trainer = Trainer(args.model, cfg, device, run_name)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()