"""Train a pet breed classifier in one of three modes:

  finetune  pretrained ResNet: train the new head first, then unfreeze and train everything
  linear    pretrained ResNet, frozen; only the new head is trained ("linear probe")
  scratch   the same architecture from random weights, as a baseline

    python -m pet_classifier.train --mode finetune
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from pet_classifier.callbacks import EarlyStopping, ModelCheckpoint
from pet_classifier.data import get_loaders
from pet_classifier.engine import Trainer
from pet_classifier.model import ARCHITECTURES, build_model, count_parameters, freeze_backbone, unfreeze
from pet_classifier.utils import get_device, set_seed

# Per-mode defaults: a pretrained network needs a much gentler learning rate than one
# learning from scratch, and the scratch baseline needs far more epochs (and patience).
MODE_DEFAULTS = {
    "finetune": {"epochs": 15, "lr": 1e-4, "patience": 4},
    "linear": {"epochs": 15, "lr": 1e-3, "patience": 4},
    "scratch": {"epochs": 60, "lr": 1e-3, "patience": 10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODE_DEFAULTS, default="finetune")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--epochs", type=int, help="maximum epochs for the main stage (default depends on mode)")
    parser.add_argument("--lr", type=float, help="peak learning rate for the main stage (default depends on mode)")
    parser.add_argument("--patience", type=int, help="early-stopping patience in epochs (default depends on mode)")
    parser.add_argument("--head-epochs", type=int, default=3, help="finetune only: epochs training just the head")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="finetune only: learning rate for the head stage")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on the GPU")
    args = parser.parse_args()
    for key, value in MODE_DEFAULTS[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def make_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int, steps_per_epoch: int):
    """AdamW over the trainable parameters, with the learning rate decaying along a cosine curve."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    return optimizer, scheduler


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, _ = get_loaders(args.data_dir, args.batch_size, num_workers=args.num_workers, seed=args.seed)
    steps = len(train_loader)

    model = build_model(args.arch, pretrained=args.mode != "scratch")
    trainer = Trainer(model, nn.CrossEntropyLoss(label_smoothing=args.label_smoothing), device, use_amp=not args.no_amp)
    checkpoint = ModelCheckpoint(f"checkpoints/{args.mode}.pt", args.arch, monitor="val_acc", mode="max")
    callbacks = [checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)]
    print(f"device: {device} | mode: {args.mode} | {args.arch}, {count_parameters(model):,} parameters")

    start = time.perf_counter()
    if args.mode in ("linear", "finetune"):
        freeze_backbone(model)
        print(f"trainable parameters with frozen backbone: {count_parameters(model, trainable_only=True):,}")

    if args.mode == "finetune":
        # Stage 1: the new head starts random, so train it alone first. Unfreezing
        # straight away would push large, noisy gradients into the pretrained layers.
        optimizer, scheduler = make_optimizer(model, args.head_lr, args.weight_decay, args.head_epochs, steps)
        trainer.fit(train_loader, val_loader, optimizer, args.head_epochs, scheduler, callbacks, stage="head")
        # Stage 2: unfreeze everything and continue with a small learning rate.
        unfreeze(model)
        print(f"unfrozen: training all {count_parameters(model, trainable_only=True):,} parameters")

    optimizer, scheduler = make_optimizer(model, args.lr, args.weight_decay, args.epochs, steps)
    trainer.fit(train_loader, val_loader, optimizer, args.epochs, scheduler, callbacks, stage=args.mode)
    minutes = (time.perf_counter() - start) / 60

    results = Path("results") / args.mode
    results.mkdir(parents=True, exist_ok=True)
    (results / "history.json").write_text(json.dumps(trainer.history, indent=2))
    summary = {
        "mode": args.mode, "arch": args.arch, "best_val_acc": checkpoint.tracker.best,
        "best_epoch": checkpoint.best_epoch, "epochs_run": len(trainer.history),
        "minutes": round(minutes, 1), "args": vars(args),
    }
    (results / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best validation accuracy: {checkpoint.tracker.best:.4f} at epoch {checkpoint.best_epoch} | {minutes:.1f} min")


if __name__ == "__main__":
    main()
