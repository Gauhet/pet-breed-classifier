"""The training loop, kept separate from the command-line scripts so tests can call it."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_classifier.callbacks import Callback


def train_mode(model: nn.Module) -> None:
    """Like model.train(), but batch-norm layers whose weights are frozen stay in eval mode.

    Otherwise a frozen backbone would still overwrite its ImageNet running
    statistics with ones estimated from our small batches.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: torch.device, use_amp: bool = True):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        # Mixed precision roughly halves training time on recent NVIDIA GPUs.
        self.scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
        self.history: list[dict] = []
        self.should_stop = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        callbacks: list[Callback] = (),
        stage: str = "train",
    ) -> list[dict]:
        """Train for up to `epochs` epochs, or until a callback sets `should_stop`.

        Can be called more than once (e.g. a frozen stage then a fine-tuning
        stage); epochs are numbered continuously across calls in `history`.
        """
        self.should_stop = False
        for callback in callbacks:
            callback.on_train_start(self)

        for _ in range(epochs):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)

            metrics = {
                "epoch": len(self.history) + 1, "stage": stage, "lr": lr,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "seconds": round(time.perf_counter() - start, 1),
            }
            for callback in callbacks:
                callback.on_epoch_end(self, metrics["epoch"], metrics)
            self.history.append(metrics)

            print(
                f"[{stage}] epoch {metrics['epoch']:2d} | "
                f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
                f"{metrics['seconds']:.0f}s{'  * saved' if metrics.get('saved') else ''}"
            )
            if self.should_stop:
                break
        return self.history

    def train_one_epoch(self, loader, optimizer, scheduler=None) -> tuple[float, float]:
        train_mode(self.model)
        total_loss, correct, seen = 0.0, 0, 0

        for inputs, targets in tqdm(loader, desc="train", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

        return total_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, targets in tqdm(loader, desc="eval", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                outputs = self.model(inputs)
            total_loss += self.loss_fn(outputs.float(), targets).item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)
        return total_loss / seen, correct / seen


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Class probabilities and true labels for every example in `loader`, both on the CPU."""
    model.eval()
    probs, labels = [], []
    for inputs, targets in tqdm(loader, desc="predict", leave=False):
        probs.append(model(inputs.to(device)).softmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(probs), torch.cat(labels)
