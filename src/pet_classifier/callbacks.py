"""Training callbacks: small objects the trainer calls after every epoch.

Each one looks at the epoch's metrics and can save files or tell the trainer to
stop, which keeps that logic out of the training loop itself.
"""

from pathlib import Path

from pet_classifier.model import save_checkpoint


class BestTracker:
    """Remembers the best value seen so far of a metric that should go down ("min") or up ("max")."""

    def __init__(self, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None

    def update(self, value: float) -> bool:
        """Record `value` and return True if it beats the best by more than `min_delta`."""
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
        return improved

    def reset(self) -> None:
        self.best = None


class Callback:
    def on_train_start(self, trainer) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training once `monitor` has gone `patience` epochs in a row without improving.

    Resets at the start of every `fit` call, so each training stage gets its own patience.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", patience: int = 4, min_delta: float = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.tracker = BestTracker(mode, min_delta)
        self.wait = 0
        self.stopped_epoch: int | None = None

    def on_train_start(self, trainer) -> None:
        self.tracker.reset()
        self.wait = 0
        self.stopped_epoch = None

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        if self.tracker.update(metrics[self.monitor]):
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            trainer.should_stop = True
            self.stopped_epoch = epoch
            print(f"early stopping: {self.monitor} has not improved for {self.patience} epochs")


class ModelCheckpoint(Callback):
    """Save the model every time `monitor` reaches a new best.

    Unlike early stopping it keeps its best value across `fit` calls, so with
    two-stage fine-tuning the file always holds the best model from either stage.
    """

    def __init__(self, path: str | Path, arch: str, monitor: str = "val_acc", mode: str = "max"):
        self.path = Path(path)
        self.arch = arch
        self.monitor = monitor
        self.tracker = BestTracker(mode)
        self.best_epoch: int | None = None

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        if self.tracker.update(metrics[self.monitor]):
            self.best_epoch = epoch
            save_checkpoint(trainer.model, self.path, self.arch, epoch=epoch, **{self.monitor: metrics[self.monitor]})
            metrics["saved"] = True
