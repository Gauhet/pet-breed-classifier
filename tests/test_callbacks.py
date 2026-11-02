import pytest
import torch
from torch import nn

from pet_classifier.callbacks import BestTracker, EarlyStopping, ModelCheckpoint


class FakeTrainer:
    def __init__(self):
        self.model = nn.Module()
        self.model.fc = nn.Linear(4, 3)
        self.should_stop = False


def run(callback, values, key):
    trainer = FakeTrainer()
    callback.on_train_start(trainer)
    for epoch, value in enumerate(values, start=1):
        callback.on_epoch_end(trainer, epoch, {key: value})
        if trainer.should_stop:
            return trainer, epoch
    return trainer, None


def test_best_tracker_respects_mode_and_min_delta():
    lower = BestTracker("min", min_delta=0.1)
    assert lower.update(1.0)
    assert not lower.update(0.95)  # better, but not by more than min_delta
    assert lower.update(0.8)
    higher = BestTracker("max")
    assert higher.update(0.5) and not higher.update(0.4) and higher.update(0.6)
    with pytest.raises(ValueError):
        BestTracker("sideways")


def test_early_stopping_waits_for_patience_epochs_without_improvement():
    _, stopped_at = run(EarlyStopping(patience=2), [1.0, 0.9, 0.95, 0.92, 0.5], "val_loss")
    assert stopped_at == 4  # epochs 3 and 4 were both worse than 0.9


def test_early_stopping_resets_counter_on_improvement():
    _, stopped_at = run(EarlyStopping(patience=2), [1.0, 1.1, 0.9, 1.0, 0.8, 0.85], "val_loss")
    assert stopped_at is None


def test_early_stopping_resets_between_fit_calls():
    callback = EarlyStopping(patience=1)
    run(callback, [0.2, 0.3], "val_loss")
    trainer, stopped_at = run(callback, [0.5, 0.4], "val_loss")  # worse than 0.2, but a fresh stage
    assert stopped_at is None


def test_checkpoint_saves_only_on_improvement(tmp_path):
    path = tmp_path / "best.pt"
    callback = ModelCheckpoint(path, arch="resnet18", monitor="val_acc", mode="max")
    run(callback, [0.5, 0.7, 0.6], "val_acc")
    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 2 and saved["val_acc"] == 0.7
    assert callback.best_epoch == 2
