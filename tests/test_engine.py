import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pet_classifier.callbacks import Callback, EarlyStopping
from pet_classifier.engine import Trainer, train_mode
from pet_classifier.model import build_model, freeze_backbone

CPU = torch.device("cpu")


def tiny_loader():
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(16, 8), torch.randint(0, 3, (16,))), batch_size=8)


class Recorder(Callback):
    def __init__(self):
        self.epochs = []

    def on_epoch_end(self, trainer, epoch, metrics):
        self.epochs.append(epoch)


def test_fit_numbers_epochs_across_stages_and_calls_callbacks():
    model = nn.Linear(8, 3)
    trainer = Trainer(model, nn.CrossEntropyLoss(), CPU)
    recorder = Recorder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.fit(tiny_loader(), tiny_loader(), optimizer, 2, callbacks=[recorder], stage="head")
    trainer.fit(tiny_loader(), tiny_loader(), optimizer, 2, callbacks=[recorder], stage="finetune")
    assert recorder.epochs == [1, 2, 3, 4]
    assert [h["stage"] for h in trainer.history] == ["head", "head", "finetune", "finetune"]


def test_early_stopping_ends_fit():
    model = nn.Linear(8, 3)
    trainer = Trainer(model, nn.CrossEntropyLoss(), CPU)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # no learning, so val loss never improves
    trainer.fit(tiny_loader(), tiny_loader(), optimizer, 10, callbacks=[EarlyStopping(patience=2)])
    assert len(trainer.history) == 3


def test_frozen_batch_norm_layers_stay_in_eval_mode():
    model = build_model("resnet18", pretrained=False)
    freeze_backbone(model)
    train_mode(model)
    assert not model.bn1.training
    assert model.fc.training
