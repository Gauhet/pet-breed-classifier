import torch

from pet_classifier.model import build_model, count_parameters, freeze_backbone, load_checkpoint, save_checkpoint, unfreeze


def test_head_is_resized_to_our_classes():
    model = build_model("resnet18", num_classes=37, pretrained=False)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 37)


def test_freeze_leaves_only_the_head_trainable():
    model = build_model("resnet18", num_classes=37, pretrained=False)
    freeze_backbone(model)
    assert count_parameters(model, trainable_only=True) == 512 * 37 + 37
    unfreeze(model)
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_checkpoint_round_trip(tmp_path):
    model = build_model("resnet18", num_classes=5, pretrained=False).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, "resnet18", epoch=3)
    loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(model(x), loaded(x))
    assert checkpoint["epoch"] == 3
