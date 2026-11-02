"""Evaluate trained models on the test set, compare them, and draw the README figures.

    python -m pet_classifier.evaluate
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

from pet_classifier.data import CLASSES, display_name, get_loaders
from pet_classifier.engine import predict_proba
from pet_classifier.model import load_checkpoint
from pet_classifier.utils import get_device

MODE_LABELS = {"linear": "Linear probe", "finetune": "Fine-tuned", "scratch": "From scratch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=["finetune", "linear", "scratch"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def evaluate_mode(mode: str, test_loader, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model, checkpoint = load_checkpoint(f"checkpoints/{mode}.pt", device)
    probs, labels = predict_proba(model, test_loader, device)
    preds = probs.argmax(dim=1)
    top5 = (probs.topk(5, dim=1).indices == labels[:, None]).any(dim=1)

    per_class = {CLASSES[i]: (preds[labels == i] == i).float().mean().item() for i in range(len(CLASSES))}
    matrix = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    off_diagonal = [
        (int(matrix[t, p]), CLASSES[t], CLASSES[p])
        for t in range(len(CLASSES)) for p in range(len(CLASSES)) if t != p and matrix[t, p] > 0
    ]
    metrics = {
        "mode": mode,
        "test_accuracy": (preds == labels).float().mean().item(),
        "test_top5_accuracy": top5.float().mean().item(),
        "best_epoch": checkpoint["epoch"],
        "per_class_accuracy": per_class,
        "most_confused": [
            {"true": t, "predicted": p, "count": n} for n, t, p in sorted(off_diagonal, reverse=True)[:10]
        ],
    }
    return metrics, probs, labels


def plot_comparison(results: list[dict], path: Path) -> None:
    names = [MODE_LABELS.get(r["mode"], r["mode"]) for r in results]
    accs = [100 * r["test_accuracy"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(names, accs, color=["#2b6cb0" if a == max(accs) else "#a0aec0" for a in accs])
    ax.bar_label(bars, labels=[f"{a:.1f}%" for a in accs], padding=4)
    ax.set(xlim=(0, 105), xlabel="test accuracy (%)", title="Same ResNet-50, three ways of training it")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(modes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in modes:
        history_path = Path("results") / mode / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        ax.plot([h["epoch"] for h in history], [100 * h["val_acc"] for h in history], marker=".",
                label=MODE_LABELS.get(mode, mode))
        head_epochs = [h["epoch"] for h in history if h["stage"] == "head"]
        if head_epochs:
            ax.axvline(head_epochs[-1] + 0.5, color="gray", linestyle=":", linewidth=1)
            ax.text(head_epochs[-1] + 0.7, 80, "fine-tune:\nbackbone\nunfrozen", fontsize=8, color="gray")
    ax.set(xlabel="epoch", ylabel="validation accuracy (%)", ylim=(0, 100), title="Validation accuracy during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(labels: torch.Tensor, preds: torch.Tensor, path: Path) -> None:
    matrix = confusion_matrix(labels, preds, labels=range(len(CLASSES)), normalize="true")
    names = [display_name(c) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set(xlabel="predicted breed", ylabel="true breed", title="Test set confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(dataset, preds: torch.Tensor, path: Path, count: int = 12, seed: int = 0) -> None:
    """Random test photos titled with the prediction; wrong ones also show the true breed, in red."""
    indices = random.Random(seed).sample(range(len(dataset)), count)
    fig, axes = plt.subplots(2, count // 2, figsize=(count * 1.25, 5.4))
    for ax, index in zip(axes.flat, indices):
        image_path, label = dataset.samples[index]
        image = ImageOps.fit(Image.open(image_path).convert("RGB"), (224, 224))
        pred = preds[index].item()
        correct = pred == label
        title = display_name(CLASSES[pred]) if correct else f"{display_name(CLASSES[pred])}\n(true: {display_name(CLASSES[label])})"
        ax.imshow(image)
        ax.set_title(title, fontsize=8, color="black" if correct else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = get_loaders(args.data_dir, num_workers=args.num_workers)

    modes = [m for m in args.modes if Path(f"checkpoints/{m}.pt").exists()]
    results, outputs = [], {}
    for mode in modes:
        metrics, probs, labels = evaluate_mode(mode, test_loader, device)
        (Path("results") / mode).mkdir(parents=True, exist_ok=True)
        (Path("results") / mode / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results.append(metrics)
        outputs[mode] = (probs.argmax(dim=1), labels)
        print(f"{MODE_LABELS.get(mode, mode):<13} top-1 {metrics['test_accuracy']:.4f} | top-5 {metrics['test_top5_accuracy']:.4f}")

    best = max(results, key=lambda r: r["test_accuracy"])
    print(f"\nmost confused pairs ({MODE_LABELS.get(best['mode'], best['mode'])}):")
    for item in best["most_confused"][:5]:
        print(f"  {item['count']:2d} x {display_name(item['true'])} -> {display_name(item['predicted'])}")

    preds, labels = outputs[best["mode"]]
    plot_comparison(results, assets / "comparison.png")
    plot_curves(modes, assets / "training_curves.png")
    plot_confusion(labels, preds, assets / "confusion_matrix.png")
    plot_samples(test_loader.dataset, preds, assets / "sample_predictions.png")
    print(f"figures written to {assets}/")


if __name__ == "__main__":
    main()
