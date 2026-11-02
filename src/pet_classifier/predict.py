"""Name the breed in your own cat or dog photos.

    python -m pet_classifier.predict path/to/photo.jpg [more photos...]
"""

import argparse

import torch
from PIL import Image

from pet_classifier.data import CLASSES, display_name, eval_transform
from pet_classifier.model import load_checkpoint
from pet_classifier.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="checkpoints/finetune.pt")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    preprocess = eval_transform()

    for path in args.images:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(image).softmax(dim=1).squeeze(0)
        top = probs.topk(args.top)
        guesses = ", ".join(
            f"{display_name(CLASSES[i])} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist())
        )
        print(f"{path}: {guesses}")


if __name__ == "__main__":
    main()
