"""A small web demo: upload a cat or dog photo, or pull a random test photo, and see the predicted breed.

    pip install -e ".[demo]"
    python app.py
"""

import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from pet_classifier.data import CLASSES, display_name, eval_transform, prepare_split
from pet_classifier.model import load_checkpoint
from pet_classifier.utils import get_device

ROOT = Path(__file__).parent
device = get_device()
model, checkpoint = load_checkpoint(ROOT / "checkpoints" / "finetune.pt", device)
preprocess = eval_transform()
test_samples = prepare_split(ROOT / "data", "test")


@torch.no_grad()
def classify(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        return {}
    probs = model(preprocess(image.convert("RGB")).unsqueeze(0).to(device)).softmax(dim=1).squeeze(0)
    return {display_name(name): p for name, p in zip(CLASSES, probs.tolist())}


def random_test_image() -> tuple[Image.Image, dict[str, float], str]:
    path, label = random.choice(test_samples)
    image = Image.open(path).convert("RGB")
    scores = classify(image)
    truth = display_name(CLASSES[label])
    verdict = "correct" if max(scores, key=scores.get) == truth else "wrong"
    return image, scores, f"True breed: **{truth}** ({verdict})"


with gr.Blocks(title="Pet Breed Classifier") as demo:
    gr.Markdown(
        "# Pet Breed Classifier\n"
        "A ResNet-50 pretrained on ImageNet and fine-tuned on the Oxford-IIIT Pet dataset. "
        "It knows 37 breeds (12 cats, 25 dogs); upload a photo or try a random one from the test set."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo", height=360)
            with gr.Row():
                random_button = gr.Button("Random test photo", variant="primary")
                classify_button = gr.Button("Classify")
            truth = gr.Markdown()
        label = gr.Label(num_top_classes=5, label="Predicted breed")

    random_button.click(random_test_image, outputs=[image, label, truth])
    classify_button.click(classify, inputs=image, outputs=label)
    image.upload(classify, inputs=image, outputs=label)
    image.upload(lambda: "", outputs=truth)


if __name__ == "__main__":
    demo.launch()
