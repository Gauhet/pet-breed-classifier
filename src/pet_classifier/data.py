"""Oxford-IIIT Pet loading: download, augmentation and a fixed train/validation split."""

import csv
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

CLASSES = (
    "abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
    "bengal", "birman", "bombay", "boxer", "british_shorthair", "chihuahua", "egyptian_mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired", "great_pyrenees",
    "havanese", "japanese_chin", "keeshond", "leonberger", "maine_coon", "miniature_pinscher",
    "newfoundland", "persian", "pomeranian", "pug", "ragdoll", "russian_blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "siamese", "sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
)

# torchvision's pretrained weights expect inputs normalised with ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The Hugging Face mirror of the dataset; the original Oxford host is often unreachable.
SOURCE_URL = "https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data/{split}-00000-of-00001.parquet"


def display_name(name: str) -> str:
    """"american_pit_bull_terrier" -> "American Pit Bull Terrier"."""
    return name.replace("_", " ").title()


def prepare_split(data_dir: str | Path, split: str) -> list[tuple[Path, int]]:
    """Return (image path, label) pairs for "train" or "test", downloading the split on first use.

    The images are unpacked into individual files so that DataLoader worker
    processes only need to be sent a list of paths, not hundreds of MB of bytes.
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / split
    index = data_dir / f"{split}.csv"

    if not index.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        parquet = data_dir / f"pets_{split}.parquet"
        if not parquet.exists():
            print(f"downloading Oxford-IIIT Pet {split} split...")
            urllib.request.urlretrieve(SOURCE_URL.format(split=split), parquet)

        rows = []
        for row in pq.read_table(parquet, columns=["image", "label", "image_id"]).to_pylist():
            name = f"{row['image_id']}.jpg"
            (image_dir / name).write_bytes(row["image"]["bytes"])
            rows.append((name, row["label"]))
        with index.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        parquet.unlink()

    with index.open(newline="") as f:
        return [(image_dir / name, int(label)) for name, label in csv.reader(f)]


class PetImages(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def train_transform(size: int = 224) -> transforms.Compose:
    """Random crops, flips and colour changes. The crop keeps at least 35% of the
    image, since breeds are told apart by details a tighter crop could cut off."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224) -> transforms.Compose:
    """Resize the short side, then take the centre square: the standard ImageNet evaluation crop."""
    return transforms.Compose([
        transforms.Resize(size * 256 // 224),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle 0..n-1 with a fixed seed and cut off the first `val_size` for validation."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def get_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    val_fraction: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation and test loaders, downloading the dataset on first use."""
    train_samples = prepare_split(data_dir, "train")
    test_samples = prepare_split(data_dir, "test")

    # Two views of the same training photos: augmented for training, plain for validation.
    train_full = PetImages(train_samples, train_transform())
    val_full = PetImages(train_samples, eval_transform())
    test_set = PetImages(test_samples, eval_transform())

    train_idx, val_idx = split_indices(len(train_full), round(len(train_full) * val_fraction), seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
