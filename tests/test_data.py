from PIL import Image

from pet_classifier.data import PetImages, display_name, eval_transform, split_indices, train_transform


def test_transforms_produce_224px_tensors_from_any_photo_size():
    image = Image.new("RGB", (500, 375), color=(120, 80, 40))
    for transform in (train_transform(), eval_transform()):
        assert transform(image).shape == (3, 224, 224)


def test_dataset_reads_images_from_disk(tmp_path):
    path = tmp_path / "cat.jpg"
    Image.new("RGB", (300, 200)).save(path)
    image, label = PetImages([(path, 7)], eval_transform())[0]
    assert image.shape == (3, 224, 224) and label == 7


def test_split_is_disjoint_complete_and_repeatable():
    train, val = split_indices(100, 10, seed=1)
    assert len(val) == 10 and set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(100))
    assert split_indices(100, 10, seed=1) == (train, val)


def test_display_name():
    assert display_name("american_pit_bull_terrier") == "American Pit Bull Terrier"
