import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from face_verif_gen.data import FaceDataset, read_image
from face_verif_gen.prepare_data import build_manifest, write_array

REPO = Path(__file__).parents[1]
CASIA = OmegaConf.load(REPO / "configs" / "data" / "casia.yaml")
# Gaps in the identity numbers on purpose: training labels must still come out as 0 and 1.
SPLIT = {10: "train", 20: "train", 30: "val", 40: "test"}


def image(n: int) -> torch.Tensor:
    """A (3, 4, 4) image, different in every pixel and channel, so a mixed-up axis shows.

    Adding n makes every file's pixels different too.
    """
    return (torch.arange(48).reshape(3, 4, 4) + n).to(torch.uint8)


@pytest.fixture
def data(tmp_path):
    """Images, a split file, a manifest and an array, laid out as prepare_data.py writes them."""
    n = 0  # numbered across people, like the real files
    for identity, count in [(10, 2), (20, 1), (30, 1), (40, 1)]:
        folder = tmp_path / "casia" / f"{identity:05d}"
        folder.mkdir(parents=True)
        for _ in range(count):
            # Pillow wants channels last, (4, 4, 3).
            Image.fromarray(image(n).permute(1, 2, 0).numpy()).save(folder / f"{n:06d}.png")
            n += 1
    rows = sorted(SPLIT.items())
    pd.DataFrame(rows, columns=["identity", "split"]).to_csv(tmp_path / "split.csv", index=False)
    manifest = build_manifest(tmp_path / "casia", SPLIT)
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    write_array(manifest, tmp_path / "casia", tmp_path / "images.npy")
    return tmp_path


def load(data: Path, split: str, from_array: bool = False) -> FaceDataset:
    array = data / "images.npy" if from_array else None
    return FaceDataset(data / "manifest.parquet", data / "casia", data / "split.csv", split, array)


def test_images_come_back_unchanged_as_uint8(data):
    train = load(data, "train")
    for n in range(len(train)):
        assert torch.equal(train[n][0], image(n))  # equal also checks the dtype is uint8


def test_each_split_holds_only_its_people(data):
    assert [len(load(data, split)) for split in ("train", "val", "test")] == [3, 1, 1]


def test_train_labels_have_no_gaps(data):
    train = load(data, "train")
    assert train.num_classes == 2
    assert [train[n][1] for n in range(len(train))] == [0, 0, 1]  # people 10, 10 and 20


def test_person_from_another_split_is_refused(data):
    # A manifest built from a different split, which puts val person 30 in training.
    stale = build_manifest(data / "casia", SPLIT | {30: "train"})
    stale.to_parquet(data / "manifest.parquet", index=False)
    with pytest.raises(ValueError, match="aren't train"):
        load(data, "train")


def test_misspelt_split_is_refused(data):
    with pytest.raises(ValueError, match="no 'trian' rows"):
        load(data, "trian")


def test_batches_through_a_dataloader(data):
    images, labels = next(iter(DataLoader(load(data, "train"), batch_size=3)))
    assert images.shape == (3, 3, 4, 4)
    assert labels.tolist() == [0, 0, 1]


# val and test are included because their rows come after train's in the array, so
# looking up the wrong row only shows there.
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_array_gives_the_same_items_as_pngs(data, split):
    pngs, array = load(data, split), load(data, split, from_array=True)
    for n in range(len(pngs)):
        assert torch.equal(array[n][0], pngs[n][0])
        assert array[n][1] == pngs[n][1]


def test_array_of_another_manifest_is_refused(data):
    manifest = pd.read_parquet(data / "manifest.parquet")
    write_array(manifest[:-1], data / "casia", data / "images.npy")  # one image short
    with pytest.raises(ValueError, match="one image per row"):
        load(data, "train", from_array=True)


def test_short_read_is_refused(data):
    test = load(data, "test", from_array=True)  # one person, in the array's last row
    array = data / "images.npy"
    os.truncate(array, array.stat().st_size - 10)  # shrinks after the dataset checked it
    with pytest.raises(OSError, match="short read"):
        test[0]


@pytest.mark.skipif(
    not (REPO / CASIA.manifest).exists(), reason="needs the real data: run prepare_data.py"
)
def test_real_train_split():
    train = FaceDataset(REPO / CASIA.manifest, REPO / CASIA.root, REPO / CASIA.split_file, "train")
    assert len(train) == 422_091
    assert set(train.labels) == set(range(9072))
    image, _ = train[0]
    assert image.shape == (3, 112, 112)
    assert image.dtype == torch.uint8


@pytest.mark.skipif(
    not (REPO / CASIA.array).exists(), reason="needs the decoded array: run prepare_data.py"
)
def test_real_array_matches_pngs():
    manifest = pd.read_parquet(REPO / CASIA.manifest)
    images = np.load(REPO / CASIA.array, mmap_mode="r")
    assert images.shape == (len(manifest), 3, 112, 112)
    assert images.dtype == np.uint8
    for row in np.random.default_rng(0).choice(len(manifest), 100, replace=False):
        assert np.array_equal(images[row], read_image(REPO / CASIA.root / manifest["path"][row]))
