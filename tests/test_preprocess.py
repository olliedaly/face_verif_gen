from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from face_verif_gen.data import FaceDataset
from face_verif_gen.model import preprocess

REPO = Path(__file__).parents[1]
CASIA = OmegaConf.load(REPO / "configs" / "data" / "casia.yaml")


@pytest.fixture
def images():
    """A (B, 3, 112, 112) uint8 batch using the whole range, both endpoints included."""
    # randint's high bound is exclusive.
    return torch.randint(0, 256, (5, 3, 112, 112), dtype=torch.uint8)


def test_every_uint8_value_maps_into_the_range():
    """All 256 inputs, not a sample: 0 becomes exactly -1.0 and 255 exactly 1.0."""
    processed = preprocess(torch.arange(256, dtype=torch.uint8))
    assert processed.min() == -1.0
    assert processed.max() == 1.0
    assert (processed.diff() > 0).all()  # strictly increasing: no floor division, no sign flip


def test_dtype_and_shape_are_unchanged(images):
    """float32 out, same shape as in."""
    processed = preprocess(images)
    assert processed.dtype == torch.float32
    assert processed.shape == images.shape


def test_channel_order_is_preserved():
    """Channel 0 is still red, so an RGB/BGR swap fails here and nowhere else."""
    image = torch.zeros(1, 3, 2, 2, dtype=torch.uint8)
    image[:, 0] = 0  # red
    image[:, 1] = 128  # green
    image[:, 2] = 255  # blue
    processed = preprocess(image)
    assert (processed[:, 0] == -1.0).all()
    assert (processed[:, 2] == 1.0).all()
    assert ((-1.0 < processed[:, 1]) & (processed[:, 1] < 1.0)).all()  # 128, between the two


def test_float_input_is_refused(images):
    """Normalising twice would give [-1.008, -0.992], and no error."""
    with pytest.raises(TypeError, match="uint8"):
        preprocess(images.float())


def test_input_is_not_modified(images):
    """The uint8 array is cached and shared, so in-place arithmetic would corrupt it."""
    before = images.clone()
    preprocess(images)
    assert torch.equal(images, before)


@pytest.fixture(scope="module")
def real_batch():
    """32 real training photos, as the loader hands them over: raw uint8 pixels, batched."""
    if not (REPO / CASIA.manifest).exists():
        pytest.skip("needs the real data: run prepare_data.py")
    dataset = FaceDataset(
        REPO / CASIA.manifest, REPO / CASIA.root, REPO / CASIA.split_file, "train"
    )
    batch, _ = next(iter(DataLoader(dataset, batch_size=32)))
    return batch


def test_real_batch_preprocesses_into_the_declared_range(real_batch):
    """Real photos, where pixel values actually sit, rather than values set by a test."""
    assert real_batch.dtype == torch.uint8  # the loader hands over raw pixels; this is the gate
    processed = preprocess(real_batch)
    assert processed.dtype == torch.float32
    assert processed.shape == real_batch.shape
    assert processed.min() >= -1.0
    assert processed.max() <= 1.0


def test_real_photos_reach_both_ends_of_the_range(real_batch):
    """Real faces hold both black and saturated pixels, so a too-large divisor would show here."""
    processed = preprocess(real_batch)
    assert processed.min() == -1.0
    assert processed.max() == 1.0
