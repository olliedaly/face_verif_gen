import pytest
import torch

from face_verif_gen.model import preprocess


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
    """Callers may reuse the uint8 batch, e.g. to save or show a face, so it must stay as it was."""
    before = images.clone()
    preprocess(images)
    assert torch.equal(images, before)
