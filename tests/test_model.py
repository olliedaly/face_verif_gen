import pytest
import torch

from face_verif_gen.model import Block, IResNet

BATCH = 2  # the smallest batch BatchNorm1d accepts in train mode
EMBEDDING_DIM = 512
# AdaFace's IResNet-18, whose weights T3 loads. Half of it is the Linear after Flatten
# (25,088 x 512); an ImageNet ResNet-18 has about 11.7M, as it averages each channel first.
PARAMETERS = 24_020_352
# Channels and height (= width) after each stage of 2 blocks.
STAGES = [(64, 56), (128, 28), (256, 14), (512, 7)]


def n_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.fixture
def model():
    torch.manual_seed(0)  # the same weights every run, so a failure can be reproduced
    return IResNet()


@pytest.fixture
def input_batch():
    return torch.randn(BATCH, 3, 112, 112)


def test_output_shape(model, input_batch):
    """(B, 3, 112, 112) images give (B, 512) float32 embeddings."""
    out = model(input_batch)
    assert out.shape == (BATCH, EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_stage_shapes(model, input_batch):
    """Each stage halves the height and width, down to the 7 x 7 that the Linear expects."""
    x = model.stem(input_batch)
    for stage, (channels, size) in enumerate(STAGES):
        x = model.body[2 * stage : 2 * stage + 2](x)
        assert x.shape == (BATCH, channels, size, size), f"stage {stage + 1}"


def test_parameter_count(model):
    assert n_parameters(model) == PARAMETERS


def test_every_parameter_gets_a_gradient(model, input_batch):
    """A layer built in __init__ but skipped in forward gets no gradient; no other test sees it."""
    out = model(input_batch)
    # Not out.sum(): the last BatchNorm centres each feature over the batch, so the sum is always
    # about 0 and every gradient is rounding noise, which doesn't matter for this test, but
    # just in case we use this for grad checking in future.
    (out * torch.randn_like(out)).sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


def test_eval_takes_one_image(model):
    """In eval mode BatchNorm uses its running averages; in train mode one image raises."""
    model.eval()
    assert model(torch.randn(1, 3, 112, 112)).shape == (1, EMBEDDING_DIM)


def test_deterministic_eval(model, input_batch):
    """Eval mode turns dropout off, so the same input gives the same output."""
    model.eval()
    model_output_1 = model(input_batch)
    model_output_2 = model(input_batch)
    assert torch.allclose(model_output_1, model_output_2, atol=1e-6)


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "stride", "parameters"),
    [(64, 64, 1, 74_176), (64, 64, 2, 74_176), (64, 128, 2, 230_400)],
    ids=["identity", "max-pool", "conv"],  # the shortcut each one needs
)
def test_block(in_channels, out_channels, stride, parameters):
    block = Block(in_channels, out_channels, stride)
    out = block(torch.randn(BATCH, in_channels, 56, 56))
    assert out.shape == (BATCH, out_channels, 56 // stride, 56 // stride)
    assert n_parameters(block) == parameters
