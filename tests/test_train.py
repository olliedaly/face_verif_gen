"""T1's dtypes, T6 and T7: train_step with the real model and both heads.

T1: under bf16 autocast the model runs in bf16, and the head, its acos and the loss in float32.
bf16 keeps about 3 significant digits: it rounds cosines within 3 degrees of 0 to 1, and a sum
of e^z over 9,072 classes loses most of its small terms. Autocast changes dtypes silently, so a
head moved inside it, or a missing .float(), would still train, just worse.

T6: train_step memorises 32 photos of 4 people, with either head.

Why: every other test checks one piece; this checks they work together. A 24M-parameter model
memorises 32 photos easily, so a working pipeline drives the loss to about 0 and gets all 32
right within a few hundred steps. Images and labels shuffled separately, a missing step or
zero_grad, a softmax applied twice or preprocessing that wipes the images all run without an
error, and all stop that from happening.

Not caught here: the model cut from the graph, with softmax (a linear head alone can separate
32 random embeddings; T7 checks every parameter gets a gradient), or labels consistently wrong
from the data (memorised like right ones; the data tests check those).

T7: after one train_step, every parameter has a gradient that is there, finite and not all zeros,
and is in the optimiser run() builds. A parameter that fails any of these never learns, and
nothing errors: the rest of the network learns around it.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.overrides import TorchFunctionMode

from face_verif_gen import train
from face_verif_gen.data import FaceDataset
from face_verif_gen.heads import ArcFaceHead, SoftmaxHead
from face_verif_gen.model import IResNet, preprocess
from face_verif_gen.train import build_optimiser, train_step

REPO = Path(__file__).parents[1]
CASIA = OmegaConf.load(REPO / "configs" / "data" / "casia.yaml")
TRAIN = OmegaConf.load(REPO / "configs" / "train" / "default.yaml")
PEOPLE, PHOTOS = 4, 8  # 8 photos each of 4 people: 32

BOTH_HEADS = pytest.mark.parametrize(
    "make_head", [SoftmaxHead, ArcFaceHead], ids=["softmax", "arcface"]
)


@pytest.fixture(scope="module")
def batch():
    """One fixed batch: 8 photos each of the first 4 training people with 8 or more, on the GPU.

    Module scope, so both heads' tests share one load.
    """
    array = REPO / CASIA.array
    dataset = FaceDataset(
        REPO / CASIA.manifest,
        REPO / CASIA.root,
        REPO / CASIA.split_file,
        split="train",
        array=array if array.exists() else None,
    )
    # Some people have only 2 photos, so pick people by how many they have.
    people = np.flatnonzero(np.bincount(dataset.labels) >= PHOTOS)[:PEOPLE]
    rows = [i for person in people for i in np.flatnonzero(dataset.labels == person)[:PHOTOS]]
    images = torch.stack([dataset[i][0] for i in rows])  # (32, 3, 112, 112) uint8
    # Renumbered 0 to 3, for a 4-row head. rows is person by person, so this lines up with it.
    labels = torch.arange(PEOPLE).repeat_interleave(PHOTOS)
    return images.cuda(), labels.cuda()


@pytest.mark.gpu  # CI runs pytest -m "not gpu"
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.skipif(not (REPO / CASIA.manifest).exists(), reason="needs the real data")
@BOTH_HEADS
def test_train_step_memorises_32_photos(batch, make_head):
    images, labels = batch
    torch.manual_seed(0)
    model = IResNet().cuda()
    head = make_head(IResNet.embedding_dim, PEOPLE).cuda()
    # Fixed here, not read from configs/train, so changing the recipe can't silently break this.
    optimiser = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=0.1, momentum=0.9)

    # The same batch every step, for the plan's 300. Not stopping once the loss is low: eval mode
    # uses BatchNorm's running averages, which trail the weights. Measured: loss under 0.01 at
    # step 24 (softmax) and 77 (ArcFace), but every photo right in eval mode only from 50 and 150.
    for _ in range(300):
        loss = train_step(model, head, optimiser, images, labels)
    assert loss < 0.01, f"loss still {loss:.3f} after 300 steps"

    # Every photo right, in eval mode: dropout off and BatchNorm on its running averages, as
    # when the model is used. The head is given the labels, so ArcFace's true class must win
    # even with its margin: stricter than a plain-cosine check.
    model.eval()
    with torch.no_grad():
        logits = head(model(preprocess(images)), labels)
    assert (logits.argmax(dim=1) == labels).all(), f"{logits.argmax(dim=1)} != {labels}"


def named_params(model, head) -> list[tuple[str, torch.nn.Parameter]]:
    """Every parameter of both, named like "model.stem.0.weight" and "head.fc.weight"."""
    params = [(f"model.{name}", p) for name, p in model.named_parameters()]
    return params + [(f"head.{name}", p) for name, p in head.named_parameters()]


@BOTH_HEADS
def test_every_parameter_gets_a_gradient(make_head):
    torch.manual_seed(0)
    # Random pixels on the CPU: needs no data or GPU. BatchNorm1d needs 2+ images in train mode.
    images = torch.randint(0, 256, (4, 3, 112, 112), dtype=torch.uint8)
    labels = torch.randint(0, 10, (4,))
    model = IResNet()
    head = make_head(IResNet.embedding_dim, 10)
    train_step(model, head, build_optimiser(model, head, TRAIN), images, labels)

    problems = {}
    for name, p in named_params(model, head):
        if p.grad is None:
            problems[name] = "no gradient: not in the graph"
        elif not torch.isfinite(p.grad).all():
            problems[name] = "NaN or inf"
        elif not p.grad.any():
            problems[name] = "all zeros"
    assert not problems, problems


@BOTH_HEADS
def test_optimiser_holds_every_parameter(make_head):
    model = IResNet()
    head = make_head(IResNet.embedding_dim, 10)
    optimiser = build_optimiser(model, head, TRAIN)
    optimised = {id(p) for group in optimiser.param_groups for p in group["params"]}
    missing = [name for name, p in named_params(model, head) if id(p) not in optimised]
    assert not missing, f"not in the optimiser, so never updated: {missing}"


class AcosInputs(TorchFunctionMode):
    """Records the dtype of every tensor torch.acos is called on, inside the head too."""

    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.acos:
            self.dtypes.append(args[0].dtype)
        return func(*args, **(kwargs or {}))


def test_dtypes_under_autocast(monkeypatch):
    torch.manual_seed(0)
    images = torch.randint(0, 256, (4, 3, 112, 112), dtype=torch.uint8)
    labels = torch.randint(0, 10, (4,))
    model = IResNet()
    head = ArcFaceHead(IResNet.embedding_dim, 10)
    seen = {}
    model.register_forward_hook(lambda module, args, out: seen.update(embeddings=out.dtype))
    head.register_forward_hook(
        lambda module, args, out: seen.update(head_input=args[0].dtype, logits=out.dtype)
    )

    def recording_loss(logits, labels):
        loss = cross_entropy_loss(logits, labels)
        seen["loss"] = loss.dtype
        return loss

    cross_entropy_loss = train.cross_entropy_loss
    monkeypatch.setattr(train, "cross_entropy_loss", recording_loss)
    with AcosInputs() as acos:
        train_step(model, head, build_optimiser(model, head, TRAIN), images, labels)

    # bf16 embeddings show autocast is on: without it, the float32 checks would pass trivially.
    assert seen == {
        "embeddings": torch.bfloat16,
        "head_input": torch.float32,
        "logits": torch.float32,
        "loss": torch.float32,
    }
    assert acos.dtypes == [torch.float32]
