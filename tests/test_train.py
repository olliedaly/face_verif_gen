"""T6: train_step memorises 32 photos of 4 people, with either head.

Why: every other test checks one piece; this checks they work together. A 24M-parameter model
memorises 32 photos easily, so a working pipeline drives the loss to about 0 and gets all 32
right within a few hundred steps. Images and labels shuffled separately, a missing step or
zero_grad, a softmax applied twice or preprocessing that wipes the images all run without an
error, and all stop that from happening.

Not caught here: the model cut from the graph, with softmax (a linear head alone can separate
32 random embeddings; T7 checks every parameter gets a gradient), or labels consistently wrong
from the data (memorised like right ones; the data tests check those).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from face_verif_gen.data import FaceDataset
from face_verif_gen.heads import ArcFaceHead, SoftmaxHead
from face_verif_gen.model import IResNet, preprocess
from face_verif_gen.train import train_step

REPO = Path(__file__).parents[1]
CASIA = OmegaConf.load(REPO / "configs" / "data" / "casia.yaml")
PEOPLE, PHOTOS = 4, 8  # 8 photos each of 4 people: 32

pytestmark = [
    pytest.mark.gpu,  # CI runs pytest -m "not gpu"
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not (REPO / CASIA.manifest).exists(), reason="needs the real data"),
]


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


@pytest.mark.parametrize("make_head", [SoftmaxHead, ArcFaceHead], ids=["softmax", "arcface"])
def test_train_step_memorises_32_photos(batch, make_head):
    images, labels = batch
    torch.manual_seed(0)
    model = IResNet().cuda()  # the full model: only the head knows how many people there are
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
