"""T3: AdaFace's released weights in IResNet give the same embeddings as AdaFace's own code.

Why: test_model.py checks shapes, parameter counts and gradients, but many wrong models pass
those. A stride on the wrong conv, two BatchNorms swapped or a misplaced PReLU change what the
model computes without changing a single shape. Given the same weights, matching a known-good
implementation to a cosine of 0.9999 shows every layer computes what it should. load_adaface
is also how M2 reproduces AdaFace's published scores, so this checks the loader as well.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from reference.adaface_net import build_model

from face_verif_gen.model import load_adaface, preprocess

CHECKPOINT = Path(__file__).parents[1] / "checkpoints" / "adaface_ir18_casia.ckpt"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="needs AdaFace's checkpoint in checkpoints/"
)


@pytest.fixture(scope="module")
def models():
    """Ours, loaded by load_adaface, and theirs, loaded as their code does; both in eval mode."""
    ours = load_adaface(CHECKPOINT)
    weights = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    theirs = build_model("ir_18")
    # Strict, so their code and the checkpoint are shown to agree before ours is compared to them.
    theirs.load_state_dict(
        {name.removeprefix("model."): w for name, w in weights.items() if name.startswith("model.")}
    )
    return ours.eval(), theirs.eval()


def test_embeddings_match_adafaces_own_code(models):
    ours, theirs = models
    torch.manual_seed(0)
    images = torch.randint(0, 256, (8, 3, 112, 112), dtype=torch.uint8)  # RGB, as the loader gives
    with torch.no_grad():
        embeddings = ours(preprocess(images))
        # Theirs takes BGR, and returns the embedding divided by its length, and the length.
        unit, length = theirs(preprocess(images)[:, [2, 1, 0]])
    assert F.cosine_similarity(embeddings, unit).min() > 0.9999
    # Cosine ignores length, so a wrong scale would pass the line above.
    torch.testing.assert_close(embeddings.norm(dim=1), length.squeeze(1))
