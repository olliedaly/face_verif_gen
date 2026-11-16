"""T8 for evaluation, T11, and the pair counting every TAR rests on.

T8: embed refuses a model in train mode and runs under torch.inference_mode. In train mode
BatchNorm uses the batch's own statistics, so a photo's embedding would depend on its batch.

T11: where pairs share people, the subject bootstrap gives a wider interval than resampling pairs.
Resampling pairs treats them as independent, which pairs of one person are not, so its interval
is too narrow.

pair_histograms puts every pair in the right bin, once, as genuine or impostor, with the
bootstrap's weights: checked against scoring each pair on its own.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from face_verif_gen.model import IResNet
from face_verif_gen.verify import embed, pair_histograms, pick_faces, subject_bootstrap

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return IResNet().eval()


@pytest.fixture
def photos():
    torch.manual_seed(0)
    images = torch.randint(0, 256, (4, 3, 112, 112), dtype=torch.uint8)
    return TensorDataset(images, torch.zeros(4, dtype=torch.long))


def test_embed_refuses_a_model_in_train_mode(model, photos):
    model.train()
    try:
        with pytest.raises(RuntimeError, match="train mode"):
            embed(model, photos, range(4), CPU)
    finally:
        model.eval()


def test_embed_runs_under_inference_mode(model, photos):
    seen = []
    hook = model.register_forward_hook(lambda *_: seen.append(torch.is_inference_mode_enabled()))
    try:
        embed(model, photos, range(4), CPU)
    finally:
        hook.remove()
    assert seen == [True]


def test_embeddings_are_unit_length_float32(model, photos):
    emb = embed(model, photos, range(4), CPU)
    assert emb.dtype == torch.float32
    torch.testing.assert_close(emb.norm(dim=1), torch.ones(4))


def test_flip_gives_a_mirrored_photo_the_same_embedding(model, photos):
    mirrored = TensorDataset(photos.tensors[0].flip(-1), photos.tensors[1])
    torch.testing.assert_close(
        embed(model, photos, range(4), CPU, flip=True),
        embed(model, mirrored, range(4), CPU, flip=True),
    )


def test_pick_faces_caps_each_person():
    labels = np.repeat([0, 1, 2], [30, 5, 1])
    faces = pick_faces(labels, per_identity=20, seed=0)
    assert np.bincount(labels[faces]).tolist() == [20, 5, 1]
    assert len(set(faces)) == len(faces)
    np.testing.assert_array_equal(faces, pick_faces(labels, per_identity=20, seed=0))


@pytest.fixture
def pairs():
    """30 unit embeddings of 6 people, in 4 dimensions so their cosines spread widely."""
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(30, 4), dim=1)
    labels = torch.arange(6).repeat_interleave(5)
    weights = torch.tensor([0, 1, 2, 1, 3, 0])  # as one bootstrap draw
    return emb, labels, weights


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_pair_histograms_match_one_pair_at_a_time(pairs, weighted):
    emb, labels, weights = pairs
    w = weights if weighted else torch.ones(6)
    bins = 50
    expected = {"genuine": np.zeros(bins), "impostor": np.zeros(bins)}
    for i in range(30):
        for j in range(i + 1, 30):
            cos = float(emb[i] @ emb[j])
            k = min(int((cos + 1) / 2 * bins), bins - 1)
            a, b = int(labels[i]), int(labels[j])
            if a == b:
                expected["genuine"][k] += float(w[a])
            else:
                expected["impostor"][k] += float(w[a] * w[b])
    genuine, impostor = pair_histograms(
        emb, labels, identity_weights=w if weighted else None, bins=bins, chunk=7
    )
    np.testing.assert_array_equal(genuine, expected["genuine"])
    np.testing.assert_array_equal(impostor, expected["impostor"])


def test_pair_histograms_refuse_scores_out_of_range(pairs):
    emb, labels, _ = pairs
    with pytest.raises(ValueError, match="outside"):
        pair_histograms(emb, labels, score=lambda rows, cos: 2 * cos)


def interval_widths(person_sd: float) -> tuple[float, float]:
    """Widths of the subject and the pair bootstrap's 95% intervals for TAR at threshold 0.5.

    200 people with 10 genuine pairs each. Each person's pairs scatter (sd 0.05) around that
    person's own level, and the levels differ between people by person_sd.
    """
    rng = np.random.default_rng(0)
    people, per_person = 200, 10
    level = rng.normal(0.5, person_sd, people)
    person = np.repeat(np.arange(people), per_person)
    scores = rng.normal(level[person], 0.05)
    accepted = scores >= 0.5

    def tar_for(weights: torch.Tensor) -> float:
        w = weights.numpy()[person]
        return (w * accepted).sum() / w.sum()

    low, high = subject_bootstrap(tar_for, people, n_resamples=1000)
    resampled = [
        accepted[rng.integers(0, len(accepted), len(accepted))].mean() for _ in range(1000)
    ]
    pair_low, pair_high = np.quantile(resampled, [0.025, 0.975])
    return high - low, pair_high - pair_low


def test_subject_bootstrap_is_wider_when_pairs_share_people():
    # People differ (sd 0.15) far more than one person's pairs do (sd 0.05), so a person's 10
    # pairs are mostly all accepted or all rejected: closer to 1 independent pair than 10.
    subject, pair = interval_widths(person_sd=0.15)
    assert subject > 2 * pair, (subject, pair)


def test_subject_bootstrap_matches_pairs_when_people_dont_matter():
    # The control: with no person effect, the pairs are independent and both agree.
    subject, pair = interval_widths(person_sd=0.0)
    assert subject == pytest.approx(pair, rel=0.25)
