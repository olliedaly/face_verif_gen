"""The unsure rule and cohort normalisation score pairs as fixes.py's docstring describes.

Each is checked on a case small enough to work by hand or by brute force: a wrong direction of
comparison, a mean taken over the wrong axis or a pair rescaled by one photo instead of both
changes every fixed TAR without raising anything.
"""

import pytest
import torch
import torch.nn.functional as F

from face_verif_gen.fixes import (
    cohort_normalisation,
    cohort_stats,
    ruin,
    unsure_cutoff,
    unsure_direction,
    unsure_rule,
)


def test_ruin_keeps_shape_and_dtype_and_is_repeatable():
    torch.manual_seed(0)
    images = torch.randint(0, 256, (12, 3, 112, 112), dtype=torch.uint8)
    once = ruin(images, torch.Generator().manual_seed(0))
    again = ruin(images, torch.Generator().manual_seed(0))
    assert once.shape == images.shape and once.dtype == torch.uint8
    assert torch.equal(once, again)
    # Every photo changed: random pixels lose most of their detail whichever way they're ruined.
    changed = (once.float() - images.float()).abs().mean(dim=(1, 2, 3))
    assert (changed > 20).all(), changed


def test_unsure_direction_is_the_unit_mean():
    ruined = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(unsure_direction(ruined), F.normalize(torch.ones(2), dim=0))


def test_unsure_cutoff_flags_the_share_asked_for():
    near = torch.linspace(-1, 1, 1001)
    cutoff = unsure_cutoff(near, share=0.01)
    assert (near >= cutoff).float().mean().item() == pytest.approx(0.01, abs=0.001)


def test_unsure_rule_rejects_every_pair_with_an_unsure_photo():
    cosines = torch.full((3, 3), 0.9)
    near = torch.tensor([0.1, 0.5, 0.2])  # photo 1 is unsure at cutoff 0.3
    scores = unsure_rule(near, cutoff=0.3)(torch.arange(3), cosines)
    expected = torch.tensor([[0.9, -1, 0.9], [-1, -1, -1], [0.9, -1, 0.9]])
    torch.testing.assert_close(scores, expected)


def test_cohort_stats_match_brute_force():
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(10, 8), dim=1)
    cohort = F.normalize(torch.randn(40, 8), dim=1)
    mean, sd = cohort_stats(emb, cohort, k=5, chunk=3)
    for i in range(10):
        top = sorted((float(emb[i] @ c) for c in cohort), reverse=True)[:5]
        assert mean[i].item() == pytest.approx(sum(top) / 5, abs=1e-6)
        assert sd[i].item() == pytest.approx(torch.tensor(top).std().item(), abs=1e-6)


def test_cohort_normalisation_by_hand():
    # Cosine 0.5. Photo 0 matches strangers at 0.2 +- 0.1, so 0.5 is 3 sds above; photo 1 at
    # 0.3 +- 0.2, so 1 sd. The pair's score is their average, 2.
    mean, sd = torch.tensor([0.2, 0.3]), torch.tensor([0.1, 0.2])
    scores = cohort_normalisation(mean, sd)(torch.tensor([0]), torch.tensor([[0.5, 0.5]]))
    assert scores[0, 1].item() == pytest.approx(2.0)
