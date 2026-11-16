"""The cluster metrics match brute force on small cases.

mean_impostor_cosine avoids forming pairs, and ui_share and top_impostor_photos work through the
pairs in chunks; all three are compared with looping over every pair one at a time.
"""

import pytest
import torch
import torch.nn.functional as F

from face_verif_gen.cluster import (
    mean_impostor_cosine,
    top_impostor_photos,
    ui_share,
    ui_tightness,
)


@pytest.fixture
def photos():
    """20 photos of 5 people, unequal numbers each, in 6 dimensions."""
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(20, 6), dim=1)
    labels = torch.tensor([0] * 7 + [1] * 2 + [2] * 5 + [3] * 1 + [4] * 5)
    return emb, labels


def impostor_pairs(labels):
    n = len(labels)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if labels[i] != labels[j]]


def test_mean_impostor_cosine_matches_brute_force(photos):
    emb, labels = photos
    cosines = [float(emb[i] @ emb[j]) for i, j in impostor_pairs(labels)]
    assert mean_impostor_cosine(emb, labels) == pytest.approx(sum(cosines) / len(cosines))


def test_ui_tightness_when_ruined_photos_coincide(photos):
    # Every ruined photo in one place: every ruined impostor pair has cosine 1.
    emb, labels = photos
    ruined = F.normalize(torch.ones(20, 6), dim=1)
    expected = 1 - mean_impostor_cosine(emb, labels)
    assert ui_tightness(emb, ruined, labels) == pytest.approx(expected)


def test_ui_share_matches_brute_force(photos):
    emb, labels = photos
    unsure = torch.zeros(20, dtype=torch.bool)
    unsure[[3, 11]] = True
    above = [(i, j) for i, j in impostor_pairs(labels) if float(emb[i] @ emb[j]) >= 0.2]
    with_unsure = [(i, j) for i, j in above if unsure[i] or unsure[j]]
    share = ui_share(emb, labels, threshold=0.2, unsure=unsure, chunk=6)
    assert share == pytest.approx(len(with_unsure) / len(above))


def test_top_impostor_photos_match_brute_force(photos):
    emb, labels = photos
    ranked = sorted(impostor_pairs(labels), key=lambda p: float(emb[p[0]] @ emb[p[1]]))
    expected = sorted({photo for pair in ranked[-15:] for photo in pair})
    assert top_impostor_photos(emb, labels, n_pairs=15, chunk=6).tolist() == expected
