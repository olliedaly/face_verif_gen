"""Regression check: the evaluation port gives i_use_arc_btw's numbers on the same embeddings.

The plan's check was to rescore the DINOv2 checkpoints and get every README number back, but the
checkpoints weren't kept. So this checks the code: the old functions, copied unchanged into
tests/reference/i_use_arc_btw_eval.py, and the new ones must count the same pairs into the same
bins, with each fix, and give the same TAR at every FAR and the same bootstrap intervals.

On structured synthetic embeddings everywhere, CI included, and on AdaFace's embeddings of CASIA
val with a GPU. AdaFace trained on the val people, so its scores there mean nothing; only the two
codes agreeing does.
"""

from pathlib import Path

import numpy as np
import pytest
import reference.i_use_arc_btw_eval as old
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from face_verif_gen import metrics
from face_verif_gen.data import FaceDataset
from face_verif_gen.fixes import (
    COHORT_RANGE,
    cohort_normalisation,
    cohort_stats,
    ruiner,
    unsure_cutoff,
    unsure_direction,
    unsure_rule,
)
from face_verif_gen.model import load_adaface
from face_verif_gen.verify import (
    bin_edges,
    cosine,
    embed,
    pair_histograms,
    pick_faces,
    subject_bootstrap,
)

REPO = Path(__file__).parents[1]
FARS = [1e-3, 1e-4, 1e-5]
FIXES = ["as is", "unsure rule", "cohort normalisation"]


def ours(fix: str, d: dict, cutoff: float, weights=None) -> tuple[np.ndarray, np.ndarray]:
    if fix == "as is":
        score, (lo, hi) = cosine, (-1.0, 1.0)
    elif fix == "unsure rule":
        score, (lo, hi) = unsure_rule(d["near"], cutoff), (-1.0, 1.0)
    else:
        score, (lo, hi) = cohort_normalisation(d["mu"], d["sd"]), COHORT_RANGE
    return pair_histograms(d["emb"], d["labels"], score, weights, lo, hi)


def our_tars(fix: str, genuine: np.ndarray, impostor: np.ndarray) -> list[float]:
    edges = bin_edges(*(COHORT_RANGE if fix == "cohort normalisation" else (-1.0, 1.0)))
    return [metrics.tar_at_far(edges, edges, far, genuine, impostor)[0] for far in FARS]


def check_fixes_agree(d: dict, cutoff: float) -> None:
    """Same counts in every bin and the same TAR at every FAR, as is and with each fix."""
    for fix in FIXES:
        genuine, impostor = ours(fix, d, cutoff)
        old_genuine, old_impostor = old.histograms(d, fix, cutoff)
        np.testing.assert_array_equal(genuine, old_genuine.numpy(), err_msg=fix)
        np.testing.assert_array_equal(impostor, old_impostor.numpy(), err_msg=fix)
        summary = old.summarise(old_genuine, old_impostor, fars=FARS)
        assert our_tars(fix, genuine, impostor) == [summary[f"TAR@FAR={f:g}"] for f in FARS], fix


def synthetic_model(seed: int, device: str = "cpu") -> dict:
    """1,200 photos of 150 people, 60 of them unreadable: pulled towards one shared direction,
    whoever they show, as the cluster does. Enough impostor pairs (715,000) for FAR 1e-5."""
    g = torch.Generator().manual_seed(seed)
    people = torch.randn(150, 64, generator=g)
    labels = torch.arange(150).repeat_interleave(8)
    emb = people[labels] + 0.9 * torch.randn(1200, 64, generator=g)
    shared = torch.randn(64, generator=torch.Generator().manual_seed(99))
    unreadable = torch.randperm(1200, generator=g)[:60]
    emb[unreadable] = shared + 0.3 * torch.randn(60, 64, generator=g)
    emb = F.normalize(emb, dim=1)
    cohort = F.normalize(torch.randn(500, 64, generator=g), dim=1)
    mu, sd = cohort_stats(emb, cohort, k=20)
    return {
        "emb": emb.to(device),
        "labels": labels.to(device),
        "cohort": cohort.to(device),
        "near": (emb @ F.normalize(shared, dim=0)).to(device),
        "mu": mu.to(device),
        "sd": sd.to(device),
    }


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(old, "DEVICE", "cpu")
    monkeypatch.setattr(old, "K", 20)


def test_cohort_stats_agree(cpu):
    d = synthetic_model(0)
    mu, sd = old.cohort_stats(d["emb"], d["cohort"])
    assert torch.equal(mu, d["mu"]) and torch.equal(sd, d["sd"])


def test_unsure_cutoff_agrees(cpu):
    # The old code took the quantile in float32, ours in float64.
    near = synthetic_model(0)["near"]
    assert unsure_cutoff(near, 0.01) == pytest.approx(float(near.quantile(0.99)), abs=1e-6)


def test_counts_and_tars_agree(cpu):
    d = synthetic_model(0)
    check_fixes_agree(d, cutoff=unsure_cutoff(d["near"], 0.01))


def test_bootstrap_intervals_agree(cpu, monkeypatch):
    # The old paired bootstrap of ArcFace - softmax, here two synthetic models, against ours
    # with the same difference as its statistic. Same seed, so the same people are drawn.
    monkeypatch.setattr(old, "N_BOOT", 200)
    a, b = synthetic_model(0), synthetic_model(1)
    embs = {"ArcFace": (a["emb"], a["labels"]), "softmax": (b["emb"], b["labels"])}
    _, old_low, old_high = old.paired_bootstrap(embs, {"ArcFace": 0, "softmax": 0}, n_ids=150)

    def difference(weights):
        tars = [our_tars("as is", *ours("as is", d, 0.0, weights)) for d in (a, b)]
        return np.subtract(*tars)

    low, high = subject_bootstrap(difference, n_identities=150, n_resamples=200, seed=0)
    np.testing.assert_allclose(low, old_low.numpy()[:3], atol=1e-12)
    np.testing.assert_allclose(high, old_high.numpy()[:3], atol=1e-12)


CASIA = OmegaConf.load(REPO / "configs" / "data" / "casia.yaml")
EVAL = OmegaConf.load(REPO / "configs" / "eval" / "default.yaml")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.skipif(not (REPO / CASIA.manifest).exists(), reason="needs the real data")
@pytest.mark.skipif(not (REPO / EVAL.adaface_checkpoint).exists(), reason="needs AdaFace's")
def test_agree_on_adaface_embeddings_of_val(monkeypatch):
    monkeypatch.setattr(old, "DEVICE", "cuda")
    device = torch.device("cuda")
    model = load_adaface(REPO / EVAL.adaface_checkpoint).to(device).eval()
    array = REPO / CASIA.array

    def load(split):
        return FaceDataset(
            REPO / CASIA.manifest,
            REPO / CASIA.root,
            REPO / CASIA.split_file,
            split,
            array=array if array.exists() else None,
        )

    train, val = load("train"), load("val")
    faces = pick_faces(val.labels, EVAL.per_identity, EVAL.face_seed)
    emb = embed(model, val, faces, device)
    # One training photo per person for each fix, rather than evaluate's 2 for the cohort:
    # agreement doesn't depend on how many.
    ruined = embed(model, train, pick_faces(train.labels, 1, 1), device, degrade=ruiner(device))
    cohort = embed(model, train, pick_faces(train.labels, 1, 0), device)
    monkeypatch.setattr(old, "K", EVAL.cohort_k)
    mu, sd = cohort_stats(emb, cohort, EVAL.cohort_k)
    old_mu, old_sd = old.cohort_stats(emb, cohort)
    assert torch.equal(mu, old_mu) and torch.equal(sd, old_sd)
    near = emb @ unsure_direction(ruined)
    d = {
        "emb": emb,
        "labels": torch.as_tensor(val.labels[faces], device=device),
        "near": near,
        "mu": mu,
        "sd": sd,
    }
    check_fixes_agree(d, cutoff=unsure_cutoff(near, EVAL.unsure_share))
