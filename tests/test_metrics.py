"""T10: our ROC, DET, TAR at FAR, EER and confusion matrix match scikit-learn's, and a hand-worked
case with tied scores comes out as worked by hand.

Why: every result in this project is a TAR at some FAR. An off-by-one at the threshold, a > for
a >= or ties split the wrong way changes TAR by a fraction of a point, which would look like a
real difference between models. Random scores check the general case against a reference; the
hand-worked case checks ties, which random floats almost never produce.
"""

import numpy as np
import pytest
import sklearn.metrics
from scipy.optimize import brentq

from face_verif_gen import metrics


@pytest.fixture
def scores():
    """Genuine and impostor scores rounded to 2 places, so many are tied, and random weights."""
    rng = np.random.default_rng(0)
    genuine = np.round(rng.normal(0.5, 0.2, 500), 2)
    impostor = np.round(rng.normal(0.0, 0.2, 2000), 2)
    weights = rng.integers(0, 4, 2500).astype(float)  # 0 to 3, as a bootstrap draws people
    return genuine, impostor, weights


def as_sklearn(genuine, impostor):
    """scikit-learn's form: one array of scores and one of labels, 1 for genuine."""
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    return labels, np.r_[genuine, impostor]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_roc_matches_sklearn(scores, weighted):
    genuine, impostor, weights = scores
    weights = weights if weighted else None
    labels, all_scores = as_sklearn(genuine, impostor)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        labels, all_scores, sample_weight=weights, drop_intermediate=False
    )
    split = len(genuine)
    gw, iw = (weights[:split], weights[split:]) if weighted else (None, None)
    ours = metrics.roc(genuine, impostor, gw, iw)
    np.testing.assert_array_equal(ours[0], thresholds)
    np.testing.assert_allclose(ours[1], fpr)
    np.testing.assert_allclose(ours[2], tpr)


def test_det_matches_sklearn(scores):
    genuine, impostor, _ = scores
    fpr, fnr, thresholds = sklearn.metrics.det_curve(*as_sklearn(genuine, impostor))
    ours_thresholds, far, frr = metrics.det(genuine, impostor)
    # scikit-learn keeps only part of the curve and orders it the other way, so match by threshold.
    at = {t: (a, r) for t, a, r in zip(ours_thresholds, far, frr, strict=True)}
    np.testing.assert_allclose([at[t] for t in thresholds], np.c_[fpr, fnr])


@pytest.mark.parametrize("far", [1e-1, 1e-2, 1e-3])
def test_tar_at_far_matches_sklearn(scores, far):
    genuine, impostor, _ = scores
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(*as_sklearn(genuine, impostor))
    within = fpr <= far
    k = np.flatnonzero(within)[tpr[within].argmax()]
    assert metrics.tar_at_far(genuine, impostor, far) == pytest.approx((tpr[k], thresholds[k]))


def test_eer_matches_sklearn(scores):
    genuine, impostor, _ = scores
    fpr, tpr, _ = sklearn.metrics.roc_curve(*as_sklearn(genuine, impostor))
    # The usual recipe: the FAR where FAR = 1 - TAR on the ROC drawn with straight lines.
    expected = brentq(lambda x: 1 - x - np.interp(x, fpr, tpr), 0, 1)
    assert metrics.eer(genuine, impostor) == pytest.approx(expected, abs=1e-9)


def test_eer_of_two_normals():
    # Genuine N(1, 1) and impostor N(-1, 1) cross at 0, where each loses Phi(-1) = 0.1587 of its
    # pairs. With 200,000 of each the estimate's sd is 0.0008.
    rng = np.random.default_rng(0)
    eer = metrics.eer(rng.normal(1, 1, 200_000), rng.normal(-1, 1, 200_000))
    assert eer == pytest.approx(0.158655, abs=0.005)


def test_confusion_matrix_matches_sklearn(scores):
    genuine, impostor, _ = scores
    labels, all_scores = as_sklearn(genuine, impostor)
    expected = sklearn.metrics.confusion_matrix(labels, all_scores >= 0.25, labels=[1, 0])
    np.testing.assert_array_equal(metrics.confusion_matrix(genuine, impostor, 0.25), expected)


# Ties of every kind: genuine with genuine (0.8, 0.8), genuine with impostor (0.8) and impostor
# with impostor (0.5, 0.5). Accepting score >= t at each threshold, worked by hand:
#   threshold   inf   0.9   0.8   0.5   0.3   0.1
#   TAR           0   1/4   3/4   3/4     1     1
#   FAR           0     0   1/4   3/4   3/4     1
GENUINE = [0.9, 0.8, 0.8, 0.3]
IMPOSTOR = [0.8, 0.5, 0.5, 0.1]


def test_roc_by_hand():
    thresholds, far, tar = metrics.roc(GENUINE, IMPOSTOR)
    np.testing.assert_array_equal(thresholds, [np.inf, 0.9, 0.8, 0.5, 0.3, 0.1])
    np.testing.assert_array_equal(tar, [0, 1 / 4, 3 / 4, 3 / 4, 1, 1])
    np.testing.assert_array_equal(far, [0, 0, 1 / 4, 3 / 4, 3 / 4, 1])


@pytest.mark.parametrize(
    ("far", "tar", "threshold"),
    [
        (0.2, 1 / 4, 0.9),  # the genuine 0.8s can't come in without the impostor 0.8
        (0.25, 3 / 4, 0.8),
        (0.5, 3 / 4, 0.8),  # the two impostor 0.5s come in together: FAR would jump to 3/4
    ],
)
def test_tar_at_far_by_hand(far, tar, threshold):
    assert metrics.tar_at_far(GENUINE, IMPOSTOR, far) == (tar, threshold)


def test_eer_by_hand():
    # FAR and FRR = 1 - TAR meet at threshold 0.8, both 1/4.
    assert metrics.eer(GENUINE, IMPOSTOR) == 0.25


def test_kfold_threshold_never_sees_its_own_fold():
    # A threshold chosen on all 8 pairs, 0.595, gets every pair right. Fold 1's, chosen on fold 2
    # alone, is the strictest that gets all of fold 2 right: 0.695, just below its genuine 0.7.
    # That rejects fold 1's genuine 0.6. Fold 2's, chosen on fold 1, is 0.595: all of fold 2 right.
    cosines = [0.9, 0.1, 0.6, 0.4, 0.7, 0.2, 0.55, 0.8]
    same = [True, False, True, False, True, False, False, True]
    np.testing.assert_array_equal(metrics.kfold_accuracy(cosines, same, n_folds=2), [0.75, 1.0])


def insightface_accuracy(emb1, emb2, same):
    """InsightFace's verification.py, condensed: squared distances, a pair the same person when
    below the threshold, thresholds 0 to 4 in steps of 0.01, 10 folds in order."""
    dist = ((emb1 - emb2) ** 2).sum(1)
    thresholds = np.arange(0, 4, 0.01)
    accuracies = []
    for test in np.array_split(np.arange(len(dist)), 10):
        train = np.setdiff1d(np.arange(len(dist)), test)
        train_accuracy = [np.mean((dist[train] < t) == same[train]) for t in thresholds]
        t = thresholds[np.argmax(train_accuracy)]
        accuracies.append(np.mean((dist[test] < t) == same[test]))
    return np.array(accuracies)


def test_kfold_accuracy_matches_insightface():
    # Pairs of unit embeddings, half of them the same person: the second photo near the first.
    rng = np.random.default_rng(0)
    emb1 = rng.normal(size=(600, 16))
    same = np.tile([True, False], 300)
    emb2 = np.where(same[:, None], emb1 + rng.normal(0, 1.2, (600, 16)), rng.normal(size=(600, 16)))
    emb1 /= np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2 /= np.linalg.norm(emb2, axis=1, keepdims=True)
    ours = metrics.kfold_accuracy((emb1 * emb2).sum(1), same)
    np.testing.assert_allclose(ours, insightface_accuracy(emb1, emb2, same))
