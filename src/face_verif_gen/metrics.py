"""Verification rates from pair scores: ROC, DET, TAR at FAR, EER, the confusion matrix and the
benchmarks' 10-fold accuracy. NumPy only.

A pair is accepted as the same person when its score is >= the threshold. Genuine pairs are two
photos of one person; impostor pairs are photos of two different people.
  TAR, true accept rate: the share of genuine pairs accepted.
  FAR, false accept rate: the share of impostor pairs accepted.
  FRR, false reject rate: 1 - TAR.
Optional weights make a pair count more than once: a person drawn twice by the subject bootstrap,
or a histogram bin standing for all the pairs counted in it.
"""

import numpy as np


def _weights(scores: np.ndarray, weights) -> np.ndarray:
    return np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=np.float64)


def roc(genuine, impostor, genuine_weights=None, impostor_weights=None):
    """(thresholds, far, tar) at every threshold that changes them, strictest first.

    As sklearn.metrics.roc_curve with drop_intermediate=False: thresholds start at inf, which
    accepts nothing, then take each distinct score in turn, down to the lowest, which accepts
    every pair. far and tar rise along the way. Pairs of weight 0, such as a person the bootstrap
    didn't draw, are left out, so they add no thresholds.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    scores = np.concatenate([genuine, impostor])
    is_genuine = np.r_[np.ones(len(genuine), bool), np.zeros(len(impostor), bool)]
    weights = np.r_[_weights(genuine, genuine_weights), _weights(impostor, impostor_weights)]
    present = weights > 0
    scores, is_genuine, weights = scores[present], is_genuine[present], weights[present]
    order = np.argsort(-scores, kind="stable")
    scores, is_genuine, weights = scores[order], is_genuine[order], weights[order]
    accepted_genuine = np.cumsum(np.where(is_genuine, weights, 0.0))
    accepted_impostor = np.cumsum(np.where(is_genuine, 0.0, weights))
    # Tied scores are accepted together: keep the last pair of each run of equal scores.
    last_of_ties = np.r_[scores[1:] != scores[:-1], True]
    thresholds = np.r_[np.inf, scores[last_of_ties]]
    tar = np.r_[0.0, accepted_genuine[last_of_ties]] / accepted_genuine[-1]
    far = np.r_[0.0, accepted_impostor[last_of_ties]] / accepted_impostor[-1]
    return thresholds, far, tar


def det(genuine, impostor, genuine_weights=None, impostor_weights=None):
    """(thresholds, far, frr): the ROC as error rates, which a DET plot draws on probit axes."""
    thresholds, far, tar = roc(genuine, impostor, genuine_weights, impostor_weights)
    return thresholds, far, 1 - tar


def tar_at_far(genuine, impostor, far: float, genuine_weights=None, impostor_weights=None):
    """(tar, threshold): the highest TAR whose FAR is at most far, and the threshold giving it.

    Tied impostor scores are accepted together, so FAR can jump past the target. The threshold
    then stays above the ties, and FAR falls short of the target rather than exceeding it.
    """
    thresholds, fars, tars = roc(genuine, impostor, genuine_weights, impostor_weights)
    k = np.flatnonzero(fars <= far)[-1]  # fars rise as the threshold falls; fars[0] is 0
    return float(tars[k]), float(thresholds[k])


def rates_at_threshold(
    genuine, impostor, threshold: float, genuine_weights=None, impostor_weights=None
):
    """(tar, far) at a given threshold, such as one chosen on val and applied to test."""
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    gw, iw = _weights(genuine, genuine_weights), _weights(impostor, impostor_weights)
    tar = gw[genuine >= threshold].sum() / gw.sum()
    far = iw[impostor >= threshold].sum() / iw.sum()
    return float(tar), float(far)


def eer(genuine, impostor, genuine_weights=None, impostor_weights=None) -> float:
    """The equal error rate: where FAR = FRR, interpolated between the two ROC points around it.

    FAR rises from 0 and FRR falls to 0 as the threshold drops, so they cross exactly once.
    """
    _, far, tar = roc(genuine, impostor, genuine_weights, impostor_weights)
    frr = 1 - tar
    k = np.flatnonzero(far >= frr)[0]  # at least 1: at the first point FAR is 0 and FRR is 1
    before, after = frr[k - 1] - far[k - 1], frr[k] - far[k]  # before > 0 >= after
    t = before / (before - after)  # how far along the segment from point k - 1 they meet
    return float(far[k - 1] + t * (far[k] - far[k - 1]))


def confusion_matrix(genuine, impostor, threshold: float) -> np.ndarray:
    """Pair counts at a threshold, as sklearn.metrics.confusion_matrix with labels [same, not]:

    [[genuine accepted, genuine rejected],
     [impostor accepted, impostor rejected]]
    """
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    accepted_genuine = int((genuine >= threshold).sum())
    accepted_impostor = int((impostor >= threshold).sum())
    return np.array(
        [
            [accepted_genuine, len(genuine) - accepted_genuine],
            [accepted_impostor, len(impostor) - accepted_impostor],
        ]
    )


# InsightFace's thresholds: squared distances 0, 0.01, ..., 3.99 between unit embeddings, as
# cosines (distance^2 = 2 - 2 cos), from 1 down to -0.995.
BENCHMARK_THRESHOLDS = 1 - np.arange(0, 4, 0.01) / 2


def kfold_accuracy(cosines, same, n_folds: int = 10) -> np.ndarray:
    """Each fold's accuracy under the benchmarks' protocol (LFW and the four after it).

    As InsightFace's verification.py, which the published numbers come from. The pairs are split
    into n_folds in file order, unshuffled: the files are laid out so each fold is balanced. A
    pair counts as the same person when its cosine is above the threshold. Each fold's threshold
    is the one on the BENCHMARK_THRESHOLDS grid with the most of the other folds' pairs right
    (the strictest of any ties), and its accuracy is the share of its own pairs that gets right.
    """
    cosines, same = np.asarray(cosines, dtype=np.float64), np.asarray(same, dtype=bool)
    accuracies = []
    for held_out in np.array_split(np.arange(len(cosines)), n_folds):
        rest = np.setdiff1d(np.arange(len(cosines)), held_out)
        right = (cosines[rest, None] > BENCHMARK_THRESHOLDS) == same[rest, None]
        threshold = BENCHMARK_THRESHOLDS[np.argmax(right.sum(0))]
        accuracies.append(np.mean((cosines[held_out] > threshold) == same[held_out]))
    return np.array(accuracies)
