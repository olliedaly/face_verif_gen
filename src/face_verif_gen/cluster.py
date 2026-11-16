"""The cluster metrics: does a model put unreadable photos of different people close together?

UI tightness: mean cosine between ruined photos of different people, minus the same for the
  clean photos. Unlike a fixed cut-off, comparable across models.
UI share: of the impostor pairs at or above the FAR 1e-5 threshold, the share that include a
  photo the unsure rule flags.
Fix lift: TAR gained at FAR 1e-5 by the unsure rule and by cohort normalisation (evaluate.py).
Culprit overlap: how many of the photos in a model's top 450 val impostor pairs are among the
  DINOv2 ArcFace run's.
Emergence: UI tightness after every epoch, on a fixed probe set (train.py).
"""

import torch


@torch.inference_mode()
def mean_impostor_cosine(emb: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean cosine over every pair of photos of different people, exactly, without forming pairs.

    The sum of e_i . e_j over all ordered pairs (i, j), i = j included, is |sum of e|^2. Over one
    person's pairs it is |sum of their e|^2. The difference is the sum over impostor pairs.
    """
    emb = emb.double()
    per_person = torch.zeros(
        int(labels.max()) + 1, emb.shape[1], dtype=emb.dtype, device=emb.device
    )
    per_person.index_add_(0, labels, emb)
    total = emb.sum(0)
    impostor_sum = total @ total - (per_person * per_person).sum()
    counts = torch.bincount(labels).double()
    n_impostor = len(emb) ** 2 - (counts * counts).sum()  # ordered pairs, as the sum is
    return float(impostor_sum / n_impostor)


def ui_tightness(clean: torch.Tensor, ruined: torch.Tensor, labels: torch.Tensor) -> float:
    """How much closer ruined photos of different people are than clean ones, in mean cosine."""
    return mean_impostor_cosine(ruined, labels) - mean_impostor_cosine(clean, labels)


@torch.inference_mode()
def ui_share(
    emb: torch.Tensor, labels: torch.Tensor, threshold: float, unsure: torch.Tensor, chunk=256
) -> float:
    """Of the impostor pairs with cosine >= threshold, the share with an unsure photo in them.

    0 when no impostor pair reaches the threshold.
    """
    n = len(emb)
    columns = torch.arange(n, device=emb.device)
    above, with_unsure = 0, 0
    for start in range(0, n, chunk):
        rows = columns[start : start + chunk]
        accepted = emb[rows] @ emb.T >= threshold
        accepted &= (columns[None] > rows[:, None]) & (labels[rows, None] != labels[None])
        above += int(accepted.sum())
        with_unsure += int((accepted & (unsure[rows, None] | unsure[None])).sum())
    return with_unsure / above if above else 0.0


@torch.inference_mode()
def top_impostor_photos(
    emb: torch.Tensor, labels: torch.Tensor, n_pairs: int, chunk: int = 256
) -> torch.Tensor:
    """The photos in the n_pairs highest-cosine impostor pairs, as sorted indices into emb."""
    n = len(emb)
    columns = torch.arange(n, device=emb.device)
    values, pairs = [], []
    for start in range(0, n, chunk):
        rows = columns[start : start + chunk]
        s = emb[rows] @ emb.T
        not_impostor = (labels[rows, None] == labels[None]) | (columns[None] <= rows[:, None])
        s[not_impostor] = -torch.inf
        # A chunk can't hold more of the overall top n_pairs than its own top n_pairs.
        v, flat = s.flatten().topk(min(n_pairs, s.numel()))
        keep = v > -torch.inf
        values.append(v[keep])
        pairs.append(torch.stack([rows[flat[keep] // n], flat[keep] % n], dim=1))
    top = torch.cat(values).topk(min(n_pairs, sum(len(v) for v in values))).indices
    return torch.cat(pairs)[top].unique()
