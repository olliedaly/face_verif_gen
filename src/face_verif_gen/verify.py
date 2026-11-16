"""Verification on people never seen in training: embed photos, score every pair, count the rates.

Every pair of the chosen photos is scored, which on test is 177M pairs: too many to keep. So each
pair's score is counted into one of BINS equal bins as it's computed. A bin's pairs all score at
least its lower edge, so metrics.roc on the edges, weighted by the counts, gives the exact rates
at every threshold on an edge.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from face_verif_gen.model import preprocess

BINS = 20_000  # over [-1, 1], candidate thresholds 1e-4 apart
# score(rows, cosines) -> scores: rows are a chunk's photo indices, cosines (len(rows), n_photos).
Score = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def pick_faces(labels: np.ndarray, per_identity: int, seed: int) -> np.ndarray:
    """Up to per_identity photos of each person, chosen at random: sorted indices into labels.

    A person with n photos gives n(n - 1) / 2 genuine pairs, so without a cap the few people with
    hundreds of photos would decide TAR.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(labels, kind="stable")
    starts = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1]])
    picked = [
        rng.choice(person, size=min(per_identity, len(person)), replace=False)
        for person in np.split(order, starts[1:])
    ]
    return np.sort(np.concatenate(picked))


@torch.inference_mode()
def embed(
    model: torch.nn.Module,
    dataset: Dataset,
    indices,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    flip: bool = False,
    degrade: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Unit-length float32 embeddings of dataset[indices], on device.

    The model must be in eval mode: in train mode, BatchNorm normalises with the batch's own
    statistics, so a photo's embedding would depend on the photos sharing its batch.
    flip: add the mirrored photo's embedding before normalising, as the benchmarks are scored.
    degrade: applied to each uint8 batch first, to embed ruined copies of the photos.
    """
    if model.training:
        raise RuntimeError("model is in train mode: call model.eval() before embedding")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=num_workers)
    embeddings = []
    for images, _ in loader:
        images = images.to(device)
        if degrade is not None:
            images = degrade(images)
        e = model(preprocess(images))
        if flip:
            e = e + model(preprocess(images.flip(-1)))
        embeddings.append(F.normalize(e.float(), dim=1))
    return torch.cat(embeddings)


def cosine(rows: torch.Tensor, cosines: torch.Tensor) -> torch.Tensor:
    """The default score: the cosine itself, clamped as rounding can take it just past +-1."""
    return cosines.clamp(-1, 1)


def bin_edges(lo: float = -1.0, hi: float = 1.0, bins: int = BINS) -> np.ndarray:
    """Each bin's lower edge: every pair counted in bin k scores at least edges[k]."""
    return lo + (hi - lo) * np.arange(bins) / bins


@torch.inference_mode()
def pair_histograms(
    emb: torch.Tensor,
    labels: torch.Tensor,
    score: Score = cosine,
    identity_weights: torch.Tensor | None = None,
    lo: float = -1.0,
    hi: float = 1.0,
    bins: int = BINS,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts of every genuine and every impostor pair's score in each bin over [lo, hi].

    Each pair (i < j) is counted once. identity_weights, one per label, come from the subject
    bootstrap: a person drawn twice counts their genuine pairs twice, and a pair of people a, b
    counts w[a] * w[b] times. Scores outside [lo, hi] raise rather than pile up in an end bin.
    Returns (genuine, impostor) as float64 arrays of length bins.
    """
    n = len(emb)
    if identity_weights is None:
        identity_weights = torch.ones(int(labels.max()) + 1, device=emb.device)
    w = identity_weights.to(emb.device, torch.float64)[labels]  # each photo carries its person's
    genuine = torch.zeros(bins, dtype=torch.float64, device=emb.device)
    impostor = torch.zeros_like(genuine)
    columns = torch.arange(n, device=emb.device)
    # Chunks of rows, so the n x n matrix of cosines is never held at once. Small enough that
    # each GPU kernel finishes quickly: the driver kills long ones on a GPU driving a display.
    for start in range(0, n, chunk):
        rows = columns[start : start + chunk]
        s = score(rows, emb[rows] @ emb.T)
        if (s < lo).any() or (s > hi).any():
            raise ValueError(
                f"scores from {float(s.min()):.3f} to {float(s.max()):.3f}: outside [{lo}, {hi}]"
            )
        index = ((s - lo) / (hi - lo) * bins).long().clamp(max=bins - 1)  # hi goes in the last
        later = columns[None] > rows[:, None]
        same = labels[rows, None] == labels[None]
        g, i = later & same, later & ~same
        genuine += torch.bincount(index[g], w[rows, None].expand_as(s)[g], minlength=bins)
        impostor += torch.bincount(index[i], (w[rows, None] * w[None])[i], minlength=bins)
    return genuine.cpu().numpy(), impostor.cpu().numpy()


def subject_bootstrap(
    statistic: Callable[[torch.Tensor], object],
    n_identities: int,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """A percentile interval for statistic over resamples of people, not of pairs.

    Pairs that share a person aren't independent: one person with unusual photos moves all their
    pairs together. Drawing people with replacement keeps each person's pairs together, so the
    interval covers which people happened to be in the split. statistic gets one weight per
    person, how many times they were drawn, and returns a number or an array.
    Returns (low, high), each shaped like statistic's result.
    """
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(n_resamples):
        drawn = torch.randint(n_identities, (n_identities,), generator=generator)
        values.append(np.asarray(statistic(torch.bincount(drawn, minlength=n_identities))))
    low, high = np.quantile(
        np.stack(values).astype(np.float64), [(1 - level) / 2, (1 + level) / 2], axis=0
    )
    return low, high
