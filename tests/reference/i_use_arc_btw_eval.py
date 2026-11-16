"""Copied unchanged from i_use_arc_btw at commit 0b31e2d: the functions that count pairs,
score them with each fix, set thresholds and bootstrap. Only these, so they run without
that repo or timm. tests/test_regression.py checks the port in face_verif_gen against them.

From src/i_use_arc_btw/verify.py, scripts/compare.py and
scripts/further_work/low_quality_fixes.py.
"""

import torch

DEVICE = "cuda"  # as in compare.py and low_quality_fixes.py; the tests set it

# --- src/i_use_arc_btw/verify.py

# Pair scores are counted into bins instead of stored (val alone has 45M pairs).
# 20,000 bins over the cosine range [-1, 1] puts candidate thresholds 1e-4 apart.
BINS = 20_000


@torch.no_grad()
def pair_histograms(
    emb: torch.Tensor, labels: torch.Tensor, identity_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Histograms of cosine similarity over every genuine and every impostor pair.

    Each pair is counted once (i < j). Rows are processed in chunks so the full
    n x n similarity matrix never has to sit in memory.

    identity_weights (one per label) resample identities for a bootstrap: an
    identity drawn twice counts its genuine pairs twice, and a pair of identities
    a, b counts w[a] * w[b] times. By default every identity counts once.
    """
    n = len(emb)
    if identity_weights is None:
        identity_weights = torch.ones(int(labels.max()) + 1, device=emb.device)
    w = identity_weights.double()[labels]  # each face carries its identity's weight
    genuine = torch.zeros(BINS, dtype=torch.float64, device=emb.device)
    impostor = torch.zeros_like(genuine)
    # 256 rows per chunk keeps each GPU kernel short. The driver kills kernels that
    # run too long on a GPU that also drives a display, and 1,024 rows of test's
    # 18,814 faces hit that limit.
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=emb.device)
        sims = emb[rows] @ emb.T
        bins = ((sims + 1) / 2 * BINS).long().clamp(0, BINS - 1)
        later = torch.arange(n, device=emb.device)[None] > rows[:, None]
        same = labels[rows, None] == labels[None]
        g, i = later & same, later & ~same
        genuine += torch.bincount(bins[g], weights=w[rows, None].expand_as(sims)[g], minlength=BINS)
        impostor += torch.bincount(bins[i], weights=(w[rows, None] * w[None])[i], minlength=BINS)
    return genuine.cpu(), impostor.cpu()


def summarise(genuine: torch.Tensor, impostor: torch.Tensor, fars=(1e-3, 1e-4, 1e-5)) -> dict:
    """ROC numbers from the two histograms.

    A threshold t accepts a pair as "same person" when its cosine is >= t.
      TAR (true accept rate):  fraction of genuine pairs accepted
      FAR (false accept rate): fraction of impostor pairs accepted
    """
    thresholds = torch.linspace(-1, 1, BINS + 1, dtype=torch.float64)[:-1]  # bins' lower edges
    # At threshold k, every pair in bin k or above is accepted: a cumulative sum from the top.
    tar = genuine.flip(0).cumsum(0).flip(0).double() / genuine.sum()
    far = impostor.flip(0).cumsum(0).flip(0).double() / impostor.sum()
    centres = thresholds + 1 / BINS
    out = {
        "genuine pairs": int(genuine.sum()),
        "impostor pairs": int(impostor.sum()),
        "mean genuine cos": float((genuine * centres).sum() / genuine.sum()),
        "mean impostor cos": float((impostor * centres).sum() / impostor.sum()),
        "genuine": genuine,
        "impostor": impostor,
        "thresholds": thresholds,
        "tar": tar,
        "far": far,
    }
    for f in fars:
        # Raising the threshold lowers both rates. The lowest threshold that keeps FAR
        # within budget accepts the most genuine pairs.
        k = int((far <= f).nonzero()[0])
        out[f"TAR@FAR={f:g}"] = float(tar[k])
    # Best accuracy on a balanced set (as many genuine as impostor pairs, like LFW's
    # 3,000 + 3,000): the average of TAR and the true reject rate, 1 - FAR.
    acc = (tar + 1 - far) / 2
    k = int(acc.argmax())
    out["best accuracy"] = float(acc[k])
    out["threshold"] = float(thresholds[k])
    return out


# --- scripts/compare.py

FARS = (1e-3, 1e-4, 1e-5)
N_BOOT = 1000  # resamples; each one rescores every pair for both heads


def balanced_acc(r: dict, k: int) -> float:
    return float((r["tar"][k] + 1 - r["far"][k]) / 2)


def paired_bootstrap(embs: dict, k_acc: dict, n_ids: int):
    """ArcFace - softmax on each metric, with a 95% interval over identities.

    Resample people with replacement, score both heads on the same resample, and
    keep the difference. This covers which people happen to be in the split, not
    the randomness of training itself.

    TAR@FAR sets its threshold again on every resample, at exactly that FAR, so
    both heads are compared at the same FAR. Val's thresholds would put them at
    slightly different FARs on another split. Accuracy uses val's threshold (k_acc).
    """

    def scores(name, weights=None):
        r = summarise(*pair_histograms(*embs[name], weights), fars=FARS)
        values = [r[f"TAR@FAR={f:g}"] for f in FARS] + [balanced_acc(r, k_acc[name])]
        return torch.tensor(values, dtype=torch.float64)

    point = scores("ArcFace") - scores("softmax")
    gen = torch.Generator().manual_seed(0)
    boot = []
    for _ in range(N_BOOT):
        drawn = torch.randint(n_ids, (n_ids,), generator=gen)
        w = torch.bincount(drawn, minlength=n_ids).to(DEVICE)
        boot.append(scores("ArcFace", w) - scores("softmax", w))
    lo, hi = torch.stack(boot).quantile(torch.tensor([0.025, 0.975]).double(), dim=0)
    return point, lo, hi


# --- scripts/further_work/low_quality_fixes.py

K = 300  # cohort normalisation uses each photo's 300 highest cosines with the cohort
# Each fix's scores are counted into BINS bins over this range. Cohort-normalised
# scores aren't cosines, so they get a wider one; histograms() checks that no
# score falls off either end.
RANGES = {"as is": (-1.0, 1.0), "unsure rule": (-1.0, 1.0), "cohort normalisation": (-60.0, 60.0)}


@torch.no_grad()
def cohort_stats(emb: torch.Tensor, cohort: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and spread (standard deviation) of each photo's K highest cosines with the cohort."""
    mus, sds = [], []
    for start in range(0, len(emb), 1024):
        top = (emb[start : start + 1024] @ cohort.T).topk(K, dim=1).values
        mus.append(top.mean(1))
        sds.append(top.std(1))
    return torch.cat(mus), torch.cat(sds)


def scorer(fix: str, d: dict, cutoff: float):
    """The function that turns a block of pair cosines into each fix's scores.
    rows are the block's photos; its columns are every photo."""
    if fix == "as is":
        return lambda rows, s: s
    if fix == "unsure rule":
        unsure = d["near"] >= cutoff
        # Pairs with an unsure photo get the lowest score, so no threshold accepts them.
        return lambda rows, s: s.masked_fill(unsure[rows, None] | unsure[None], -1.0)
    mu, sd = d["mu"], d["sd"]
    # How many spreads the cosine sits above each photo's usual score with
    # strangers, averaged over the two photos.
    return lambda rows, s: 0.5 * ((s - mu[rows, None]) / sd[rows, None] + (s - mu[None]) / sd[None])


@torch.no_grad()
def histograms(d: dict, fix: str, cutoff: float, identity_weights=None):
    """verify.pair_histograms, with each pair scored by the fix instead of by its cosine."""
    emb, labels = d["emb"], d["labels"]
    score, (lo, hi) = scorer(fix, d, cutoff), RANGES[fix]
    n = len(emb)
    if identity_weights is None:
        identity_weights = torch.ones(int(labels.max()) + 1, device=DEVICE)
    w = identity_weights.double()[labels]
    genuine = torch.zeros(BINS, dtype=torch.float64, device=DEVICE)
    impostor = torch.zeros_like(genuine)
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=DEVICE)
        s = score(rows, emb[rows] @ emb.T)
        bins = ((s - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1)
        later = torch.arange(n, device=DEVICE)[None] > rows[:, None]
        same = labels[rows, None] == labels[None]
        g, i = later & same, later & ~same
        genuine += torch.bincount(bins[g], weights=w[rows, None].expand_as(s)[g], minlength=BINS)
        impostor += torch.bincount(bins[i], weights=(w[rows, None] * w[None])[i], minlength=BINS)
    if fix == "cohort normalisation":
        assert genuine[[0, -1]].sum() + impostor[[0, -1]].sum() == 0, "scores ran off RANGES"
    return genuine.cpu(), impostor.cpu()
