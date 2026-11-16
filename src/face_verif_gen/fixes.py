"""Two fixes for unreadable photos of different people matching each other. Neither retrains the
model: both change only how a pair's cosine becomes a score. Both are built from training photos.

Unsure rule (Deng et al., WACV 2023). Ruin training photos so no one could tell who they are, and
average their embeddings: the model's "can't tell who this is" direction. A photo whose cosine
with that direction is at or above a cutoff is unsure, and any pair containing it is rejected.

Cohort normalisation (AS-norm, from speaker verification). Compare each photo with a fixed set of
training photos, the cohort. The mean and standard deviation of its k highest cosines say how
readily it matches strangers. A pair's score is how many standard deviations its cosine sits
above each photo's usual match with strangers, averaged over the two photos. Unreadable photos
match strangers widely, so their high cosines count for less.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from face_verif_gen.verify import Score

BLOCK = 80  # side of the grey square: about half of a 112 x 112 face
GREY = 128  # mid grey, which preprocess maps to about 0
# Cohort-normalised scores are standard deviations, not cosines, so they need a wider range.
COHORT_RANGE = (-60.0, 60.0)


def ruin(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Each uint8 photo ruined one of three ways, at random, each enough to hide who it is:
    shrunk to 8 x 8 pixels and back, blurred (sigma 6), or half covered by a grey square."""
    n, size = len(images), images.shape[-1]
    x = images.float()
    kind = torch.randint(3, (n,), generator=generator, device=x.device)[:, None, None, None]
    small = F.interpolate(F.interpolate(x, size=8, mode="area"), size=size, mode="bilinear")
    blurred = gaussian_blur(x, kernel_size=31, sigma=6.0)
    top, left = torch.randint(size - BLOCK + 1, (2, n), generator=generator, device=x.device)
    p = torch.arange(size, device=x.device)
    in_rows = (p[None] >= top[:, None]) & (p[None] < top[:, None] + BLOCK)  # (n, size)
    in_columns = (p[None] >= left[:, None]) & (p[None] < left[:, None] + BLOCK)
    square = (in_rows[:, :, None] & in_columns[:, None, :])[:, None]  # (n, 1, size, size)
    covered = x.masked_fill(square, GREY)
    ruined = torch.where(kind == 0, small, torch.where(kind == 1, blurred, covered))
    return ruined.round().clamp(0, 255).to(torch.uint8)


def ruiner(device: torch.device, seed: int = 0):
    """ruin with its own generator, so the same photos are ruined the same way on every call."""
    return partial(ruin, generator=torch.Generator(device).manual_seed(seed))


def unsure_direction(ruined: torch.Tensor) -> torch.Tensor:
    """The unit-length mean of ruined photos' embeddings: "can't tell who this is"."""
    return F.normalize(ruined.mean(0), dim=0)


def unsure_cutoff(val_near: torch.Tensor, share: float) -> float:
    """The cosine with the unsure direction above which share of val photos lie."""
    return float(torch.quantile(val_near.double(), 1 - share))


def unsure_rule(near: torch.Tensor, cutoff: float) -> Score:
    """Scores as cosines, except that a pair with an unsure photo gets -1: no threshold above -1
    accepts it. near is each photo's cosine with the unsure direction."""
    unsure = near >= cutoff

    def score(rows: torch.Tensor, cosines: torch.Tensor) -> torch.Tensor:
        return cosines.clamp(-1, 1).masked_fill(unsure[rows, None] | unsure[None], -1.0)

    return score


@torch.inference_mode()
def cohort_stats(
    emb: torch.Tensor, cohort: torch.Tensor, k: int, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of each photo's k highest cosines with the cohort's photos."""
    means, sds = [], []
    for start in range(0, len(emb), chunk):
        top = (emb[start : start + chunk] @ cohort.T).topk(k, dim=1).values
        means.append(top.mean(1))
        sds.append(top.std(1))
    return torch.cat(means), torch.cat(sds)


def cohort_normalisation(mean: torch.Tensor, sd: torch.Tensor) -> Score:
    """Each pair's cosine as standard deviations above each photo's cohort mean, averaged."""

    def score(rows: torch.Tensor, cosines: torch.Tensor) -> torch.Tensor:
        from_row = (cosines - mean[rows, None]) / sd[rows, None]
        from_column = (cosines - mean[None]) / sd[None]
        return (from_row + from_column) / 2

    return score
