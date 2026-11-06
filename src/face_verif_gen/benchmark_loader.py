"""Time loading the training split from PNGs and from the decoded array, with no model attached."""

import os
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

from face_verif_gen.data import FaceDataset

N_IMAGES = 50_000  # a random sample of the training split, long enough for a steady rate
BATCH_SIZE = 256
WORKERS = 8


def evict(files: list[Path]) -> None:
    """Drop the files from the OS page cache, so the next read comes from disk."""
    for file in files:
        fd = os.open(file, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # unlike drop_caches, needs no root
        os.close(fd)


def images_per_second(dataset: Dataset) -> float:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)
    start = time.perf_counter()
    for _ in loader:
        pass
    return len(dataset) / (time.perf_counter() - start)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    array_file = Path(d.array)
    pngs = FaceDataset(d.manifest, d.root, d.split_file, "train")
    array = FaceDataset(d.manifest, d.root, d.split_file, "train", array_file)
    # Random order, as in shuffled training. Both sources read the same images.
    sample = np.random.default_rng(cfg.seed).choice(len(pngs), N_IMAGES, replace=False)
    print(f"{N_IMAGES} training images, batches of {BATCH_SIZE}, {WORKERS} workers")
    for name, dataset, files in [
        ("PNGs", pngs, [pngs.root / path for path in pngs.paths[sample]]),
        ("array", array, [array_file]),
    ]:
        evict(files)
        cold = images_per_second(Subset(dataset, sample))  # read from disk
        warm = images_per_second(Subset(dataset, sample))  # read again, now from RAM
        print(f"{name:>5}: {cold:7,.0f} images/s cold, {warm:7,.0f} images/s warm")


if __name__ == "__main__":
    main()
