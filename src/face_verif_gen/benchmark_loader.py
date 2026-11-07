"""Time loading the training split from PNGs and from the decoded array, with no model attached."""

import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from face_verif_gen.data import FaceDataset

BATCH_SIZE = 256
WORKERS = 8
EPOCHS = 2  # the first reads from disk; the second shows how much of the split stays in RAM


def evict(files: list[Path]) -> None:
    """Drop the files from the OS page cache, so the next read comes from disk."""
    for file in files:
        fd = os.open(file, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # unlike drop_caches, needs no root
        os.close(fd)


def available_ram_gb() -> float:
    """RAM the kernel could give to programs or the page cache, which decides the second epoch."""
    with open("/proc/meminfo") as f:
        kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
    return kb * 1024 / 1e9


def images_per_second(dataset: Dataset, seed: int) -> list[float]:
    """The rate of each epoch through the whole dataset in shuffled order, as in training."""
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        persistent_workers=True,  # workers start once, as they would in training
        generator=torch.Generator().manual_seed(seed),  # same order for both sources
    )
    rates = []
    for _ in range(EPOCHS):
        start = time.perf_counter()
        for _ in loader:
            pass
        rates.append(len(dataset) / (time.perf_counter() - start))
    return rates


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    array_file = Path(d.array)
    pngs = FaceDataset(d.manifest, d.root, d.split_file, "train")
    array = FaceDataset(d.manifest, d.root, d.split_file, "train", array_file)
    png_files = [pngs.root / path for path in pngs.paths]
    print(
        f"{len(pngs):,} training images, batches of {BATCH_SIZE}, {WORKERS} workers, "
        f"{available_ram_gb():.1f} GB of RAM available"
    )
    for name, dataset in [("PNGs", pngs), ("array", array)]:
        # Both sources, so pages left from the other's run don't take RAM this one could use.
        evict([*png_files, array_file])
        rates = images_per_second(dataset, cfg.seed)
        epochs = ", ".join(f"epoch {e} {rate:7,.0f}" for e, rate in enumerate(rates, start=1))
        print(f"{name:>5}: {epochs} images/s")


if __name__ == "__main__":
    main()
