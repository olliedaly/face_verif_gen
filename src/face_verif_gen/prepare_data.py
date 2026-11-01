from pathlib import Path

import hydra
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from omegaconf import DictConfig


def download(repo: str, revision: str) -> Path:
    """Download the parquet shards at a pinned commit and return the folder holding them."""
    folder = snapshot_download(
        repo, repo_type="dataset", revision=revision, allow_patterns=["data/*.parquet"]
    )
    return Path(folder)


def unpack(shard_dir: Path, root: Path) -> None:
    """Write every image to root/<label:05d>/<n:06d>.png, then mark root as complete."""
    n = 0  # counts across all shards, so every image gets its own number
    for shard in sorted(shard_dir.glob("data/*.parquet")):  # glob order isn't guaranteed
        for row in pq.read_table(shard).to_pylist():
            png = row["image"]["bytes"]
            if not png.startswith(b"\x89PNG"):
                raise ValueError(f"image {n} in {shard.name} is not a PNG")
            folder = root / f"{row['label']:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            # The bytes are written as stored: no decoding, so every pixel stays exact.
            (folder / f"{n:06d}.png").write_bytes(png)
            n += 1
        print(f"unpacked {shard.name} ({n} images so far)")
    (root / ".complete").touch()  # last, so a crash halfway leaves no marker


def split_identities(identities: list[int], n_val, n_test, seed) -> dict[int, str]:
    """Maps each person to train, val or test."""


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.data.root)
    if (root / ".complete").exists():
        print(f"{root} is already unpacked")
        return
    unpack(download(cfg.data.hf_repo, cfg.data.revision), root)


if __name__ == "__main__":
    main()
