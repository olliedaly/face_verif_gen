import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import wandb
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from face_verif_gen.data import read_image
from face_verif_gen.run_meta import file_sha256


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


def split_identities(identities: list[int], n_val: int, n_test: int, seed: int) -> dict[int, str]:
    """Map each person to train, val or test, so no person's photos are in two splits."""
    # Sorted first because the shuffle depends on input order, and folder listings have none.
    shuffled = random.Random(seed).sample(sorted(identities), len(identities))
    test, val = shuffled[:n_test], shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return dict.fromkeys(test, "test") | dict.fromkeys(val, "val") | dict.fromkeys(train, "train")


def build_manifest(root: Path, split: dict[int, str]) -> pd.DataFrame:
    """One row per image: its path under root, its person, their split and the image's sha256."""
    rows = []
    for png in sorted(root.glob("*/*.png")):  # sorted so the same images give the same file
        identity = int(png.parent.name)
        rows.append(
            {
                "path": png.relative_to(root).as_posix(),  # relative, so root can move
                "identity": identity,
                "split": split[identity],  # KeyError if a person has no split
                "sha256": file_sha256(png),
            }
        )
    return pd.DataFrame(rows)


def log_manifest(manifest_file: Path, sha256: str, cfg: DictConfig) -> None:
    """Log the manifest as a W&B Artifact. An unchanged file adds no new version."""
    with wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        job_type="prepare-data",
        config=OmegaConf.to_container(cfg.data, resolve=True),
    ) as run:
        artifact = wandb.Artifact(
            cfg.data.artifact,
            type="dataset",
            metadata={"sha256": sha256, "hf_repo": cfg.data.hf_repo, "revision": cfg.data.revision},
        )
        artifact.add_file(str(manifest_file))
        version = run.log_artifact(artifact).wait().version  # known once the upload finishes
        print(f"logged {cfg.data.artifact}:{version}; runs use data.artifact_version")


def write_array(manifest: pd.DataFrame, root: Path, array_file: Path) -> None:
    """Decode every image into one uint8 (N, 3, 112, 112) .npy file, row i being manifest row i."""
    shape = (len(manifest), *read_image(root / manifest["path"].iloc[0]).shape)
    # Filled under a temporary name, then renamed: the file starts as all zeros, so an
    # interrupted run would otherwise leave black images in a file that looks finished.
    partial = array_file.with_name(array_file.name + ".partial")
    images = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
    for i, path in enumerate(manifest["path"]):
        images[i] = read_image(root / path)
    images.flush()
    partial.replace(array_file)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.data.root)
    if (root / ".complete").exists():
        print(f"{root} is already unpacked")
    else:
        unpack(download(cfg.data.hf_repo, cfg.data.revision), root)

    identities = [int(p.name) for p in root.iterdir() if p.is_dir()]  # is_dir skips .complete
    split = split_identities(identities, cfg.data.n_val, cfg.data.n_test, cfg.data.split_seed)
    split_file = Path(cfg.data.split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(split.items())
    pd.DataFrame(rows, columns=["identity", "split"]).to_csv(split_file, index=False)
    print(f"wrote {split_file}")

    manifest_file = Path(cfg.data.manifest)
    manifest = build_manifest(root, split)
    manifest.to_parquet(manifest_file, index=False)
    sha256 = file_sha256(manifest_file)
    print(f"wrote {manifest_file}: {len(manifest)} images, sha256 {sha256}")
    log_manifest(manifest_file, sha256, cfg)

    array_file = Path(cfg.data.array)
    print(f"decoding {len(manifest)} images into {array_file}, which can take 15 minutes")
    write_array(manifest, root, array_file)
    print(f"wrote {array_file}: {array_file.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
