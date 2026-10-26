import hashlib
import platform
import random
import subprocess

import numpy as np
import torch


def _cmd(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_metadata(manifest_path: str) -> dict:
    meta = {
        "git_commit": _cmd("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_cmd("git", "status", "--porcelain")),
        "manifest_sha256": file_sha256(manifest_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["driver"] = _cmd("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    return meta


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
