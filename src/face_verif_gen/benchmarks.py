"""The five standard benchmarks: LFW, CFP-FP, CPLFW, CALFW and AgeDB-30, as InsightFace .bin files.

Each file holds image pairs and whether each pair is the same person, scored by 10-fold accuracy
(metrics.kfold_accuracy). Each embedding is the photo's plus its mirror image's, as AdaFace and
InsightFace score them. A sanity check against published numbers only: this project's results
are TAR at FAR on CASIA's own held-out people.

  uv run python -m face_verif_gen.benchmarks eval.weights=adaface
"""

import io
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy._core.multiarray import _reconstruct
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import TensorDataset

from face_verif_gen import metrics
from face_verif_gen.evaluate import load_model
from face_verif_gen.verify import embed

# AdaFace's published accuracies for its IResNet-18 trained on CASIA-WebFace (their README).
ADAFACE_R18_CASIA = {
    "lfw": 0.9913,
    "cfp_fp": 0.9259,
    "cplfw": 0.8700,
    "calfw": 0.9265,
    "agedb_30": 0.9272,
}


class ArrayUnpickler(pickle.Unpickler):
    """Unpickles NumPy arrays and nothing else.

    A pickle can name any Python function to call while loading, which runs code from the file.
    These files only need the three names that rebuild a NumPy array.
    """

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct,  # as NumPy 1 named it
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a benchmark file")
        return self.ALLOWED[module, name]


def load_bin(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """A .bin file's images as uint8 RGB (2 * pairs, 3, 112, 112), photos 2i and 2i + 1 being pair
    i, and whether each pair is the same person."""
    with open(path, "rb") as f:
        jpegs, same = ArrayUnpickler(f, encoding="bytes").load()
    images = np.stack(
        [np.asarray(Image.open(io.BytesIO(bytes(jpeg))).convert("RGB")) for jpeg in jpegs]
    )
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), np.asarray(same, dtype=bool)


def accuracy(model, images: torch.Tensor, same: np.ndarray, device, batch_size=256) -> float:
    """Mean 10-fold accuracy of a model in eval mode on one benchmark."""
    photos = TensorDataset(images, torch.zeros(len(images)))
    emb = embed(model, photos, range(len(images)), device, batch_size, flip=True)
    cosines = (emb[0::2] * emb[1::2]).sum(1).cpu().numpy()
    return float(metrics.kfold_accuracy(cosines, same).mean())


def download(cfg: DictConfig) -> Path:
    """The .bin files at the pinned commit, in cfg.eval.benchmarks.root; returns that folder."""
    b = cfg.eval.benchmarks
    for name in b.names:
        hf_hub_download(
            b.hf_repo, f"{name}.bin", repo_type="dataset", revision=b.revision, local_dir=b.root
        )
    return Path(b.root)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg).to(device).eval()
    root = download(cfg)
    results = {}
    for name in cfg.eval.benchmarks.names:
        images, same = load_bin(root / f"{name}.bin")
        results[name] = accuracy(model, images, same, device, cfg.eval.batch_size)
        published = ADAFACE_R18_CASIA[name]
        print(f"{name:9s} {results[name]:.2%}  (AdaFace R18 published {published:.2%})")
    mean = np.mean(list(results.values()))
    print(f"{'average':9s} {mean:.2%}  (published {np.mean(list(ADAFACE_R18_CASIA.values())):.2%})")
    print(
        "AdaFace's published CFP-FP matches its model fed swapped colour channels; see "
        "docs/casia_datasheet.md"
    )


if __name__ == "__main__":
    main()
