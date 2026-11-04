from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """One split's faces as uint8 (3, 112, 112) tensors, labelled 0 to num_classes - 1.

    Labels are numbered within the split, so label 5 in train and label 5 in val are
    different people. They have no gaps because the head has one row per label.
    """

    def __init__(self, manifest_file: Path, root: Path, split_file: Path, split: str):
        rows = pd.read_parquet(manifest_file)
        rows = rows[rows["split"] == split]
        if rows.empty:
            raise ValueError(f"no {split!r} rows in {manifest_file}")
        # Checked against the committed split, not only the manifest's own column, so a
        # manifest built from another split can't bring val or test people into training.
        committed = pd.read_csv(split_file)
        allowed = set(committed.loc[committed["split"] == split, "identity"])
        strays = set(rows["identity"]) - allowed
        if strays:
            raise ValueError(
                f"{len(strays)} people in the manifest's {split} rows "
                f"aren't {split} in {split_file}"
            )

        identities = sorted(rows["identity"].unique())
        self.num_classes = len(identities)
        label_of = {identity: label for label, identity in enumerate(identities)}
        self.root = Path(root)
        # NumPy arrays, not lists: a DataLoader worker reading a Python object updates its
        # reference count, which copies that memory into every worker.
        self.paths = rows["path"].to_numpy(dtype=str)
        self.labels = rows["identity"].map(label_of).to_numpy()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.paths[i]) as png:
            pixels = np.array(png.convert("RGB"))  # (112, 112, 3): rows, columns, channels
        # PyTorch wants channels first: (3, 112, 112).
        return torch.from_numpy(pixels).permute(2, 0, 1), int(self.labels[i])
