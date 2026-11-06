from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(path: Path) -> np.ndarray:
    """A PNG as a uint8 (3, height, width) array: channels first, as PyTorch wants."""
    with Image.open(path) as png:
        pixels = np.array(png.convert("RGB"))  # (height, width, 3): rows, columns, channels
    return pixels.transpose(2, 0, 1)


class FaceDataset(Dataset):
    """One split's faces as uint8 (3, 112, 112) tensors, labelled 0 to num_classes - 1.

    Labels are numbered within the split, so label 5 in train and label 5 in val are
    different people. They have no gaps because the head has one row per label.

    Images are decoded from the PNGs, or read from the array prepare_data.py decodes
    them into once, if one is given.
    """

    def __init__(
        self,
        manifest_file: Path,
        root: Path,
        split_file: Path,
        split: str,
        array: Path | None = None,
    ):
        manifest = pd.read_parquet(manifest_file)
        in_split = (manifest["split"] == split).to_numpy()
        rows = manifest[in_split]
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
        if array is not None:
            images = np.load(array, mmap_mode="r")  # reads the header, not the images
            if len(images) != len(manifest):
                raise ValueError(
                    f"{array} doesn't hold one image per row of {manifest_file}: "
                    "rebuild it with prepare_data.py"
                )
            # Where the images start in the file, after its header, and the size of each.
            self.offset, self.image_shape = images.offset, images.shape[1:]
            self.image_bytes = images[0].nbytes

        identities = sorted(rows["identity"].unique())
        self.num_classes = len(identities)
        label_of = {identity: label for label, identity in enumerate(identities)}
        self.root = Path(root)
        # NumPy arrays, not lists: a DataLoader worker reading a Python object updates its
        # reference count, which copies that memory into every worker.
        self.paths = rows["path"].to_numpy(dtype=str)
        self.labels = rows["identity"].map(label_of).to_numpy()
        self.rows = np.flatnonzero(in_split)  # the manifest's row numbers, which are the array's
        self.array = array

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self.array is None:
            return torch.from_numpy(read_image(self.root / self.paths[i])), int(self.labels[i])
        # One read per image. Reading random rows through a memmap instead makes Linux
        # fetch about 2 MB around each 37 KB image, which made it slower than the PNGs.
        pixels = bytearray(self.image_bytes)
        with open(self.array, "rb", buffering=0) as f:  # unbuffered, so a single read
            f.seek(self.offset + self.rows[i] * self.image_bytes)
            # One read can return fewer bytes than asked, which would leave the rest black.
            if f.readinto(pixels) != self.image_bytes:
                raise OSError(f"short read of row {self.rows[i]} from {self.array}")
        image = torch.frombuffer(pixels, dtype=torch.uint8).reshape(self.image_shape)
        return image, int(self.labels[i])
