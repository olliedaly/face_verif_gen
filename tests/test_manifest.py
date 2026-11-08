import hashlib
from pathlib import Path

import pandas as pd
import pytest

from face_verif_gen.prepare_data import build_manifest
from face_verif_gen.run_meta import file_sha256

REPO = Path(__file__).parents[1]
MANIFEST = REPO / "data" / "casia" / "manifest.parquet"
SPLIT = {0: "train", 1: "test"}


@pytest.fixture
def root(tmp_path):
    """A tiny dataset laid out like data/casia: person 0 has two images, person 1 has one."""
    # The bytes don't have to be real PNGs, because the manifest only hashes them.
    for path, data in [
        ("00000/000000.png", b"a"),
        ("00000/000001.png", b"b"),
        ("00001/000002.png", b"c"),
    ]:
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_bytes(data)
    (tmp_path / ".complete").touch()  # not an image, so it must not get a row
    return tmp_path


def test_one_row_per_image(root):
    def row(path, identity, split, data):
        return {
            "path": path,
            "identity": identity,
            "split": split,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    assert build_manifest(root, SPLIT).to_dict("records") == [
        row("00000/000000.png", 0, "train", b"a"),
        row("00000/000001.png", 0, "train", b"b"),
        row("00001/000002.png", 1, "test", b"c"),
    ]


def test_person_without_a_split_is_refused(root):
    with pytest.raises(KeyError):
        build_manifest(root, {0: "train"})


def test_rebuilding_gives_the_same_file(root):
    # The manifest's hash names the data version, so the same images must give the same hash.
    build_manifest(root, SPLIT).to_parquet(root / "a.parquet", index=False)
    build_manifest(root, SPLIT).to_parquet(root / "b.parquet", index=False)
    assert file_sha256(root / "a.parquet") == file_sha256(root / "b.parquet")


@pytest.mark.skipif(not MANIFEST.exists(), reason="needs the real data: run prepare_data.py")
def test_real_manifest_matches_committed_split():
    manifest = pd.read_parquet(MANIFEST)
    committed = pd.read_csv(REPO / "splits" / "casia.csv").set_index("identity")["split"]
    assert len(manifest) == 490_592
    assert manifest["identity"].nunique() == len(committed)  # every person has images
    assert (manifest["split"] == manifest["identity"].map(committed)).all()
