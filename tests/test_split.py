from pathlib import Path

import pandas as pd
import pytest

from face_verif_gen.prepare_data import split_identities

SPLIT_FILE = Path(__file__).parents[1] / "splits" / "casia.csv"
N_IDENTITIES = 10572


@pytest.fixture
def split():
    return pd.read_csv(SPLIT_FILE)


def test_every_person_appears_once(split):
    # The file is sorted by identity, so this one check means none missing and none in two splits.
    assert split["identity"].tolist() == list(range(N_IDENTITIES))


def test_split_sizes(split):
    assert split["split"].value_counts().to_dict() == {"train": 9072, "val": 500, "test": 1000}


def test_file_matches_code(split):
    # Fails if the CSV is edited by hand, or the code changes so it would write a different split.
    expected = split_identities(list(range(N_IDENTITIES)), n_val=500, n_test=1000, seed=0)
    assert dict(zip(split["identity"].tolist(), split["split"], strict=True)) == expected


def test_input_order_does_not_matter():
    identities = list(range(100))
    forwards = split_identities(identities, n_val=10, n_test=20, seed=0)
    assert split_identities(identities[::-1], n_val=10, n_test=20, seed=0) == forwards
