import pandas as pd

from face_verif_gen.find_duplicates import shared_between_people, summarise

# Letters stand in for hashes. Person 1 has image a twice, people 1 and 2 share b and c, and
# people 3 and 4 share e across test and val. Person 2 comes first, so the order within a
# pair has to come from the code, not from the order of the rows.
MANIFEST = pd.DataFrame(
    [
        (2, "train", "b"),
        (2, "train", "c"),
        (1, "train", "a"),
        (1, "train", "a"),
        (1, "train", "b"),
        (1, "train", "c"),
        (3, "test", "d"),
        (3, "test", "e"),
        (4, "val", "e"),
    ],
    columns=["identity", "split", "sha256"],
)


def test_summary_of_a_hand_worked_example():
    assert summarise(MANIFEST).to_dict("index") == {
        "different people, train": {"images": 2, "rows": 4},  # b and c
        "different splits": {"images": 1, "rows": 2},  # e
        "same person, train": {"images": 1, "rows": 2},  # a
    }


def test_pairs_of_people_sharing_images():
    assert shared_between_people(MANIFEST).to_dict("records") == [
        {"person_a": 1, "person_b": 2, "split_a": "train", "split_b": "train", "shared": 2},
        {"person_a": 3, "person_b": 4, "split_a": "test", "split_b": "val", "shared": 1},
    ]


def test_no_copies_gives_no_pairs():
    assert shared_between_people(MANIFEST.drop_duplicates("sha256")).empty
