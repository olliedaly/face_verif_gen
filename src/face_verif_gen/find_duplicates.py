"""Find images stored more than once: rows of the manifest with the same sha256."""

import itertools
from collections import Counter

import hydra
import pandas as pd
from omegaconf import DictConfig


def summarise(manifest: pd.DataFrame) -> pd.DataFrame:
    """Images stored more than once, grouped by where their copies are.

    Each image is counted in exactly one group, so the groups add up to the total.
    """
    copies = manifest[manifest["sha256"].duplicated(keep=False)]  # every row of a repeated image
    per_image = copies.groupby("sha256").agg(
        rows=("split", "size"),
        people=("identity", "nunique"),
        splits=("split", "nunique"),
        split=("split", "first"),
    )
    who = per_image["people"].gt(1).map({True: "different people", False: "same person"})
    # Copies in two splits are in neither split's group: no single split holds them all.
    where = (who + ", " + per_image["split"]).where(per_image["splits"] == 1, "different splits")
    return per_image.groupby(where).agg(images=("rows", "size"), rows=("rows", "sum"))


def shared_between_people(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per pair of people filed with the same image, most shared images first."""
    copies = manifest[manifest["sha256"].duplicated(keep=False)]
    shared = Counter(
        pair
        for people in copies.groupby("sha256")["identity"].unique()
        for pair in itertools.combinations(sorted(people), 2)  # sorted, so (a, b) never (b, a)
    )
    split_of = dict(zip(manifest["identity"], manifest["split"], strict=True))
    rows = [(a, b, split_of[a], split_of[b], n) for (a, b), n in shared.items()]
    pairs = pd.DataFrame(rows, columns=["person_a", "person_b", "split_a", "split_b", "shared"])
    return pairs.sort_values("shared", ascending=False, kind="stable")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    manifest = pd.read_parquet(cfg.data.manifest)
    table = summarise(manifest)
    table.loc["total"] = table.sum()
    print("images stored more than once, by where their copies are:")
    print(table.to_string())
    print("\npeople filed with the same image:")
    print(shared_between_people(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
