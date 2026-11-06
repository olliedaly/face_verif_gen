"""Plot how many images each person has, to show how long the dataset's tail is."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from omegaconf import DictConfig

SURFACE = "#fcfcfb"  # background
INK = "#0b0b0b"  # title
INK_2 = "#52514e"  # subtitle, labels and axis titles
MUTED = "#898781"  # tick labels
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"  # the data


def summarise(counts: np.ndarray) -> dict:
    """Headline numbers for a group of people, given how many images each one has."""
    counts = np.sort(counts)[::-1]  # largest first
    largest = counts[: len(counts) // 10]  # the 10% of people with the most images
    return {
        "people": len(counts),
        "images": counts.sum(),
        "min": counts.min(),
        "median": np.median(counts),
        "mean": counts.mean(),
        "max": counts.max(),
        "% in largest 10%": 100 * largest.sum() / counts.sum(),
        "% under 20": 100 * (counts < 20).mean(),
    }


def compact(value: float, _) -> str:
    """Axis tick text: 500, 2K, 10K."""
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


def plot(counts: np.ndarray, out: Path) -> None:
    """Draw people ranked by image count, largest first, with images on a log scale."""
    counts = np.sort(counts)[::-1]
    summary = summarise(counts)
    rank = np.arange(1, len(counts) + 1)

    # Figure directly, not pyplot: it only saves a file, so it needs no window or backend.
    fig = Figure(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes((0.08, 0.13, 0.88, 0.65))  # left, bottom, width, height
    ax.set_facecolor(SURFACE)
    ax.fill_between(rank, counts, 1, color=SERIES, alpha=0.1, linewidth=0)
    ax.plot(rank, counts, color=SERIES, linewidth=1, solid_capstyle="round")

    # Label three people: the most images, the median and the fewest.
    middle = len(counts) // 2
    for i, text, offset, align in [
        (0, f"{counts[0]} images", (8, -3), "left"),
        (middle, f"median: {summary['median']:g} images", (6, 8), "left"),
        (len(counts) - 1, f"{counts[-1]} images", (-8, 8), "right"),
    ]:
        # clip_on=False, because the end dots sit on the plot's edges and would be cut in half.
        ax.plot(rank[i], counts[i], "o", color=SERIES, markersize=5, mec=SURFACE, clip_on=False)
        ax.annotate(
            text,
            (rank[i], counts[i]),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            fontsize=9,
            color=INK_2,
        )

    ax.set_yscale("log")
    ax.set_ylim(1.5, 1000)
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500, 1000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, len(counts))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(compact))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, labelcolor=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_xlabel("People, ranked by number of images", fontsize=9, color=INK_2)
    ax.set_ylabel("Images (log scale)", fontsize=9, color=INK_2)

    fig.text(0.08, 0.96, "Images per person", fontsize=13, weight="bold", color=INK, va="top")
    subtitle = (
        f"{summary['people']:,} people, ranked by image count.\n"
        f"The largest 10% hold {summary['% in largest 10%']:.0f}% of the images, "
        f"and {summary['% under 20']:.0f}% have fewer than 20."
    )
    fig.text(0.08, 0.895, subtitle, fontsize=9.5, color=INK_2, va="top", linespacing=1.4)
    fig.savefig(out)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    manifest = pd.read_parquet(cfg.data.manifest)
    counts = {"all": manifest["identity"].value_counts().to_numpy()}
    for split, rows in manifest.groupby("split"):
        counts[split] = rows["identity"].value_counts().to_numpy()
    table = pd.DataFrame.from_dict({name: summarise(c) for name, c in counts.items()}, "index")
    print(table.to_string(float_format="{:.1f}".format))

    out = Path("docs") / f"{cfg.data.name}_images_per_identity.png"
    out.parent.mkdir(exist_ok=True)
    plot(counts["all"], out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
