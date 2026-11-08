import numpy as np
import pytest

from face_verif_gen.plot_long_tail import plot, summarise

# 10 people with 100 images in all, unsorted. Worked by hand: the largest person holds half
# the images, and 8 have fewer than 20, because the person with exactly 20 doesn't count.
COUNTS = np.array([5, 20, 1, 50, 5, 4, 3, 5, 2, 5])


def test_summary_of_a_hand_worked_example():
    expected = {"people": 10, "images": 100, "min": 1, "median": 5, "mean": 10, "max": 50}
    expected |= {"% in largest 10%": 50, "% under 20": 80}
    assert summarise(COUNTS) == pytest.approx(expected)  # approx: the percentages are floats


def test_plot_writes_a_png(tmp_path):
    plot(COUNTS, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").read_bytes().startswith(b"\x89PNG")
