"""Benchmark files load safely and correctly, and AdaFace's released model reproduces its published
benchmark accuracies with our evaluation code.

The reproduction is M2's check that evaluation is right: the same weights and protocol must give
the same numbers. Measured: CPLFW, CALFW and AgeDB-30 exactly, LFW 0.09 points high (5 of 6,000
pairs; their copy or JPEG decoder may differ slightly).

CFP-FP is the exception. Read as RGB, like every other file, AdaFace scores 94.99%, 2.4 points
above the published 92.59%; with its colour channels swapped it scores 92.60%, and the other four
get worse when swapped. The file decodes with red above green above blue, as natural faces do and
as CASIA and the other four do, so RGB is right for this file. AdaFace's convert.py reads all five
files the same way, so their copy of cfp_fp.bin most likely stores its channels the other way
round from this one. It is reproduced here with them swapped.
"""

import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from face_verif_gen.benchmarks import ADAFACE_R18_CASIA, accuracy, load_bin
from face_verif_gen.model import load_adaface

REPO = Path(__file__).parents[1]
EVAL = OmegaConf.load(REPO / "configs" / "eval" / "default.yaml")
BENCHMARKS = REPO / EVAL.benchmarks.root


def encoded(image: np.ndarray) -> np.ndarray:
    """An (H, W, 3) image as the files store each one: encoded bytes in a uint8 array."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")  # lossless, unlike the files' JPEG
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)[:, None]


def test_load_bin_gives_rgb_pairs(tmp_path):
    # Two pairs. Each image is one flat colour, red, green and blue different, so a swapped or
    # misplaced channel shows.
    colours = [(200, 100, 50), (10, 20, 30), (1, 2, 3), (250, 0, 125)]
    images = [np.full((112, 112, 3), colour, dtype=np.uint8) for colour in colours]
    path = tmp_path / "toy.bin"
    path.write_bytes(pickle.dumps(([encoded(image) for image in images], [True, False])))
    loaded, same = load_bin(path)
    assert loaded.shape == (4, 3, 112, 112) and loaded.dtype == torch.uint8
    assert [tuple(image[:, 0, 0].tolist()) for image in loaded] == colours
    assert same.tolist() == [True, False]


class RunsCode:
    """Unpickling this would call print: a stand-in for anything a malicious file could run."""

    def __reduce__(self):
        return print, ("this ran",)


def test_load_bin_refuses_anything_but_arrays(tmp_path, capsys):
    path = tmp_path / "bad.bin"
    path.write_bytes(pickle.dumps(([RunsCode()], [True])))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_bin(path)
    assert capsys.readouterr().out == ""


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.skipif(not (REPO / EVAL.adaface_checkpoint).exists(), reason="needs AdaFace's")
@pytest.mark.skipif(not (BENCHMARKS / "lfw.bin").exists(), reason="needs the benchmark files")
def test_adaface_reproduces_its_published_accuracy():
    device = torch.device("cuda")
    model = load_adaface(REPO / EVAL.adaface_checkpoint).to(device).eval()
    found = {}
    for name in ADAFACE_R18_CASIA:
        images, same = load_bin(BENCHMARKS / f"{name}.bin")
        if name == "cfp_fp":
            images = images[:, [2, 1, 0]]  # as AdaFace's published number was measured
        found[name] = accuracy(model, images, same, device)
    # 0.15 points: LFW's 0.09 plus room for a pair or two flipping on another GPU or driver.
    assert found == pytest.approx(ADAFACE_R18_CASIA, abs=0.0015)
    average = np.mean(list(found.values()))
    assert average == pytest.approx(0.9282, abs=0.0005)  # the plan's 92.82%
