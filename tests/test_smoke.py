import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from face_verif_gen import evaluate, train
from face_verif_gen.prepare_data import build_manifest

CONFIGS = (Path(__file__).parents[1] / "configs").resolve()
# Every option of every config group, e.g. ("head", "arcface"), read from disk so a new
# config file is tested without anyone editing this list.
OPTIONS = sorted((p.parent.name, p.stem) for p in CONFIGS.glob("*/*.yaml"))


def load_config(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose("config", overrides=overrides)


@pytest.mark.parametrize(("group", "option"), OPTIONS, ids=[f"{g}={o}" for g, o in OPTIONS])
def test_every_config_composes(group, option):
    cfg = load_config([f"{group}={option}"])
    OmegaConf.to_container(cfg, resolve=True)  # raises if any ${...} reference is broken
    # A run launched as head=softmax must not train whatever softmax.yaml's name says instead.
    if "name" in cfg[group]:
        assert cfg[group].name == option


def test_dirty_tree_is_refused(monkeypatch):
    # Patch the name train.py looks up, not the original in run_meta.
    monkeypatch.setattr(train, "run_metadata", lambda manifest_path: {"git_dirty": True})
    cfg = load_config(["wandb.mode=disabled"])
    with pytest.raises(SystemExit, match="Uncommitted changes"):
        train.run(cfg)


def test_dirty_tree_runs_when_allowed(monkeypatch):
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true"])
    # The pinned hash, so the manifest check passes and only the dirty tree is tested.
    meta = {"git_dirty": True, "manifest_sha256": cfg.data.manifest_sha256}
    monkeypatch.setattr(train, "run_metadata", lambda manifest_path: meta)
    train.check_inputs(cfg)  # not run(): that would go on to train on CASIA


def fake_run(sha256: str):
    """Stands in for a W&B run whose use_artifact returns an artifact recording this hash."""
    artifact = SimpleNamespace(metadata={"sha256": sha256})
    return SimpleNamespace(use_artifact=lambda _: artifact)


def test_matching_artifact_is_linked():
    train.link_manifest(fake_run("abc"), "casia-manifest:v0", sha256="abc")


def test_artifact_of_other_data_is_refused():
    with pytest.raises(SystemExit, match="not the local manifest"):
        train.link_manifest(fake_run("abc"), "casia-manifest:v0", sha256="def")


@pytest.fixture
def fake_manifest(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"fake")
    return manifest


def test_wrong_manifest_is_refused(fake_manifest):
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true", f"data.manifest={fake_manifest}"])
    with pytest.raises(SystemExit, match="config pins"):
        train.run(cfg)


def test_new_manifest_is_accepted_once_pinned(fake_manifest):
    sha256 = hashlib.sha256(b"fake").hexdigest()  # pin the fake file, as a data change would
    cfg = load_config(
        [
            "wandb.mode=disabled",
            "allow_dirty=true",
            f"data.manifest={fake_manifest}",
            f"data.manifest_sha256={sha256}",
        ]
    )
    train.check_inputs(cfg)


@pytest.fixture
def tiny_casia(tmp_path):
    """Images, split file and manifest as prepare_data.py lays them out: 2 training people with
    2 photos each, and 6 val and 6 test people with 3 each. Random pixels."""
    rng = np.random.default_rng(0)
    split = {0: "train", 1: "train"} | dict.fromkeys(range(2, 8), "val")
    split |= dict.fromkeys(range(8, 14), "test")
    n = 0
    for identity, which in split.items():
        folder = tmp_path / "casia" / f"{identity:05d}"
        folder.mkdir(parents=True)
        for _ in range(2 if which == "train" else 3):
            pixels = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{n:06d}.png")
            n += 1
    rows = sorted(split.items())
    pd.DataFrame(rows, columns=["identity", "split"]).to_csv(tmp_path / "split.csv", index=False)
    build_manifest(tmp_path / "casia", split).to_parquet(tmp_path / "manifest.parquet", index=False)
    return tmp_path


def test_train_then_evaluate(tiny_casia, capsys):
    """T14: a 2-step run of train, its checkpoint, evaluate and the results table, on the CPU."""
    manifest = tiny_casia / "manifest.parquet"
    run_dir = tiny_casia / "run"
    cfg = load_config(
        [
            "wandb.mode=disabled",
            "allow_dirty=true",
            f"out_dir={run_dir}",
            f"data.root={tiny_casia / 'casia'}",
            f"data.manifest={manifest}",
            f"data.manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
            f"data.split_file={tiny_casia / 'split.csv'}",
            "data.array=null",
            "train.batch_size=2",  # 4 photos: 2 steps
            "train.num_workers=0",
            "eval.num_workers=0",
            "eval.probe_size=6",
            "eval.cohort_k=2",  # of a cohort of 4
            "eval.n_resamples=5",
            f"eval.weights={run_dir / 'checkpoint.pt'}",
        ]
    )
    train.run(cfg)
    results = evaluate.run(cfg)
    assert (run_dir / "results_val.json").exists()
    for fix in evaluate.FIXES:
        for far in cfg.eval.far_targets:
            assert 0 <= results[f"{fix}: TAR at FAR {far:g}"] <= 1
    assert "| as is |" in capsys.readouterr().out
