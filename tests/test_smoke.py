import hashlib
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import wandb
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
    6 photos each, and 6 val and 6 test people with 3 each. Random pixels."""
    rng = np.random.default_rng(0)
    split = {0: "train", 1: "train"} | dict.fromkeys(range(2, 8), "val")
    split |= dict.fromkeys(range(8, 14), "test")
    n = 0
    for identity, which in split.items():
        folder = tmp_path / "casia" / f"{identity:05d}"
        folder.mkdir(parents=True)
        for _ in range(6 if which == "train" else 3):
            pixels = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{n:06d}.png")
            n += 1
    rows = sorted(split.items())
    pd.DataFrame(rows, columns=["identity", "split"]).to_csv(tmp_path / "split.csv", index=False)
    build_manifest(tmp_path / "casia", split).to_parquet(tmp_path / "manifest.parquet", index=False)
    return tmp_path


@pytest.fixture
def cpu_only(monkeypatch):
    """Train and evaluate on the CPU even where there is a GPU, as CI does. There every op is
    deterministic, and dropout and the flips draw from the same generator as the DataLoader."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def tiny_run(tiny_casia: Path, run_dir: Path, *overrides: str):
    """The config of a CPU run on tiny_casia, with W&B off, writing to run_dir."""
    manifest = tiny_casia / "manifest.parquet"
    return load_config(
        [
            "wandb.mode=disabled",
            "allow_dirty=true",
            f"out_dir={run_dir}",
            f"data.root={tiny_casia / 'casia'}",
            f"data.manifest={manifest}",
            f"data.manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
            f"data.split_file={tiny_casia / 'split.csv'}",
            "data.array=null",
            "train.batch_size=4",  # 12 photos: 3 steps an epoch
            "train.num_workers=0",
            "eval.num_workers=0",
            "eval.probe_size=6",
            *overrides,
        ]
    )


def test_train_then_evaluate(tiny_casia, cpu_only, capsys, monkeypatch):
    """T14: a 3-step run of train, its checkpoint, evaluate and the results table, on the CPU."""
    # A model trained 3 steps on random pixels, scored against a cohort of 4, can give cohort
    # scores in the hundreds, past the range a trained model's fit in, depending on how those 3
    # steps went. This test checks the pieces connect, so it lets any score through.
    monkeypatch.setattr(evaluate, "COHORT_RANGE", (-1e6, 1e6))
    run_dir = tiny_casia / "run"
    cfg = tiny_run(
        tiny_casia,
        run_dir,
        "train.epochs=1",
        "eval.cohort_k=2",  # of a cohort of 4
        "eval.n_resamples=5",
        f"eval.weights={run_dir / 'checkpoint.pt'}",
    )
    train.run(cfg)
    results = evaluate.run(cfg)
    assert (run_dir / "results_val.json").exists()
    for fix in evaluate.FIXES:
        for far in cfg.eval.far_targets:
            assert 0 <= results[f"{fix}: TAR at FAR {far:g}"] <= 1
    assert "| as is |" in capsys.readouterr().out


def same(a, b) -> bool:
    """Whether two checkpoint entries are equal all the way down, tensors bit for bit."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def test_stopped_and_resumed_run_matches_one_run(tiny_casia, cpu_only, monkeypatch):
    """T12: a run stopped part-way through an epoch, as by Ctrl+C or kill, then resumed, ends
    exactly where one uninterrupted run does: weights, momentum, schedule, random state and
    position. On the CPU every op here is deterministic, so any difference is a bug.

    Why: each piece left out of a checkpoint still resumes without an error. Momentum restarted
    at 0, the schedule back at warmup, a different shuffle or other dropout masks each just make
    the resumed run a slightly different run, which no loss curve would show.
    """
    epochs = "train.epochs=3"  # 9 steps, evaluated between epochs
    train.run(tiny_run(tiny_casia, tiny_casia / "straight", epochs))

    real_step = train.train_step
    steps = 0

    def step_then_kill(*args):
        nonlocal steps
        loss = real_step(*args)
        steps += 1
        if steps == 4:  # epoch 1, batch 0: part-way through an epoch
            os.kill(os.getpid(), signal.SIGTERM)  # what kill sends; run() stops after the step
        return loss

    monkeypatch.setattr(train, "train_step", step_then_kill)
    stopped = tiny_casia / "stopped"
    train.run(tiny_run(tiny_casia, stopped, epochs))
    saved = torch.load(stopped / "checkpoint.pt")
    assert (saved["epoch"], saved["batch"]) == (1, 1)
    monkeypatch.setattr(train, "train_step", real_step)
    train.run(tiny_run(tiny_casia, stopped, epochs, "resume=true"))

    one_run = torch.load(tiny_casia / "straight" / "checkpoint.pt")
    resumed = torch.load(stopped / "checkpoint.pt")
    for key in ("model", "head", "optimiser", "scheduler", "random", "epoch", "batch"):
        assert same(one_run[key], resumed[key]), key


def test_resume_with_other_settings_is_refused(tiny_casia, cpu_only):
    run_dir = tiny_casia / "run"
    train.run(tiny_run(tiny_casia, run_dir, "train.epochs=1"))
    with pytest.raises(SystemExit, match="train differ"):
        train.run(tiny_run(tiny_casia, run_dir, "train.epochs=1", "train.lr=0.05", "resume=true"))


class FakeArtifact:
    """Stands in for wandb.Artifact, keeping what was added."""

    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path):
        self.files.append(path)


class FakeRun:
    """Stands in for a W&B run, keeping what was logged."""

    id = "fake"
    disabled = False

    def __init__(self):
        self.logs, self.artifacts = [], []

    def log(self, data, step):
        self.logs.append((step, data))

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))

    def use_artifact(self, name):
        return None  # as when W&B is disabled: link_manifest skips its check

    def finish(self):
        pass


def test_each_epoch_is_scored_and_its_checkpoint_logged(tiny_casia, cpu_only, monkeypatch):
    wb = FakeRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: wb)
    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    run_dir = tiny_casia / "run"
    cfg = tiny_run(tiny_casia, run_dir, "train.epochs=2")
    train.run(cfg)

    steps = [step for step, _ in wb.logs]
    assert steps == sorted(steps)  # W&B drops anything logged at an earlier step than the last
    scored = [data for _, data in wb.logs if "val/UI tightness" in data]
    assert [data["epoch"] for data in scored] == [0, 1]
    for data in scored:
        for far in cfg.eval.far_targets:
            assert 0 <= data[f"val/TAR at FAR {far:g}"] <= 1
        assert data["speed/photos per second"] > 0
    logged = [(a.type, a.files, a.metadata, aliases) for a, aliases in wb.artifacts]
    checkpoint = [str(run_dir / "checkpoint.pt")]
    assert logged == [
        ("model", checkpoint, {"epochs": 1}, ["epoch-1", "latest"]),
        ("model", checkpoint, {"epochs": 2}, ["epoch-2", "latest"]),
    ]
