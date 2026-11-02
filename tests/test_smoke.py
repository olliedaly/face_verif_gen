from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_verif_gen import train

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
    meta = {"git_dirty": True, "manifest_sha256": "fake"}  # the run now reads the hash too
    monkeypatch.setattr(train, "run_metadata", lambda manifest_path: meta)
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true"])
    train.run(cfg)


def fake_run(sha256: str):
    """Stands in for a W&B run whose use_artifact returns an artifact recording this hash."""
    artifact = SimpleNamespace(metadata={"sha256": sha256})
    return SimpleNamespace(use_artifact=lambda _: artifact)


def test_matching_artifact_is_linked():
    train.link_manifest(fake_run("abc"), "casia-manifest:v0", sha256="abc")


def test_artifact_of_other_data_is_refused():
    with pytest.raises(SystemExit, match="not the local manifest"):
        train.link_manifest(fake_run("abc"), "casia-manifest:v0", sha256="def")


def test_dummy_run(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"fake")
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true", f"data.manifest={manifest}"])
    train.run(cfg)
