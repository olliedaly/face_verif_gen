from pathlib import Path

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
    monkeypatch.setattr(train, "run_metadata", lambda manifest_path: {"git_dirty": True})
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true"])
    train.run(cfg)


def test_dummy_run(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"fake")
    cfg = load_config(["wandb.mode=disabled", "allow_dirty=true", f"data.manifest={manifest}"])
    train.run(cfg)
