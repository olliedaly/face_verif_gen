from pathlib import Path

from hydra import compose, initialize_config_dir
from yourpkg.train import run

CONFIGS = str((Path(__file__).parents[1] / "configs").resolve())


def test_every_head_composes():
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        for head in ["softmax", "arcface"]:
            assert compose("config", overrides=[f"head={head}"]).head.name == head


def test_dummy_run(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"fake")
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "wandb.mode=disabled",
                "allow_dirty=true",
                f"data.manifest={manifest}",
            ],
        )
    run(cfg)
