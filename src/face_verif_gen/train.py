import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from face_verif_gen.run_meta import run_metadata, seed_everything


def link_manifest(wb, name: str, sha256: str) -> None:
    """Record the manifest's W&B Artifact as the run's input, refusing one of different data."""
    artifact = wb.use_artifact(name)
    # use_artifact returns None when W&B is disabled, as in the tests.
    if artifact is not None and artifact.metadata["sha256"] != sha256:
        raise SystemExit(f"{name} is not the local manifest: check data.artifact_version")


def run(cfg: DictConfig) -> None:
    meta = run_metadata(cfg.data.manifest)
    if meta["git_dirty"] and not cfg.allow_dirty:
        raise SystemExit("Uncommitted changes: commit them or pass allow_dirty=true")
    seed_everything(cfg.seed)
    wb = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config={**OmegaConf.to_container(cfg, resolve=True), "meta": meta},
    )
    link_manifest(wb, f"{cfg.data.artifact}:{cfg.data.artifact_version}", meta["manifest_sha256"])
    for step in range(10):  # dummy loop for M0; real training arrives in M2
        wb.log({"loss": 1.0 / (step + 1)}, step=step)
    wb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
