import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from face_verif_gen.run_meta import run_metadata, seed_everything


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
    for step in range(10):  # dummy loop for M0; real training arrives in M2
        wb.log({"loss": 1.0 / (step + 1)}, step=step)
    wb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
