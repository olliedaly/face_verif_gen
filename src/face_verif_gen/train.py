import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from face_verif_gen.data import FaceDataset
from face_verif_gen.heads import build_head, cross_entropy_loss
from face_verif_gen.model import IResNet, preprocess
from face_verif_gen.run_meta import run_metadata, seed_everything


def link_manifest(wb, name: str, sha256: str) -> None:
    """Record the manifest's W&B Artifact as the run's input, refusing one of different data."""
    artifact = wb.use_artifact(name)
    # use_artifact returns None when W&B is disabled, as in the tests.
    if artifact is not None and artifact.metadata["sha256"] != sha256:
        raise SystemExit(f"{name} is not the local manifest: check data.artifact_version")


def check_inputs(cfg: DictConfig) -> dict:
    """Refuse to start from uncommitted code or unpinned data; return the run's metadata."""
    meta = run_metadata(cfg.data.manifest)
    if meta["git_dirty"] and not cfg.allow_dirty:
        raise SystemExit("Uncommitted changes: commit them or pass allow_dirty=true")
    found, pinned = meta["manifest_sha256"], cfg.data.manifest_sha256
    if found != pinned:
        raise SystemExit(
            f"{cfg.data.manifest} has sha256 {found[:12]}… but the config pins {pinned[:12]}…: "
            "rebuild it with prepare_data.py, or update data.manifest_sha256 if the data changed"
        )
    return meta


def build_optimiser(model, head, cfg: DictConfig) -> torch.optim.Optimizer:
    """SGD over the model's parameters and the head's; cfg is the train config, configs/train."""
    return torch.optim.SGD(
        [*model.parameters(), *head.parameters()],  # the head's rows are learned too
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def train_step(model, head, optimiser, images: torch.Tensor, labels: torch.Tensor) -> float:
    """One optimisation step on a batch of uint8 images, as the loader gives them; the loss."""
    model.train()
    with torch.autocast(images.device.type, dtype=torch.bfloat16):
        embeddings = model(preprocess(images))
    logits = head(embeddings.float(), labels)  # labels: ArcFace puts its margin on each true class
    loss = cross_entropy_loss(logits, labels)
    optimiser.zero_grad()  # clear the last step's gradients
    loss.backward()
    optimiser.step()  # update the model's weights and the head's
    return loss.item()


def run(cfg: DictConfig) -> None:
    meta = check_inputs(cfg)
    seed_everything(cfg.seed)  # before the model is built: its starting weights are random
    wb = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config={**OmegaConf.to_container(cfg, resolve=True), "meta": meta},
    )
    link_manifest(wb, f"{cfg.data.artifact}:{cfg.data.artifact_version}", meta["manifest_sha256"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FaceDataset(
        cfg.data.manifest, cfg.data.root, cfg.data.split_file, split="train", array=cfg.data.array
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,  # page-locked memory: faster copies to the GPU
        drop_last=True,  # a last batch of 1 would break BatchNorm1d in train mode
    )
    model = IResNet().to(device)
    head = build_head(cfg.head, embed_dim=model.embedding_dim, num_classes=dataset.num_classes)
    head = head.to(device)
    optimiser = build_optimiser(model, head, cfg.train)

    step = 0  # counts batches across epochs: W&B keeps one point per step
    for epoch in range(cfg.train.epochs):
        for images, labels in loader:
            # Still uint8, a quarter of float32's bytes to copy; train_step preprocesses on the GPU.
            images, labels = images.to(device), labels.to(device)
            loss = train_step(model, head, optimiser, images, labels)
            wb.log({"loss": loss, "epoch": epoch}, step=step)
            step += 1
    wb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
