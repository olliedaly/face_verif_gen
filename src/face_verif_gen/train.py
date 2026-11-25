import signal
import time
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from face_verif_gen import metrics
from face_verif_gen.cluster import ui_tightness
from face_verif_gen.data import FaceDataset
from face_verif_gen.fixes import ruiner
from face_verif_gen.heads import build_head, cross_entropy_loss
from face_verif_gen.model import IResNet, preprocess
from face_verif_gen.run_meta import run_metadata, seed_everything
from face_verif_gen.verify import bin_edges, embed, pair_histograms, pick_faces


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
    """SGD over the model's parameters and the head's; cfg is the train config, configs/train.

    As AdaFace's: weight decay on every parameter but BatchNorm's scales and shifts, the head's
    rows included.
    """
    norms = (nn.BatchNorm1d, nn.BatchNorm2d)
    in_norms = {id(p) for m in model.modules() if isinstance(m, norms) for p in m.parameters()}
    decayed = [p for p in model.parameters() if id(p) not in in_norms]
    not_decayed = [p for p in model.parameters() if id(p) in in_norms]
    return torch.optim.SGD(
        [
            {"params": decayed + [*head.parameters()], "weight_decay": cfg.weight_decay},
            {"params": not_decayed, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
    )


def build_schedule(optimiser, cfg: DictConfig, steps_per_epoch: int) -> LambdaLR:
    """The rate at each step, as a multiple of cfg.lr; scheduler.step() after every step.

    A linear rise from 1% over the warmup epochs, then cut by lr_gamma at each milestone epoch.
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def multiple(step: int) -> float:
        epoch = step // steps_per_epoch
        cut = cfg.lr_gamma ** sum(epoch >= milestone for milestone in cfg.lr_milestones)
        if step < warmup_steps:
            return cut * (0.01 + 0.99 * step / warmup_steps)
        return cut

    return LambdaLR(optimiser, multiple)


def random_flip(images: torch.Tensor) -> torch.Tensor:
    """images, (N, 3, H, W), with a random half of them mirrored left to right."""
    flip = torch.rand(len(images), device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], images.flip(-1), images)


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


class EpochSampler(Sampler[int]):
    """Each epoch's shuffle, fixed by the seed and the epoch alone, so a resumed run gets the same.

    set_epoch's start skips that many photos of the epoch: where a stopped run left off.
    """

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return iter(order[self.start :].tolist())

    def __len__(self) -> int:
        return self.n - self.start


class StopRequest:
    """While open, Ctrl+C (SIGINT) and kill (SIGTERM) set .requested instead of ending the process.

    The training loop checks it after each step, so a run is never stopped halfway through one:
    after optimiser.step() but before scheduler.step(), a saved run would be out of step.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self):
        self.requested = False

    def __enter__(self):
        self.previous = {s: signal.signal(s, self.handle) for s in self.SIGNALS}
        return self

    def handle(self, signum, frame) -> None:
        self.requested = True

    def __exit__(self, *exc) -> None:
        for s, handler in self.previous.items():
            signal.signal(s, handler)


def check_resumable(saved_config: dict, cfg: DictConfig) -> None:
    """Refuse to resume with different settings: the run would silently become a mix of two."""
    now = OmegaConf.to_container(cfg, resolve=True)
    keys = ("seed", "data", "model", "head", "train")
    changed = [key for key in keys if saved_config[key] != now[key]]
    if changed:
        raise SystemExit(f"Can't resume: {', '.join(changed)} differ from the checkpoint's config")


def epoch_scores(model, val: FaceDataset, faces, probe, device, cfg: DictConfig) -> dict:
    """Scores after an epoch, on val people the model never trains on; puts it in eval mode.

    TAR at each FAR on val's faces, up to eval.per_identity photos a person as evaluate.py
    picks them; and UI tightness on the probe photos, the emergence metric.
    """
    model.eval()
    ev = cfg.eval
    labels = torch.as_tensor(val.labels[faces], device=device)
    emb = embed(model, val, faces, device, ev.batch_size, ev.num_workers)
    genuine, impostor = pair_histograms(emb, labels)
    edges = bin_edges()
    scores = {
        f"val/TAR at FAR {far:g}": metrics.tar_at_far(edges, edges, far, genuine, impostor)[0]
        for far in ev.far_targets
    }
    embed_probe = partial(embed, model, val, probe, device, ev.batch_size)
    probe_labels = torch.as_tensor(val.labels[probe], device=device)
    scores["val/UI tightness"] = ui_tightness(
        embed_probe(), embed_probe(degrade=ruiner(device)), probe_labels
    )
    return scores


def log_checkpoint(wb, path: Path, epochs_done: int) -> None:
    """The checkpoint as a version of the run's model Artifact, linked to the run that made it,
    and through it to the config and the data. W&B copies the file at once, so the next epoch can
    overwrite it while this one uploads."""
    if wb.disabled:  # tests: no copy of a 200 MB file each epoch
        return
    artifact = wandb.Artifact(f"checkpoint-{wb.id}", type="model", metadata={"epochs": epochs_done})
    artifact.add_file(str(path))
    wb.log_artifact(artifact, aliases=[f"epoch-{epochs_done}", "latest"])


def save_checkpoint(path: Path, run_state: dict) -> None:
    """Written to a temporary file and renamed, so a kill mid-write leaves the last one whole."""
    temporary = path.with_suffix(".tmp")
    torch.save(run_state, temporary)
    temporary.replace(path)


def run(cfg: DictConfig) -> None:
    """Train, saving a checkpoint after every epoch and on Ctrl+C or kill.

    With resume=true, carry on from out_dir's checkpoint.pt, exactly where it stopped.
    """
    meta = check_inputs(cfg)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu") if cfg.resume else None
    if saved is not None:
        check_resumable(saved["config"], cfg)
    seed_everything(cfg.seed)  # before the model is built: its starting weights are random
    wb = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config={**OmegaConf.to_container(cfg, resolve=True), "meta": meta},
        id=saved["wandb_id"] if saved is not None else None,  # a resumed run carries on its curve
        resume="allow",
    )
    link_manifest(wb, f"{cfg.data.artifact}:{cfg.data.artifact_version}", meta["manifest_sha256"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FaceDataset(
        cfg.data.manifest,
        cfg.data.root,
        cfg.data.split_file,
        split="train",
        array=cfg.data.array,
        people=cfg.data.people,
    )
    # Scored after every epoch, on the same val photos every epoch and run.
    val = FaceDataset(
        cfg.data.manifest, cfg.data.root, cfg.data.split_file, split="val", array=cfg.data.array
    )
    faces = pick_faces(val.labels, cfg.eval.per_identity, cfg.eval.face_seed)
    probe = np.random.default_rng(0).choice(len(val), cfg.eval.probe_size, replace=False)
    sampler = EpochSampler(len(dataset), cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        # Each epoch's iterator draws a number to seed its workers, and a resumed run makes one
        # more iterator. Drawn from the global generator, that would shift dropout's and the
        # flips' numbers for the rest of the run. The workers draw nothing: FaceDataset has no
        # randomness.
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",  # page-locked memory: faster copies to the GPU
        drop_last=True,  # a last batch of 1 would break BatchNorm1d in train mode
    )
    # Not len(loader): that is shorter when a run resumes part-way through an epoch.
    steps_per_epoch = len(dataset) // cfg.train.batch_size
    model = IResNet().to(device)
    head = build_head(cfg.head, embed_dim=model.embedding_dim, num_classes=dataset.num_classes)
    head = head.to(device)
    optimiser = build_optimiser(model, head, cfg.train)
    scheduler = build_schedule(optimiser, cfg.train, steps_per_epoch)

    epoch, batch = 0, 0  # the next step to take: batch number within its epoch
    if saved is not None:
        model.load_state_dict(saved["model"])
        head.load_state_dict(saved["head"])
        optimiser.load_state_dict(saved["optimiser"])  # SGD's momentum, one buffer per parameter
        scheduler.load_state_dict(saved["scheduler"])  # how many steps it has counted
        # The only random numbers training draws: dropout's and the flips'.
        torch.set_rng_state(saved["random"]["torch"])
        if saved["random"]["cuda"]:
            torch.cuda.set_rng_state_all(saved["random"]["cuda"])
        epoch, batch = saved["epoch"], saved["batch"]
        print(f"Resuming at epoch {epoch}, batch {batch}")

    def save() -> None:
        save_checkpoint(
            checkpoint,
            {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "random": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                "epoch": epoch,
                "batch": batch,
                "wandb_id": wb.id,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "meta": meta,
            },
        )

    with StopRequest() as stop:
        while epoch < cfg.train.epochs and not stop.requested:
            sampler.set_epoch(epoch, start=batch * cfg.train.batch_size)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started, losses = time.perf_counter(), []
            for images, labels in loader:
                step = epoch * steps_per_epoch + batch  # W&B keeps one point per step
                # Still uint8, a quarter of float32's bytes to copy; train_step preprocesses on
                # the GPU.
                images, labels = images.to(device), labels.to(device)
                if cfg.train.flip:
                    images = random_flip(images)
                lr = scheduler.get_last_lr()[0]  # the rate this step uses
                loss = train_step(model, head, optimiser, images, labels)
                scheduler.step()
                wb.log({"loss": loss, "lr": lr, "epoch": epoch}, step=step)
                losses.append(loss)
                batch += 1
                if stop.requested:  # Ctrl+C has also ended the loader's workers: no next batch
                    break
            if stop.requested:
                break
            # train_step's loss.item() waits for the GPU each step, so the clock is not ahead.
            seconds = time.perf_counter() - started
            logs = {
                "epoch": epoch,
                "speed/photos per second": len(losses) * cfg.train.batch_size / seconds,
                "speed/minutes training": seconds / 60,
            }
            if device.type == "cuda":
                logs["speed/peak GPU memory GB"] = torch.cuda.max_memory_allocated(device) / 1e9
            started = time.perf_counter()
            logs |= epoch_scores(model, val, faces, probe, device, cfg)
            logs["speed/minutes scoring"] = (time.perf_counter() - started) / 60
            wb.log(logs, step=(epoch + 1) * steps_per_epoch)
            tars = ", ".join(
                f"{far:g}: {logs[f'val/TAR at FAR {far:g}']:.3f}" for far in cfg.eval.far_targets
            )
            print(
                f"epoch {epoch + 1}/{cfg.train.epochs}: loss {np.mean(losses):.2f}; "
                f"val TAR at FAR {tars}; {logs['speed/photos per second']:,.0f} photos/s, "
                f"{seconds / 60:.1f} min"
            )
            epoch, batch = epoch + 1, 0
            save()
            log_checkpoint(wb, checkpoint, epochs_done=epoch)
        if stop.requested:
            save()
            print(f"Stopped at epoch {epoch}, batch {batch}: resume with out_dir={out} resume=true")
    wb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
