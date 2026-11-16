"""Evaluate a model on people it never saw in training.

TAR at each FAR on the evaluated split, with 95% intervals over its people; the same with the
unsure rule and with cohort normalisation; and the cluster metrics. Thresholds are chosen on val
and applied to the evaluated split, as a deployed system sets its threshold before it meets the
people it will see. Each split's own TAR at FAR is reported too.

  uv run python -m face_verif_gen.evaluate eval.weights=outputs/<date>/<time>/checkpoint.pt
  uv run python -m face_verif_gen.evaluate eval.weights=... eval.split=test   (once, at the end)
"""

import json
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from face_verif_gen import metrics
from face_verif_gen.cluster import top_impostor_photos, ui_share, ui_tightness
from face_verif_gen.data import FaceDataset
from face_verif_gen.fixes import (
    COHORT_RANGE,
    cohort_normalisation,
    cohort_stats,
    ruiner,
    unsure_cutoff,
    unsure_direction,
    unsure_rule,
)
from face_verif_gen.model import IResNet, load_adaface
from face_verif_gen.verify import (
    bin_edges,
    cosine,
    embed,
    pair_histograms,
    pick_faces,
    subject_bootstrap,
)

FIXES = ("as is", "unsure rule", "cohort normalisation")


def load_model(cfg: DictConfig) -> IResNet:
    """The model cfg.eval.weights names: a train.py checkpoint, or adaface."""
    if cfg.eval.weights == "adaface":
        return load_adaface(cfg.eval.adaface_checkpoint)
    model = IResNet()
    model.load_state_dict(torch.load(cfg.eval.weights, map_location="cpu")["model"])
    return model


def dataset(cfg: DictConfig, split: str) -> FaceDataset:
    d = cfg.data
    return FaceDataset(d.manifest, d.root, d.split_file, split, array=d.array)


def evaluate(model: torch.nn.Module, cfg: DictConfig, device: torch.device) -> dict:
    """Every number in the results table, for cfg.eval.split. The model must be in eval mode."""
    ev = cfg.eval
    fars = list(ev.far_targets)
    strictest = min(fars)
    embed_ = partial(
        embed, model, device=device, batch_size=ev.batch_size, num_workers=ev.num_workers
    )

    # Both fixes are built from training photos only.
    train = dataset(cfg, "train")
    ruined = embed_(
        train,
        pick_faces(train.labels, ev.ruined_per_identity, ev.ruined_seed),
        degrade=ruiner(device),
    )
    direction = unsure_direction(ruined)
    cohort = embed_(train, pick_faces(train.labels, ev.cohort_per_identity, ev.cohort_seed))

    splits = {}
    for name in dict.fromkeys(["val", ev.split]):  # val once when evaluating val
        ds = dataset(cfg, name)
        faces = pick_faces(ds.labels, ev.per_identity, ev.face_seed)
        emb = embed_(ds, faces)
        mean, sd = cohort_stats(emb, cohort, ev.cohort_k)
        splits[name] = {
            "ds": ds,
            "faces": faces,
            "emb": emb,
            "near": emb @ direction,
            "labels": torch.as_tensor(ds.labels[faces], device=device),
            "mean": mean,
            "sd": sd,
        }
    cutoff = unsure_cutoff(splits["val"]["near"], ev.unsure_share)

    def scorer(fix: str, s: dict):
        if fix == "as is":
            return cosine, (-1.0, 1.0)
        if fix == "unsure rule":
            return unsure_rule(s["near"], cutoff), (-1.0, 1.0)
        return cohort_normalisation(s["mean"], s["sd"]), COHORT_RANGE

    def histograms(fix: str, split: str, weights=None):
        score, (lo, hi) = scorer(fix, splits[split])
        s = splits[split]
        return pair_histograms(s["emb"], s["labels"], score, weights, lo, hi)

    def edges(fix: str) -> np.ndarray:
        return bin_edges(*scorer(fix, splits["val"])[1])

    ev_split = splits[ev.split]
    results = {
        "split": ev.split,
        "FAR targets": fars,
        "photos": len(ev_split["faces"]),
        "people": int(ev_split["labels"].max()) + 1,
        "unsure cutoff": cutoff,
    }
    thresholds = {}  # (fix, far) -> chosen on val
    for fix in FIXES:
        on_val, on_split = histograms(fix, "val"), histograms(fix, ev.split)
        results.setdefault("genuine pairs", int(on_split[0].sum()))
        results.setdefault("impostor pairs", int(on_split[1].sum()))
        e = edges(fix)
        for far in fars:
            _, thresholds[fix, far] = metrics.tar_at_far(e, e, far, *on_val)
            tar, achieved = metrics.rates_at_threshold(e, e, thresholds[fix, far], *on_split)
            own_tar, _ = metrics.tar_at_far(e, e, far, *on_split)
            results[f"{fix}: TAR at FAR {far:g}, val threshold"] = tar
            results[f"{fix}: FAR at FAR {far:g}, val threshold"] = achieved
            results[f"{fix}: TAR at FAR {far:g}"] = own_tar

    # Intervals over the evaluated split's people. Each resample sets every threshold again at
    # exactly its FAR, so all three are compared at the same FAR. The fix lifts come from the
    # same resamples, so their intervals are paired.
    def statistic(weights: torch.Tensor) -> np.ndarray:
        tars = {}
        for fix in FIXES:
            e = edges(fix)
            on_split = histograms(fix, ev.split, weights)
            for far in fars:
                tars[fix, far] = metrics.tar_at_far(e, e, far, *on_split)[0]
        lifts = [tars[fix, strictest] - tars["as is", strictest] for fix in FIXES[1:]]
        return np.array([tars[fix, far] for fix in FIXES for far in fars] + lifts)

    low, high = subject_bootstrap(statistic, results["people"], ev.n_resamples, seed=0)
    names = [f"{fix}: TAR at FAR {far:g}" for fix in FIXES for far in fars]
    names += [f"{fix}: lift at FAR {strictest:g}" for fix in FIXES[1:]]
    for fix in FIXES[1:]:
        lift = results[f"{fix}: TAR at FAR {strictest:g}"]
        lift -= results[f"as is: TAR at FAR {strictest:g}"]
        results[f"{fix}: lift at FAR {strictest:g}"] = lift
    for name, lo, hi in zip(names, low, high, strict=True):
        results[f"{name}, 95% interval"] = [float(lo), float(hi)]

    # Cluster metrics. Emergence is tracked during training instead.
    ruined_split = embed_(ev_split["ds"], ev_split["faces"], degrade=ruiner(device))
    results["UI tightness"] = ui_tightness(ev_split["emb"], ruined_split, ev_split["labels"])
    results["UI share"] = ui_share(
        ev_split["emb"],
        ev_split["labels"],
        thresholds["as is", strictest],
        ev_split["near"] >= cutoff,
    )
    val = splits["val"]
    top = top_impostor_photos(val["emb"], val["labels"], ev.culprit_pairs).cpu().numpy()
    culprits = sorted(val["ds"].paths[val["faces"][top]].tolist())
    results["culprit photos"] = culprits
    if ev.culprit_reference:
        reference = set(Path(ev.culprit_reference).read_text().split())
        results["culprit overlap"] = len(reference.intersection(culprits))
    return results


def table(results: dict) -> str:
    """The results as Markdown: one row per fix, TAR at each FAR with its interval."""
    fars = [f"{far:g}" for far in results["FAR targets"]]
    lines = [
        f"{results['split']}: {results['people']:,} people, {results['photos']:,} photos, "
        f"{results['genuine pairs']:,} genuine and {results['impostor pairs']:,} impostor "
        "pairs",
        "",
        "| | " + " | ".join(f"TAR at FAR {f}" for f in fars) + " |",
        "|---" * (len(fars) + 1) + "|",
    ]
    for fix in FIXES:
        cells = []
        for f in fars:
            tar = results[f"{fix}: TAR at FAR {f}"]
            lo, hi = results[f"{fix}: TAR at FAR {f}, 95% interval"]
            cells.append(f"{tar:.1%} [{lo:.1%}, {hi:.1%}]")
        lines.append(f"| {fix} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"UI tightness {results['UI tightness']:+.3f}, UI share {results['UI share']:.1%}",
    ]
    for fix in FIXES[1:]:
        key = f"{fix}: lift at FAR {fars[-1]}"
        lo, hi = results[f"{key}, 95% interval"]
        lines.append(f"{key}: {results[key]:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    if "culprit overlap" in results:
        lines.append(
            f"culprit overlap: {results['culprit overlap']} of "
            f"{len(results['culprit photos'])} photos"
        )
    return "\n".join(lines)


def run(cfg: DictConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg).to(device).eval()
    results = evaluate(model, cfg, device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{cfg.eval.split}.json").write_text(json.dumps(results, indent=1))
    print(table(results))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
