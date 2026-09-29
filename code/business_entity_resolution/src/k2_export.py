"""R4: stage-2 matcher features for chosen folds, exported compactly for R5 (Kaggle CPU notebook).

Features are computed by the same ``k2_features.process_shard`` code path as K2:

* fold 0 exactly as in K2 (out-of-fold filter scores for train S1, the final filter for validation and
  holdout);
* extra training folds 1-4 exactly as ``k2_experiments.prepare_extra_folds`` (the final filter for every S1).

Only what R5 needs is written, as compressed ``chunkNNN.npz`` files under ``<output>/fold<k>/``:

* fold-0 train and every extra fold: K2's sampled training rows (all positives, the top filter-ranked
  negatives, random negatives with their natural-rate weights), drawn with the same seeds as K2's
  ``load_split``, so R5 trains on exactly the rows K2 would;
* fold-0 validation and holdout: every kept row (weight 1).

``load_export`` reads them back into the dictionary layout of ``k2_experiments.load_split``.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .k1_filter import build_s1_store, build_stores, load_context
from .k2_experiments import choose_filter_k, extra_fold_context
from .k2_features import build_features
from .phase2a_env import ResourceGuard, ResourceStop, log

OFFSET = 10_000_000          # extra-fold S1 positions are offset by fold * OFFSET (as in K2)


def export_fold(config: dict, ctx0: dict, fold: int, phase1c_work0: Path, work: Path, k1_dir: Path, keep_k: int,
                out_dir: Path, root: Path = ROOT) -> dict:
    s = config["train_sampling"]
    export = {"seed": s["seed"], "top": s["top_filter_negatives"], "random": s["random_negatives"],
              "offset": fold * OFFSET, "train_all": fold != 0, "train_end": config["split"]["train"][1]}
    if fold == 0:
        cfg, ctx = config, ctx0
        build_features(config, ctx0, phase1c_work0, work, k1_dir, keep_k, config["workers"], features_dir=str(out_dir),
                       export=export)
    else:
        work_k = root / "artifacts" / f"phase1c_fold{fold}" / "work"
        cfg, ctx = extra_fold_context(config, fold, work_k, root)
        s1_dir = work / f"stores_fold{fold}"
        if not (s1_dir / "s1_name.npy").exists():
            build_s1_store(cfg, s1_dir, ctx)
        build_features(cfg, ctx, work_k, work, k1_dir, keep_k, config["workers"], s1_dir=s1_dir,
                       features_dir=str(out_dir), filter_all_final=True, export=export)
    chunks = sorted(out_dir.glob("chunk*.npz"))
    rows, positives, s1 = [], 0, 0
    for path in chunks:
        with np.load(path) as data:
            rows.append(int(len(data["label"])))
            positives += int(data["label"].sum())
            s1 += int(len(data["s1_positions"]))
    info = {"fold": fold, "offset": fold * OFFSET, "s1": s1, "chunks": len(chunks), "chunk_rows": rows,
            "rows": int(sum(rows)), "positives": positives, "keep_k": keep_k, "split": cfg["split"],
            "chunk_s1": config["chunk_s1"], "sampling": s, "phase1c_manifest_state": ctx["manifest"]["state"],
            "order_sha256": hashlib.sha256("\n".join(ctx["ordered"]).encode()).hexdigest()}
    if s1 != len(ctx["ordered"]):
        raise RuntimeError(f"fold {fold}: exported {s1:,} S1 of {len(ctx['ordered']):,}")
    atomic_write_json(out_dir / "index.json", info)
    log(f"R4: fold {fold} exported: {s1:,} S1, {info['rows']:,} rows, {positives:,} positives, {len(chunks)} chunks")
    return info


def load_export(fold_dir: Path, split: str | None = None, *, chunk_range: tuple[int, int] | None = None) -> dict:
    """An exported fold as ``k2_experiments.load_split``'s layout. ``split`` selects fold-0 train/validation/
    holdout chunks from the index; extra folds are all training (``split`` ignored). Positions are offset."""
    info = json.loads((fold_dir / "index.json").read_text())
    chunk, offset = info["chunk_s1"], info["offset"]
    if chunk_range is None:
        if info["fold"] == 0 and split:
            low, high = info["split"][split]
            chunk_range = (low // chunk, (high + chunk - 1) // chunk)
        else:
            chunk_range = (0, info["chunks"])
    parts = {name: [] for name in ("X", "label", "s1_pos", "cand", "weight", "country")}
    entities = {name: [] for name in ("ent_positions", "ent_truth_len", "ent_kept_truth", "ent_retrieved_truth",
                                      "ent_cand_count", "ent_country", "ent_key")}
    for index in range(*chunk_range):
        with np.load(fold_dir / f"chunk{index:03d}.npz", allow_pickle=False) as data:
            parts["X"].append(data["X"]); parts["label"].append(data["label"])
            parts["s1_pos"].append(data["s1_pos"] + offset); parts["cand"].append(data["cand"])
            parts["weight"].append(data["weight"]); parts["country"].append(data["country"])
            entities["ent_positions"].append(data["s1_positions"] + offset)
            entities["ent_truth_len"].append(data["s1_truth_len"]); entities["ent_kept_truth"].append(data["s1_kept_truth"])
            entities["ent_retrieved_truth"].append(data["s1_retrieved_truth"])
            entities["ent_cand_count"].append(data["s1_cand_count"]); entities["ent_country"].append(data["s1_country"])
            entities["ent_key"].append(data["s1_key"])
    out = {name: np.concatenate(values) for name, values in parts.items()}
    out.update({name: np.concatenate(values) for name, values in entities.items()})
    out["group"] = np.searchsorted(out["ent_positions"], out["s1_pos"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1c-work-dir", type=Path, required=True, help="fold-0 Phase 1C work directory")
    parser.add_argument("--k1-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True,
                        help="0 and/or extra folds; extra fold k needs artifacts/phase1c_fold<k>/ (manifest and work/)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    guard = ResourceGuard(1.0, 8.0, "python3 -m src.k2_export ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = load_context(config, ROOT, args.phase1c_work_dir)
        ctx["train_dir"] = ROOT / config["inputs"]["train_dir"]
        k1 = json.loads((args.k1_dir / "k1_results.json").read_text())
        chosen, gate_table = choose_filter_k(k1, config["filter"])
        keep_k = config["filter"]["k_override"] or chosen or config["filter"]["fallback_k"]
        atomic_write_json(args.output_dir / "filter_choice.json", {"keep_k": keep_k, "gate_table": gate_table})
        log(f"R4: keep top-{keep_k} per S1; folds {args.folds}")
        build_stores(config, ROOT, args.work_dir, ctx, guard)
        summary = {str(fold): export_fold(config, ctx, fold, args.phase1c_work_dir, args.work_dir, args.k1_dir, keep_k,
                                          args.output_dir.resolve() / f"fold{fold}")
                   for fold in args.folds}
        atomic_write_json(args.output_dir / f"r4_summary_folds_{'_'.join(map(str, args.folds))}.json", summary)
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
