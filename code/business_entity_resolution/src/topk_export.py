"""Top-K candidate lists for the cross-encoder (CPU; the AWS node or a Kaggle CPU notebook).

Each S1 keeps its ``keep_k`` best candidates by the K1 filter (the frozen B+C+E blocker's candidate union,
re-ranked by the cheap learned filter), exactly as K2/K3 select them but without computing the stage-2 features:

* ``--split fold0``: out-of-fold filter models for fold-0 train S1 and the final filter for validation/holdout (K2's
  rule, so train-time lists look like test-time lists); labels come from the training ground truth.
* ``--split test``: the final filter on the verified test route shards (K3's driver); no labels are read.

Output (the cross-encoder's input contract), per split directory under ``--output-dir``:

* ``part-NNNNN.parquet``: s1_id, t_id, label (int8; -1 for test), filter_score (float32), filter_rank (int16),
  grouped by S1 and sorted by filter_rank within an S1;
* ``s1.parquet``: s1_id, truth_len (int32; -1 for test), n_cand (int16), retrieved_truth (int32; -1 for test),
  one row per S1 including those without candidates.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import json
from pathlib import Path

import numpy as np

from .blocking import decode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .k1_filter import build_stores, load_context
from .k2_features import build_features
from .phase2a_env import ResourceGuard, log

PART_ROWS = 4_000_000


def decode(values: np.ndarray, exceptions: dict[str, str]) -> np.ndarray:
    return np.asarray([exceptions.get(str(v)) or decode_id(v) for v in values.tolist()], dtype=object)


def write_split(out_dir: Path, ordered: list[str], exceptions: dict[str, str], s1_pos: np.ndarray, cand: np.ndarray,
                label: np.ndarray, score: np.ndarray, rank: np.ndarray, positions: np.ndarray,
                truth_len: np.ndarray, retrieved: np.ndarray) -> dict:
    """Write one split: rows sorted by (S1 position, filter rank); ``positions`` are all S1 of the split."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((rank, s1_pos))
    s1_pos, cand, label, score, rank = s1_pos[order], cand[order], label[order], score[order], rank[order]
    ids = np.asarray(ordered, dtype=object)
    for part, start in enumerate(range(0, max(len(s1_pos), 1), PART_ROWS)):
        stop = min(start + PART_ROWS, len(s1_pos))
        table = pa.table({"s1_id": pa.array(ids[s1_pos[start:stop]].tolist(), pa.string()),
                          "t_id": pa.array(decode(cand[start:stop], exceptions).tolist(), pa.string()),
                          "label": pa.array(label[start:stop], pa.int8()),
                          "filter_score": pa.array(score[start:stop], pa.float32()),
                          "filter_rank": pa.array(rank[start:stop], pa.int16())})
        temporary = out_dir / f"part-{part:05d}.parquet.tmp"
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(out_dir / f"part-{part:05d}.parquet")
    n_cand = np.bincount(np.searchsorted(positions, s1_pos), minlength=len(positions))
    s1 = pa.table({"s1_id": pa.array(ids[positions].tolist(), pa.string()), "truth_len": pa.array(truth_len, pa.int32()),
                   "n_cand": pa.array(n_cand, pa.int16()), "retrieved_truth": pa.array(retrieved, pa.int32())})
    pq.write_table(s1, out_dir / "s1.parquet", compression="zstd")
    info = {"s1": int(len(positions)), "rows": int(len(s1_pos)), "positives": int((label == 1).sum()),
            "s1_without_candidates": int((n_cand == 0).sum()), "max_cand": int(n_cand.max()) if len(n_cand) else 0}
    if (truth_len >= 0).all() and len(truth_len):
        info["kept_truth_share"] = float((label == 1).sum() / max(int(truth_len.sum()), 1))
    atomic_write_json(out_dir / "info.json", info)
    return info


def export_fold0(config: dict, phase1c_work: Path, k1_dir: Path, work: Path, out: Path, keep_k: int, workers: int,
                 root: Path = ROOT) -> dict:
    guard = ResourceGuard(1.0, 8.0, "rerun the same topk_export command")
    ctx = load_context(config, root, phase1c_work)
    ctx["train_dir"] = root / config["inputs"]["train_dir"]
    build_stores(config, root, work, ctx, guard)
    chunks = work / f"topk_fold0_k{keep_k}"
    build_features(config, ctx, phase1c_work, work, k1_dir, keep_k, workers, features_dir=str(chunks),
                   export={"topk_only": True})
    arrays: dict[str, list] = {k: [] for k in ("s1_pos", "cand", "label", "filter_score", "filter_rank", "s1_positions",
                                               "s1_truth_len", "s1_retrieved_truth")}
    for path in sorted(chunks.glob("chunk*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for name in arrays:
                arrays[name].append(data[name])
    a = {name: np.concatenate(values) for name, values in arrays.items()}
    exceptions = json.loads((work / "stores" / "id_exceptions.json").read_text())
    summary = {}
    for split in ("train", "validation", "holdout"):
        low, high = config["split"][split]
        rows = (a["s1_pos"] >= low) & (a["s1_pos"] < high)
        ent = (a["s1_positions"] >= low) & (a["s1_positions"] < high)
        summary[split] = write_split(out / split, ctx["ordered"], exceptions, a["s1_pos"][rows], a["cand"][rows],
                                     a["label"][rows].astype(np.int8), a["filter_score"][rows], a["filter_rank"][rows],
                                     a["s1_positions"][ent], a["s1_truth_len"][ent], a["s1_retrieved_truth"][ent])
        log(f"topk: fold0 {split}: {summary[split]}")
    return summary


def export_test(k3_config: dict, test_dir: Path, blocking_root: Path, k1_dir: Path, work: Path, out: Path, keep_k: int,
                workers: int, root: Path = ROOT) -> dict:
    from .k1_filter import Stores
    from .k2_features import density_table
    from .k3_inference import load_test_context, score_all

    guard = ResourceGuard(1.0, 8.0, "rerun the same topk_export command")
    ctx = load_test_context(k3_config, root, test_dir)
    build_stores(k3_config, root, work, ctx, guard, targets_dir=test_dir, prefix="test")
    density_table(Stores(work), work)
    models = {"topk_only": True, "keep_k": keep_k, "filter_model": str(k1_dir / "models" / "filter_final.txt")}
    size = ctx["p1c"]["algorithm"]["shard_size"]
    n = len(ctx["ordered"])
    score_all(k3_config, ctx, work, blocking_root, models, workers, list(range((n + size - 1) // size)))
    parts: dict[str, list] = {k: [] for k in ("s1_pos", "cand", "filter_score", "filter_rank")}
    for shard in range((n + size - 1) // size):
        with np.load(work / "topk" / f"shard{shard:03d}.npz", allow_pickle=False) as data:
            for name in parts:
                parts[name].append(data[name])
    a = {name: np.concatenate(values) for name, values in parts.items()}
    exceptions = json.loads((work / "stores" / "id_exceptions.json").read_text())
    minus = np.full(n, -1, np.int64)
    summary = {"test": write_split(out / "test", ctx["ordered"], exceptions, a["s1_pos"], a["cand"],
                                   np.full(len(a["cand"]), -1, np.int8), a["filter_score"], a["filter_rank"],
                                   np.arange(n, dtype=np.int64), minus, minus)}
    log(f"topk: test: {summary['test']}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("fold0", "test"), required=True)
    parser.add_argument("--config", type=Path, required=True, help="k2_matcher.json (fold0) or k3_inference.json (test)")
    parser.add_argument("--k1-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-k", type=int, default=40)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--phase1c-work-dir", type=Path, help="fold0: fold-0 Phase 1C work directory")
    parser.add_argument("--test-dir", type=Path, help="test: directory with test_source*.tsv")
    parser.add_argument("--blocking-root", type=Path, help="test: directory holding test_range_*_manifest.json + work/")
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    workers = args.workers or config["workers"]
    if args.split == "fold0":
        summary = export_fold0(config, args.phase1c_work_dir, args.k1_dir, args.work_dir, args.output_dir, args.keep_k, workers)
    else:
        summary = export_test(config, args.test_dir, args.blocking_root, args.k1_dir, args.work_dir, args.output_dir,
                              args.keep_k, workers)
    atomic_write_json(args.output_dir / f"topk_{args.split}_summary.json", {"keep_k": args.keep_k, **summary})


if __name__ == "__main__":
    main()
