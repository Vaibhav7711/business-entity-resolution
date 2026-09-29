"""K3: test inference — filter, stage-2 champion, set policy, both submission files, official validator.

Run from code/business_entity_resolution (resumable per test shard):

    python3 -m src.k3_inference --config ../../configs/k3_inference.json \
        --test-dir <test TSV dir> --blocking-root <dir holding test_range_*_manifest.json + work/> \
        --k1-dir <K1 results> --k2-dir <K2 results> --work-dir /tmp/k3 --output-dir /kaggle/working/k3 --stage all

Stages:
* ``stores``: test S2/S3 and S1 token/text stores; IDF and number-key density fit on the test
  corpus, mirroring how training fit them on the training corpus.
* ``score``: per test shard, rebuild each S1's candidate union from the verified test route
  shards, keep the K1 filter's top K, compute the champion's features, and store calibrated
  scores.
* ``write``: apply the frozen threshold and empty-set guard (plus one-S1-per-record
  reassignment when K2 adopted it), write ``candidate_pairs.tsv`` (the filtered list the
  matcher scored) and ``matching_results.tsv``, run the official validator, and write
  unlabeled per-country diagnostics.

Ground truth is never read. Test records are only scored, never used to choose anything.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .block_test import ordered_test_ids
from .blocking import decode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import ROUTES, Workspace, atomic_savez, atomic_write_json, content_hash, country_slug, count_summary
from .k1_filter import Stores, build_chunk, build_stores, cheap_features, rank_within_s1, shard_keys
from .k2_features import (
    ALL_FEATURES, CHEAP_INDEX, NORM_EXTRA, TargetAdapter, context_features, density_table, v2_features,
)
from .normalization import has_non_ascii, normalize_text
from .phase2a_env import log
from .phase2a_eval import apply_policy
from .phase2a_pairs import compute_features


class TestShards:
    """Test route shards spread over several shard-range work directories, hash-verified."""

    def __init__(self, blocking_root: Path, n_shards: int):
        self.by_shard: dict[int, tuple[Path, dict]] = {}
        manifests = sorted(glob.glob(str(blocking_root / "**" / "test_range_*_manifest.json"), recursive=True))
        for path in manifests:
            manifest = json.loads(Path(path).read_text())
            if manifest.get("status") != "complete" or manifest.get("labels_read") is not False:
                raise RuntimeError(f"Incomplete or invalid test blocking manifest: {path}")
            work = Path(path).parent / "work"
            first, last = manifest["shards"]
            for shard in range(first, last + 1):
                if shard in self.by_shard:
                    raise RuntimeError(f"Test shard {shard} appears in two blocking ranges")
                self.by_shard[shard] = (work, manifest["tasks"])
        missing = sorted(set(range(n_shards)) - set(self.by_shard))
        if missing:
            raise RuntimeError(f"Test blocking is missing shards {missing[:10]} (found {len(manifests)} range manifests)")

    def load(self, shard: int, keys: list[str]) -> tuple[dict, dict]:
        work, expected = self.by_shard[shard]
        ws = Workspace(work)
        loaded_by_key, locator = {}, {}
        for key in keys:
            loaded = {}
            for route in ROUTES:
                for source in (2, 3):
                    path = ws.task_path(source, country_slug(key), route, shard)
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {name: data[name] for name in data.files}
                    rel = str(path.with_suffix(".json").relative_to(ws.work))
                    if content_hash(arrays) != expected.get(rel):
                        raise RuntimeError(f"Test shard content differs from its range manifest: {rel}")
                    loaded[route, source] = arrays
            positions = loaded["exact_name", 2]["positions"]
            loaded_by_key[key] = loaded
            for local, position in enumerate(positions):
                locator[int(position)] = (key, local)
        return loaded_by_key, locator


def load_test_context(config: dict, root: Path, test_dir: Path) -> dict:
    p1c = json.loads((root / config["inputs"]["phase1c_config"]).read_text())
    ordered = ordered_test_ids(test_dir, p1c["algorithm"]["seed"])
    position = {entity_id: i for i, entity_id in enumerate(ordered)}
    n = len(ordered)
    queries = {name: [None] * n for name in ("country", "country_key", "name", "address")}
    non_ascii = np.zeros(n, bool)
    file_order = []
    with (test_dir / "test_source1.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            i = position[row["entity_id"]]
            file_order.append(i)
            queries["country"][i] = row["country"]
            queries["country_key"][i] = normalize_text(row["country"])
            queries["name"][i] = normalize_text(row["business_name"])
            queries["address"][i] = normalize_text(row["business_address"])
            non_ascii[i] = has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"])
    queries["non_ascii"] = non_ascii
    keys = sorted(set(queries["country_key"]))
    raw_by_key = {}
    for raw, key in zip(queries["country"], queries["country_key"]):
        raw_by_key.setdefault(key, raw)
    return {"p1c": p1c, "ordered": ordered, "queries": queries, "file_order": np.asarray(file_order, np.int64),
            "truth": (np.zeros(n + 1, np.int64), np.zeros(0, np.int64)), "keys": keys, "raw_by_key": raw_by_key,
            "country_code": np.asarray([keys.index(k) for k in queries["country_key"]], np.int8)}


def load_models(k1_dir: Path, k2_dir: Path) -> dict:
    policy = json.loads((k2_dir / "models" / "champion_policy.json").read_text())
    calibrator = json.loads((k2_dir / "models" / "champion_calibrator.json").read_text())
    canon_path = next(iter(sorted(k2_dir.rglob("canon_map.json"))), None)
    if canon_path is None:
        raise FileNotFoundError("canon_map.json missing from K2 results")
    filter_choice = json.loads((k2_dir / "filter_choice.json").read_text())
    e4_path = k2_dir / "experiments" / "E4_reassignment.json"
    e4 = json.loads(e4_path.read_text()) if e4_path.exists() else {"adopted": False}
    return {"policy": policy, "calibrator": calibrator, "canon": json.loads(canon_path.read_text()),
            "keep_k": int(filter_choice["keep_k"]), "reassign": bool(e4.get("adopted")),
            "filter_model": str(k1_dir / "models" / "filter_final.txt"), "champion_model": str(k2_dir / "models" / "champion.txt")}


_STATE: dict = {}


def init_worker(config: dict, ctx: dict, work: str, blocking_root: str, models: dict) -> None:
    import lightgbm as lgb

    stores = Stores(Path(work))
    size = ctx["p1c"]["algorithm"]["shard_size"]
    _STATE.update(config=config, ctx=ctx, stores=stores, adapter=TargetAdapter(stores), models=models,
                  shards=TestShards(Path(blocking_root), (len(ctx["ordered"]) + size - 1) // size),
                  density=density_table(stores, Path(work)), work=Path(work),
                  filter=lgb.Booster(model_file=models["filter_model"]))
    if not models.get("topk_only"):
        _STATE.update(champion=lgb.Booster(model_file=models["champion_model"]),
                      feature_idx=np.asarray([ALL_FEATURES.index(name) for name in models["policy"]["features"]], np.int64))
    q = ctx["queries"]
    _STATE["s1"] = {"name": q["name"], "address": q["address"],
                    "address_missing": np.asarray([not a for a in q["address"]]), "non_ascii": q["non_ascii"]}


def score_shard(shard: int) -> dict:
    s = _STATE
    config, ctx, stores, models = s["config"], s["ctx"], s["stores"], s["models"]
    size = ctx["p1c"]["algorithm"]["shard_size"]
    topk_only = bool(models.get("topk_only"))
    out = s["work"] / ("topk" if topk_only else "scores") / f"shard{shard:03d}.npz"
    if out.exists():
        return {"shard": shard, "skipped": True}
    started = time.perf_counter()
    n = len(ctx["ordered"])
    low, high = shard * size, min((shard + 1) * size, n)
    loaded, locator = s["shards"].load(shard, shard_keys(ctx, shard, size))
    cache = {"s1": {}, "t": {}}
    parts = defaultdict(list)
    if not topk_only:
        cal_x, cal_y = np.asarray(models["calibrator"]["x"]), np.asarray(models["calibrator"]["y"])
    for start in range(low, high, config["chunk_s1"]):
        stop = min(start + config["chunk_s1"], high)
        arrays = build_chunk(start, stop, loaded, locator, ctx, config["rrf_constant"])
        rows = np.arange(len(arrays["cand"]))
        cheap = cheap_features(arrays, rows, stores, start)
        fscore = s["filter"].predict(cheap, num_threads=1)
        rank = rank_within_s1(arrays["s1_pos"].astype(np.int64), fscore, arrays["cand"].astype(np.int64))
        keep = rank < models["keep_k"]
        if topk_only:
            parts["s1_pos"].append(arrays["s1_pos"][keep].astype(np.int64))
            parts["cand"].append(arrays["cand"][keep])
            parts["filter_rank"].append(rank[keep].astype(np.int16))
            parts["filter_score"].append(fscore[keep].astype(np.float32))
            parts["cand_count"].append(arrays["s1_cand_count"])
            continue
        kept = {name: arrays[name][keep] for name in ("s1_pos", "cand", "label", "route_bits", "rrf", "rrf_rank",
                                                      "cand_count", *[k for k in arrays if k.startswith(("score_", "rank_"))])}
        kept["s1_positions"], kept["s1_exact_hits"] = arrays["s1_positions"], arrays["s1_exact_hits"]
        v1 = compute_features(kept, np.arange(len(kept["cand"])), s["s1"], s["adapter"])
        extra = cheap[keep][:, [CHEAP_INDEX[name] for name in NORM_EXTRA]]
        v2 = v2_features(kept, ctx, stores, models["canon"], cache)
        context = context_features(kept, fscore[keep].astype(np.float32), rank[keep].astype(np.float32), cheap[keep],
                                   stores, s["density"])
        X = np.column_stack([v1, extra, v2, context]).astype(np.float32)
        raw = s["champion"].predict(X[:, s["feature_idx"]], num_threads=1)
        parts["s1_pos"].append(kept["s1_pos"].astype(np.int64))
        parts["cand"].append(kept["cand"])
        parts["filter_rank"].append(rank[keep].astype(np.int16))
        parts["score"].append(np.interp(raw, cal_x, cal_y).astype(np.float32))
        parts["cand_count"].append(arrays["s1_cand_count"])
        if len(cache["t"]) > 400_000:
            cache["t"].clear()
    atomic_savez(out, {name: np.concatenate(values) for name, values in parts.items()} |
                 {"positions": np.arange(low, high, dtype=np.int64)})
    return {"shard": shard, "seconds": time.perf_counter() - started}


def score_all(config: dict, ctx: dict, work: Path, blocking_root: Path, models: dict, workers: int, shards: list[int]) -> None:
    import multiprocessing as mp

    (work / ("topk" if models.get("topk_only") else "scores")).mkdir(parents=True, exist_ok=True)
    args = (config, ctx, str(work), str(blocking_root), models)
    started = time.perf_counter()
    if workers <= 1:
        init_worker(*args)
        for shard in shards:
            info = score_shard(shard)
            status = "done earlier" if info.get("skipped") else f"{info['seconds']:.0f}s"
            log(f"score: shard {shard} ({status})")
    else:
        with mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=args) as pool:
            for done, info in enumerate(pool.imap_unordered(score_shard, shards), 1):
                status = "done earlier" if info.get("skipped") else f"{info.get('seconds', 0):.0f}s"
                log(f"score: shard {info['shard']} ({done}/{len(shards)}, {status})")
    log(f"score: {len(shards)} shards in {(time.perf_counter() - started) / 60:.1f} min")


def decode_all(values: np.ndarray, exceptions: dict[str, str]) -> list[str]:
    return [exceptions.get(str(v)) or decode_id(v) for v in values.tolist()]


def write_outputs(config: dict, ctx: dict, work: Path, output: Path, models: dict, test_dir: Path) -> dict:
    n = len(ctx["ordered"])
    size = ctx["p1c"]["algorithm"]["shard_size"]
    parts = defaultdict(list)
    for shard in range((n + size - 1) // size):
        with np.load(work / "scores" / f"shard{shard:03d}.npz", allow_pickle=False) as data:
            for name in ("s1_pos", "cand", "score", "filter_rank"):
                parts[name].append(data[name])
    s1_pos, cand, score = (np.concatenate(parts[name]) for name in ("s1_pos", "cand", "score"))
    policy = models["policy"]
    predicted = apply_policy(s1_pos, score, n, policy["threshold"], policy["empty_threshold"])
    reassigned = 0
    if models["reassign"]:
        claim = np.flatnonzero(predicted)
        order = claim[np.lexsort((s1_pos[claim], -score[claim], cand[claim]))]
        first = np.r_[True, cand[order][1:] != cand[order][:-1]]
        losers = order[~first]
        predicted[losers] = False
        reassigned = int(len(losers))
    exceptions = json.loads((work / "stores" / "id_exceptions.json").read_text())
    order = np.lexsort((-score, s1_pos))
    s1_sorted, cand_sorted, pred_sorted = s1_pos[order], cand[order], predicted[order]
    starts = np.searchsorted(s1_sorted, np.arange(n + 1))
    out_dir = output / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ctx["ordered"]
    with (out_dir / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as cfile, \
            (out_dir / "matching_results.tsv").open("w", encoding="utf-8", newline="") as mfile:
        cfile.write("source1_entity_id\tcandidate_entity_ids\n")
        mfile.write("source1_entity_id\tmatched_entity_ids\n")
        for position in ctx["file_order"].tolist():
            left, right = starts[position], starts[position + 1]
            c_ids = decode_all(cand_sorted[left:right], exceptions)
            m_ids = [c for c, p in zip(c_ids, pred_sorted[left:right].tolist()) if p]
            cfile.write(f"{ids[position]}\t{','.join(c_ids)}\n")
            mfile.write(f"{ids[position]}\t{','.join(m_ids)}\n")
    n_pred = np.bincount(s1_pos, weights=predicted, minlength=n)
    n_cand = np.bincount(s1_pos, minlength=n)
    best = np.full(n, -1.0)
    np.maximum.at(best, s1_pos, score)
    diagnostics = {}
    for code, key in enumerate(ctx["keys"]):
        mask = ctx["country_code"] == code
        diagnostics[ctx["raw_by_key"][key]] = {
            "s1": int(mask.sum()), "candidates_kept": count_summary(n_cand[mask]),
            "predicted_match_rate": float(np.mean(n_pred[mask] > 0)), "mean_predicted_set_size": float(n_pred[mask].mean()),
            "best_score_quantiles": {str(q): float(np.quantile(best[mask], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "empty_prediction_rate": float(np.mean(n_pred[mask] == 0))}
    summary = {"s1": n, "candidate_pairs": int(len(cand)), "matched_pairs": int(predicted.sum()), "policy": policy,
               "keep_k": models["keep_k"], "reassignment_applied": models["reassign"], "claims_reassigned": reassigned,
               "diagnostics_by_country": diagnostics, "labels_read": False}
    return summary


def run_validator(output: Path, test_dir: Path) -> dict:
    validator = ROOT / "student_resource" / "utils" / "validate_submission.py"
    out_dir = output / "output"
    attempts = [["--check-ids", "--candidate", str(out_dir / "candidate_pairs.tsv")], ["--check-ids"]]
    for extra in attempts:
        command = [sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
                   "--test-dir", str(test_dir), *extra]
        if "--candidate" not in extra:
            command += ["--candidate", str(out_dir / "does_not_exist.tsv")]
        result = subprocess.run(command, capture_output=True, text=True)
        text = result.stdout + result.stderr
        if result.returncode == 0 or "FAIL" in text:
            return {"command": extra, "exit_code": result.returncode, "passed": result.returncode == 0,
                    "output_tail": text[-3000:]}
    return {"command": attempts[-1], "exit_code": result.returncode, "passed": False, "output_tail": text[-3000:]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--blocking-root", type=Path, required=True)
    parser.add_argument("--k1-dir", type=Path, required=True)
    parser.add_argument("--k2-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stores", "score", "write", "all"), required=True)
    parser.add_argument("--shards", default=None, help="score only this inclusive range, e.g. 0-34")
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    started = time.perf_counter()
    ctx = load_test_context(config, ROOT, args.test_dir)
    models = load_models(args.k1_dir, args.k2_dir)
    size = ctx["p1c"]["algorithm"]["shard_size"]
    n_shards = (len(ctx["ordered"]) + size - 1) // size
    log(f"K3: {len(ctx['ordered']):,} test S1 in {n_shards} shards; countries {ctx['keys']}; keep top-{models['keep_k']}; "
        f"threshold {models['policy']['threshold']:.4f}, empty guard {models['policy']['empty_threshold']}, "
        f"reassignment {models['reassign']}")
    from .phase2a_env import ResourceGuard
    guard = ResourceGuard(1.0, 8.0, "rerun the same K3 command")
    if args.stage in ("stores", "all"):
        build_stores(config, ROOT, args.work_dir, ctx, guard, targets_dir=args.test_dir, prefix="test")
        density_table(Stores(args.work_dir), args.work_dir)
    if args.stage in ("score", "all"):
        if args.shards:
            first, _, last = args.shards.partition("-")
            shards = list(range(int(first), int(last or first) + 1))
        else:
            shards = list(range(n_shards))
        score_all(config, ctx, args.work_dir, args.blocking_root, models, config["workers"], shards)
    if args.stage in ("write", "all"):
        summary = write_outputs(config, ctx, args.work_dir, args.output_dir, models, args.test_dir)
        summary["validator"] = run_validator(args.output_dir, args.test_dir)
        summary["seconds"] = time.perf_counter() - started
        atomic_write_json(args.output_dir / "k3_summary.json", summary)
        log(f"write: {summary['candidate_pairs']:,} candidate pairs, {summary['matched_pairs']:,} matches; "
            f"validator passed = {summary['validator']['passed']}")
        print(summary["validator"]["output_tail"][-1500:], flush=True)


if __name__ == "__main__":
    main()
