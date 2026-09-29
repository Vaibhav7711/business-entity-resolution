"""Dense retrieval route: a bi-encoder fine-tuned on gold (S1, target) pairs retrieves the top-K targets of every S1
among the targets with the same country label (Kaggle GPU notebook; one process that uses every visible GPU).

The string blocker misses about 2.3% of the gold links, many of them cross-script (a Latin S1 name against a
Devanagari or Odia target name), the rest name/address variants. This route embeds every record once and searches by
cosine, so a candidate needs no shared token or character n-gram.

Model input is ``src.ce_model.record_text`` (``name | address``, case/accents/scripts kept) with the prefix ``query: ``
on both sides, at most ``max_length`` tokens per text; the vector is the attention-masked mean of the last hidden layer,
L2-normalised.

Stages (``--stage all`` runs them in order; each skips work whose outputs exist):

* ``train``: pairs (S1 text, one gold target text) for the S1 of ``train_folds`` (folds 1-4; fold 0 holds the
  validation and holdout S1 and is refused, and any validation/holdout S1 among the training S1 is an error). Pairs are
  drawn in rounds (seeded): one random gold target of every S1, then a second one, ... until ``max_pairs``. Batches are
  cut inside one round, so no batch holds two pairs of the same S1; no target is shared between S1, so no in-batch
  negative is a true match. Symmetric InfoNCE over the in-batch negatives at ``temperature``; AdamW, linear
  warmup/decay, fp16 autocast with a gradient scaler on CUDA. One epoch or ``train_minutes`` (counted from the start of
  the stage), then the model is saved either way. Training runs in this process on cuda:0.
* ``embed``: float16 vectors of the fold-0 validation+holdout S1 (ids from ``<pairs_root>/<split>/s1.parquet``, texts
  from train_source1), the training S2+S3 targets, every test S1 and the test S2+S3 targets: per store
  ``<emb>/<store>/vectors.npy`` (memory-mapped, rows in descending text length) and ``meta.parquet`` (id, country,
  row in the source files). Batches of similar length go from a shared queue to one thread per visible GPU, each with
  its own fp16 model copy and tokenizer. RAM holds one corpus's texts, never all its vectors.
* ``retrieve``: per split and per S1 country label, the cosine top-``top_k`` among the targets with exactly that label:
  fp16 matmul in blocks with ``torch.topk``, merged across target chunks; a country's S1 are split between the GPUs.
  ``<out>/dense/<split>/part-NNNNN.parquet``: s1_id, t_id, dense_score (float32 cosine), dense_rank (int16, 0 = best),
  grouped by S1 in the split's S1 order (test: test_source1 order), rank order inside. An S1 whose country has no
  target has no rows. ``_manifest.json`` marks a finished split.
* ``report``: on validation and holdout, over all gold links of the split's S1: recall@k of the dense lists, recall of
  the filter lists (``<pairs_root>/<split>/part-*.parquet``), the union recall (filter top-40 or dense top-k) and the
  new coverage (share of the gold missing from the filter lists that dense top-k finds), overall, per S1 country and
  (new coverage) per target-name script. ``<out>/dense_report.json``.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import contextlib
import csv
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .ce_model import (autocast, corpus_kind, environment, grad_scaler, length_batches, load_corpus, param_groups,
                       prefetch, quiet_transformers, rank_bounds, read_part, record_text, to_device, to_numpy,
                       within_group)
from .embed_names import resolve_model
from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

LABELLED_SPLITS = ("validation", "holdout")
REPORT_K = (5, 10, 20, 50)
LOG_SECONDS = 60
SUPERBLOCK = 262_144              # queries whose running top-K stays on the device while the target chunks stream past
MANIFEST = "_manifest.json"       # leading underscore: parquet dataset readers skip it
NO_RANK = np.iinfo(np.int32).max


# ---------------------------------------------------------------------------
# Records


class Records:
    """Ids, model texts and country labels of TSV rows in file order (only the ``keep`` ids when given)."""

    def __init__(self, paths: list[Path], keep: set | None = None):
        import pandas as pd

        ids, texts, countries, labels = [], [], [], {}
        for path in paths:
            with path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    if keep is None or row["entity_id"] in keep:
                        ids.append(row["entity_id"])
                        texts.append(record_text(row["business_name"] or "", row["business_address"] or ""))
                        countries.append(labels.setdefault(row["country"] or "", len(labels)))
        if pd.Index(ids).has_duplicates:
            raise ValueError(f"duplicate entity ids in {[p.name for p in paths]}")
        if keep is not None and len(ids) != len(keep):
            missing = sorted(keep.difference(ids))
            raise KeyError(f"{len(missing):,} ids are not in {[p.name for p in paths]}, e.g. {missing[:5]}")
        self.ids = np.asarray(ids, dtype=object)
        self.texts = np.asarray(texts, dtype=object)
        self.lengths = np.fromiter(map(len, texts), np.int32, len(texts))
        self.country = np.asarray(countries, np.int32)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.ids)


def read_folds(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as file:
        return {row["source1_entity_id"]: int(row["fold"]) for row in csv.DictReader(file, delimiter="\t")}


def read_gold(train_dir: Path, keep) -> dict[str, list[str]]:
    """S1 id -> its distinct gold target ids, for the S1 with gold that ``keep(s1_id)`` accepts."""
    gold = {}
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["matched_entity_ids"] and keep(row["source1_entity_id"]):
                gold[row["source1_entity_id"]] = list(dict.fromkeys(row["matched_entity_ids"].split(",")))
    return gold


def read_field(paths: list[Path], keep: set, field: str) -> dict[str, str]:
    out = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if row["entity_id"] in keep:
                    out[row["entity_id"]] = row[field] or ""
    return out


def split_s1(pairs_root: Path, split: str) -> list[str]:
    return to_numpy(read_part(pairs_root / split / "s1.parquet", ["s1_id"]).column("s1_id")).tolist()


def labelled_splits(pairs_root: Path) -> list[str]:
    return [s for s in LABELLED_SPLITS if (pairs_root / s / "s1.parquet").is_file()]


def apply_split_config(cfg: dict) -> tuple[str, ...]:
    """Config ``labelled_splits`` (default validation, holdout); adding "train" also builds dense lists for the fold-0
    training S1 (the model trains on folds 1-4 only, so they are unseen), e.g. to train a cross-encoder on them."""
    global LABELLED_SPLITS
    LABELLED_SPLITS = tuple(cfg.get("labelled_splits", ("validation", "holdout")))
    return LABELLED_SPLITS


# ---------------------------------------------------------------------------
# Model


def devices() -> list:
    import torch

    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def device_scope(device):
    import torch

    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def load_encoder(source: str, revision: str | None, device, *, inference: bool = False):
    """Tokenizer + encoder (no task head); SDPA attention when supported; fp16 weights for inference on CUDA."""
    from transformers import AutoModel, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    try:
        model = AutoModel.from_pretrained(source, attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError, ImportError):
        model = AutoModel.from_pretrained(source, **kwargs)
    model = model.half() if inference and device.type == "cuda" else model.float()
    return tokenizer, model.to(device)


def encode_texts(tokenizer, texts, prefix: str, max_length: int, pin: bool) -> dict:
    batch = tokenizer([prefix + text for text in texts], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt")
    return {k: (v.pin_memory() if pin else v) for k, v in batch.items()}


def pooled(model, inputs: dict):
    """Attention-masked mean of the last hidden layer, L2-normalised, float32."""
    import torch

    hidden = model(**inputs).last_hidden_state.float()
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return torch.nn.functional.normalize(mean, dim=-1)


def info_nce(a, b, temperature: float):
    """Symmetric InfoNCE: row i of ``a`` belongs to row i of ``b``; every other row of the batch is a negative."""
    import torch

    logits = a.float() @ b.float().T / temperature
    target = torch.arange(len(a), device=a.device)
    return (torch.nn.functional.cross_entropy(logits, target) +
            torch.nn.functional.cross_entropy(logits.T, target)) / 2


# ---------------------------------------------------------------------------
# Train


def pair_rounds(group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """0 for one random pair of each group, 1 for a second one, and so on."""
    order = np.lexsort((rng.random(len(group)), group))
    rounds = np.empty(len(group), np.int64)
    rounds[order] = within_group(group[order])
    return rounds


def training_pairs(cfg: dict, train_dir: Path, folds_path: Path, pairs_root: Path) -> dict:
    """Seeded (S1, gold target) pairs of the training folds, their texts and batches without a repeated S1."""
    import pyarrow as pa

    train_folds = {int(f) for f in cfg["train_folds"]}
    if 0 in train_folds:
        raise ValueError("fold 0 holds the validation and holdout S1 and is never trained on")
    folds = read_folds(folds_path)
    gold = read_gold(train_dir, lambda s1: folds.get(s1) in train_folds)
    held_out = {s for split in labelled_splits(pairs_root) for s in split_s1(pairs_root, split)}
    leaked = held_out.intersection(gold)
    if leaked:
        raise ValueError(f"{len(leaked):,} validation/holdout S1 are in the training folds, e.g. {sorted(leaked)[:3]}")
    s1_ids = np.asarray(sorted(gold), dtype=object)
    counts = np.fromiter((len(gold[s]) for s in s1_ids), np.int64, len(s1_ids))
    group = np.repeat(np.arange(len(s1_ids)), counts)
    targets = np.asarray([t for s in s1_ids for t in gold[s]], dtype=object)
    if not len(group):
        raise ValueError(f"no gold pairs for the S1 of folds {sorted(train_folds)}")
    rng = np.random.default_rng(cfg["seed"])
    rounds = pair_rounds(group, rng)
    chosen = np.lexsort((rng.random(len(group)), rounds))[:cfg["max_pairs"]]
    s1_col, t_col = s1_ids[group[chosen]], targets[chosen]
    s1_texts, t_texts = load_corpus("train", train_dir, set(s1_col), set(t_col))
    data = {"a": s1_texts.texts, "b": t_texts.texts, "s1": s1_col, "round": rounds[chosen],
            "ia": s1_texts.positions(pa.array(s1_col.tolist(), pa.string()), "train s1_id"),
            "ib": t_texts.positions(pa.array(t_col.tolist(), pa.string()), "train t_id"),
            "gold_pairs": int(len(group)), "s1_count": int(len(np.unique(group[chosen]))),
            "folds_seen": sorted({folds[s] for s in set(s1_col)})}
    data["length"] = s1_texts.lengths[data["ia"]] + t_texts.lengths[data["ib"]]
    batches = []
    for r in np.unique(data["round"]):
        batches += length_batches(np.flatnonzero(data["round"] == r), data["length"], cfg["batch_size"], rng)
    data["batches"] = [batches[i] for i in rng.permutation(len(batches))]
    data["round_pairs"] = np.bincount(data["round"]).tolist()
    log(f"train: {len(s1_ids):,} S1 of folds {sorted(train_folds)} with {len(group):,} gold pairs; "
        f"{len(chosen):,} pairs of {data['s1_count']:,} S1 sampled (per round {data['round_pairs']}), "
        f"{len(data['batches']):,} batches")
    return data


def pair_batches(tokenizer, data: dict, batches: list, cfg: dict, pin: bool):
    """One forward per batch: the S1 texts, then their targets' texts."""
    for rows in batches:
        texts = np.concatenate([data["a"][data["ia"][rows]], data["b"][data["ib"][rows]]])
        yield len(rows), encode_texts(tokenizer, texts, cfg["prefix"], cfg["max_length"], pin)


def train(cfg: dict, dirs: dict, folds_path: Path, pairs_root: Path, out: Path) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    log_path = out / "train_log.json"
    if log_path.is_file() and (out / "model" / "config.json").is_file():
        log(f"train: {log_path} exists; skipping")
        return json.loads(log_path.read_text())
    started = time.perf_counter()
    spec = resolve_model({"repo": cfg["backbone"]}, cfg["allowed_licenses"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"train: backbone {spec['repo']} @ {spec.get('revision')} (license {spec['license']}) on {device}")
    data = training_pairs(cfg, dirs["train"], folds_path, pairs_root)
    torch.manual_seed(cfg["seed"])
    tokenizer, model = load_encoder(spec["repo"], spec.get("revision"), device)
    batches = data["batches"]
    total_steps = len(batches)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    schedule = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup"] * total_steps), total_steps)
    scaler = grad_scaler(device.type == "cuda")
    budget = cfg["train_minutes"] * 60
    step, seen, stopped, curve = 0, 0, False, []
    window_loss, window_steps = torch.zeros((), device=device), 0
    loop_started = time.perf_counter()
    model.train()
    for n, inputs in prefetch(pair_batches(tokenizer, data, batches, cfg, device.type == "cuda")):
        if time.perf_counter() - started > budget:
            stopped = True
            break
        with autocast(device):
            vectors = pooled(model, to_device(inputs, device))
        loss = info_nce(vectors[:n], vectors[n:], cfg["temperature"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        step += 1
        seen += n
        window_loss += loss.detach()
        window_steps += 1
        if step % cfg["log_every"] == 0 or step == total_steps:
            curve.append([step, float(window_loss.item()) / window_steps])
            window_loss.zero_()
            window_steps = 0
            rate = seen / max(time.perf_counter() - loop_started, 1e-9)
            log(f"train: step {step:,}/{total_steps:,} loss {curve[-1][1]:.4f} {rate:,.0f} pairs/s "
                f"lr {schedule.get_last_lr()[0]:.2e}")
    if window_steps:
        curve.append([step, float(window_loss.item()) / window_steps])
    loop_seconds = time.perf_counter() - loop_started
    if stopped:
        log(f"train: time budget of {cfg['train_minutes']} min reached at step {step:,}/{total_steps:,}")
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    info = {"pairs": int(len(data["s1"])), "s1": data["s1_count"], "gold_pairs_available": data["gold_pairs"],
            "round_pairs": data["round_pairs"], "train_folds_seen": data["folds_seen"], "pairs_seen": seen,
            "steps": step, "planned_steps": total_steps, "stopped_early": stopped,
            "seconds": time.perf_counter() - started, "train_loop_seconds": loop_seconds,
            "throughput": seen / max(loop_seconds, 1e-9),
            "loss_first": curve[0][1] if curve else None, "loss_last": curve[-1][1] if curve else None,
            "loss_curve": curve, "backbone": spec["repo"], "revision": spec.get("revision"),
            "license": spec["license"], "device": str(device), "model_stamp": str(time.time_ns()),
            "config": cfg, "environment": environment()}
    atomic_write_json(log_path, info)
    log(f"train: saved {out / 'model'} after {step:,} steps, {seen:,} pairs, {info['seconds']:,.0f}s "
        f"({info['throughput']:,.0f} pairs/s)")
    return info


# ---------------------------------------------------------------------------
# Embed


def store_plan(dirs: dict, pairs_root: Path) -> dict[str, tuple[list[Path], set | None]]:
    """Embedding stores: name -> (source files, ids to keep or None for all)."""
    plan = {}
    splits = labelled_splits(pairs_root)
    if splits:
        plan["train_query"] = ([dirs["train"] / "train_source1.tsv"],
                               {s for split in splits for s in split_s1(pairs_root, split)})
        plan["train_target"] = ([dirs["train"] / f"train_source{k}.tsv" for k in (2, 3)], None)
    if (dirs["test"] / "test_source1.tsv").is_file():
        plan["test_query"] = ([dirs["test"] / "test_source1.tsv"], None)
        plan["test_target"] = ([dirs["test"] / f"test_source{k}.tsv" for k in (2, 3)], None)
    return plan


def model_stamp(out: Path) -> str:
    if not (out / "model" / "config.json").is_file():
        raise FileNotFoundError(f"{out / 'model'}: no trained model (run --stage train first)")
    path = out / "train_log.json"
    return json.loads(path.read_text()).get("model_stamp", "") if path.is_file() else ""


def encode_all(units: list, texts: np.ndarray, vectors: np.ndarray, cfg: dict, label: str) -> None:
    """``vectors[i]`` = the embedding of ``texts[i]``. Batches (contiguous rows) are handed out from one queue to one
    thread per (device, tokenizer, model); each thread tokenizes ahead in its own prefetch thread."""
    import torch

    work: queue.Queue = queue.Queue()
    for start in range(0, len(texts), cfg["embed_batch"]):
        work.put((start, min(start + cfg["embed_batch"], len(texts))))
    lock, failed, errors, done = threading.Lock(), threading.Event(), [], [0]

    def batches(tokenizer, pin: bool):
        while not failed.is_set():
            try:
                start, stop = work.get_nowait()
            except queue.Empty:
                return
            yield start, stop, encode_texts(tokenizer, texts[start:stop], cfg["prefix"], cfg["max_length"], pin)

    def run(device, tokenizer, model) -> None:
        try:
            model.eval()
            with device_scope(device), torch.inference_mode():
                for start, stop, inputs in prefetch(batches(tokenizer, device.type == "cuda")):
                    with autocast(device):
                        embedded = pooled(model, to_device(inputs, device))
                    vectors[start:stop] = embedded.to(torch.float16).cpu().numpy()
                    with lock:
                        done[0] += stop - start
        except BaseException as error:  # noqa: BLE001  (re-raised by the caller)
            failed.set()
            errors.append(error)

    started = time.perf_counter()
    threads = [threading.Thread(target=run, args=unit, daemon=True) for unit in units]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(timeout=LOG_SECONDS)
            if thread.is_alive():
                rate = done[0] / max(time.perf_counter() - started, 1e-9)
                log(f"embed {label}: {done[0]:,}/{len(texts):,} texts, {rate:,.0f}/s, "
                    f"ETA {(len(texts) - done[0]) / max(rate, 1e-9) / 60:,.1f} min")
    if errors:
        raise errors[0]
    if done[0] != len(texts):
        raise RuntimeError(f"embed {label}: {done[0]:,} of {len(texts):,} texts embedded")


def embed_store(name: str, paths: list[Path], keep: set | None, out: Path, cfg: dict, emb_root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    folder, stamp = emb_root / name, model_stamp(out)
    ids_digest = hashlib.sha256("\n".join(sorted(keep)).encode()).hexdigest() if keep is not None else None
    done_path = folder / "done.json"
    done = json.loads(done_path.read_text()) if done_path.is_file() else {}
    if done and done.get("model") == stamp and done.get("ids_digest") == ids_digest:
        log(f"embed {name}: {done['rows']:,} vectors exist in {folder}; skipping")
        return done
    started = time.perf_counter()
    records = Records(paths, keep)
    order = np.argsort(-records.lengths, kind="stable")
    units = [(device, *load_encoder(str(out / "model"), None, device, inference=True)) for device in devices()]
    dim = int(units[0][2].config.hidden_size)
    folder.mkdir(parents=True, exist_ok=True)
    done_path.unlink(missing_ok=True)
    (folder / "vectors.npy").unlink(missing_ok=True)
    need, free = len(order) * dim * 2, shutil.disk_usage(folder).free
    if need + 2**30 > free:
        raise OSError(f"embed {name}: {need / 2**30:.1f} GiB of vectors do not fit in {free / 2**30:.1f} GiB free "
                      f"under {folder}")
    labels = np.asarray(records.labels, dtype=object)
    pq.write_table(pa.table({"id": pa.array(records.ids[order].tolist(), pa.string()),
                             "country": pa.array(labels[records.country[order]].tolist(), pa.string()),
                             "row": pa.array(order.astype(np.int64))}), folder / "meta.parquet")
    log(f"embed {name}: {len(order):,} texts from {[p.name for p in paths]} on {[str(u[0]) for u in units]}")
    if len(order):
        vectors = np.lib.format.open_memmap(folder / "vectors.npy", mode="w+", dtype=np.float16,
                                            shape=(len(order), dim))
        encode_all(units, records.texts[order], vectors, cfg, name)
        vectors.flush()
        del vectors
    else:
        np.save(folder / "vectors.npy", np.zeros((0, dim), np.float16))
    del units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seconds = time.perf_counter() - started
    done = {"rows": int(len(order)), "dim": dim, "model": stamp, "ids_digest": ids_digest, "seconds": seconds,
            "texts_per_s": len(order) / max(seconds, 1e-9), "devices": [str(d) for d in devices()],
            "countries": dict(zip(records.labels, np.bincount(records.country, minlength=len(labels)).tolist()))}
    atomic_write_json(done_path, done)
    log(f"embed {name}: {len(order):,} vectors in {seconds:,.0f}s ({done['texts_per_s']:,.0f} texts/s) -> {folder}")
    return done


def open_store(folder: Path) -> dict:
    """ids (Arrow strings), country codes + labels, source-file rows and the memory-mapped vectors of one store."""
    import pyarrow.parquet as pq

    if not (folder / "done.json").is_file():
        raise FileNotFoundError(f"{folder}: no finished embedding store (run --stage embed first)")
    meta = pq.read_table(folder / "meta.parquet")
    country = meta.column("country").combine_chunks().dictionary_encode()
    return {"ids": meta.column("id").combine_chunks(), "country": to_numpy(country.indices).astype(np.int32),
            "labels": country.dictionary.to_pylist(), "row": to_numpy(meta.column("row")),
            "vectors": np.load(folder / "vectors.npy", mmap_mode="r"),
            "done": json.loads((folder / "done.json").read_text())}


# ---------------------------------------------------------------------------
# Retrieve


def search_on(device, queries: np.ndarray, targets: np.ndarray, k: int, cfg: dict, index_out: np.ndarray,
              score_out: np.ndarray) -> None:
    """Exact top-``k`` dot products of ``queries`` against ``targets`` on one device (fp16 on CUDA, fp32 on CPU)."""
    import torch

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    block, chunk = cfg["search_query_block"], cfg["search_target_chunk"]
    with device_scope(device), torch.inference_mode():
        for s0 in range(0, len(queries), SUPERBLOCK):
            q = torch.from_numpy(np.ascontiguousarray(queries[s0:s0 + SUPERBLOCK])).to(device=device, dtype=dtype)
            best_s = torch.full((len(q), k), -float("inf"), device=device)
            best_i = torch.full((len(q), k), -1, dtype=torch.int64, device=device)
            for c0 in range(0, len(targets), chunk):
                t = torch.from_numpy(np.ascontiguousarray(targets[c0:c0 + chunk])).to(device=device, dtype=dtype)
                kk = min(k, len(t))
                for b0 in range(0, len(q), block):
                    top_s, top_i = torch.topk(q[b0:b0 + block] @ t.T, kk, dim=1)
                    merged_s = torch.cat([best_s[b0:b0 + block], top_s.float()], dim=1)
                    merged_i = torch.cat([best_i[b0:b0 + block], top_i + c0], dim=1)
                    new_s, where = torch.topk(merged_s, k, dim=1)
                    best_s[b0:b0 + block] = new_s
                    best_i[b0:b0 + block] = merged_i.gather(1, where)
                del t
            index_out[s0:s0 + len(q)] = best_i.cpu().numpy()
            score_out[s0:s0 + len(q)] = best_s.cpu().numpy()


def search(queries: np.ndarray, targets: np.ndarray, k: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` (``k <= len(targets)``) target positions (int32) and scores (float32) per query, best first; the
    queries are split between the visible devices, one thread each."""
    units = devices()
    index, score = np.full((len(queries), k), -1, np.int32), np.zeros((len(queries), k), np.float32)
    errors = []

    def run(device, start: int, stop: int) -> None:
        try:
            search_on(device, queries[start:stop], targets, k, cfg, index[start:stop], score[start:stop])
        except BaseException as error:  # noqa: BLE001  (re-raised below)
            errors.append(error)

    threads = [threading.Thread(target=run, args=(device, *rank_bounds(len(queries), len(units), r)), daemon=True)
               for r, device in enumerate(units)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return index, score


def write_dense(folder: Path, s1_ids: np.ndarray, index: np.ndarray, score: np.ndarray, t_ids, part_rows: int) -> int:
    """Parquet parts of (s1_id, t_id, dense_score, dense_rank), grouped by S1 in ``s1_ids`` order, rank order inside;
    ``index`` -1 marks no row. Returns the number of parts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    per_part = max(1, part_rows // max(index.shape[1], 1))
    starts = range(0, max(len(s1_ids), 1), per_part)
    for part, start in enumerate(starts):
        rows, ranks = np.nonzero(index[start:start + per_part] >= 0)
        found = index[start:start + per_part][rows, ranks]
        table = pa.table({"s1_id": pa.array(s1_ids[start:start + per_part][rows].tolist(), pa.string()),
                          "t_id": t_ids.take(pa.array(found.astype(np.int64))),
                          "dense_score": pa.array(score[start:start + per_part][rows, ranks], pa.float32()),
                          "dense_rank": pa.array(ranks.astype(np.int16), pa.int16())})
        pq.write_table(table, folder / f"part-{part:05d}.parquet", compression="zstd")
    return len(starts)


def split_done(out: Path, split: str) -> bool:
    """The split's dense lists are finished for the current model."""
    path = out / "dense" / split / MANIFEST
    return path.is_file() and json.loads(path.read_text()).get("model") == model_stamp(out)


def retrieve_split(cfg: dict, split: str, pairs_root: Path, out: Path, emb_root: Path) -> dict:
    import pandas as pd

    kind, folder, stamp = corpus_kind(split), out / "dense" / split, model_stamp(out)
    if split_done(out, split):
        log(f"retrieve {split}: {folder} exists; skipping")
        return json.loads((folder / MANIFEST).read_text())
    queries, targets = open_store(emb_root / f"{kind}_query"), open_store(emb_root / f"{kind}_target")
    for name, store in (("query", queries), ("target", targets)):
        if store["done"]["model"] != stamp:
            raise RuntimeError(f"retrieve {split}: the {kind}_{name} vectors come from another model than "
                               f"{out / 'model'} (re-run --stage embed)")
    started = time.perf_counter()
    q_ids = to_numpy(queries["ids"])
    if split == "test":
        q_pos = np.argsort(queries["row"], kind="stable")                   # test_source1 order
    else:
        wanted = split_s1(pairs_root, split)
        q_pos = pd.Index(q_ids).get_indexer(wanted)
        if (q_pos < 0).any():
            raise KeyError(f"retrieve {split}: {int((q_pos < 0).sum()):,} S1 have no vector (re-run --stage embed)")
    top_k = cfg["top_k"]
    index, score = np.full((len(q_pos), top_k), -1, np.int32), np.zeros((len(q_pos), top_k), np.float32)
    q_codes = queries["country"][q_pos]
    t_code_of = {label: code for code, label in enumerate(targets["labels"])}
    countries = {}
    for code in np.unique(q_codes):
        label, members = queries["labels"][code], np.flatnonzero(q_codes == code)
        t_code = t_code_of.get(label)
        rows = np.flatnonzero(targets["country"] == t_code) if t_code is not None else np.zeros(0, np.int64)
        countries[label] = {"s1": int(len(members)), "targets": int(len(rows))}
        if not len(rows):
            log(f"retrieve {split}: {len(members):,} S1 of country {label!r} have no target of that country")
            continue
        began, k = time.perf_counter(), min(top_k, len(rows))
        found, sims = search(queries["vectors"][q_pos[members]], targets["vectors"][rows], k, cfg)
        index[members, :k], score[members, :k] = rows[found], sims
        countries[label] |= {"k": k, "seconds": time.perf_counter() - began}
        log(f"retrieve {split} {label}: {len(members):,} S1 x {len(rows):,} targets, top-{k} in "
            f"{countries[label]['seconds']:,.0f}s")
        del found, sims
    temporary = folder.with_name(f"{folder.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    parts = write_dense(temporary, q_ids[q_pos], index, score, targets["ids"], cfg["part_rows"])
    manifest = {"split": split, "model": stamp, "top_k": top_k, "s1_requested": int(len(q_pos)),
                "s1_written": int((index[:, 0] >= 0).sum()), "rows": int((index >= 0).sum()), "parts": parts,
                "seconds": time.perf_counter() - started, "devices": [str(d) for d in devices()],
                "countries": countries}
    atomic_write_json(temporary / MANIFEST, manifest)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(temporary, folder)
    log(f"retrieve {split}: {manifest['s1_written']:,}/{manifest['s1_requested']:,} S1, {manifest['rows']:,} rows in "
        f"{parts} parts, {manifest['seconds']:,.0f}s -> {folder}")
    return manifest


# ---------------------------------------------------------------------------
# Report


def read_pairs(split_dir: Path, columns: list[str]):
    import pyarrow as pa

    return pa.concat_tables([read_part(p, columns) for p in sorted(split_dir.glob("part-*.parquet"))])


def target_codes(values) -> np.ndarray:
    from .blocking import encode_id                            # lazy: blocking imports scikit-learn

    return np.fromiter((encode_id(v) for v in values), np.int64, len(values))


def link_keys(s1_column, t_column, position: dict) -> tuple[np.ndarray, np.ndarray]:
    """int64 key (S1 position << 33 | encoded target id) per row, and whether the row's S1 is in ``position``."""
    s1 = s1_column.combine_chunks().dictionary_encode()
    t = t_column.combine_chunks().dictionary_encode()
    s1_pos = np.fromiter((position.get(s, -1) for s in s1.dictionary.to_pylist()), np.int64,
                         len(s1.dictionary))[to_numpy(s1.indices)]
    t_code = target_codes(t.dictionary.to_pylist())[to_numpy(t.indices)]
    return (np.maximum(s1_pos, 0) << 33) | t_code, s1_pos >= 0


def ranks_of(keys: np.ndarray, list_keys: np.ndarray, list_ranks: np.ndarray) -> np.ndarray:
    """The rank each key has in a list (keys unique there), NO_RANK when absent."""
    rank = np.full(len(keys), NO_RANK, np.int64)
    if len(list_keys):
        order = np.argsort(list_keys, kind="stable")
        where = np.minimum(np.searchsorted(list_keys[order], keys), len(order) - 1)
        hit = list_keys[order][where] == keys
        rank[hit] = list_ranks[order][where][hit]
    return rank


def coverage(in_filter: np.ndarray, rank: np.ndarray, ks: list[int]) -> dict:
    """Recall of gold links: filter lists, dense top-k, their union, and dense top-k on the links the filter missed."""
    missing = ~in_filter
    share = (lambda hit: float(hit.mean())) if len(rank) else (lambda hit: None)
    out = {"gold_links": int(len(rank)), "filter_recall": share(in_filter), "missing_from_filter": int(missing.sum())}
    for k in ks:
        hit = rank < k
        out[f"dense_recall@{k}"] = share(hit)
        out[f"union_recall@{k}"] = share(hit | in_filter)
        out[f"new_coverage@{k}"] = float(hit[missing].mean()) if missing.any() else None
        out[f"new_links@{k}"] = int((hit & missing).sum())
    return out


def script_group(text: str) -> str:
    return "non_latin" if any(ch.isalpha() and ord(ch) > 0x24F for ch in text) else "latin"


def report(cfg: dict, pairs_root: Path, out: Path, train_dir: Path) -> dict:
    splits = [s for s in labelled_splits(pairs_root) if (out / "dense" / s / MANIFEST).is_file()]
    if not splits:
        log("report: no finished validation/holdout dense lists; nothing to report")
        return {}
    started = time.perf_counter()
    ks = [k for k in REPORT_K if k <= cfg["top_k"]]
    ids = {split: split_s1(pairs_root, split) for split in splits}
    wanted = {s for split in splits for s in ids[split]}
    gold = read_gold(train_dir, wanted.__contains__)
    country = read_field([train_dir / "train_source1.tsv"], wanted, "country")
    result, missed = {}, {}
    for split in splits:
        position = {s: i for i, s in enumerate(ids[split])}
        g_s1 = np.asarray([position[s] for s in ids[split] for _ in gold.get(s, ())], np.int64)
        g_t = np.asarray([t for s in ids[split] for t in gold.get(s, ())], dtype=object)
        g_keys = (g_s1 << 33) | target_codes(g_t.tolist())
        filt = read_pairs(pairs_root / split, ["s1_id", "t_id"])
        f_keys, f_ok = link_keys(filt.column("s1_id"), filt.column("t_id"), position)
        dense = read_pairs(out / "dense" / split, ["s1_id", "t_id", "dense_rank"])
        d_keys, d_ok = link_keys(dense.column("s1_id"), dense.column("t_id"), position)
        d_keys, d_rank = d_keys[d_ok], to_numpy(dense.column("dense_rank")).astype(np.int64)[d_ok]
        in_filter, rank = np.isin(g_keys, f_keys[f_ok]), ranks_of(g_keys, d_keys, d_rank)
        g_country = np.asarray([country.get(ids[split][i], "") for i in g_s1], dtype=object)
        result[split] = {"s1": len(ids[split]), "s1_with_gold": sum(s in gold for s in ids[split]),
                         "s1_with_dense_rows": int(len(np.unique(d_keys >> 33))), **coverage(in_filter, rank, ks),
                         "by_country": {c: coverage(in_filter[g_country == c], rank[g_country == c], ks)
                                        for c in sorted(set(g_country.tolist()))}}
        missed[split] = (g_t[~in_filter], rank[~in_filter])
    names = read_field([train_dir / f"train_source{k}.tsv" for k in (2, 3)],
                       {t for t_ids, _ in missed.values() for t in t_ids}, "business_name")
    for split, (t_ids, rank) in missed.items():
        script = np.asarray([script_group(names.get(t, "")) for t in t_ids], dtype=object)
        result[split]["missing_by_target_script"] = {
            group: {"links": int((script == group).sum()),
                    **{f"new_coverage@{k}": float((rank[script == group] < k).mean()) for k in ks}}
            for group in sorted(set(script.tolist()))}
        r = result[split]
        log(f"report {split}: {r['gold_links']:,} gold links of {r['s1']:,} S1, filter recall {r['filter_recall']}, "
            f"{r['missing_from_filter']:,} missed by the filter; " +
            "; ".join(f"@{k} dense {r[f'dense_recall@{k}']} union {r[f'union_recall@{k}']} "
                      f"new {r[f'new_coverage@{k}']}" for k in ks))
    result["seconds"] = time.perf_counter() - started
    atomic_write_json(out / "dense_report.json", result)
    return result


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "student_resource/dataset/train")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "student_resource/dataset/test")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/folds.tsv")
    parser.add_argument("--pairs-root", type=Path, required=True, help="top-40 pairs root with validation/holdout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emb-dir", type=Path, help="embedding stores (default <out>/emb; ~17 GB for the full data)")
    parser.add_argument("--stage", choices=("train", "embed", "retrieve", "report", "all"), required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.resolve().read_text())
    apply_split_config(cfg)
    pairs_root, out = args.pairs_root.resolve(), args.out.resolve()
    emb_root = (args.emb_dir or out / "emb").resolve()
    dirs = {"train": args.train_dir.resolve(), "test": args.test_dir.resolve()}
    quiet_transformers(0)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in ("train", "all"):
        train(cfg, dirs, args.folds.resolve(), pairs_root, out)
    plan = store_plan(dirs, pairs_root)
    for kind, splits in (("train", labelled_splits(pairs_root)), ("test", ["test"])):
        names = [name for name in plan if name.startswith(f"{kind}_")]
        if not names:
            if args.stage in ("embed", "retrieve", "all"):
                log(f"no {kind} queries (pairs root / TSVs absent); skipping {kind} embedding and retrieval")
            continue
        if kind == "test" and not cfg.get("retrieve_test", True):
            log("retrieve_test is false: skipping test embedding and retrieval")
            continue
        if args.stage == "all" and all(split_done(out, split) for split in splits):
            log(f"dense lists of {splits} exist for this model; skipping {kind} embedding and retrieval")
            continue
        if args.stage in ("embed", "all"):
            for name in names:
                embed_store(name, *plan[name], out, cfg, emb_root)
        if args.stage in ("retrieve", "all"):
            for split in splits:
                retrieve_split(cfg, split, pairs_root, out, emb_root)
    if args.stage in ("report", "all"):
        report(cfg, pairs_root, out, dirs["train"])
    log(f"bi_encoder {args.stage} done in {time.perf_counter() - started:,.0f}s")


if __name__ == "__main__":
    main()
