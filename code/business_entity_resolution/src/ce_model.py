"""Cross-encoder reranker: fine-tune a small multilingual transformer on (S1, candidate) text pairs and score every
top-K candidate row (Kaggle GPU notebook; ``torchrun`` for several GPUs).

Input is the ``topk_export`` contract: ``<pairs_root>/<split>/part-NNNNN.parquet`` rows (s1_id, t_id, label,
filter_score, filter_rank; grouped by S1, filter-rank order) and ``s1.parquet``. The model reads one pair per row:
the S1 record text first, the candidate second, each ``name | address`` from the challenge TSVs with whitespace
collapsed (case, accents and scripts kept). S1 text comes from source 1, candidate text from sources 2 and 3 of the
same corpus (training TSVs for train/validation/holdout, test TSVs for test); an id missing from them is an error.

Stages (``--stage all`` runs train, score, evaluate):

* ``train``: a seeded sample of ``train_s1`` S1 from the ``train`` split only (validation and holdout are never
  seen). Per S1: every positive, the ``neg_top`` best filter-ranked negatives (the hard ones a reranker must
  separate) and ``neg_random`` random other negatives, shuffled. Plain PyTorch: AdamW, linear warmup/decay, fp16
  autocast with a gradient scaler on CUDA, binary cross-entropy on one logit. ``accumulate`` micro-batches make one
  optimizer step (DDP syncs gradients only then, which PCIe-linked GPUs need for a large model). After
  ``FIT_SCHEDULE_STEPS`` steps the linear schedule is shortened to the steps that fit the time budget at the measured
  rate, so the learning rate still decays to zero when throughput is lower than planned. Batches hold pairs of similar length
  (in shuffled order) to cut padding; ``freeze_word_embeddings`` keeps the token embedding table fixed (96M of
  multilingual-e5-small's 118M parameters, which DDP would otherwise all-reduce over PCIe every step). A
  wall-clock budget (``train_minutes``, counted from the start of the stage) stops every rank at the same step;
  the model is saved either way.
* ``score``: one float32 logit per pairs row, in concatenated part order (``<out>/scores/<split>.npy``). Each rank
  takes a contiguous row slice, scores it in descending text-length order (little padding, long batches first so
  memory problems show at once) and writes it back in row order. Pair ids become integer text positions per
  parquet part, so memory is the text lookups plus a few bytes per row.
  Tokenisation runs in ``tokenizer_threads()`` background threads (one tokenizer copy each) so a fast GPU is not
  starved. Test can be scored in ``CE_TEST_SHARDS`` contiguous row shards (``<split>.shardIofN.npy``, each saved as it
  finishes, merged into ``test.npy`` once all exist): ``CE_TEST_ONLY=i,j`` scores only those shards (another machine
  does the rest), finished shards are skipped on a rerun, and no shard starts that would pass ``CE_DEADLINE`` (epoch
  seconds) at the measured rate, so a paid GPU never loses finished work to a stopped machine.
* ``evaluate``: AUC and top-1 hit rate of the logits on validation/holdout, next to the filter score's.

Multi-GPU: ``torchrun --nproc_per_node=N -m src.ce_model ...`` gives DDP over NCCL (WORLD_SIZE/RANK/LOCAL_RANK);
plain ``python -m src.ce_model`` runs one process on one GPU or the CPU. Only rank 0 writes the model and logs. A
rank that fails exits at once without NCCL teardown, so torchrun stops the group instead of the others waiting on it.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import contextlib
import copy
import csv
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np

from .embed_names import resolve_model
from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

SCORED_SPLITS = ("validation", "holdout", "test")
PREFETCH = 8
SCORE_LOG_SECONDS = 60
BUCKET_BATCHES = 50                      # training batches sorted by length together (then shuffled)
STOP_CHECK_STEPS = 10                    # ranks agree on the time budget every this many steps
FIT_SCHEDULE_STEPS = 100                 # optimizer steps measured before the schedule is fitted to the budget
GROUP_TIMEOUT = timedelta(hours=6)       # a barrier may wait for the slowest rank's whole scoring slice
MAX_TOKENIZER_THREADS = 4                # scoring: the HF tokenizer call is ~9k pairs/s per thread


def log0(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        log(message)


# ---------------------------------------------------------------------------
# Processes


class Dist:
    """One process, or one torchrun worker of a DDP group (NCCL on GPUs, gloo on CPU)."""

    def __init__(self):
        import torch
        import torch.distributed

        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.set_device(self.local)
        self.device = torch.device(f"cuda:{self.local}" if self.cuda else "cpu")
        self.group = torch.distributed if self.world > 1 else None
        if self.group is not None:
            self.group.init_process_group("nccl" if self.cuda else "gloo", timeout=GROUP_TIMEOUT)

    def barrier(self) -> None:
        if self.group is not None:
            self.group.barrier(**({"device_ids": [self.local]} if self.cuda else {}))

    def any(self, flag: bool) -> bool:
        if self.group is None:
            return flag
        import torch

        value = torch.tensor([int(flag)], device=self.device)
        self.group.all_reduce(value, op=self.group.ReduceOp.MAX)
        return bool(value.item())

    def share(self, value):
        """Rank 0's ``value`` on every rank."""
        if self.group is None:
            return value
        box = [value]
        self.group.broadcast_object_list(box, src=0)
        return box[0]

    def close(self) -> None:
        if self.group is not None and self.group.is_initialized():
            self.group.destroy_process_group()


# ---------------------------------------------------------------------------
# Record text


def record_text(name: str, address: str) -> str:
    name, address = " ".join(name.split()), " ".join(address.split())
    return f"{name} | {address}" if address else name


class Texts:
    """Record id -> model text for one side of a corpus (S1, or the S2+S3 targets)."""

    def __init__(self, paths: list[Path], keep: set | None = None):
        import pandas as pd

        ids, texts = [], []
        for path in paths:
            with path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    if keep is None or row["entity_id"] in keep:
                        ids.append(row["entity_id"])
                        texts.append(record_text(row["business_name"] or "", row["business_address"] or ""))
        self.index = pd.Index(ids, dtype=object)
        if not self.index.is_unique:
            raise ValueError(f"duplicate entity ids in {[p.name for p in paths]}")
        self.texts = np.asarray(texts, dtype=object)
        self.lengths = np.fromiter(map(len, texts), np.int32, len(texts))

    def __len__(self) -> int:
        return len(self.texts)

    def positions(self, column, what: str) -> np.ndarray:
        """int32 positions of an Arrow string column's ids (dictionary-encoded, so repeats cost nothing)."""
        column = column.combine_chunks() if hasattr(column, "combine_chunks") else column
        if column.null_count:
            raise ValueError(f"{what}: {column.null_count:,} null ids")
        encoded = column.dictionary_encode()
        values = encoded.dictionary.to_numpy(zero_copy_only=False)
        where = self.index.get_indexer(values)
        if (where < 0).any():
            raise KeyError(f"{what}: {int((where < 0).sum()):,} ids are not in the source files, "
                           f"e.g. {values[where < 0][:5].tolist()}")
        return where.astype(np.int32)[encoded.indices.to_numpy(zero_copy_only=False)]


def corpus_kind(split: str) -> str:
    return "test" if split == "test" else "train"


def load_corpus(kind: str, directory: Path, keep_s1: set | None = None,
                keep_t: set | None = None) -> tuple[Texts, Texts]:
    started = time.perf_counter()
    s1 = Texts([directory / f"{kind}_source1.tsv"], keep_s1)
    targets = Texts([directory / f"{kind}_source{k}.tsv" for k in (2, 3)], keep_t)
    log0(f"texts: {kind} corpus {len(s1):,} S1 and {len(targets):,} targets in {time.perf_counter() - started:,.0f}s")
    return s1, targets


# ---------------------------------------------------------------------------
# Pairs files


def part_paths(split_dir: Path) -> list[Path]:
    paths = sorted(split_dir.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{split_dir}: no part-*.parquet files")
    return paths


def part_rows(paths: list[Path]) -> np.ndarray:
    import pyarrow.parquet as pq

    return np.asarray([pq.ParquetFile(p).metadata.num_rows for p in paths], np.int64)


def read_part(path: Path, columns: list[str]):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read(columns=columns)


def read_slice(paths: list[Path], counts: np.ndarray, start: int, stop: int, columns: list[str]):
    """Rows ``[start, stop)`` of the concatenated parts, one Arrow table per overlapping part."""
    offsets = np.concatenate([[0], np.cumsum(counts)])
    for path, low, high in zip(paths, offsets[:-1], offsets[1:]):
        if high <= start or low >= stop:
            continue
        begin, end = max(start, low) - low, min(stop, high) - low
        yield read_part(path, columns).slice(begin, end - begin)


def read_split(split_dir: Path, columns: list[str]):
    import pyarrow as pa

    return pa.concat_tables([read_part(p, columns) for p in part_paths(split_dir)])


def to_numpy(column) -> np.ndarray:
    """An Arrow Array or ChunkedArray as numpy."""
    column = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    return column.to_numpy(zero_copy_only=False)


def rank_bounds(total: int, world: int, rank: int) -> tuple[int, int]:
    return total * rank // world, total * (rank + 1) // world


def run_starts(codes: np.ndarray) -> np.ndarray:
    """First row of each run of equal values."""
    if not len(codes):
        return np.zeros(0, np.int64)
    return np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])


def within_group(sorted_groups: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... inside each run of equal values."""
    starts = run_starts(sorted_groups)
    return np.arange(len(sorted_groups)) - np.repeat(starts, np.diff(np.r_[starts, len(sorted_groups)]))


# ---------------------------------------------------------------------------
# Training pairs


def select_rows(group: np.ndarray, label: np.ndarray, rank: np.ndarray, neg_top: int, neg_random: int,
                rng: np.random.Generator) -> np.ndarray:
    """Every positive, the ``neg_top`` best-ranked negatives and ``neg_random`` random other negatives per group."""
    order = np.lexsort((rank, group))
    negatives = order[label[order] == 0]
    ordinal = within_group(group[negatives])
    top, rest = negatives[ordinal < neg_top], negatives[ordinal >= neg_top]
    rest = rest[np.lexsort((rng.random(len(rest)), group[rest]))]
    extra = rest[within_group(group[rest]) < neg_random]
    return np.sort(np.concatenate([np.flatnonzero(label == 1), top, extra]))


def training_pairs(cfg: dict, split_dir: Path, train_dir: Path) -> dict:
    """Identical on every rank (seeded): sampled S1, their selected rows shuffled, and text positions."""
    import pyarrow as pa
    import pyarrow.compute as pc

    rng = np.random.default_rng(cfg["seed"])
    s1 = read_part(split_dir / "s1.parquet", ["s1_id", "n_cand"])
    eligible = to_numpy(s1.column("s1_id"))[to_numpy(s1.column("n_cand")) > 0]
    take = np.sort(rng.choice(len(eligible), size=min(cfg["train_s1"], len(eligible)), replace=False))
    chosen = pa.array(eligible[take].tolist(), pa.string())
    tables = []
    for path in part_paths(split_dir):
        table = read_part(path, ["s1_id", "t_id", "label", "filter_rank"])
        tables.append(table.filter(pc.is_in(table.column("s1_id"), value_set=chosen)))
    rows = pa.concat_tables(tables)
    label = to_numpy(rows.column("label")).astype(np.int8)
    if len(label) == 0 or not np.isin(label, (0, 1)).all():
        raise ValueError(f"{split_dir}: training rows need labels 0/1 ({len(label):,} rows)")
    group = to_numpy(rows.column("s1_id").combine_chunks().dictionary_encode().indices)
    picked = select_rows(group, label, to_numpy(rows.column("filter_rank")), cfg["neg_top"], cfg["neg_random"], rng)
    shuffle = rng.permutation(len(picked))
    if cfg.get("max_train_pairs"):
        shuffle = shuffle[:cfg["max_train_pairs"]]
    rows = rows.take(pa.array(picked[shuffle]))
    s1_texts, t_texts = load_corpus("train", train_dir, set(rows.column("s1_id").to_pylist()),
                                    set(rows.column("t_id").to_pylist()))
    data = {"a": s1_texts.texts, "b": t_texts.texts, "ia": s1_texts.positions(rows.column("s1_id"), "train s1_id"),
            "ib": t_texts.positions(rows.column("t_id"), "train t_id"), "label": to_numpy(rows.column("label")) == 1,
            "s1": len(chosen)}
    data["length"] = s1_texts.lengths[data["ia"]] + t_texts.lengths[data["ib"]]
    log0(f"train: {data['s1']:,} S1 sampled, {len(data['label']):,} pairs, {int(data['label'].sum()):,} positives")
    return data


# ---------------------------------------------------------------------------
# Model and batches


def quiet_transformers(rank: int) -> None:
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        if rank:
            hf_logging.set_verbosity_error()
    except (ImportError, AttributeError):
        pass


def load_model(source: str, revision: str | None, device, *, inference: bool = False):
    """Tokenizer + model; SDPA attention when the installed transformers supports it for this architecture.
    For inference on CUDA the weights are cast to fp16 once (no per-batch autocast casts)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1, attn_implementation="sdpa",
                                                                   **kwargs)
    except (TypeError, ValueError, ImportError):
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1, **kwargs)
    model = model.half() if inference and device.type == "cuda" else model.float()
    return tokenizer, model.to(device)


def encode(tokenizer, a: np.ndarray, b: np.ndarray, max_length: int, pin: bool) -> dict:
    batch = tokenizer(a.tolist(), b.tolist(), truncation="longest_first", max_length=max_length, padding=True,
                      return_tensors="pt")
    return {k: (v.pin_memory() if pin else v) for k, v in batch.items()}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast(device):
    import torch

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def grad_scaler(enabled: bool):
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def tokenizer_threads() -> int:
    return max(1, min(MAX_TOKENIZER_THREADS, (os.cpu_count() or 2) // 2))


def prefetch(*sources, depth: int = PREFETCH):
    """Run batch generators in background threads (one each) so tokenisation overlaps the GPU. With several sources the
    item order across them is not kept (scoring batches carry their own row ids)."""
    box: queue.Queue = queue.Queue(maxsize=depth * len(sources))
    stop, done = threading.Event(), object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                box.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def work(items) -> None:
        try:
            for item in items:
                if not put(item):
                    return
            put(done)
        except BaseException as error:  # noqa: BLE001  (re-raised in the consumer)
            put(error)

    threads = [threading.Thread(target=work, args=(items,), daemon=True) for items in sources]
    for thread in threads:
        thread.start()
    remaining = len(threads)
    try:
        while remaining:
            item = box.get()
            if item is done:
                remaining -= 1
                continue
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=60)                               # no tokenizer call outlives the loop


def param_groups(model, weight_decay: float) -> list[dict]:
    decay, plain = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (plain if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]


def environment() -> dict:
    info = {"python": sys.version.split()[0]}
    for module in ("torch", "transformers", "tokenizers", "numpy", "pyarrow"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    import torch

    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


# ---------------------------------------------------------------------------
# Train


def length_batches(rows: np.ndarray, lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list:
    """Batches of similar-length pairs (little padding): sort windows of ``BUCKET_BATCHES`` batches by length,
    cut, and shuffle the batch order. The batch count is ``ceil(len(rows) / batch_size)`` whatever the order."""
    window = batch_size * BUCKET_BATCHES
    ordered = np.concatenate([w[np.argsort(lengths[w], kind="stable")]
                              for w in np.split(rows, range(window, len(rows), window))])
    batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def train_batches(tokenizer, data: dict, batches: list, max_length: int, pin: bool):
    import torch

    for chunk in batches:
        target = torch.from_numpy(data["label"][chunk].astype(np.float32))
        yield (encode(tokenizer, data["a"][data["ia"][chunk]], data["b"][data["ib"][chunk]], max_length, pin),
               target.pin_memory() if pin else target)


def backbone_spec(backbone: str, allowed: list[str]) -> dict:
    """A Hub model (license and revision checked) or a local fine-tuned model directory to warm-start from; the latter
    inherits the license recorded in its train_log.json (next to it or one level up), which must also be allowed."""
    path = Path(backbone)
    if not path.is_dir():
        return resolve_model({"repo": backbone}, allowed)
    logs = [p for p in (path / "train_log.json", path.parent / "train_log.json") if p.is_file()]
    if not logs:
        raise FileNotFoundError(f"{path}: a local backbone needs its train_log.json (license and origin)")
    parent = json.loads(logs[0].read_text())
    if parent.get("license") not in allowed:
        raise RuntimeError(f"{path}: license {parent.get('license')!r} is not one of {allowed}")
    return {"repo": str(path), "license": parent["license"], "revision": None,
            "warm_start_from": parent.get("backbone"), "warm_start_revision": parent.get("revision")}


def train(cfg: dict, pairs_root: Path, out: Path, train_dir: Path, backbone: str, dist: Dist) -> dict:
    import torch

    started = time.perf_counter()
    spec = dist.share(backbone_spec(backbone, cfg["allowed_licenses"]) if dist.rank == 0 else None)
    log0(f"train: backbone {spec['repo']} @ {spec.get('revision')} (license {spec['license']}), "
         f"{dist.world} process(es)")
    data = training_pairs(cfg, pairs_root / "train", train_dir)
    torch.manual_seed(cfg["seed"])
    if dist.rank == 0:
        tokenizer, model = load_model(spec["repo"], spec.get("revision"), dist.device)
    dist.barrier()
    if dist.rank != 0:
        tokenizer, model = load_model(spec["repo"], spec.get("revision"), dist.device)
    if cfg.get("freeze_word_embeddings"):
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log0(f"train: {trainable:,} trainable parameters of {sum(p.numel() for p in model.parameters()):,}")
    net = model
    if dist.world > 1:
        from torch.nn.parallel import DistributedDataParallel

        net = DistributedDataParallel(model, device_ids=[dist.local] if dist.cuda else None,
                                      gradient_as_bucket_view=True)
    per_rank = len(data["label"]) // dist.world
    if per_rank == 0:
        raise ValueError("fewer training pairs than processes")
    mine = np.arange(dist.rank, per_rank * dist.world, dist.world)
    batch_size, accumulate = cfg["batch_size"], max(1, int(cfg.get("accumulate", 1)))
    total_steps = -(-(-(-per_rank // batch_size)) // accumulate) * cfg["epochs"]
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    plan = {"total": total_steps, "warmup": int(cfg["warmup"] * total_steps)}

    def lr_factor(current: int) -> float:                     # linear warmup and decay, as transformers' schedule
        if current < plan["warmup"]:
            return current / max(1, plan["warmup"])
        return max(0.0, (plan["total"] - current) / max(1, plan["total"] - plan["warmup"]))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler, loss_fn = grad_scaler(dist.cuda), torch.nn.BCEWithLogitsLoss()
    budget = cfg["train_minutes"] * 60
    step, seen, stopped, fitted_at = 0, 0, False, None
    window_loss, window_steps, last_loss = torch.zeros((), device=dist.device), 0, float("nan")
    loop_started = time.perf_counter()
    net.train()
    for epoch in range(cfg["epochs"]):
        rng = np.random.default_rng([cfg["seed"], epoch, dist.rank])
        batches = length_batches(mine if epoch == 0 else rng.permutation(mine), data["length"], batch_size, rng)
        for index, (inputs, target) in enumerate(prefetch(train_batches(tokenizer, data, batches, cfg["max_length"],
                                                                        dist.cuda))):
            if (index % accumulate == 0 and step % STOP_CHECK_STEPS == 0
                    and dist.any(time.perf_counter() - started > budget)):
                stopped = True
                break
            sync = (index + 1) % accumulate == 0 or index + 1 == len(batches)
            inputs, target = to_device(inputs, dist.device), target.to(dist.device, non_blocking=True)
            with net.no_sync() if dist.world > 1 and not sync else contextlib.nullcontext():
                with autocast(dist.device):
                    logits = net(**inputs).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target)
                scaler.scale(loss / accumulate).backward()
            seen += len(target) * dist.world
            window_loss += loss.detach()
            window_steps += 1
            if not sync:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            if step == FIT_SCHEDULE_STEPS and fitted_at is None:
                per_step = (time.perf_counter() - loop_started) / step
                fits = step + int((budget - (time.perf_counter() - started)) / per_step * 0.97)
                plan["total"] = dist.share(max(step + 1, min(plan["total"], fits)) if dist.rank == 0 else None)
                fitted_at = step
                log0(f"train: {per_step:.3f}s per step; schedule {plan['total']:,} steps (planned {total_steps:,})")
            if step % cfg["log_every"] == 0 or step == plan["total"]:
                last_loss = float(window_loss.item()) / window_steps
                window_loss.zero_()
                window_steps = 0
                rate = seen / max(time.perf_counter() - loop_started, 1e-9)
                log0(f"train: step {step:,}/{plan['total']:,} loss {last_loss:.4f} {rate:,.0f} pairs/s "
                     f"lr {schedule.get_last_lr()[0]:.2e}")
            if step >= plan["total"]:
                break
        if stopped or step >= plan["total"]:
            break
    if window_steps:
        last_loss = float(window_loss.item()) / window_steps
    loop_seconds = time.perf_counter() - loop_started
    if stopped:
        log0(f"train: time budget of {cfg['train_minutes']} min reached at step {step:,}/{total_steps:,}")
    dist.barrier()
    info = {}
    if dist.rank == 0:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
        info = {"pairs": int(len(data["label"])), "positives": int(data["label"].sum()), "s1": data["s1"],
                "pairs_seen": seen, "steps": step, "planned_steps": total_steps, "fitted_steps": plan["total"],
                "accumulate": accumulate,
                "seconds": time.perf_counter() - started,
                "train_loop_seconds": loop_seconds, "throughput": seen / max(loop_seconds, 1e-9),
                "stopped_early": stopped, "final_window_loss": last_loss, "backbone": spec["repo"],
                "revision": spec.get("revision"), "license": spec["license"], "world_size": dist.world,
                "warm_start_from": spec.get("warm_start_from"), "warm_start_revision": spec.get("warm_start_revision"),
                "trainable_parameters": trainable,
                "config": cfg, "environment": environment()}
        atomic_write_json(out / "train_log.json", info)
        log0(f"train: saved {out / 'model'} after {step:,} steps, {seen:,} pairs, {info['seconds']:,.0f}s")
    dist.barrier()
    return info


# ---------------------------------------------------------------------------
# Score


def score_batches(tokenizer, a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunks: list,
                  max_length: int, pin: bool):
    for rows in chunks:
        yield rows, encode(tokenizer, a[ia[rows]], b[ib[rows]], max_length, pin)


def score_rows(model, tokenizer, device, s1_texts: Texts, t_texts: Texts, ia: np.ndarray, ib: np.ndarray,
               batch_size: int, max_length: int, label: str = "") -> np.ndarray:
    """Logits in row order; batches run longest-first so similar lengths share padding."""
    import torch

    order = np.argsort(-(s1_texts.lengths[ia] + t_texts.lengths[ib]), kind="stable")
    out = np.full(len(order), np.nan, np.float32)
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    threads = max(1, min(tokenizer_threads(), len(chunks)))
    streams = [score_batches(tokenizer if k == 0 else copy.deepcopy(tokenizer), s1_texts.texts, t_texts.texts, ia, ib,
                             chunks[k::threads], max_length, device.type == "cuda") for k in range(threads)]
    started = last = time.perf_counter()
    done = 0
    model.eval()
    with torch.inference_mode():
        for rows, inputs in prefetch(*streams):
            with autocast(device):
                logits = model(**to_device(inputs, device)).logits
            out[rows] = logits.float().squeeze(-1).cpu().numpy()
            done += len(rows)
            if time.perf_counter() - last > SCORE_LOG_SECONDS:
                last = time.perf_counter()
                rate = done / (last - started)
                log0(f"score {label}: {done:,}/{len(order):,} rows on rank 0, {rate:,.0f} rows/s, "
                     f"ETA {(len(order) - done) / max(rate, 1e-9) / 60:,.1f} min")
    if not np.isfinite(out).all():
        raise RuntimeError(f"score {label}: {int((~np.isfinite(out)).sum()):,} non-finite logits")
    return out


def score_split(model, tokenizer, device, split_dir: Path, s1_texts: Texts, t_texts: Texts, rank: int, world: int,
                batch_size: int, max_length: int, max_rank: int | None = None,
                row_range: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    """This rank's contiguous slice of the split's rows (of ``row_range`` if given), scored; and the split's total row
    count. With ``max_rank`` only rows with filter_rank < max_rank are scored; the others are NaN (not scored)."""
    paths = part_paths(split_dir)
    counts = part_rows(paths)
    total = int(counts.sum())
    low, high = row_range if row_range is not None else (0, total)
    start, stop = rank_bounds(high - low, world, rank)
    start, stop = start + low, stop + low
    ia, ib, fr = [np.zeros(0, np.int32)], [np.zeros(0, np.int32)], [np.zeros(0, np.int16)]
    for table in read_slice(paths, counts, start, stop, ["s1_id", "t_id", "filter_rank"]):
        ia.append(s1_texts.positions(table.column("s1_id"), f"{split_dir.name} s1_id"))
        ib.append(t_texts.positions(table.column("t_id"), f"{split_dir.name} t_id"))
        fr.append(to_numpy(table.column("filter_rank")).astype(np.int16))
    ia, ib, fr = np.concatenate(ia), np.concatenate(ib), np.concatenate(fr)
    if len(ia) != stop - start:
        raise RuntimeError(f"{split_dir.name}: read {len(ia):,} rows for slice [{start:,}, {stop:,})")
    logits = np.full(len(ia), np.nan, np.float32)
    keep = np.ones(len(ia), bool) if max_rank is None else fr < max_rank
    logits[keep] = score_rows(model, tokenizer, device, s1_texts, t_texts, ia[keep], ib[keep], batch_size, max_length,
                              split_dir.name)
    return logits, total


def choose_test_k(cfg: dict, split_dir: Path, rows_per_s: float) -> dict:
    """Largest K in [min_k, max_k] whose test rows (filter_rank < K) fit ``test_budget_minutes`` at the measured rate."""
    ranks = np.concatenate([to_numpy(read_part(path, ["filter_rank"]).column("filter_rank")).astype(np.int64)
                            for path in part_paths(split_dir)])
    cumulative = np.cumsum(np.bincount(ranks, minlength=cfg["max_k"] + 1))
    budget_rows = rows_per_s * cfg["test_budget_minutes"] * 60
    k = cfg["min_k"]
    for candidate in range(cfg["min_k"], cfg["max_k"] + 1):
        if cumulative[candidate - 1] <= budget_rows:
            k = candidate
    return {"k": int(k), "rows_scored": int(cumulative[k - 1]), "rows_total": int(len(ranks)),
            "rows_per_s_measured": float(rows_per_s), "budget_minutes": cfg["test_budget_minutes"],
            "expected_minutes": float(cumulative[k - 1] / max(rows_per_s, 1e-9) / 60)}


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def shard_path(scores_dir: Path, split: str, index: int, count: int) -> Path:
    return scores_dir / f"{split}.shard{index}of{count}.npy"


def merge_shards(scores_dir: Path, split: str, count: int) -> bool:
    """Write ``<split>.npy`` from the ``count`` shard files in row order, if every shard exists."""
    paths = [shard_path(scores_dir, split, i, count) for i in range(count)]
    if not all(p.exists() for p in paths):
        return False
    save_npy(scores_dir / f"{split}.npy", np.concatenate([np.load(p) for p in paths]).astype(np.float32))
    return True


def shard_settings() -> tuple[int, set | None, float]:
    """(CE_TEST_SHARDS, CE_TEST_ONLY as a set or None, CE_DEADLINE in epoch seconds)."""
    only = os.environ.get("CE_TEST_ONLY", "").strip()
    return (int(os.environ.get("CE_TEST_SHARDS", "1")), {int(x) for x in only.split(",")} if only else None,
            float(os.environ.get("CE_DEADLINE", "inf")))


def previous_rate(out: Path) -> float | None:
    path = out / "score_log.json"
    if not path.exists():
        return None
    rates = [v["rows_per_s"] for v in json.loads(path.read_text()).values() if isinstance(v, dict) and v.get("rows_per_s")]
    return min(rates) if rates else None


def score_shards(model, tokenizer, dist: Dist, split_dir: Path, texts: tuple, scores_dir: Path, split: str, cfg: dict,
                 max_rank: int | None, rate: float | None) -> dict:
    """Score the split's shards one by one (each saved when done); stop before a shard that would pass the deadline."""
    count, only, deadline = shard_settings()
    total = int(part_rows(part_paths(split_dir)).sum())
    report = {}
    for index in range(count):
        path = shard_path(scores_dir, split, index, count)
        if (only is not None and index not in only) or path.exists():
            continue
        low, high = rank_bounds(total, count, index)
        need = (high - low) / rate if rate else 0.0
        if not dist.share(time.time() + need <= deadline if dist.rank == 0 else None):
            log0(f"score {split}: shard {index}/{count} not started, ~{need / 60:.1f} min would pass the deadline")
            break
        started = time.perf_counter()
        logits, _ = score_split(model, tokenizer, dist.device, split_dir, *texts, dist.rank, dist.world,
                                cfg["score_batch_size"], cfg["max_length"], max_rank, (low, high))
        save_npy(scores_dir / f"{path.stem}.part{dist.rank}.npy", logits)
        dist.barrier()
        if dist.rank == 0:
            parts = [scores_dir / f"{path.stem}.part{r}.npy" for r in range(dist.world)]
            merged = np.concatenate([np.load(p) for p in parts])
            if len(merged) != high - low:
                raise RuntimeError(f"{path.stem}: merged {len(merged):,} scores for {high - low:,} rows")
            save_npy(path, merged.astype(np.float32))
            for part in parts:
                part.unlink()
            seconds = time.perf_counter() - started
            rate = (high - low) / max(seconds, 1e-9)
            report[path.stem] = {"rows": high - low, "first_row": low, "seconds": seconds, "rows_per_s": rate,
                                 "world_size": dist.world, "batch_size": cfg["score_batch_size"]}
            log0(f"score {path.stem}: rows [{low:,}, {high:,}) in {seconds:,.0f}s ({rate:,.0f} rows/s)")
        rate = dist.share(rate if dist.rank == 0 else None)
        dist.barrier()
    if dist.rank == 0 and merge_shards(scores_dir, split, count):
        report[split] = {"rows": total, "shards": count}
        log0(f"score {split}: all {count} shards present -> {scores_dir / f'{split}.npy'}")
    return report


def score(cfg: dict, pairs_root: Path, out: Path, dirs: dict, splits: list[str], dist: Dist) -> dict:
    tokenizer, model = load_model(str(out / "model"), None, dist.device, inference=True)
    scores_dir, report = out / "scores", {}
    k_path = out / "test_k.json"
    cached: dict[str, tuple[Texts, Texts]] = {}
    for split in splits:
        kind = corpus_kind(split)
        if kind not in cached:
            cached.clear()
            cached[kind] = load_corpus(kind, dirs[kind])
        started = time.perf_counter()
        max_rank = None
        if split == "test" and cfg.get("test_budget_minutes"):
            if dist.rank == 0:
                rate = next((report[s]["rows_per_s"] for s in ("validation", "holdout") if s in report), None)
                choice = (choose_test_k(cfg, pairs_root / split, rate) if rate else
                          {"k": cfg["max_k"], "note": "no measured rate; scoring every row"})
                atomic_write_json(k_path, choice)
                log0(f"score test: top-{choice['k']} per S1 by the filter fits the budget: {choice}")
            dist.barrier()
            max_rank = json.loads(k_path.read_text())["k"]
        if split == "test" and shard_settings()[0] > 1:
            rate = next((report[s]["rows_per_s"] for s in ("validation", "holdout") if s in report), None)
            report |= score_shards(model, tokenizer, dist, pairs_root / split, cached[kind], scores_dir, split, cfg,
                                   max_rank, rate or previous_rate(out))
            continue
        logits, total = score_split(model, tokenizer, dist.device, pairs_root / split, *cached[kind], dist.rank,
                                    dist.world, cfg["score_batch_size"], cfg["max_length"], max_rank)
        save_npy(scores_dir / f"{split}.part{dist.rank}.npy", logits)
        dist.barrier()
        if dist.rank == 0:
            merged = np.concatenate([np.load(scores_dir / f"{split}.part{r}.npy") for r in range(dist.world)])
            if len(merged) != total:
                raise RuntimeError(f"{split}: merged {len(merged):,} scores for {total:,} rows")
            save_npy(scores_dir / f"{split}.npy", merged.astype(np.float32))
            seconds = time.perf_counter() - started
            report[split] = {"rows": total, "seconds": seconds, "rows_per_s": total / max(seconds, 1e-9),
                             "world_size": dist.world, "batch_size": cfg["score_batch_size"]}
            log0(f"score {split}: {total:,} rows in {seconds:,.0f}s ({report[split]['rows_per_s']:,.0f} rows/s, "
                 f"{dist.world} process(es)) -> {scores_dir / f'{split}.npy'}")
        dist.barrier()
    if dist.rank == 0:
        path = out / "score_log.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        atomic_write_json(path, previous | report)
    return report


# ---------------------------------------------------------------------------
# Evaluate


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if labels.all() or not labels.any():
        return None
    return float(roc_auc_score(labels, scores))


def top1_hit(starts: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Share of S1 with a true candidate whose best-scored candidate is true (rows grouped by S1)."""
    if not len(starts):
        return None
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(labels)]))
    best = np.lexsort((-scores, group))[starts]
    has = np.maximum.reduceat(labels.astype(np.int8), starts) > 0
    return float(labels[best][has].mean()) if has.any() else None


def evaluate(out: Path, pairs_root: Path, splits: list[str]) -> dict:
    report = {}
    for split in splits:
        path = out / "scores" / f"{split}.npy"
        if not path.exists():
            continue
        table = read_split(pairs_root / split, ["s1_id", "label", "filter_score"])
        label = to_numpy(table.column("label"))
        if (label < 0).any():
            continue
        logits, filter_score, labels = np.load(path), to_numpy(table.column("filter_score")), label == 1
        if len(logits) != len(labels):
            raise RuntimeError(f"{split}: {len(logits):,} scores for {len(labels):,} rows")
        starts = run_starts(to_numpy(table.column("s1_id").combine_chunks().dictionary_encode().indices))
        report[split] = {"rows": int(len(labels)), "positives": int(labels.sum()), "s1_with_rows": int(len(starts)),
                         "auc": auc(labels, logits), "auc_filter_score": auc(labels, filter_score),
                         "top1_hit": top1_hit(starts, labels, logits),
                         "top1_hit_filter_score": top1_hit(starts, labels, filter_score)}
        r = report[split]
        log0(f"evaluate {split}: AUC {r['auc']} (filter {r['auc_filter_score']}), top-1 hit {r['top1_hit']} "
             f"(filter {r['top1_hit_filter_score']})")
    atomic_write_json(out / "eval.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "student_resource/dataset/train")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "student_resource/dataset/test")
    parser.add_argument("--stage", choices=("train", "score", "evaluate", "all"), required=True)
    parser.add_argument("--splits", nargs="+", help="splits to score/evaluate (default: validation, holdout, test "
                                                    "that exist under --pairs-root)")
    parser.add_argument("--backbone", help="Hub repo id overriding the config (license-checked the same way)")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.resolve().read_text())
    pairs_root, out = args.pairs_root.resolve(), args.out.resolve()
    dirs = {"train": args.train_dir.resolve(), "test": args.test_dir.resolve()}
    splits = args.splits or [s for s in SCORED_SPLITS if (pairs_root / s).is_dir()]
    missing = [s for s in splits if not (pairs_root / s).is_dir()]
    if missing:
        raise SystemExit(f"--splits {missing}: not under {pairs_root}")
    dist = Dist()
    quiet_transformers(dist.rank)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if args.stage in ("train", "all"):
            train(cfg, pairs_root, out, dirs["train"], args.backbone or cfg["backbone"], dist)
        if args.stage in ("score", "all"):
            score(cfg, pairs_root, out, dirs, splits, dist)
        if args.stage in ("evaluate", "all") and dist.rank == 0:
            evaluate(out, pairs_root, splits)
        log0(f"ce_model {args.stage} done in {time.perf_counter() - started:,.0f}s")
    except BaseException:
        if dist.world > 1:
            # NCCL communicator teardown is an intra-node collective: a failed rank that tears down waits for the
            # others, which may block for GROUP_TIMEOUT. Exit now so torchrun stops the whole group.
            traceback.print_exc()
            log(f"rank {dist.rank}: failed; exiting without process-group teardown")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    dist.close()


if __name__ == "__main__":
    main()
