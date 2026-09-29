"""R3: multilingual name vectors for every train and test record (Kaggle GPU notebook).

Stages (``--stage all`` runs them in order):

* ``diagnose``: labelled training pairs from folds outside ``diagnostic.exclude_folds`` (fold 0 stays
  untouched for K2's validation and holdout). For each license-checked candidate model, measure how
  well name cosine separates true pairs from the main false-positive pattern (another business at
  the same address), alone and on top of string similarity. The model is chosen by the rule in the
  config.
* ``vocab``: every distinct business-name string of the six source files, plus each file's row
  mapping and row keys. Distinct strings are only a compute cache: every record keeps its own output
  row and no IDs are merged.
* ``pca``: embed a sample of distinct *training* names and fit an uncentered PCA (preserves dot
  products, hence cosines). Test names never influence it.
* ``encode``: embed every distinct name, project, L2-normalise, split across the visible GPUs.
* ``finalize``: one float16 array per source file in the file's row order, with row keys, the PCA,
  a manifest of hashes, and a short report.

Rows are parsed with ``csv.DictReader`` exactly like the K1/K2 stores. Keys: S2/S3 rows use
``blocking.encode_id``; S1 rows use the integer after ``S1-``.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .blocking import encode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .normalization import normalize_text
from .phase2a_analysis import script_of
from .phase2a_env import log

CODE_DIR = Path(__file__).resolve().parents[1]
LICENSE_TAG = "license:"


# ---------------------------------------------------------------------------
# Source files


def source_path(cfg: dict, root: Path, split: str, source: int) -> Path:
    return root / cfg["inputs"][f"{split}_dir"] / f"{split}_source{source}.tsv"


def file_label(split: str, source: int) -> str:
    return f"{split}_s{source}"


def record_key(entity_id: str) -> int:
    """S2/S3 rows: the pipeline's ``encode_id``. S1 rows: the integer after ``S1-``."""
    if entity_id.startswith("S1-"):
        return int(entity_id[3:])
    return encode_id(entity_id)


def name_text(raw: str) -> str:
    """Model input: the raw business name (case, accents, and scripts kept), whitespace collapsed."""
    return " ".join(raw.split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file, delimiter="\t")


# ---------------------------------------------------------------------------
# Models


def resolve_model(spec: dict, allowed: list[str]) -> dict:
    """Pin the exact Hub revision and check the model-card license; refuses anything not allowed."""
    from huggingface_hub import model_info

    info = model_info(spec["repo"])
    card = getattr(info, "card_data", None)
    license_id = getattr(card, "license", None) if card is not None else None
    if license_id is None and isinstance(card, dict):
        license_id = card.get("license")
    if not license_id:
        license_id = next((tag[len(LICENSE_TAG):] for tag in (info.tags or []) if tag.startswith(LICENSE_TAG)), "")
    license_id = str(license_id).lower()
    if license_id not in allowed:
        raise RuntimeError(f"{spec['repo']}: license {license_id!r} is not one of {allowed}")
    return spec | {"license": license_id, "revision": info.sha}


def load_encoder(spec: dict, device: str, enc: dict):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["repo"], device=device, revision=spec.get("revision"))
    model.max_seq_length = enc["max_seq_length"]
    if enc["fp16"] and device.startswith("cuda"):
        model.half()
    return model


def devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def release(model) -> None:
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def embed(model, texts: list[str], prefix: str, enc: dict) -> np.ndarray:
    """Normalised float32 embeddings of ``prefix + text`` (one chunk; the caller streams)."""
    return np.asarray(model.encode([prefix + text for text in texts], batch_size=enc["batch_size"],
                                   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# Diagnostic: which model separates true pairs from same-address other businesses?


def diagnostic_pairs(cfg: dict, root: Path) -> dict:
    """True pairs of sampled S1 (folds outside ``exclude_folds``) plus, for each true pair, one target at
    the same normalised address that is not a true match of that S1, when such a target exists."""
    d = cfg["diagnostic"]
    rng = np.random.default_rng(d["seed"])
    train = root / cfg["inputs"]["train_dir"]
    folds = {row["source1_entity_id"]: int(row["fold"]) for row in read_rows(root / cfg["inputs"]["folds"])}
    gold = {row["source1_entity_id"]: row["matched_entity_ids"].split(",")
            for row in read_rows(train / "train_ground_truth.tsv") if row["matched_entity_ids"]}
    excluded = set(d["exclude_folds"])
    eligible = sorted(s1 for s1 in gold if s1 in folds and folds[s1] not in excluded)
    take = rng.choice(len(eligible), size=min(d["s1_sample"], len(eligible)), replace=False)
    chosen = sorted(eligible[i] for i in take)
    chosen_set = set(chosen)
    s1_rec = {row["entity_id"]: (row["business_name"], row["country"])
              for row in read_rows(train / "train_source1.tsv") if row["entity_id"] in chosen_set}
    wanted = {t for s1 in chosen for t in gold[s1]}
    t_rec = {}
    for source in (2, 3):
        for row in read_rows(train / f"train_source{source}.tsv"):
            if row["entity_id"] in wanted:
                t_rec[row["entity_id"]] = (row["business_name"], normalize_text(row["business_address"]))
    keys = {key for _, key in t_rec.values() if key}
    groups: dict[str, list[tuple[str, str]]] = {}
    for source in (2, 3):
        for row in read_rows(train / f"train_source{source}.tsv"):
            if row["business_address"]:
                key = normalize_text(row["business_address"])
                if key in keys:
                    groups.setdefault(key, []).append((row["entity_id"], row["business_name"]))
    pairs: dict[str, list] = {"s1": [], "s1_name": [], "t_name": [], "label": [], "country": []}

    def add(index: int, s1_name: str, t_name: str, label: bool, country: str) -> None:
        for name, value in zip(pairs, (index, s1_name, t_name, label, country)):
            pairs[name].append(value)

    for index, s1 in enumerate(chosen):
        if s1 not in s1_rec:
            continue
        s1_name, country = s1_rec[s1]
        truth, used = set(gold[s1]), set()
        for target in gold[s1]:
            if target not in t_rec:
                continue
            t_name, key = t_rec[target]
            add(index, s1_name, t_name, True, country)
            pool = [x for x in groups.get(key, ()) if x[0] not in truth and x[0] not in used] if key else []
            if pool:
                negative = pool[int(rng.integers(len(pool)))]
                used.add(negative[0])
                add(index, s1_name, negative[1], False, country)
    out = {name: np.asarray(values) for name, values in pairs.items()}
    out["label"] = out["label"].astype(bool)
    out["s1_script"] = np.asarray([script_of(x) for x in out["s1_name"]])
    out["t_script"] = np.asarray([script_of(x) for x in out["t_name"]])
    return out


def string_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The K2 ``name_token_set`` definition: token-set ratio of normalised names, in [0, 1]."""
    from rapidfuzz import fuzz, process

    return np.asarray(process.cpdist([normalize_text(x) for x in a], [normalize_text(x) for x in b],
                                     scorer=fuzz.token_set_ratio, workers=-1), np.float64) / 100


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, bool)
    if labels.all() or not labels.any():
        return None
    return float(roc_auc_score(labels, scores))


def oof_auc(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float | None:
    """Out-of-fold AUC of a logistic regression, two folds split by S1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    labels = np.asarray(labels, bool)
    if labels.all() or not labels.any() or len(np.unique(groups)) < 2:
        return None
    scores = np.zeros(len(labels))
    for train, test in GroupKFold(n_splits=2).split(features, labels, groups):
        if labels[train].all() or not labels[train].any():
            return None
        model = LogisticRegression(max_iter=1000).fit(features[train], labels[train])
        scores[test] = model.decision_function(features[test])
    return auc(labels, scores)


def pair_metrics(pairs: dict, string_sim: np.ndarray, cosine: np.ndarray | None, weak: float) -> dict:
    labels, groups, negative = pairs["label"], pairs["s1"], ~pairs["label"]
    cross = pairs["label"] & (pairs["s1_script"] != pairs["t_script"])
    low = pairs["label"] & (string_sim < weak)
    if cosine is None:
        return {"auc_alone": auc(labels, string_sim),
                "auc_combined_oof": oof_auc(string_sim[:, None], labels, groups),
                "auc_cross_script_positives": auc(labels[cross | negative], string_sim[cross | negative])}
    both = np.column_stack([string_sim, cosine, string_sim * cosine])
    return {"auc_alone": auc(labels, cosine),
            "auc_combined_oof": oof_auc(both, labels, groups),
            "auc_cross_script_positives": auc(labels[cross | negative], cosine[cross | negative]),
            "auc_weak_string_positives": auc(labels[low | negative], cosine[low | negative]),
            "mean_cosine_true": float(cosine[labels].mean()), "mean_cosine_same_address": float(cosine[negative].mean())}


def select(results: list[dict], margin: float) -> dict:
    """Highest combined AUC; a cheaper model within ``margin`` of the best wins."""
    def value(result: dict) -> float:
        score = result["metrics"]["auc_combined_oof"]
        return -1.0 if score is None else score

    best = max(value(r) for r in results)
    near = [r for r in results if value(r) >= best - margin]
    return min(near, key=lambda r: (r["cost"], -value(r)))


def diagnose(cfg: dict, root: Path, output: Path) -> dict:
    path = output / "diagnostic.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    pairs = diagnostic_pairs(cfg, root)
    labels = pairs["label"]
    log(f"diagnose: {int(labels.sum()):,} true pairs, {int((~labels).sum()):,} same-address negatives, "
        f"{len(np.unique(pairs['s1'])):,} S1")
    string_sim = string_similarity(pairs["s1_name"], pairs["t_name"])
    weak = cfg["diagnostic"]["weak_string_threshold"]
    texts = sorted(set(pairs["s1_name"].tolist()) | set(pairs["t_name"].tolist()))
    where = {text: i for i, text in enumerate(texts)}
    left = np.asarray([where[x] for x in pairs["s1_name"]])
    right = np.asarray([where[x] for x in pairs["t_name"]])
    device = devices()[0]
    results = []
    for spec in cfg["candidates"]:
        resolved = resolve_model(spec, cfg["allowed_licenses"])
        t0 = time.perf_counter()
        model = load_encoder(resolved, device, cfg["encode"])
        vectors = embed(model, [name_text(t) for t in texts], resolved["prefix"], cfg["encode"])
        seconds = time.perf_counter() - t0
        release(model)
        cosine = np.einsum("ij,ij->i", vectors[left], vectors[right])
        metrics = pair_metrics(pairs, string_sim, cosine, weak)
        results.append(resolved | {"metrics": metrics, "texts": len(texts), "seconds_load_and_embed": seconds})
        log(f"diagnose: {spec['name']}: combined AUC {fmt(metrics['auc_combined_oof'])}, "
            f"cosine alone {fmt(metrics['auc_alone'])}, {len(texts) / seconds:,.0f} names/s incl. loading")
    chosen = select(results, cfg["diagnostic"]["speed_tie_margin"])
    baseline = pair_metrics(pairs, string_sim, None, weak)
    report = {
        "pairs": {"true": int(labels.sum()), "same_address_negative": int((~labels).sum()),
                  "s1": int(len(np.unique(pairs["s1"]))), "cross_script_true": int((labels & (pairs["s1_script"] != pairs["t_script"])).sum()),
                  "weak_string_true": int((labels & (string_sim < weak)).sum()),
                  "by_country": {str(c): int((pairs["country"] == c).sum()) for c in np.unique(pairs["country"])}},
        "string_baseline": baseline, "candidates": results, "selection_rule": cfg["diagnostic"]["selection"],
        "selected": {k: chosen[k] for k in ("name", "repo", "prefix", "revision", "license", "cost")},
        "gain_over_string_auc": (chosen["metrics"]["auc_combined_oof"] or 0) - (baseline["auc_combined_oof"] or 0),
        "seconds": time.perf_counter() - started}
    atomic_write_json(path, report)
    log(f"diagnose: selected {chosen['name']} (combined AUC {fmt(chosen['metrics']['auc_combined_oof'])} vs "
        f"string only {fmt(baseline['auc_combined_oof'])})")
    return report


# ---------------------------------------------------------------------------
# Vocabulary, PCA, encoding


def build_vocab(cfg: dict, root: Path, work: Path) -> dict:
    done = work / "vocab.json"
    if done.exists():
        return json.loads(done.read_text())
    work.mkdir(parents=True, exist_ok=True)
    index: dict[str, int] = {}
    files = {}
    for split, source in cfg["files"]:
        label = file_label(split, source)
        path = source_path(cfg, root, split, source)
        ids, keys = [], []
        for row in read_rows(path):
            text = name_text(row["business_name"])
            j = index.get(text)
            if j is None:
                j = index[text] = len(index)
            ids.append(j)
            keys.append(record_key(row["entity_id"]))
        keys_array = np.asarray(keys, np.uint64)
        if len(np.unique(keys_array)) != len(keys_array):
            raise ValueError(f"{path.name}: duplicate record keys")
        np.save(work / f"map_{label}.npy", np.asarray(ids, np.int32))
        np.save(work / f"keys_{label}.npy", keys_array)
        files[label] = {"rows": len(ids), "tsv": path.name, "tsv_sha256": sha256_file(path),
                        "keys_sha256": hashlib.sha256(keys_array.tobytes()).hexdigest()}
        log(f"vocab: {label} {len(ids):,} rows, {len(index):,} distinct names so far")
    with (work / "vocab.pkl").open("wb") as file:
        pickle.dump(list(index), file, protocol=pickle.HIGHEST_PROTOCOL)
    info = {"files": files, "distinct_names": len(index)}
    atomic_write_json(done, info)
    return info


def load_vocab(work: Path) -> list[str]:
    with (work / "vocab.pkl").open("rb") as file:
        return pickle.load(file)


def fit_pca(sample: np.ndarray, dims: int) -> tuple[np.ndarray, float]:
    """Top eigenvectors of the (uncentered) second-moment matrix: the best rank-``dims`` map for dot products."""
    x = np.asarray(sample, np.float64)
    moment = x.T @ x / max(len(x), 1)
    values, vectors = np.linalg.eigh(moment)
    order = np.argsort(values)[::-1][:dims]
    return vectors[:, order].astype(np.float32), float(values[order].sum() / values.sum())


def project(vectors: np.ndarray, components: np.ndarray) -> np.ndarray:
    reduced = np.asarray(vectors, np.float32) @ components
    return (reduced / np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12)).astype(np.float16)


def train_pca(cfg: dict, work: Path, spec: dict) -> dict:
    done = work / "pca.npz"
    if done.exists():
        with np.load(done) as data:
            return {"components": data["components"], "energy": float(data["energy"]), "rows": int(data["rows"])}
    texts = load_vocab(work)
    train_ids = np.unique(np.concatenate([np.load(work / f"map_{file_label(split, source)}.npy")
                                          for split, source in cfg["files"] if split == "train"]))
    rng = np.random.default_rng(cfg["pca"]["seed"])
    sample = np.sort(rng.choice(train_ids, size=min(cfg["pca"]["fit_sample"], len(train_ids)), replace=False))
    model = load_encoder(spec, devices()[0], cfg["encode"])
    chunk = cfg["encode"]["chunk_texts"]
    vectors = np.concatenate([embed(model, [texts[i] for i in sample[s:s + chunk]], spec["prefix"], cfg["encode"])
                              for s in range(0, len(sample), chunk)])
    release(model)
    dims = min(cfg["pca"]["dims"], vectors.shape[1])
    components, energy = fit_pca(vectors, dims)
    np.savez(done, components=components, energy=energy, rows=len(sample))
    log(f"pca: {len(sample):,} training names, {vectors.shape[1]} -> {dims} dims, {energy:.4f} of energy kept")
    return {"components": components, "energy": energy, "rows": len(sample)}


def encode_part(cfg: dict, work: Path, spec: dict, part: int, parts: int, device: str) -> None:
    """Embed this part's slice of distinct names, project, and write ``reduced_part<k>.npy``."""
    marker = work / f"reduced_part{part}.json"
    if marker.exists():
        return
    texts = load_vocab(work)
    bounds = np.linspace(0, len(texts), parts + 1).astype(np.int64)
    mine = texts[bounds[part]:bounds[part + 1]]
    del texts
    with np.load(work / "pca.npz") as data:
        components = data["components"]
    out_path = work / f"reduced_part{part}.npy"
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(mine), components.shape[1]))
    model = load_encoder(spec, device, cfg["encode"])
    chunk = cfg["encode"]["chunk_texts"]
    started = time.perf_counter()
    for start in range(0, len(mine), chunk):
        out[start:start + chunk] = project(embed(model, mine[start:start + chunk], spec["prefix"], cfg["encode"]), components)
        done = min(start + chunk, len(mine))
        log(f"encode part {part}: {done:,}/{len(mine):,} ({done / max(time.perf_counter() - started, 1e-9):,.0f} names/s)")
    out.flush()
    del out
    release(model)
    atomic_write_json(marker, {"rows": len(mine), "device": device, "seconds": time.perf_counter() - started})


def encode_all(cfg: dict, config_path: Path, work: Path, output: Path, spec: dict) -> dict:
    targets = devices()
    parts = len(targets)
    atomic_write_json(work / "model.json", spec)
    started = time.perf_counter()
    if parts == 1:
        encode_part(cfg, work, spec, 0, 1, targets[0])
    else:
        procs = [subprocess.Popen([sys.executable, "-m", "src.embed_names", "--config", str(config_path),
                                   "--work-dir", str(work), "--output-dir", str(output), "--stage", "encode-part",
                                   "--part", str(i), "--parts", str(parts), "--device", device], cwd=CODE_DIR)
                 for i, device in enumerate(targets)]
        failed = [i for i, proc in enumerate(procs) if proc.wait() != 0]
        if failed:
            raise RuntimeError(f"encode parts {failed} failed")
    return {"parts": parts, "devices": targets, "seconds": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Outputs


def environment() -> dict:
    info = {"python": sys.version.split()[0]}
    for module in ("torch", "sentence_transformers", "transformers", "huggingface_hub", "numpy"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    try:
        import torch

        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["gpus"] = []
    return info


def finalize(cfg: dict, work: Path, output: Path, spec: dict, vocab: dict, pca: dict, diagnostic: dict,
             encoding: dict) -> dict:
    parts = sorted(work.glob("reduced_part*.npy"), key=lambda p: int(p.stem.removeprefix("reduced_part")))
    reduced = np.concatenate([np.load(p, mmap_mode="r") for p in parts])
    if len(reduced) != vocab["distinct_names"]:
        raise RuntimeError(f"{len(reduced):,} vectors for {vocab['distinct_names']:,} distinct names")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, source in cfg["files"]:
        label = file_label(split, source)
        ids = np.load(work / f"map_{label}.npy")
        keys = np.load(work / f"keys_{label}.npy")
        vectors = reduced[ids]
        np.save(output / f"names_{label}.npy", vectors)
        np.save(output / f"keys_{label}.npy", keys)
        files[label] = vocab["files"][label] | {"vectors": f"names_{label}.npy", "keys": f"keys_{label}.npy",
                                                "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()}
        log(f"finalize: {label} {len(ids):,} rows")
    np.savez(output / "pca.npz", components=pca["components"], energy=pca["energy"], rows=pca["rows"])
    manifest = {
        "run_id": cfg["run_id"],
        "model": {k: spec[k] for k in ("name", "repo", "revision", "license", "prefix")},
        "vector": {"dims": int(pca["components"].shape[1]), "dtype": "float16", "normalised": True,
                   "pca": "uncentered, fitted on distinct training names only", "pca_rows": pca["rows"],
                   "energy_kept": pca["energy"], "max_seq_length": cfg["encode"]["max_seq_length"]},
        "row_order": "csv.DictReader order of each source TSV (same as the K1/K2 stores)",
        "keys": "S2/S3: blocking.encode_id(entity_id); S1: int(entity_id[3:])",
        "distinct_names": vocab["distinct_names"], "files": files, "encoding": encoding,
        "diagnostic_selected": diagnostic.get("selected"), "environment": environment()}
    atomic_write_json(output / "manifest.json", manifest)
    (output / "EMB_REPORT.md").write_text(make_report(manifest, diagnostic))
    return manifest


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def make_report(manifest: dict, diagnostic: dict) -> str:
    p = diagnostic["pairs"]
    lines = [
        "# R3 name vectors",
        "",
        f"Selected model: **{manifest['model']['name']}** (`{manifest['model']['repo']}` @ "
        f"`{manifest['model']['revision']}`, license {manifest['model']['license']}).",
        "",
        f"Diagnostic pairs (training folds outside fold 0): {p['true']:,} true pairs and "
        f"{p['same_address_negative']:,} same-address negatives from {p['s1']:,} S1 "
        f"(cross-script true pairs: {p['cross_script_true']:,}; weak-string true pairs: {p['weak_string_true']:,}).",
        "",
        "| Model | Combined AUC (string + cosine, out-of-fold) | Cosine alone | Cross-script true vs negatives | "
        "Weak-string true vs negatives | Mean cosine true / same-address |",
        "|---|---:|---:|---:|---:|---|",
        f"| string similarity only | {fmt(diagnostic['string_baseline']['auc_combined_oof'])} | "
        f"{fmt(diagnostic['string_baseline']['auc_alone'])} | "
        f"{fmt(diagnostic['string_baseline']['auc_cross_script_positives'])} | — | — |",
    ]
    for r in diagnostic["candidates"]:
        m = r["metrics"]
        lines.append(f"| {r['name']} | {fmt(m['auc_combined_oof'])} | {fmt(m['auc_alone'])} | "
                     f"{fmt(m['auc_cross_script_positives'])} | {fmt(m['auc_weak_string_positives'])} | "
                     f"{m['mean_cosine_true']:.3f} / {m['mean_cosine_same_address']:.3f} |")
    v = manifest["vector"]
    lines += ["", f"Rule: {diagnostic['selection_rule']}.", "",
              f"Vectors: {v['dims']} dims ({v['dtype']}, L2-normalised), uncentered PCA on {v['pca_rows']:,} distinct "
              f"training names keeping {v['energy_kept']:.1%} of the energy; {manifest['distinct_names']:,} distinct names "
              "embedded once and written per record.", "",
              "| File | Rows |", "|---|---:|"]
    lines += [f"| {label} | {info['rows']:,} |" for label, info in manifest["files"].items()]
    lines += ["", "These vectors become a candidate feature group in K2-final and are adopted only under the "
              "pre-registered ablation rule (≥ +0.002 macro F0.5, CI > 0, no slice drop > 0.005)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def selected_spec(cfg: dict, output: Path, override: str | None) -> dict:
    if override:
        spec = next((c for c in cfg["candidates"] if override in (c["name"], c["repo"])), None)
        if spec is None:
            raise SystemExit(f"--model {override} is not a configured candidate")
        return resolve_model(spec, cfg["allowed_licenses"])
    chosen = json.loads((output / "diagnostic.json").read_text())["selected"]
    return next(c for c in cfg["candidates"] if c["repo"] == chosen["repo"]) | chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("diagnose", "vocab", "pca", "encode", "finalize", "all", "encode-part"),
                        required=True)
    parser.add_argument("--model", help="skip the diagnostic choice and use this configured candidate")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    work, output, root = args.work_dir.resolve(), args.output_dir.resolve(), args.root
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "encode-part":
        spec = json.loads((work / "model.json").read_text())
        encode_part(cfg, work, spec, args.part, args.parts, args.device)
        return
    started = time.perf_counter()
    diagnostic = {}
    if args.stage in ("diagnose", "all") and not args.model:
        diagnostic = diagnose(cfg, root, output)
        if args.stage == "diagnose":
            return
    elif (output / "diagnostic.json").exists():
        diagnostic = json.loads((output / "diagnostic.json").read_text())
    spec = selected_spec(cfg, output, args.model)
    vocab = build_vocab(cfg, root, work)
    if args.stage == "vocab":
        return
    pca = train_pca(cfg, work, spec)
    if args.stage == "pca":
        return
    encoding = {}
    if args.stage in ("encode", "all"):
        encoding = encode_all(cfg, args.config.resolve(), work, output, spec)
        if args.stage == "encode":
            return
    if not diagnostic:
        diagnostic = {"pairs": {"true": 0, "same_address_negative": 0, "s1": 0, "cross_script_true": 0,
                                "weak_string_true": 0}, "string_baseline": {"auc_combined_oof": None, "auc_alone": None,
                                                                            "auc_cross_script_positives": None},
                      "candidates": [], "selection_rule": f"model set explicitly: {spec['name']}", "selected": None}
    manifest = finalize(cfg, work, output, spec, vocab, pca, diagnostic, encoding)
    log(f"R3 done in {time.perf_counter() - started:,.0f}s: {manifest['model']['name']}, "
        f"{manifest['distinct_names']:,} distinct names, {manifest['vector']['dims']} dims")


if __name__ == "__main__":
    main()
