"""Run the deterministic fold-0 candidate generation pilot.

From code/business_entity_resolution: python3 -m src.evaluate_blocking
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import threading
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

from .blocking import encode_id, exact_name_lookup, sparse_topk_rows, union_candidates
from .normalization import has_non_ascii, normalize_text

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ("exact_name", "name_tfidf", "address_tfidf", "union")
K_VALUES = (10, 20, 50, 100, 200)


class PeakRSS:
    """Sample whole-process resident RAM during a single pipeline stage."""

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()

        def monitor():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def selected_ids(folds_path: Path, fold: int, size: int, seed: int) -> list[str]:
    scored = []
    with folds_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if int(row["fold"]) == fold:
                entity_id = row["source1_entity_id"]
                digest = hashlib.blake2b(f"{seed}:{entity_id}".encode(), digest_size=8).digest()
                scored.append((digest, entity_id))
    if len(scored) < size:
        raise ValueError("Fold has fewer rows than the requested sample")
    return [entity_id for _, entity_id in sorted(scored)[:size]]


def load_sample(ids: list[str]) -> tuple[list[dict], dict[str, list[int]]]:
    selected = set(ids)
    records = {}
    with (ROOT / "student_resource/dataset/train/train_source1.tsv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["entity_id"] in selected:
                records[row["entity_id"]] = {
                    "id": row["entity_id"], "country": row["country"],
                    "country_key": normalize_text(row["country"]),
                    "name": normalize_text(row["business_name"]),
                    "address": normalize_text(row["business_address"]),
                    "non_ascii": has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"]),
                }
    if len(records) != len(ids):
        raise RuntimeError("Sampled S1 IDs missing from the training file")
    truth = {}
    with (ROOT / "student_resource/dataset/train/train_ground_truth.tsv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["source1_entity_id"] in selected:
                truth[row["source1_entity_id"]] = [
                    encode_id(value) for value in row["matched_entity_ids"].split(",") if value
                ]
    if len(truth) != len(ids):
        raise RuntimeError("Sampled S1 IDs missing from ground truth")
    return [records[id] for id in ids], truth


def recall_at(candidates: list[int], truth: list[int], k: int | None = None) -> int:
    return len(set(candidates if k is None else candidates[:k]) & set(truth))


def ranked_union(exact: list[int], name: list[int], address: list[int]) -> list[int]:
    """Deterministic reciprocal-rank fusion; every unique ID retained."""
    scores = defaultdict(float)
    for route in (exact, name, address):
        for rank, candidate in enumerate(dict.fromkeys(route), 1):
            scores[candidate] += 1.0 / (60 + rank)
    return sorted(scores, key=lambda candidate: (-scores[candidate], candidate))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_route(
    ranked: list[list[int]], truths: list[list[int]], countries: list[str],
    non_ascii_links: list[set[int]], missing_address_links: list[set[int]],
    retrieval_pool_sizes_by_country: dict[str, int] | None = None,
) -> dict:
    counts = np.fromiter((len(row) for row in ranked), dtype=np.int32)
    total_links = sum(map(len, truths))
    hits = sum(recall_at(row, truth) for row, truth in zip(ranked, truths))
    all_hits = sum(set(truth).issubset(row) for row, truth in zip(ranked, truths))
    positive_indices = [i for i, truth in enumerate(truths) if truth]
    result = {
        "positive_link_candidate_recall": ratio(hits, total_links),
        "retrieved_true_links": hits,
        "true_links": total_links,
        "entities_with_every_true_match_pct": 100 * all_hits / len(truths),
        "positive_entities_with_every_true_match_pct": 100 * sum(
            set(truths[i]).issubset(ranked[i]) for i in positive_indices
        ) / len(positive_indices),
        "recall_at": {str(k): ratio(sum(recall_at(row, truth, k) for row, truth in zip(ranked, truths)), total_links)
                      for k in K_VALUES},
        "candidate_count_per_s1": {
            "median": float(np.median(counts)), "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)), "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
        },
        "zero_candidate_rate": float(np.mean(counts == 0)),
    }
    if retrieval_pool_sizes_by_country is not None:
        compared_pairs = sum(retrieval_pool_sizes_by_country[country] for country in countries)
        result["candidate_reduction_ratio"] = 1 - int(counts.sum()) / compared_pairs
        result["candidate_count_total"] = int(counts.sum())
        result["country_partition_pair_space"] = compared_pairs
    for source, parity in (("S2", 0), ("S3", 1)):
        denominator = sum(sum((v & 1) == parity for v in truth) for truth in truths)
        numerator = sum(sum((v & 1) == parity for v in set(row) & set(truth))
                        for row, truth in zip(ranked, truths))
        result.setdefault("candidate_recall_by_source", {})[source] = ratio(numerator, denominator)
    by_country = {}
    for country in sorted(set(countries)):
        indices = [i for i, value in enumerate(countries) if value == country]
        denominator = sum(len(truths[i]) for i in indices)
        numerator = sum(recall_at(ranked[i], truths[i]) for i in indices)
        by_country[country] = {"sampled_s1": len(indices), "true_links": denominator,
                               "candidate_recall": ratio(numerator, denominator)}
    result["candidate_recall_by_country"] = by_country
    for key, subsets in (("non_ascii_records", non_ascii_links),
                         ("target_address_missing", missing_address_links)):
        denominator = sum(map(len, subsets))
        numerator = sum(len(set(row) & subset) for row, subset in zip(ranked, subsets))
        result[key] = {"true_links": denominator, "retrieved_true_links": numerator,
                       "candidate_recall": ratio(numerator, denominator)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1a_pilot.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    started = time.perf_counter()
    np.random.seed(config["seed"])
    ids = selected_ids(ROOT / "artifacts/folds.tsv", config["validation_fold"],
                       config["sample_size"], config["seed"])
    sample, truth = load_sample(ids)
    k = config["top_k_per_source"]
    n = len(sample)
    exact: list[array] = [array("Q") for _ in range(n)]
    # Fixed-width arrays avoid millions of Python tuple objects in the final state.
    candidate_ids = {route: np.zeros((n, 2, k), dtype=np.uint64)
                     for route in ("name_tfidf", "address_tfidf")}
    candidate_scores = {route: np.zeros((n, 2, k), dtype=np.float32)
                        for route in candidate_ids}
    target_truth = set(value for links in truth.values() for value in links)
    target_meta = {}
    grouped = defaultdict(list)
    for i, row in enumerate(sample):
        grouped[row["country_key"]].append(i)
    stage_times = {}
    stage_peaks = {}
    corpus_counts = defaultdict(int)
    for source_number in (2, 3):
        source_started = time.perf_counter()
        path = ROOT / f"student_resource/dataset/train/train_source{source_number}.tsv"
        for country, query_indices in sorted(grouped.items()):
            group_started = time.perf_counter()
            print(f"S{source_number} country={country!r}: loading complete target partition", flush=True)
            target_ids, names, addresses = [], [], []
            with PeakRSS() as exact_memory:
                with path.open(encoding="utf-8", newline="") as file:
                    for row in csv.DictReader(file, delimiter="\t"):
                        if normalize_text(row["country"]) != country:
                            continue
                        target_id = encode_id(row["entity_id"])
                        target_ids.append(target_id)
                        names.append(normalize_text(row["business_name"]))
                        addresses.append(normalize_text(row["business_address"]))
                        if target_id in target_truth:
                            target_meta[target_id] = {
                                "non_ascii": has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"]),
                                "address_missing": not row["business_address"].strip(),
                            }
                queries = [sample[i] for i in query_indices]
                for local_i, hits in enumerate(exact_name_lookup(
                    [row["name"] for row in queries], names, target_ids
                )):
                    exact[query_indices[local_i]].extend(hits)
            stage_times[f"S{source_number}/{country}/exact_name"] = time.perf_counter() - group_started
            stage_peaks[f"S{source_number}/{country}/exact_name"] = exact_memory.peak
            corpus_counts[country] += len(target_ids)
            for route, field, target_texts in (
                ("name_tfidf", "name", names),
                ("address_tfidf", "address", addresses),
            ):
                route_started = time.perf_counter()
                print(f"S{source_number} country={country!r}: {route}, {len(target_ids):,} targets", flush=True)
                with PeakRSS() as route_memory:
                    rows = sparse_topk_rows(
                        [row[field] for row in queries], target_texts, target_ids,
                        k=k, n_features=config["hash_features"],
                        ngram_range=tuple(config["ngram_range"]),
                        max_document_frequency=config["max_document_frequency"],
                        query_batch_size=config["query_batch_size"], threads=config["threads"],
                        progress_label=f"S{source_number}/{country}/{route}",
                    )
                    source_index = source_number - 2
                    for local_i, hits in enumerate(rows):
                        global_i = query_indices[local_i]
                        length = len(hits)
                        candidate_ids[route][global_i, source_index, :length] = [v for v, _ in hits]
                        candidate_scores[route][global_i, source_index, :length] = [s for _, s in hits]
                stage_times[f"S{source_number}/{country}/{route}"] = time.perf_counter() - route_started
                stage_peaks[f"S{source_number}/{country}/{route}"] = route_memory.peak
            stage_times[f"S{source_number}/{country}/total"] = time.perf_counter() - group_started
            del target_ids, names, addresses
        stage_times[f"S{source_number}/total"] = time.perf_counter() - source_started
    if set(target_meta) != target_truth:
        raise RuntimeError(f"Missing metadata for {len(target_truth - set(target_meta))} sampled true targets")

    def ranked_sparse(route: str, index: int) -> list[int]:
        ids_row = candidate_ids[route][index].ravel()
        scores_row = candidate_scores[route][index].ravel()
        pairs = [(int(value), float(score)) for value, score in zip(ids_row, scores_row) if value]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        return union_candidates([value for value, _ in pairs])

    rankings = {route: [] for route in ROUTES}
    additional = Counter()
    for i in range(n):
        exact_row = sorted(set(exact[i]))
        name_row = ranked_sparse("name_tfidf", i)
        address_row = ranked_sparse("address_tfidf", i)
        union_row = ranked_union(exact_row, name_row, address_row)
        for route, row in zip(ROUTES, (exact_row, name_row, address_row, union_row)):
            rankings[route].append(row)
        true = set(truth[ids[i]])
        route_sets = [set(exact_row), set(name_row), set(address_row)]
        for route_index, route in enumerate(ROUTES[:3]):
            other = set().union(*(route_sets[j] for j in range(3) if j != route_index))
            additional[route] += len(true & (route_sets[route_index] - other))
    truths = [truth[id] for id in ids]
    non_ascii = [{value for value in links if sample[i]["non_ascii"] or target_meta[value]["non_ascii"]}
                 for i, links in enumerate(truths)]
    missing = [{value for value in links if target_meta[value]["address_missing"]} for links in truths]
    countries = [row["country"] for row in sample]
    pool_sizes = {raw: corpus_counts[normalize_text(raw)] for raw in set(countries)}
    metrics = {
        "config": config, "sample": {"size": n, "validation_fold": config["validation_fold"],
            "selection": "25,000 smallest BLAKE2b(seed:S1 ID) digests within fold 0",
            "country_counts": dict(Counter(countries)),
            "true_links": sum(map(len, truths)), "singleton_count": sum(not links for links in truths)},
        "routes": {route: summarize_route(rankings[route], truths, countries, non_ascii, missing, pool_sizes)
                   for route in ROUTES},
        "retrieval_pool_sizes_by_country": pool_sizes,
        "unique_additional_true_links": dict(additional),
        "runtime_seconds": time.perf_counter() - started,
        "stage_runtime_seconds": stage_times,
        "stage_peak_rss_bytes": stage_peaks,
        "route_resources": {
            route: {
                "runtime_seconds": sum(seconds for key, seconds in stage_times.items()
                                       if key.endswith(f"/{route}")),
                "peak_process_rss_bytes": max(value for key, value in stage_peaks.items()
                                              if key.endswith(f"/{route}")),
            } for route in ROUTES[:3]
        },
        "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    metrics["route_resources"]["union"] = {
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_process_rss_bytes": metrics["peak_ram_bytes"],
    }
    output = ROOT / "artifacts/phase1a"
    output.mkdir(parents=True, exist_ok=True)
    exact_lengths = np.fromiter((len(row) for row in exact), dtype=np.int64, count=n)
    exact_indptr = np.concatenate(([0], np.cumsum(exact_lengths)))
    exact_values = np.fromiter((value for row in exact for value in row),
                               dtype=np.uint64, count=int(exact_indptr[-1]))
    sorted_targets = sorted(target_meta)
    np.savez(
        output / "pilot_candidates.npz",
        sample_ids=np.asarray(ids),
        exact_indptr=exact_indptr,
        exact_values=exact_values,
        name_ids=candidate_ids["name_tfidf"],
        name_scores=candidate_scores["name_tfidf"],
        address_ids=candidate_ids["address_tfidf"],
        address_scores=candidate_scores["address_tfidf"],
        true_target_ids=np.asarray(sorted_targets, dtype=np.uint64),
        true_target_non_ascii=np.asarray([target_meta[value]["non_ascii"] for value in sorted_targets], dtype=np.bool_),
        true_target_address_missing=np.asarray([target_meta[value]["address_missing"] for value in sorted_targets], dtype=np.bool_),
    )
    (output / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = make_report(metrics)
    (output / "pilot_report.md").write_text(report)
    print(report, flush=True)


def make_report(metrics: dict) -> str:
    lines = ["# Phase 1A candidate-generation pilot", "",
             "## Scope and method", "",
             f"- Sample: {metrics['sample']['size']:,} S1 entities from validation fold 0, selected by fixed-seed BLAKE2b rank.",
             "- Corpus: all training S2 and S3 records, searched separately within dynamic country partitions.",
             "- Text: Unicode NFKC, case-folding, punctuation and whitespace normalization; separate name and address views.",
             f"- Sparse search: hashed character {metrics['config']['ngram_range'][0]}–{metrics['config']['ngram_range'][1]}-gram TF-IDF ({metrics['config']['hash_features']:,} features), corpus-fit IDF per source/country/field, and removal of grams occurring in more than {100*metrics['config']['max_document_frequency']:g}% of target records; bounded top-k and no true-link injection.",
             "- Union ranking: reciprocal-rank fusion (constant 60) of exact name, name TF-IDF, and address TF-IDF; unique target IDs retained.",
             "- Recall@k uses the first k candidates per S1 across both target sources. Full candidate recall uses every emitted candidate.",
             "- Every-match percentage includes singleton S1 entities, for which the condition is vacuously true; positive-only rate is also shown.",
             "- Non-ASCII subset: true links where S1 or target name/address contains non-ASCII characters.", ""]
    lines += ["## Results", "", "| Route | Link recall | Every match % | Positive-only every match % | R@10 | R@20 | R@50 | R@100 | R@200 | S2 | S3 | Zero candidate % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for route in ROUTES:
        r = metrics["routes"][route]
        fmt = lambda v: "n/a" if v is None else f"{v:.3%}"
        lines.append("| " + " | ".join([route, fmt(r["positive_link_candidate_recall"]),
            f"{r['entities_with_every_true_match_pct']:.2f}", f"{r['positive_entities_with_every_true_match_pct']:.2f}",
            *[fmt(r["recall_at"][str(k)]) for k in K_VALUES],
            fmt(r["candidate_recall_by_source"]["S2"]), fmt(r["candidate_recall_by_source"]["S3"]),
            f"{100*r['zero_candidate_rate']:.2f}"]) + " |")
    lines += ["", "### Candidate counts and reduction", "", "| Route | Median | p90 | p95 | p99 | Max | Reduction vs country pool |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for route in ROUTES:
        c = metrics["routes"][route]["candidate_count_per_s1"]
        reduction = metrics['routes'][route]['candidate_reduction_ratio']
        lines.append(f"| {route} | {c['median']:g} | {c['p90']:g} | {c['p95']:g} | {c['p99']:g} | {c['max']} | {reduction:.5%} |")
    lines += ["", "### Runtime and peak RAM", "", "| Route | Runtime (min) | Peak process RSS (GiB) |",
              "|---|---:|---:|"]
    for route in ROUTES:
        r = metrics["route_resources"][route]
        lines.append(f"| {route} | {r['runtime_seconds']/60:.1f} | {r['peak_process_rss_bytes']/2**30:.2f} |")
    lines += ["", "Per-route RAM is the observed process RSS during that stage, including data retained from earlier stages. "
              "The union row covers the full pipeline. Exact-name runtime includes the streaming corpus reads shared with the TF-IDF routes."]
    lines += ["", "### Subsets and marginal contribution", "",
              "| Route | Country recall | Non-ASCII recall (links) | Missing target address recall (links) | Unique extra true links |",
              "|---|---|---:|---:|---:|"]
    for route in ROUTES:
        r = metrics["routes"][route]
        country = ", ".join(f"{key}: {value['candidate_recall']:.3%} ({value['true_links']:,})"
                            for key, value in r["candidate_recall_by_country"].items())
        def subset(key: str) -> str:
            s = r[key]
            return f"{s['candidate_recall']:.3%} ({s['true_links']:,})" if s['candidate_recall'] is not None else "n/a (0)"
        extra = str(metrics["unique_additional_true_links"].get(route, 0)) if route != "union" else "—"
        lines.append(f"| {route} | {country} | {subset('non_ascii_records')} | {subset('target_address_missing')} | {extra} |")
    lines += ["", "Unique extra true links means links found by that route and by neither of the other two routes.",
              "", "## Resources and reproducibility", "",
              f"- Wall time: {metrics['runtime_seconds']/60:.1f} minutes; peak process RAM: {metrics['peak_ram_bytes']/2**30:.2f} GiB.",
              f"- Config: `configs/phase1a_pilot.json`; seed: {metrics['config']['seed']}; top-k: {metrics['config']['top_k_per_source']} per source and sparse route.",
              "- Run locally or in Colab from `code/business_entity_resolution` with `python3 -m pip install -r requirements.txt` and `python3 -m src.evaluate_blocking`.",
              "- sparse-dot-topn 1.2.0 license was checked before installation: Apache-2.0 ([PyPI](https://pypi.org/project/sparse-dot-topn/)).",
              "- No test data was read. No matching model, encoder, reranker, or ensemble was trained.", "",
              "## Recommendation", ""]
    union = metrics["routes"]["union"]
    lines.append("Use the exact-name, name TF-IDF, and address TF-IDF union in the next phase, "
                 f"with top-{metrics['config']['top_k_per_source']} per target source for each sparse route. "
                 f"This pilot recovered {union['positive_link_candidate_recall']:.2%} of sampled true links "
                 f"with median {union['candidate_count_per_s1']['median']:g} candidates per S1. "
                 "Reassess k and route value on the full validation fold before any matching model work.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
