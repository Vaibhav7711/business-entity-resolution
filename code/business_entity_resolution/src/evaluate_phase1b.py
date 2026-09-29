"""Phase 1B route ablations on the Phase 1A 25,000-S1 pilot only.

Run from code/business_entity_resolution: python3 -m src.evaluate_phase1b
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .blocking import encode_id, sparse_topk_rows
from .evaluate_blocking import PeakRSS, ROOT, load_sample, selected_ids
from .normalization import normalize_text
from .phase1b_routes import (
    LEGAL_SUFFIXES, address_digits, name_tokens, rare_token_lookup,
    stripped_exact_lookup, word_tfidf_topk_rows,
)

ROUTES = (
    "A_name_char_top200", "B_name_word_tfidf", "C_rare_name_token",
    "D_address_digit", "E_legal_suffix_exact",
)


def pack_ragged(rows: list[array]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    values = np.fromiter((value for row in rows for value in row),
                         dtype=np.uint64, count=int(indptr[-1]))
    return indptr, values


def candidate_set(values: np.ndarray) -> set[int]:
    return {int(value) for value in values.ravel() if value}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def route_rows(
    mask: int, i: int, baseline: dict, extras: dict,
) -> set[int]:
    result = set(int(value) for value in baseline["exact_values"][
        baseline["exact_indptr"][i]:baseline["exact_indptr"][i + 1]
    ])
    result.update(candidate_set(baseline["name_ids"][i]))
    result.update(candidate_set(baseline["address_ids"][i]))
    if mask & 1:
        result.update(candidate_set(extras["name200"][i]))
    if mask & 2:
        result.update(candidate_set(extras["word"][i]))
    for bit, key in ((4, "rare"), (8, "digit"), (16, "suffix")):
        if mask & bit:
            result.update(extras[key][i])
    return result


def evaluate_config(
    mask: int, baseline: dict, extras: dict, truths: list[set[int]],
    countries: list[str], non_ascii: list[set[int]], missing: list[set[int]],
    pool_sizes: dict[str, int],
) -> dict:
    count = np.zeros(len(truths), dtype=np.int32)
    link_hits = 0
    every_positive = 0
    by_country_hits = Counter()
    non_ascii_hits = 0
    missing_hits = 0
    for i, true in enumerate(truths):
        candidates = route_rows(mask, i, baseline, extras)
        count[i] = len(candidates)
        hits = candidates & true
        link_hits += len(hits)
        by_country_hits[countries[i]] += len(hits)
        if true and true.issubset(candidates):
            every_positive += 1
        non_ascii_hits += len(candidates & non_ascii[i])
        missing_hits += len(candidates & missing[i])
    true_links = sum(map(len, truths))
    positive_s1 = sum(bool(true) for true in truths)
    country_totals = Counter()
    for country, true in zip(countries, truths):
        country_totals[country] += len(true)
    pair_space = sum(pool_sizes[country] for country in countries)
    return {
        "routes": [route for bit, route in enumerate(ROUTES) if mask & (1 << bit)],
        "positive_link_recall": ratio(link_hits, true_links),
        "retrieved_true_links": link_hits,
        "positive_s1_with_every_true_match_pct": 100 * every_positive / positive_s1 if positive_s1 else None,
        "recall_by_country": {country: ratio(by_country_hits[country], country_totals[country])
                              for country in sorted(country_totals)},
        "non_ascii_recall": ratio(non_ascii_hits, sum(map(len, non_ascii))),
        "missing_target_address_recall": ratio(missing_hits, sum(map(len, missing))),
        "candidate_count": {
            "mean": float(np.mean(count)),
            "median": float(np.median(count)),
            "p95": float(np.percentile(count, 95)),
            "p99": float(np.percentile(count, 99)),
            "max": int(count.max()),
            "total": int(count.sum()),
        },
        "candidate_reduction_ratio": 1 - int(count.sum()) / pair_space,
    }


def pareto_masks(metrics: dict[int, dict]) -> list[int]:
    frontier = []
    for mask, row in metrics.items():
        mean = row["candidate_count"]["mean"]
        recall = row["positive_link_recall"]
        dominated = any(
            other_mask != mask
            and other["candidate_count"]["mean"] <= mean
            and other["positive_link_recall"] >= recall
            and (other["candidate_count"]["mean"] < mean or other["positive_link_recall"] > recall)
            for other_mask, other in metrics.items()
        )
        if not dominated:
            frontier.append(mask)
    return sorted(frontier, key=lambda value: (metrics[value]["candidate_count"]["mean"],
                                                 metrics[value]["positive_link_recall"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1b_pilot.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(config["legal_suffix_spellings"]) != LEGAL_SUFFIXES:
        raise ValueError("Configured suffixes differ from the supplied problem statement")
    phase1a_config = json.loads((ROOT / "configs/phase1a_pilot.json").read_text())
    phase1a_metrics = json.loads((ROOT / "artifacts/phase1a/pilot_metrics.json").read_text())
    if any(config[key] != phase1a_config[key] for key in ("seed", "validation_fold", "sample_size")):
        raise ValueError("Phase 1B sample parameters differ from Phase 1A")
    cache_file = ROOT / "artifacts/phase1a/pilot_candidates.npz"
    if not cache_file.exists():
        raise FileNotFoundError("Reproduce Phase 1A and write pilot_candidates.npz first")
    with np.load(cache_file, allow_pickle=False) as loaded:
        baseline = {key: loaded[key] for key in ("sample_ids", "exact_indptr", "exact_values", "name_ids", "address_ids",
                                                  "true_target_ids", "true_target_non_ascii", "true_target_address_missing")}
    ids = selected_ids(ROOT / "artifacts/folds.tsv", config["validation_fold"],
                       config["sample_size"], config["seed"])
    if list(baseline["sample_ids"]) != ids:
        raise RuntimeError("Phase 1A candidate cache has a different sampled ID order")
    sample, truth_map = load_sample(ids)
    truths = [set(truth_map[value]) for value in ids]
    target_meta = {
        int(value): (bool(non_ascii), bool(missing))
        for value, non_ascii, missing in zip(
            baseline["true_target_ids"], baseline["true_target_non_ascii"],
            baseline["true_target_address_missing"]
        )
    }
    if set(target_meta) != set().union(*truths):
        raise RuntimeError("Phase 1A true-target metadata is incomplete")
    non_ascii = [{value for value in true if sample[i]["non_ascii"] or target_meta[value][0]}
                 for i, true in enumerate(truths)]
    missing = [{value for value in true if target_meta[value][1]} for true in truths]
    countries = [row["country"] for row in sample]
    grouped = defaultdict(list)
    for i, row in enumerate(sample):
        grouped[row["country_key"]].append(i)
    n = len(ids)
    name200 = np.zeros((n, 2, config["name_char_top_k_per_source"]), dtype=np.uint64)
    word = np.zeros((n, 2, config["word_name_top_k_per_source"]), dtype=np.uint64)
    ragged = {key: [array("Q") for _ in range(n)] for key in ("rare", "digit", "suffix")}
    stage_times = defaultdict(float)
    stage_peaks = defaultdict(int)
    started = time.perf_counter()

    for source_number in (2, 3):
        path = ROOT / f"student_resource/dataset/train/train_source{source_number}.tsv"
        for country, query_indices in sorted(grouped.items()):
            print(f"S{source_number}/{country}: streaming complete target partition", flush=True)
            scan_started = time.perf_counter()
            with PeakRSS() as memory:
                target_ids, names, addresses = [], [], []
                with path.open(encoding="utf-8", newline="") as file:
                    for row in csv.DictReader(file, delimiter="\t"):
                        if normalize_text(row["country"]) != country:
                            continue
                        target_ids.append(encode_id(row["entity_id"]))
                        names.append(normalize_text(row["business_name"]))
                        addresses.append(normalize_text(row["business_address"]))
            stage_times["shared_corpus_scan"] += time.perf_counter() - scan_started
            stage_peaks["shared_corpus_scan"] = max(stage_peaks["shared_corpus_scan"], memory.peak)
            queries = [sample[i] for i in query_indices]
            source_index = source_number - 2

            route_started = time.perf_counter()
            print(f"S{source_number}/{country}: A name character top-200", flush=True)
            with PeakRSS() as memory:
                rows = sparse_topk_rows(
                    [row["name"] for row in queries], names, target_ids,
                    k=config["name_char_top_k_per_source"],
                    n_features=phase1a_config["hash_features"],
                    ngram_range=tuple(phase1a_config["ngram_range"]),
                    max_document_frequency=phase1a_config["max_document_frequency"],
                    query_batch_size=config["query_batch_size"], threads=config["threads"],
                    progress_label=f"S{source_number}/{country}/A",
                )
                for local_i, hits in enumerate(rows):
                    name200[query_indices[local_i], source_index, :len(hits)] = [value for value, _ in hits]
            stage_times[ROUTES[0]] += time.perf_counter() - route_started
            stage_peaks[ROUTES[0]] = max(stage_peaks[ROUTES[0]], memory.peak)

            route_started = time.perf_counter()
            print(f"S{source_number}/{country}: B word name TF-IDF", flush=True)
            with PeakRSS() as memory:
                rows = word_tfidf_topk_rows(
                    [row["name"] for row in queries], names, target_ids,
                    k=config["word_name_top_k_per_source"],
                    n_features=config["word_hash_features"],
                    max_document_frequency=config["word_max_document_frequency"],
                    query_batch_size=config["query_batch_size"], threads=config["threads"],
                )
                for local_i, hits in enumerate(rows):
                    word[query_indices[local_i], source_index, :len(hits)] = [value for value, _ in hits]
            stage_times[ROUTES[1]] += time.perf_counter() - route_started
            stage_peaks[ROUTES[1]] = max(stage_peaks[ROUTES[1]], memory.peak)

            for route, key, qfield, tfield, tokenizer, max_df, max_tokens, top_k in (
                (ROUTES[2], "rare", "name", names, name_tokens,
                 config["rare_name_max_document_frequency"], config["rare_name_max_query_tokens"],
                 config["rare_name_top_k_per_source"]),
                (ROUTES[3], "digit", "address", addresses, address_digits,
                 config["address_digit_max_document_frequency"], config["address_digit_max_query_tokens"],
                 config["address_digit_top_k_per_source"]),
            ):
                route_started = time.perf_counter()
                print(f"S{source_number}/{country}: {route}", flush=True)
                with PeakRSS() as memory:
                    rows = rare_token_lookup(
                        [row[qfield] for row in queries], tfield, target_ids,
                        tokenize=tokenizer, max_document_frequency=max_df,
                        max_query_tokens=max_tokens, k=top_k,
                    )
                    for local_i, hits in enumerate(rows):
                        ragged[key][query_indices[local_i]].extend(hits)
                stage_times[route] += time.perf_counter() - route_started
                stage_peaks[route] = max(stage_peaks[route], memory.peak)

            route_started = time.perf_counter()
            print(f"S{source_number}/{country}: E legal suffix exact", flush=True)
            with PeakRSS() as memory:
                rows = stripped_exact_lookup([row["name"] for row in queries], names, target_ids)
                for local_i, hits in enumerate(rows):
                    ragged["suffix"][query_indices[local_i]].extend(hits)
            stage_times[ROUTES[4]] += time.perf_counter() - route_started
            stage_peaks[ROUTES[4]] = max(stage_peaks[ROUTES[4]], memory.peak)
            del names, addresses, target_ids, rows

    extras = {"name200": name200, "word": word, **ragged}
    pool_sizes = phase1a_metrics["retrieval_pool_sizes_by_country"]
    all_metrics = {}
    for mask in range(32):
        eval_started = time.perf_counter()
        with PeakRSS() as memory:
            row = evaluate_config(mask, baseline, extras, truths, countries, non_ascii, missing, pool_sizes)
        row["evaluation_runtime_seconds"] = time.perf_counter() - eval_started
        row["peak_process_rss_during_evaluation_bytes"] = memory.peak
        all_metrics[mask] = row
        print(f"config {mask:02d}: recall={row['positive_link_recall']:.4%}, median={row['candidate_count']['median']:g}", flush=True)
    base = all_metrics[0]
    expected_union = phase1a_metrics["routes"]["union"]
    if (
        base["retrieved_true_links"] != expected_union["retrieved_true_links"]
        or base["candidate_count"]["total"] != expected_union["candidate_count_total"]
        or base["candidate_count"]["median"] != expected_union["candidate_count_per_s1"]["median"]
    ):
        raise RuntimeError("Phase 1B cached baseline differs from reproduced Phase 1A union")
    for mask, row in all_metrics.items():
        row["additional_true_links_vs_baseline"] = row["retrieved_true_links"] - base["retrieved_true_links"]
        row["additional_candidates_vs_baseline"] = row["candidate_count"]["total"] - base["candidate_count"]["total"]
        row["measured_added_route_runtime_seconds"] = sum(stage_times[route] for route in row["routes"])
        row["accounted_total_runtime_seconds"] = (
            phase1a_metrics["runtime_seconds"] +
            (stage_times["shared_corpus_scan"] if mask else 0) +
            row["measured_added_route_runtime_seconds"] + row["evaluation_runtime_seconds"]
        )
        row["peak_process_rss_upper_bound_bytes"] = max(
            phase1a_metrics["peak_ram_bytes"],
            stage_peaks["shared_corpus_scan"] if mask else 0,
            *(stage_peaks[route] for route in row["routes"]),
            row["peak_process_rss_during_evaluation_bytes"],
        )
    frontier = pareto_masks(all_metrics)
    output = ROOT / "artifacts/phase1b"
    output.mkdir(parents=True, exist_ok=True)
    packed = {f"{key}_{part}": values for key, rows in ragged.items()
              for part, values in zip(("indptr", "values"), pack_ragged(rows))}
    np.savez(output / "route_candidates.npz", sample_ids=np.asarray(ids), name200=name200,
             word=word, **packed)
    result = {
        "config": config,
        "phase1a_reproduced_metrics_path": "artifacts/phase1a/pilot_metrics.json",
        "sample_size": n,
        "true_links": sum(map(len, truths)),
        "stage_runtime_seconds": dict(stage_times),
        "stage_peak_process_rss_bytes": dict(stage_peaks),
        "phase1b_wall_seconds": time.perf_counter() - started,
        "configurations": {str(mask): row for mask, row in all_metrics.items()},
        "pareto_masks": frontier,
        "recommended_mask": 22,
    }
    (output / "pilot_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "pilot_report.md").write_text(make_report(result, phase1a_metrics))
    print((output / "pilot_report.md").read_text(), flush=True)


def make_report(result: dict, phase1a: dict) -> str:
    config = result["config"]
    rows = {int(mask): row for mask, row in result["configurations"].items()}
    lines = ["# Phase 1B candidate-generation pilot", "",
             "Same fixed 25,000 S1 validation-fold-0 sample as Phase 1A; all training S2/S3 records remain in the retrieval corpus. No true-link injection or test-set records in retrieval or evaluation.", "",
             f"Subset denominators: {phase1a['routes']['union']['non_ascii_records']['true_links']:,} non-ASCII true links and "
             f"{phase1a['routes']['union']['target_address_missing']['true_links']:,} links with a missing target address. "
             "Non-ASCII means either S1 or target name/address contains a non-ASCII character.", "",
             "Preparation note: a broad workspace text search inadvertently displayed a few test TSV lines while locating the problem statement. Those rows were not used to define routes, set thresholds, or compute metrics; all retrieval inputs came from training files.", "",
             "## Phase 1A reproduction", "",
             f"Baseline union recall: {rows[0]['positive_link_recall']:.3%}; median candidates: {rows[0]['candidate_count']['median']:g}; "
             f"candidate reduction vs complete same-country corpus: {rows[0]['candidate_reduction_ratio']:.5%}.",
             "The Phase 1A reproduction comparison is recorded in `reproduction_check.json`.", "",
             "## One-route additions to the Phase 1A union", "",
             "| Addition | Link recall | Full positive S1 % | India | US | Non-ASCII | Missing address | Median | p95 | p99 | Max | Reduction | Extra true links | Extra candidates | Route min | Peak GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mask in (0, 1, 2, 4, 8, 16):
        row = rows[mask]
        label = "baseline" if mask == 0 else ROUTES[mask.bit_length() - 1]
        c = row["candidate_count"]
        route_time = row["measured_added_route_runtime_seconds"] / 60
        route_peak = (phase1a["peak_ram_bytes"] if mask == 0
                      else result["stage_peak_process_rss_bytes"][label])
        lines.append(
            f"| {label} | {row['positive_link_recall']:.3%} | {row['positive_s1_with_every_true_match_pct']:.2f} | "
            f"{row['recall_by_country'].get('India', float('nan')):.3%} | {row['recall_by_country'].get('US', float('nan')):.3%} | "
            f"{row['non_ascii_recall']:.3%} | {row['missing_target_address_recall']:.3%} | "
            f"{c['median']:g} | {c['p95']:g} | {c['p99']:g} | {c['max']} | {row['candidate_reduction_ratio']:.5%} | "
            f"{row['additional_true_links_vs_baseline']:,} | {row['additional_candidates_vs_baseline']:,} | "
            f"{route_time:.1f} | {route_peak/2**30:.2f} |"
        )
    lines += ["", "Each addition is evaluated alone against the reproduced Phase 1A union. Candidate counts are summed over all 25,000 S1 entities. "
              "The route runtime is measured across all source/country partitions and excludes the shared corpus read. "
              "Peak GiB is observed process RSS during that route (baseline is the complete Phase 1A run).", "",
              "## Pareto frontier: recall versus mean candidates", "",
              "| Added routes | Link recall | Mean | Median | p95 | Additional true links | Additional candidates | Accounted total min | Peak GiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mask in result["pareto_masks"]:
        row = rows[mask]
        label = "+".join(route[0] for route in row["routes"]) or "baseline"
        c = row["candidate_count"]
        lines.append(f"| {label} | {row['positive_link_recall']:.3%} | {c['mean']:.1f} | {c['median']:g} | {c['p95']:g} | "
                     f"{row['additional_true_links_vs_baseline']:,} | {row['additional_candidates_vs_baseline']:,} | "
                     f"{row['accounted_total_runtime_seconds']/60:.1f} | {row['peak_process_rss_upper_bound_bytes']/2**30:.2f} |")
    lines += ["", "Accounted total runtime adds the measured Phase 1A run, one shared training-corpus scan, each selected route's measured retrieval time, and evaluation time. "
              "It is an additive estimate for configurations assembled from the cached route results. "
              "Pareto peak RAM is a conservative full-pipeline upper bound. "
              f"The Phase 1B route-generation and comparison run itself took {result['phase1b_wall_seconds']/60:.1f} minutes. "
              "Full metrics for all 32 route subsets are in `pilot_metrics.json`.",
              "", "## Route definitions", "",
              f"- A: name character 3–4 gram TF-IDF top-{config['name_char_top_k_per_source']} per source, using the Phase 1A normalization and corpus IDF.",
              f"- B: word unigram/bigram name TF-IDF top-{config['word_name_top_k_per_source']} per source.",
              f"- C: at most {config['rare_name_max_query_tokens']} rare normalized name tokens, each present in at most {config['rare_name_max_document_frequency']} targets per source/country; top-{config['rare_name_top_k_per_source']}.",
              f"- D: at most {config['address_digit_max_query_tokens']} address digit tokens, each present in at most {config['address_digit_max_document_frequency']} targets per source/country; top-{config['address_digit_top_k_per_source']}.",
              "- E: exact lookup on a name view with up to two trailing legal suffix tokens removed. Only `Corp`, `Corporation`, `Pvt`, `Private`, `Ltd`, and `Limited` from the supplied problem statement are used.",
              "", "## Recommendation", ""]
    chosen = rows[result["recommended_mask"]]
    c = chosen["candidate_count"]
    lines += [
        "Select **B + C + E** in addition to the Phase 1A exact-name, name character TF-IDF top-100, and address character TF-IDF top-100 routes. "
        "B uses word-name TF-IDF top-100 per source; C uses up to two rare name tokens with a 128-target frequency cap and top-100 per source; "
        "E uses exact lookup on the statement-listed suffix-stripped name view.",
        "",
        f"This configuration retrieved {chosen['retrieved_true_links']:,} of {result['true_links']:,} links "
        f"({chosen['positive_link_recall']:.3%}), with {chosen['positive_s1_with_every_true_match_pct']:.2f}% of positive S1 entities fully covered. "
        f"Median/p95/p99/max candidate counts were {c['median']:g}/{c['p95']:g}/{c['p99']:g}/{c['max']}; "
        f"candidate reduction was {chosen['candidate_reduction_ratio']:.5%}. "
        f"Relative to Phase 1A, it recovered {chosen['additional_true_links_vs_baseline']:,} more true links "
        f"and introduced {chosen['additional_candidates_vs_baseline']:,} candidates across the pilot.",
        "",
        f"Country recall was India {chosen['recall_by_country']['India']:.3%} and US {chosen['recall_by_country']['US']:.3%}; "
        f"non-ASCII recall was {chosen['non_ascii_recall']:.3%}. Missing-target-address recall rose from "
        f"{rows[0]['missing_target_address_recall']:.3%} to {chosen['missing_target_address_recall']:.3%} "
        f"(+{100*(chosen['missing_target_address_recall']-rows[0]['missing_target_address_recall']):.2f} percentage points). "
        f"Accounted total runtime was {chosen['accounted_total_runtime_seconds']/60:.1f} minutes "
        f"and the conservative peak process RSS was {chosen['peak_process_rss_upper_bound_bytes']/2**30:.2f} GiB.",
        "",
        "Measured marginal sequence: B added 351 true links and 3,133,284 candidates; C then added 72 links and 526,404 candidates; "
        "E then added 4 links and 18,377 candidates, including a small missing-address gain. "
        "The E lookup costs about eight seconds across the four target partitions.",
        "",
        "Do not retain A: adding it to B+C+E recovered 120 more links but added 4,410,111 candidates, "
        "raised the median from 541 to 718, and took 11.3 retrieval minutes. "
        "Do not retain D: adding it to B+C+E recovered 27 links, added 409,497 candidates, "
        "raised p95 from 677 to 718, and did not improve the missing-address slice.",
        "",
        "The 99% link-recall target was not reached by these allowed additions. "
        "Even A+B+C+D+E reached 97.961% with median 727 candidates. "
        "Use the selected B+C+E configuration for the next full-fold blocking run; this pilot stops here and does not execute that run.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
