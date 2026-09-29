import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.blocking import encode_id, exact_name_lookup, sparse_topk_rows
from src.evaluate_phase1c import (
    ROUTES, Workspace, candidate_union, fill_topk, main, rank_route, read_jsonl, run, summarize,
)
from src.normalization import normalize_text
from src.phase1b_routes import (
    name_tokens, rare_token_lookup, stripped_exact_lookup, word_tfidf_topk_rows,
)

WORDS = ["acme", "blue", "sky", "rama", "kirana", "mart", "traders", "global", "sun", "star",
         "corp", "ltd", "pvt", "private", "limited", "corporation", "foods", "tech", "zen", "om"]
STREETS = ["main st", "mg road", "elm st", "park ave", "station rd", "rue de la paix"]
COUNTRIES = ["US", "India", "Atlantis"]


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def make_dataset(root: Path, n_s1: int = 40, seed: int = 7) -> dict:
    rng = random.Random(seed)
    s1, s2, s3, truth, folds = [], [], [], [], []
    next_id = {2: 100, 3: 100}

    def add_target(source: int, name: str, address: str, country: str) -> str:
        next_id[source] += rng.randint(1, 5)
        entity_id = f"S{source}-{next_id[source]}"
        (s2 if source == 2 else s3).append([entity_id, name, address, country])
        return entity_id

    for i in range(n_s1):
        country = COUNTRIES[i % 3]
        name = " ".join(rng.sample(WORDS, rng.randint(1, 3)))
        address = f"{rng.randint(1, 99)} {rng.choice(STREETS)}"
        entity_id = f"S1-{1000 + i}"
        s1.append([entity_id, name.upper() if i % 4 == 0 else name, address, country])
        matches = []
        for _ in range(rng.choice([0, 1, 2, 3])):
            source = rng.choice([2, 3])
            noisy = name + " ltd" if rng.random() < 0.3 else name
            matches.append(add_target(source, noisy, address if rng.random() < 0.7 else "", country))
        if i == 4:
            # Same text as a matched target but a distinct record; must survive as its own ID.
            add_target(2, name, address, country)
        if i == 8:
            # Cross-country gold link: unreachable by dynamic-country blocking and never injected.
            matches.append(add_target(3, name, address, "Elsewhere"))
        truth.append([entity_id, ",".join(matches)])
        folds.append([entity_id, str(i % 2)])
    for _ in range(60):
        country = rng.choice(COUNTRIES)
        add_target(rng.choice([2, 3]), " ".join(rng.sample(WORDS, 2)),
                   f"{rng.randint(1, 99)} {rng.choice(STREETS)}", country)
    train = root / "student_resource/dataset/train"
    header = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(train / "train_source1.tsv", header, s1)
    write_tsv(train / "train_source2.tsv", header, s2)
    write_tsv(train / "train_source3.tsv", header, s3)
    write_tsv(train / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], truth)
    write_tsv(root / "artifacts/folds.tsv", ["source1_entity_id", "fold"], folds)
    return {"s1": s1, "s2": s2, "s3": s3, "truth": truth}


def small_config() -> dict:
    config = json.loads((Path(__file__).resolve().parents[3] / "configs/phase1c_fold0.json").read_text())
    config = copy.deepcopy(config)
    config["algorithm"].update(shard_size=6, char_hash_features=4096, char_max_document_frequency=0.5,
                               char_top_k_per_source=4, word_hash_features=4096, word_max_document_frequency=0.9,
                               word_top_k_per_source=3, rare_name_max_document_frequency=4,
                               rare_name_top_k_per_source=5)
    config["operational"].update(query_batch_size=4, threads=1, benchmark_shards=2,
                                 min_available_memory_gib=0)
    return config


def reference_unions(root: Path, config: dict, ids: list[str]) -> dict[str, set[int]]:
    """Brute-force pilot-style union using the original Phase 1A/1B route functions."""
    algo = config["algorithm"]
    train = root / config["paths"]["train_dir"]
    with (train / "train_source1.tsv").open(encoding="utf-8", newline="") as file:
        s1 = {row["entity_id"]: row for row in csv.DictReader(file, delimiter="\t")}
    unions = {entity_id: set() for entity_id in ids}
    by_country = {}
    for entity_id in ids:
        by_country.setdefault(normalize_text(s1[entity_id]["country"]), []).append(entity_id)
    for source in (2, 3):
        with (train / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        for country, members in by_country.items():
            part = [row for row in rows if normalize_text(row["country"]) == country]
            tids = [encode_id(row["entity_id"]) for row in part]
            names = [normalize_text(row["business_name"]) for row in part]
            addresses = [normalize_text(row["business_address"]) for row in part]
            qnames = [normalize_text(s1[m]["business_name"]) for m in members]
            qaddresses = [normalize_text(s1[m]["business_address"]) for m in members]
            route_rows = [
                exact_name_lookup(qnames, names, tids),
                [[v for v, _ in hits] for hits in sparse_topk_rows(
                    qnames, names, tids, k=algo["char_top_k_per_source"], n_features=algo["char_hash_features"],
                    ngram_range=tuple(algo["char_ngram_range"]), max_document_frequency=algo["char_max_document_frequency"],
                    query_batch_size=2, threads=1)],
                [[v for v, _ in hits] for hits in sparse_topk_rows(
                    qaddresses, addresses, tids, k=algo["char_top_k_per_source"], n_features=algo["char_hash_features"],
                    ngram_range=tuple(algo["char_ngram_range"]), max_document_frequency=algo["char_max_document_frequency"],
                    query_batch_size=2, threads=1)],
                [[v for v, _ in hits] for hits in word_tfidf_topk_rows(
                    qnames, names, tids, k=algo["word_top_k_per_source"], n_features=algo["word_hash_features"],
                    max_document_frequency=algo["word_max_document_frequency"], query_batch_size=2, threads=1)],
                rare_token_lookup(qnames, names, tids, tokenize=name_tokens,
                                  max_document_frequency=algo["rare_name_max_document_frequency"],
                                  max_query_tokens=algo["rare_name_max_query_tokens"],
                                  k=algo["rare_name_top_k_per_source"]),
                stripped_exact_lookup(qnames, names, tids),
            ]
            for rows_for_route in route_rows:
                for member, hits in zip(members, rows_for_route):
                    unions[member].update(hits)
    return unions


def union_shards(work: Path) -> dict[int, set[int]]:
    result = {}
    for path in sorted((work / "union").glob("shard*.npz")):
        with np.load(path) as data:
            for i, position in enumerate(data["positions"]):
                left, right = data["indptr"][i:i + 2]
                result[int(position)] = {int(v) for v in data["ids"][left:right]}
    return result


def shard_hashes(work: Path) -> dict[str, str]:
    return {str(path.relative_to(work)): json.loads(path.read_text())["content_sha256"]
            for path in sorted((work / "shards").rglob("*.json"))}


@pytest.fixture()
def dataset(tmp_path):
    make_dataset(tmp_path)
    return tmp_path


def test_fill_topk_orders_by_score_then_target_id():
    product = csr_matrix((np.asarray([0.5, 0.9, 0.5, 0.2], dtype=np.float32),
                          np.asarray([0, 1, 2, 0]), np.asarray([0, 3, 4])), shape=(2, 3))
    target_ids = np.asarray([30, 10, 20], dtype=np.uint32)
    ids = np.zeros((2, 3), np.uint32)
    scores = np.zeros((2, 3), np.float32)
    counts = np.zeros(2, np.uint16)
    fill_topk(product, target_ids, ids, scores, counts)
    assert ids[0].tolist() == [10, 20, 30]
    assert scores[0].tolist() == pytest.approx([0.9, 0.5, 0.5])
    assert ids[1, :counts[1]].tolist() == [30]


def test_candidate_union_keeps_distinct_ids_and_route_bits():
    ranked = [rank_route([(np.asarray([8, 4], np.uint32), None), (np.asarray([5], np.uint32), None)]),
              rank_route([(np.asarray([4, 6], np.uint32), np.asarray([0.2, 0.9], np.float32)),
                          (np.zeros(0, np.uint32), np.zeros(0, np.float32))])]
    assert ranked[0].tolist() == [4, 5, 8]
    assert ranked[1].tolist() == [6, 4]
    uids, bits, rank, _ = candidate_union(ranked, 60)
    assert uids.tolist() == [4, 5, 6, 8]
    assert bits.tolist() == [3, 1, 2, 1]
    assert rank[0] == 0  # the ID retrieved by both routes ranks first


def test_summarize_singleton_convention_and_reduction():
    ev = {
        "country": ["X", "Y"], "truth_len": np.asarray([2, 0]), "pool": np.asarray([10, 20]),
        "link_s1": np.asarray([0, 0]), "link_ids": np.asarray([2, 5]), "link_bits": np.asarray([1, 0], np.uint8),
        "link_rrf_rank": np.asarray([0, 99]), "link_non_ascii": np.asarray([False, True]),
        "link_missing": np.asarray([True, False]), "link_cross_country": np.asarray([False, False]),
        "counts": {"S2": np.asarray([1, 1]), "S3": np.asarray([0, 0])},
    }
    result = summarize(ev, 1, np.asarray([2, 0]), detailed=True, recall_at_k=(1,))
    assert result["positive_link_recall"] == 0.5
    assert result["positive_s1_with_every_true_match_pct"] == 0
    assert result["all_s1_every_match_pct"] == 50
    assert result["zero_candidate_rate"] == 0.5
    assert result["candidate_reduction_ratio"] == 1 - 2 / 30
    assert result["recall_by_source"] == {"S2": 1.0, "S3": 0.0}
    assert result["non_ascii_recall"] == 0 and result["missing_target_address_recall"] == 1
    assert result["by_match_count_bucket"]["0"]["every_match_pct"] == 100


def test_end_to_end_matches_reference_union_without_gold_injection(dataset):
    config = small_config()
    result = run(config, dataset, "full", require_gate=False, emit_union=True)
    assert result["status"] == "complete"
    ids = result["_ordered"]
    reference = reference_unions(dataset, config, ids)
    observed = union_shards(dataset / config["paths"]["work_dir"])
    assert set(observed) == set(range(len(ids)))
    for position, entity_id in enumerate(ids):
        assert observed[position] == reference[entity_id], entity_id
    selected = result["metrics"]["selected_bce"]
    # Dynamic third country is evaluated like any other label.
    assert "Atlantis" in selected["recall_by_country"]
    # The cross-country gold link is reported as missed, not injected.
    assert selected["cross_country_true_links"] == 1
    assert selected["cross_country_true_links_retrieved"] == 0
    assert selected["retrieved_true_links"] < selected["true_links"]
    ev = result["_ev"]
    total_hits = sum(len(observed[p] & set(ev["link_ids"][ev["link_s1"] == s].tolist()))
                     for s, p in enumerate(ev["positions"]))
    assert total_hits == selected["retrieved_true_links"]
    assert selected["candidate_count"]["total"] == sum(map(len, observed.values()))


def test_resume_after_stop_matches_uninterrupted_run(dataset, tmp_path_factory):
    config = small_config()
    interrupted = dataset / "work_interrupted"
    first = run(config, dataset, "benchmark", stop_after_tasks=5, work_dir=interrupted)
    assert first["status"] == "stopped"
    partial = shard_hashes(interrupted)
    assert len(partial) == 5
    resumed = run(config, dataset, "benchmark", work_dir=interrupted)
    assert resumed["status"] == "complete"
    invocations = read_jsonl(interrupted / "invocations.jsonl")
    assert [row["status"] for row in invocations] == ["stopped", "complete"]
    assert invocations[1]["tasks_skipped"] == 5
    clean = dataset / "work_clean"
    config_other_batch = copy.deepcopy(config)
    config_other_batch["operational"].update(query_batch_size=1, threads=2)
    uninterrupted = run(config_other_batch, dataset, "benchmark", work_dir=clean)
    assert shard_hashes(interrupted) == shard_hashes(clean)
    assert {k: v for k, v in partial.items()} == {k: shard_hashes(clean)[k] for k in partial}
    assert resumed["metrics"] == uninterrupted["metrics"]
    # A third invocation skips every partition scan.
    again = run(config, dataset, "benchmark", work_dir=interrupted)
    assert again["invocation"]["partitions_scanned"] == []
    assert again["metrics"] == resumed["metrics"]


def test_work_dir_rejects_changed_algorithm(dataset):
    config = small_config()
    work = dataset / "work_state"
    run(config, dataset, "pilot", stop_after_tasks=1, work_dir=work)
    changed = copy.deepcopy(config)
    changed["algorithm"]["char_top_k_per_source"] = 5
    with pytest.raises(RuntimeError, match="different config"):
        run(changed, dataset, "pilot", work_dir=work)


def test_frozen_route_set_and_gate_are_enforced(dataset):
    config = small_config()
    broken = copy.deepcopy(config)
    broken["algorithm"]["routes"] = list(ROUTES) + ["address_digit"]
    with pytest.raises(ValueError, match="frozen"):
        run(broken, dataset, "pilot")
    with pytest.raises(RuntimeError, match="requires pilot_reproduction"):
        run(config, dataset, "full")
    with pytest.raises(SystemExit):
        main(["--config", "configs/phase1c_fold0.json"])  # --scope is mandatory


def test_workspace_task_layout_is_per_source_country_route_shard(tmp_path):
    ws = Workspace(tmp_path)
    path = ws.task_path(3, "india-abcd1234", "rare_name", 7)
    assert path.relative_to(tmp_path).parts == ("shards", "S3", "india-abcd1234", "rare_name", "shard007.npz")
    assert not ws.task_done(3, "india-abcd1234", "rare_name", 7)


def test_partition_workers_then_unfiltered_evaluation_match_single_process(dataset):
    config = small_config()
    workers = dataset / "work_workers"
    for source in (2, 3):
        for routes in (["address_char"], [r for r in ROUTES if r != "address_char"]):
            result = run(config, dataset, "full", require_gate=False, work_dir=workers,
                         only_sources=[source], only_routes=routes)
            assert result["status"] == "generated" and "metrics" not in result
    final = run(config, dataset, "full", require_gate=False, work_dir=workers)
    assert final["invocation"]["tasks_completed"] == 0
    assert final["invocation"]["partitions_scanned"] == []
    single = run(config, dataset, "full", require_gate=False, work_dir=dataset / "work_single")
    assert shard_hashes(workers) == shard_hashes(dataset / "work_single")
    assert final["metrics"] == single["metrics"]
    with pytest.raises(ValueError, match="Unknown routes"):
        run(config, dataset, "full", require_gate=False, work_dir=workers, only_routes=["address_digit"])


def test_memory_guard_stops_cleanly_and_resumes(dataset, monkeypatch):
    import psutil

    config = small_config()
    config["operational"].update(min_available_memory_gib=1.0, max_memory_wait_seconds=0)
    work = dataset / "work_memory"
    low = type("Memory", (), {"available": 0})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: low)
    stopped = run(config, dataset, "pilot", work_dir=work)
    assert stopped["status"] == "stopped" and stopped["invocation"]["tasks_completed"] == 0
    monkeypatch.undo()
    resumed = run(config, dataset, "pilot", work_dir=work)
    assert resumed["status"] == "complete"


def test_platform_tolerance_bounds_float_near_tie_effects():
    from src.evaluate_phase1c import platform_tolerance

    limits = {"max_retrieved_link_diff": 10, "max_every_match_pp_diff": 0.05,
              "max_candidate_total_rel_diff": 1e-4, "require_equal_percentiles": True}
    counts = {"mean": 544.76, "median": 541.0, "p95": 677.0, "p99": 729.0, "max": 791, "total": 13618996}
    expected = {"retrieved_true_links": 84465, "positive_s1_with_every_true_match_pct": 93.5, "candidate_count": counts}
    colab = {"retrieved_true_links": 84464, "positive_s1_with_every_true_match_pct": 93.4957627118644,
             "candidate_count": counts | {"total": 13618606}}
    assert platform_tolerance({"expected": expected, "observed": colab}, limits)["within"]
    drifted = colab | {"retrieved_true_links": 84400}
    assert not platform_tolerance({"expected": expected, "observed": drifted}, limits)["within"]
    shifted = colab | {"candidate_count": counts | {"p95": 690.0}}
    assert not platform_tolerance({"expected": expected, "observed": shifted}, limits)["within"]
    assert not platform_tolerance({"expected": expected, "observed": colab}, {})["within"]
