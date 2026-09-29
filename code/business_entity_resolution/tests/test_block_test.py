import copy
import shutil

import numpy as np
import pytest

from src.block_test import ordered_test_ids, run_test, shard_unions
from src.evaluate_phase1c import Workspace
from tests.test_phase1c_evaluation import make_dataset, reference_unions, shard_hashes, small_config


@pytest.fixture()
def test_split(tmp_path):
    """Synthetic 'test' split: copies of the synthetic sources, with no ground truth or folds nearby."""
    make_dataset(tmp_path, n_s1=60, seed=5)
    test_dir = tmp_path / "test_only"
    test_dir.mkdir()
    for i in (1, 2, 3):
        shutil.copy(tmp_path / f"student_resource/dataset/train/train_source{i}.tsv", test_dir / f"test_source{i}.tsv")
    return tmp_path, test_dir


def no_training_config():
    config = small_config()
    config["paths"].update(train_dir="does/not/exist/train", folds="does/not/exist/folds.tsv")
    return config


def test_test_blocking_matches_reference_without_labels_and_resumes(test_split):
    root, test_dir = test_split
    config = no_training_config()          # any training/fold/truth read would fail loudly
    work = root / "work_test"
    ids = ordered_test_ids(test_dir, config["algorithm"]["seed"])
    n_shards = (len(ids) + 5) // 6
    stopped = run_test(config, root, test_dir, f"0-{n_shards - 1}", work, root / "out", stop_after_tasks=7)
    assert stopped["status"] == "stopped"
    for source in (2, 3):
        run_test(config, root, test_dir, f"0-{n_shards - 1}", work, root / "out", only_sources=[source])
    result = run_test(config, root, test_dir, f"0-{n_shards - 1}", work, root / "out", finalize=True)
    assert result["status"] == "complete" and result["manifest"]["labels_read"] is False
    assert "Atlantis" in result["manifest"]["s1_by_country"]
    reference_config = small_config()
    reference = reference_unions(root, reference_config, ids)   # original Phase 1A/1B route functions
    ws = Workspace(work)
    positions_by_shard = {}
    for position, entity_id in enumerate(ids):
        positions_by_shard.setdefault(position // 6, []).append(position)
    for shard, positions in positions_by_shard.items():
        by_key = {}
        for p in positions:
            by_key.setdefault(ids[p], None)
        # Rebuild the per-country grouping exactly as run_test does.
        from src.block_test import load_test_queries
        queries = load_test_queries(test_dir, ids, positions)
        grouped = {}
        for p in positions:
            grouped.setdefault(queries["country_key"][p], []).append(p)
        for position, uids in shard_unions(ws, shard, {k: np.asarray(v) for k, v in grouped.items()}, 60):
            assert set(uids.tolist()) == reference[ids[position]], ids[position]
    assert sum(result["counts"].values()) == sum(len(v) for v in reference.values())


def test_split_ranges_equal_single_range(test_split):
    root, test_dir = test_split
    config = no_training_config()
    ids = ordered_test_ids(test_dir, config["algorithm"]["seed"])
    last = (len(ids) + 5) // 6 - 1
    one = root / "work_one"
    run_test(config, root, test_dir, f"0-{last}", one, root / "out1")
    split = root / "work_split"
    run_test(config, root, test_dir, "0-1", split, root / "out2")
    run_test(copy.deepcopy(config), root, test_dir, f"2-{last}", split, root / "out2")
    assert shard_hashes(one) == shard_hashes(split)
    with pytest.raises(ValueError, match="outside"):
        run_test(config, root, test_dir, f"0-{last + 1}", split, root / "out2")
