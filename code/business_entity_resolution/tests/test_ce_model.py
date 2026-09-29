import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import ce_model

CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG = CODE_DIR.parents[1] / "configs" / "ce.json"
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "jasper", "kestrel",
         "lotus", "maple", "nimbus", "onyx", "pepper", "quartz", "raven", "sierra", "tango", "umber", "violet"]
SPLITS = {"train": range(1, 37), "validation": range(37, 49), "holdout": range(49, 61)}
HEAD = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(HEAD) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def make_corpus(directory: Path) -> None:
    s1, s2, s3 = [], [], []
    for i in range(1, 62):
        word = WORDS[i % len(WORDS)]
        tail = " ".join(WORDS[:i % 5])                                  # varied lengths, so sorting permutes rows
        address = f"{i}  main road, town{i % 4} {tail}"
        s1.append([f"S1-{i}", f"{word.title()}   Traders {tail}", address, "US"])
        s2.append([f"S2-{1000 + i}", f"{word.title()} Traders Pvt", address if i % 3 else "", "US"])
        s3.append([f"S3-{2000 + i}", "शर्मा ट्रेडर्स" if i == 38 else f"{word.upper()} TRADING CO", address, "India"])
        s3.append([f"S3-{3000 + i}", f"Other Shop {word}", address, "US"])
    write_tsv(directory / "train_source1.tsv", s1)
    write_tsv(directory / "train_source2.tsv", s2)
    write_tsv(directory / "train_source3.tsv", s3)


def candidates(i: int) -> list[tuple[str, int]]:
    rows = [(f"S2-{1000 + i}", 1), (f"S3-{3000 + i}", 0)]
    if i % 2 == 0:
        rows.insert(1, (f"S3-{2000 + i}", 1))
    rows += [(f"S2-{1000 + (i + k) % 60 + 1}", 0) for k in (3, 7, 11)]
    if i % 4 == 0:
        rows = [rows[-1]] + rows[:-1]                                   # a negative ranked first
    return rows


def make_pairs(root: Path, part_rows: int = 25) -> dict[str, int]:
    totals = {}
    for split, ids in SPLITS.items():
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        cols = {"s1_id": [], "t_id": [], "label": [], "filter_score": [], "filter_rank": []}
        s1 = {"s1_id": [], "truth_len": [], "n_cand": [], "retrieved_truth": []}
        for i in ids:
            rows = candidates(i) if i % 11 else []                      # some S1 have no candidates
            for rank, (target, label) in enumerate(rows):
                for name, value in zip(cols, (f"S1-{i}", target, label, 1.0 - rank / 10, rank)):
                    cols[name].append(value)
            positives = sum(label for _, label in rows)
            for name, value in zip(s1, (f"S1-{i}", 1 + (i % 2 == 0), len(rows), positives)):
                s1[name].append(value)
        n = len(cols["s1_id"])
        for part, start in enumerate(range(0, n, part_rows)):
            chunk = slice(start, start + part_rows)
            pq.write_table(pa.table({"s1_id": pa.array(cols["s1_id"][chunk], pa.string()),
                                     "t_id": pa.array(cols["t_id"][chunk], pa.string()),
                                     "label": pa.array(cols["label"][chunk], pa.int8()),
                                     "filter_score": pa.array(cols["filter_score"][chunk], pa.float32()),
                                     "filter_rank": pa.array(cols["filter_rank"][chunk], pa.int16())}),
                           out / f"part-{part:05d}.parquet")
        pq.write_table(pa.table({"s1_id": pa.array(s1["s1_id"], pa.string()),
                                 "truth_len": pa.array(s1["truth_len"], pa.int32()),
                                 "n_cand": pa.array(s1["n_cand"], pa.int16()),
                                 "retrieved_truth": pa.array(s1["retrieved_truth"], pa.int32())}), out / "s1.parquet")
        totals[split] = n
    return totals


def make_tiny_backbone(directory: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + list("|,.-") + [f"##{c}" for c in chars]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"), do_lower_case=True)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128, num_labels=1)
    model_dir = directory / "tiny"
    BertForSequenceClassification(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def read_rows(split_dir: Path) -> dict:
    table = pa.concat_tables([pq.read_table(p) for p in sorted(split_dir.glob("part-*.parquet"))])
    return table.to_pydict()


def test_select_rows_keeps_positives_hardest_and_random_negatives():
    group = np.array([0] * 8 + [1] * 3)
    label = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    rank = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])
    picked = ce_model.select_rows(group, label, rank, 2, 1, np.random.default_rng(0))
    assert {1, 6, 0, 2, 8, 9} <= set(picked.tolist())                    # positives + two best-ranked negatives
    assert len(picked) == 8 and (np.diff(picked) > 0).all()               # plus one random negative per S1
    assert set(picked[group[picked] == 0].tolist()) - {0, 1, 2, 6} <= {3, 4, 5, 7} and 10 in picked


def test_length_batches_cover_every_row_once_with_similar_lengths():
    rows = np.arange(1000, 2037)
    lengths = np.zeros(3000, np.int32)
    lengths[rows] = np.random.default_rng(1).integers(5, 200, len(rows))
    batches = ce_model.length_batches(rows, lengths, 8, np.random.default_rng(2))
    assert len(batches) == -(-len(rows) // 8) and sorted(np.concatenate(batches).tolist()) == rows.tolist()
    spread = np.mean([np.ptp(lengths[b]) for b in batches])
    assert spread < 0.2 * np.ptp(lengths[rows])                           # far less padding than random batches


def test_missing_ids_raise(tmp_path):
    make_corpus(tmp_path)
    s1, targets = ce_model.load_corpus("train", tmp_path)
    assert targets.positions(pa.array(["S3-2007", "S2-1001", "S3-2007"]), "t").tolist() == [
        targets.index.get_loc("S3-2007"), targets.index.get_loc("S2-1001"), targets.index.get_loc("S3-2007")]
    assert targets.texts[targets.index.get_loc("S2-1003")] == "Delta Traders Pvt"             # empty address
    assert s1.texts[s1.index.get_loc("S1-2")] == "Cedar Traders alpha bravo | 2 main road, town2 alpha bravo"
    with pytest.raises(KeyError, match="S2-404"):
        targets.positions(pa.array(["S2-1001", "S2-404"]), "t")


def test_ce_stage_all_end_to_end(tmp_path, monkeypatch):
    import torch

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(ce_model, "backbone_spec", lambda backbone, allowed: {"repo": backbone, "license": "mit", "revision": None})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)        # CPU fp32 even on a GPU image
    torch.manual_seed(0)
    data_dir, pairs, out = tmp_path / "data", tmp_path / "pairs", tmp_path / "out"
    make_corpus(data_dir)
    totals = make_pairs(pairs)
    cfg = json.loads(CONFIG.read_text()) | {
        "backbone": str(make_tiny_backbone(tmp_path)), "max_length": 48, "train_s1": 30, "neg_top": 2,
        "neg_random": 1, "batch_size": 16, "score_batch_size": 7, "epochs": 2, "lr": 1e-3, "log_every": 3}
    (tmp_path / "ce.json").write_text(json.dumps(cfg))
    ce_model.main(["--config", str(tmp_path / "ce.json"), "--pairs-root", str(pairs), "--out", str(out),
                   "--train-dir", str(data_dir), "--test-dir", str(tmp_path / "no_test"), "--stage", "all"])

    log = json.loads((out / "train_log.json").read_text())
    assert log["pairs"] > 0 and 0 < log["positives"] < log["pairs"] and log["s1"] == 30
    assert log["steps"] == 2 * -(-log["pairs"] // 16) and not log["stopped_early"]
    assert log["license"] == "mit" and log["backbone"] == cfg["backbone"]
    assert (out / "model" / "config.json").is_file()
    for split in ("validation", "holdout"):
        scores = np.load(out / "scores" / f"{split}.npy")
        assert scores.dtype == np.float32 and len(scores) == totals[split] and np.isfinite(scores).all()
    report = json.loads((out / "eval.json").read_text())
    assert set(report) == {"validation", "holdout"} and report["validation"]["rows"] == totals["validation"]
    assert report["validation"]["auc"] is not None

    trained = np.load(out / "scores" / "validation.npy")

    # Row alignment needs logits that differ between rows, which the briefly trained tiny model barely gives: swap in a
    # randomly initialised model with a wide weight scale, rescore with another batch size, and compare every logit
    # with that row's own pair scored alone.
    tokenizer, model = ce_model.load_model(str(out / "model"), None, torch.device("cpu"))
    assert np.allclose(expected_logits(model, tokenizer, data_dir, pairs / "validation"), trained, atol=1e-4)
    torch.manual_seed(1)
    wild = type(model)(model.config.__class__.from_dict(model.config.to_dict() | {"initializer_range": 0.5}))
    wild.save_pretrained(out / "model")
    cfg["score_batch_size"] = 3
    (tmp_path / "ce.json").write_text(json.dumps(cfg))
    ce_model.main(["--config", str(tmp_path / "ce.json"), "--pairs-root", str(pairs), "--out", str(out),
                   "--train-dir", str(data_dir), "--stage", "score", "--splits", "validation"])
    validation = np.load(out / "scores" / "validation.npy")
    tokenizer, model = ce_model.load_model(str(out / "model"), None, torch.device("cpu"))
    expected = expected_logits(model, tokenizer, data_dir, pairs / "validation")
    assert validation.std() > 0.05 and len(np.unique(np.round(validation, 3))) > len(validation) // 3
    assert np.allclose(validation, expected, atol=1e-4)

    # Batch size and rank slicing change nothing.
    s1_texts, t_texts = ce_model.load_corpus("train", data_dir)
    split_dir = pairs / "validation"
    big, total = ce_model.score_split(model, tokenizer, torch.device("cpu"), split_dir, s1_texts, t_texts, 0, 1, 64, 48)
    thirds = [ce_model.score_split(model, tokenizer, torch.device("cpu"), split_dir, s1_texts, t_texts, r, 3, 5, 48)[0]
              for r in range(3)]
    assert total == totals["validation"]
    assert np.allclose(big, validation, atol=1e-5) and np.allclose(np.concatenate(thirds), validation, atol=1e-5)


def expected_logits(model, tokenizer, data_dir: Path, split_dir: Path) -> np.ndarray:
    """Each row's pair scored alone, in file order (texts built independently of ``ce_model``)."""
    import torch

    texts = {}
    for source in (1, 2, 3):
        lines = (data_dir / f"train_source{source}.tsv").read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            entity, name, address, _ = line.split("\t")
            name, address = " ".join(name.split()), " ".join(address.split())
            texts[entity] = f"{name} | {address}" if address else name
    rows = read_rows(split_dir)
    model.eval()
    out = []
    with torch.inference_mode():
        for s1_id, t_id in zip(rows["s1_id"], rows["t_id"]):
            batch = tokenizer([texts[s1_id]], [texts[t_id]], truncation="longest_first", max_length=48,
                              return_tensors="pt")
            out.append(float(model(**batch).logits[0, 0]))
    return np.asarray(out, np.float32)


def test_failed_rank_exits_without_process_group_teardown(tmp_path, monkeypatch, capsys):
    class TwoRanks:
        world, rank, closed = 2, 1, False

        def close(self):
            self.closed = True

    def fail(*args):
        raise KeyError("validation t_id: 1 ids are not in the source files, e.g. ['S2-404']")

    def hard_exit(code):
        raise SystemExit(code)

    dist = TwoRanks()
    (tmp_path / "pairs" / "validation").mkdir(parents=True)
    argv = ["--config", str(CONFIG), "--pairs-root", str(tmp_path / "pairs"), "--out", str(tmp_path / "out"),
            "--stage", "score"]
    monkeypatch.setattr(ce_model, "Dist", lambda: dist)
    monkeypatch.setattr(ce_model, "score", fail)
    monkeypatch.setattr(ce_model.os, "_exit", hard_exit)
    with pytest.raises(SystemExit) as stop:
        ce_model.main(argv)
    assert stop.value.code == 1 and not dist.closed and "S2-404" in capsys.readouterr().err
    dist.world = 1                                                        # one process: the error propagates
    with pytest.raises(KeyError):
        ce_model.main(argv)
    dist.world = 2
    monkeypatch.setattr(ce_model, "score", lambda *args: {})
    ce_model.main(argv)
    assert dist.closed


def test_kernel_finds_split_pairs_code_and_data(tmp_path):
    spec = importlib.util.spec_from_file_location("ce_kernel", CODE_DIR / "kaggle_ce" / "ce_kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    inputs = tmp_path / "input"
    src = inputs / "ber-code" / "repo" / "code" / "business_entity_resolution" / "src"
    src.mkdir(parents=True)
    (src / "ce_model.py").write_text("")
    for dataset, split in (("pairs-fold0", "validation"), ("pairs-fold0", "train"), ("pairs-test", "test")):
        (inputs / dataset / "top40" / split).mkdir(parents=True)
        (inputs / dataset / "top40" / split / "s1.parquet").write_text("")
        (inputs / dataset / "top40" / split / "part-00000.parquet").write_text("")
    for kind in ("train", "test"):
        (inputs / "tsvs" / kind).mkdir(parents=True)
        (inputs / "tsvs" / kind / f"{kind}_source2.tsv").write_text("")
    assert kernel.find_repo(inputs) == inputs / "ber-code" / "repo"
    found = kernel.link_pairs(inputs, tmp_path / "working" / "pairs")
    assert set(found) == {"train", "validation", "test"}
    assert (tmp_path / "working" / "pairs" / "test" / "s1.parquet").is_file()
    linked = (tmp_path / "working" / "pairs" / "train").resolve()
    assert linked == (inputs / "pairs-fold0" / "top40" / "train").resolve()
    assert kernel.find_dir(inputs, "test_source2.tsv") == inputs / "tsvs" / "test"
    cmd = kernel.ce_command(Path("/w/ber"), Path("/w/pairs"), Path("/w/ce"), Path("/d/train"), None, 2)
    assert cmd[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    assert cmd[5:7] == ["-m", "src.ce_model"] and "--test-dir" not in cmd and cmd[cmd.index("--stage") + 1] == "all"
    assert "--backbone" not in cmd
    warm = kernel.ce_command(Path("/w/ber"), Path("/w/pairs"), Path("/w/ce"), None, None, 2, Path("/tmp/warm/model"))
    assert warm[warm.index("--backbone") + 1] == "/tmp/warm/model"
    assert kernel.find_previous_output(inputs) is None
    flat = inputs / "ber-ce-rerank-model"                                 # a model dataset: config.json + train_log.json
    flat.mkdir()
    (flat / "config.json").write_text("{}")
    (flat / "train_log.json").write_text("{}")
    assert kernel.find_previous_output(inputs) == (flat, flat / "train_log.json")
    nested = inputs / "a-earlier-output" / "ce"                           # an earlier kernel output: model/ + log
    (nested / "model").mkdir(parents=True)
    (nested / "model" / "config.json").write_text("{}")
    (nested / "train_log.json").write_text("{}")
    assert kernel.find_previous_output(inputs) == (nested / "model", nested / "train_log.json")


def test_choose_test_k_fits_the_budget(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.ce_model import choose_test_k

    split = tmp_path / "test"
    split.mkdir()
    ranks = np.tile(np.arange(40, dtype=np.int16), 1000)                  # 1,000 S1 x 40 candidates
    for part, start in enumerate(range(0, len(ranks), 15000)):
        pq.write_table(pa.table({"filter_rank": pa.array(ranks[start:start + 15000], pa.int16())}),
                       split / f"part-{part:05d}.parquet")
    cfg = {"test_budget_minutes": 1, "min_k": 10, "max_k": 40}
    assert choose_test_k(cfg, split, rows_per_s=25000 / 60)["k"] == 25       # 25 rows per S1 fit exactly
    assert choose_test_k(cfg, split, rows_per_s=1e9)["k"] == 40
    assert choose_test_k(cfg, split, rows_per_s=1.0)["k"] == 10              # never below min_k


def test_sharded_test_scoring_resumes_respects_the_deadline_and_matches_one_pass(tmp_path, monkeypatch):
    import shutil
    import time

    import torch

    for name in ("WORLD_SIZE", "RANK", "CE_TEST_SHARDS", "CE_TEST_ONLY", "CE_DEADLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data_dir, test_dir, pairs, out = tmp_path / "data", tmp_path / "test", tmp_path / "pairs", tmp_path / "out"
    make_corpus(data_dir)
    for source in (1, 2, 3):
        test_dir.mkdir(exist_ok=True)
        shutil.copy(data_dir / f"train_source{source}.tsv", test_dir / f"test_source{source}.tsv")
    totals = make_pairs(pairs, part_rows=13)
    shutil.copytree(pairs / "validation", pairs / "test")
    torch.manual_seed(3)
    tiny = make_tiny_backbone(tmp_path)
    tokenizer, model = ce_model.load_model(str(tiny), None, torch.device("cpu"))
    wild = type(model)(model.config.__class__.from_dict(model.config.to_dict() | {"initializer_range": 0.5}))
    wild.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    cfg = json.loads(CONFIG.read_text()) | {"max_length": 48, "score_batch_size": 4, "test_budget_minutes": None}
    (tmp_path / "ce.json").write_text(json.dumps(cfg))
    args = ["--config", str(tmp_path / "ce.json"), "--pairs-root", str(pairs), "--out", str(out), "--train-dir",
            str(data_dir), "--test-dir", str(test_dir), "--stage", "score", "--splits", "test"]
    scores = out / "scores"

    monkeypatch.setattr(ce_model, "MAX_TOKENIZER_THREADS", 1)
    ce_model.main(args)                                                   # one pass, one tokenizer thread
    reference = np.load(scores / "test.npy")
    assert len(reference) == totals["validation"] and reference.std() > 0.05
    (scores / "test.npy").unlink()

    monkeypatch.setattr(ce_model, "MAX_TOKENIZER_THREADS", 4)
    monkeypatch.setattr(ce_model.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("CE_TEST_SHARDS", "3")
    monkeypatch.setenv("CE_TEST_ONLY", "0,2")                             # another machine takes shard 1
    ce_model.main(args)
    assert [p.name for p in sorted(scores.glob("test.shard*"))] == ["test.shard0of3.npy", "test.shard2of3.npy"]
    assert not (scores / "test.npy").exists() and not list(scores.glob("test.shard*.part*.npy"))

    monkeypatch.delenv("CE_TEST_ONLY")
    monkeypatch.setenv("CE_DEADLINE", str(time.time() - 1))               # past: a known rate means no shard fits
    ce_model.main(args)
    assert not (scores / "test.shard1of3.npy").exists() and not (scores / "test.npy").exists()

    monkeypatch.delenv("CE_DEADLINE")
    ce_model.main(args)                                                   # resume: only shard 1 is scored, then merged
    merged = np.load(scores / "test.npy")
    bounds = [ce_model.rank_bounds(len(reference), 3, i) for i in range(3)]
    assert [len(np.load(scores / f"test.shard{i}of3.npy")) for i in range(3)] == [hi - lo for lo, hi in bounds]
    assert np.allclose(merged, reference, atol=1e-5)
    log = json.loads((out / "score_log.json").read_text())
    assert {"test.shard0of3", "test.shard1of3", "test.shard2of3"} <= set(log) and log["test"]["shards"] == 3


def test_accumulated_steps_and_the_budget_fitted_schedule(tmp_path, monkeypatch):
    import torch

    for name in ("WORLD_SIZE", "RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ce_model, "backbone_spec", lambda backbone, allowed: {"repo": backbone, "license": "mit", "revision": None})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data_dir, pairs = tmp_path / "data", tmp_path / "pairs"
    make_corpus(data_dir)
    make_pairs(pairs)
    base = json.loads(CONFIG.read_text()) | {"backbone": str(make_tiny_backbone(tmp_path)), "max_length": 48,
                                             "train_s1": 30, "neg_top": 2, "neg_random": 1, "batch_size": 8,
                                             "epochs": 1, "lr": 1e-3, "log_every": 1}
    logs = {}
    for name, extra, fit_at in (("acc", {"accumulate": 3}, 100), ("fit", {"accumulate": 2, "train_minutes": 60}, 2)):
        monkeypatch.setattr(ce_model, "FIT_SCHEDULE_STEPS", fit_at)
        (tmp_path / f"{name}.json").write_text(json.dumps(base | extra))
        ce_model.main(["--config", str(tmp_path / f"{name}.json"), "--pairs-root", str(pairs), "--out",
                       str(tmp_path / name), "--train-dir", str(data_dir), "--stage", "train"])
        logs[name] = json.loads((tmp_path / name / "train_log.json").read_text())
    batches = -(-logs["acc"]["pairs"] // 8)
    assert logs["acc"]["steps"] == logs["acc"]["planned_steps"] == -(-batches // 3) and logs["acc"]["accumulate"] == 3
    assert logs["acc"]["pairs_seen"] == logs["acc"]["pairs"]              # every pair once, the last window included
    fit = logs["fit"]                                                     # an ample budget keeps the whole plan
    assert fit["fitted_steps"] == fit["planned_steps"] == fit["steps"] == -(-batches // 2) and not fit["stopped_early"]


def test_warm_start_backbone_inherits_the_parent_license(tmp_path):
    model = tmp_path / "ce" / "model"
    model.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ce_model.backbone_spec(str(model), ["mit"])
    (tmp_path / "ce" / "train_log.json").write_text(json.dumps({"license": "mit", "backbone": "intfloat/x", "revision": "abc"}))
    spec = ce_model.backbone_spec(str(model), ["mit", "apache-2.0"])
    assert spec == {"repo": str(model), "license": "mit", "revision": None, "warm_start_from": "intfloat/x",
                    "warm_start_revision": "abc"}
    with pytest.raises(RuntimeError):
        ce_model.backbone_spec(str(model), ["apache-2.0"])


def test_kernel_links_a_train_only_pairs_dataset(tmp_path):
    spec = importlib.util.spec_from_file_location("ce_kernel", CODE_DIR / "kaggle_ce" / "ce_kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    inputs = tmp_path / "input"
    for dataset, split in (("ber-union-train", "train"), ("ber-dense-new", "validation"), ("ber-dense-new", "holdout"),
                           ("ber-dense-new", "test")):
        (inputs / dataset / split).mkdir(parents=True)
        (inputs / dataset / split / "s1.parquet").write_text("")
        (inputs / dataset / split / "part-00000.parquet").write_text("")
    (inputs / "ber-union-train" / "notes").mkdir()
    found = kernel.link_pairs(inputs, tmp_path / "pairs")
    assert set(found) == {"train", "validation", "holdout", "test"}
    assert (tmp_path / "pairs" / "train").resolve() == (inputs / "ber-union-train" / "train").resolve()
    assert (tmp_path / "pairs" / "validation").resolve() == (inputs / "ber-dense-new" / "validation").resolve()
