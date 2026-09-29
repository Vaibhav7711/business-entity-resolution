import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import bi_encoder

CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG = CODE_DIR.parents[1] / "configs" / "bi.json"
HEAD = ["entity_id", "business_name", "business_address", "country"]
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "jasper", "kestrel",
         "lotus", "maple", "nimbus", "onyx", "pepper", "quartz", "raven", "sierra", "tango", "umber", "violet"]
N_TRAIN, K = 80, 5


def write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(HEAD) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def fold(i: int) -> int:
    return i % 5


def make_train(directory: Path) -> dict[str, list[str]]:
    """S1-i with gold S2-(1000+i) (the identical record when i % 7 == 0) and S3-(2000+i) (Devanagari for India); S1
    with i % 10 == 9 are singletons. S3-(3000+i) are other businesses."""
    s1, s2, s3, gold = [], [], [], {}
    for i in range(1, N_TRAIN + 1):
        word, country = WORDS[i % len(WORDS)], "India" if i % 3 == 0 else "US"
        name, address = f"{word.title()}  Traders {i}", f"{i} main road, town{i % 4}"
        s1.append([f"S1-{i}", name, address, country])
        s2.append([f"S2-{1000 + i}", *((name, address) if i % 7 == 0 else (f"{word.upper()} TRADERS PVT", address)),
                   country])
        s3.append([f"S3-{2000 + i}", "शर्मा ट्रेडर्स" if country == "India" else f"{word} trading co", address, country])
        s3.append([f"S3-{3000 + i}", f"Other Shop {WORDS[(i * 5) % len(WORDS)]} {i * 7}", f"{i * 3} lake view",
                   country])
        gold[f"S1-{i}"] = [] if i % 10 == 9 else [f"S2-{1000 + i}", f"S3-{2000 + i}"]
    write_tsv(directory / "train_source1.tsv", s1)
    write_tsv(directory / "train_source2.tsv", s2)
    write_tsv(directory / "train_source3.tsv", s3)
    (directory / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{s}\t{','.join(t)}\n" for s, t in gold.items()))
    return gold


def make_test(directory: Path) -> dict[str, str]:
    """Test S1-(500+j): US/India/France, one Monaco S1 with two targets, one Atlantis S1 with none."""
    s1, s2, s3, countries = [], [], [], {}
    for j in range(1, 25):
        word = WORDS[(j * 3) % len(WORDS)]
        country = "Monaco" if j == 23 else "Atlantis" if j == 24 else ("US", "India", "France")[j % 3]
        name, address = f"{word.title()} Bakery {j}", f"{j} rue du port, zone{j % 3}"
        s1.append([f"S1-{500 + j}", name, address, country])
        countries[f"S1-{500 + j}"] = country
        if country != "Atlantis":
            s2.append([f"S2-{5000 + j}", *((name, address) if j % 4 == 0 else (f"{word} bakery ltd", address)),
                       country])
            s3.append([f"S3-{6000 + j}", f"Corner Store {j * 11}", f"{j * 5} hill street", country])
    write_tsv(directory / "test_source1.tsv", s1)
    write_tsv(directory / "test_source2.tsv", s2)
    write_tsv(directory / "test_source3.tsv", s3)
    return countries


def filter_list(i: int) -> list[str]:
    """The string blocker's list: misses the S2 gold when i % 4 == 0 and the S3 gold when i % 3 == 0."""
    rows = [f"S3-{3000 + i}"]
    if i % 4:
        rows.append(f"S2-{1000 + i}")
    if i % 3:
        rows.append(f"S3-{2000 + i}")
    return rows


def make_pairs(root: Path) -> dict[str, list[str]]:
    splits = {"validation": [f"S1-{i}" for i in range(1, N_TRAIN + 1) if fold(i) == 0 and i % 10 == 5],
              "holdout": [f"S1-{i}" for i in range(1, N_TRAIN + 1) if fold(i) == 0 and i % 10 == 0]}
    for split, ids in splits.items():
        (root / split).mkdir(parents=True)
        rows = [(s, t, r) for s in ids for r, t in enumerate(filter_list(int(s[3:])))]
        pq.write_table(pa.table({"s1_id": pa.array([r[0] for r in rows], pa.string()),
                                 "t_id": pa.array([r[1] for r in rows], pa.string()),
                                 "label": pa.array([0] * len(rows), pa.int8()),
                                 "filter_score": pa.array([1.0 - r[2] / 10 for r in rows], pa.float32()),
                                 "filter_rank": pa.array([r[2] for r in rows], pa.int16())}),
                       root / split / "part-00000.parquet")
        pq.write_table(pa.table({"s1_id": pa.array(ids, pa.string()),
                                 "n_cand": pa.array([len(filter_list(int(s[3:]))) for s in ids], pa.int16())}),
                       root / split / "s1.parquet")
    return splits


def make_folds(path: Path) -> None:
    path.write_text("source1_entity_id\tfold\n" + "".join(f"S1-{i}\t{fold(i)}\n" for i in range(1, N_TRAIN + 1)))


def make_tiny_backbone(directory: Path) -> Path:
    from transformers import BertConfig, BertModel, BertTokenizerFast

    # Whole words as tokens: with characters only, every mean-pooled vector is nearly the same.
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
    words = WORDS + ["query", "traders", "trading", "co", "pvt", "main", "road", "town", "other", "shop", "lake",
                     "view", "bakery", "ltd", "rue", "du", "port", "zone", "corner", "store", "hill", "street"]
    vocab = (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + list("|,.-:") + [f"##{c}" for c in chars] +
             [w for w in words if w not in chars])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    # Positional: the keyword is vocab_file in transformers 4.x but vocab in 5.x (where vocab_file= is ignored).
    tokenizer = BertTokenizerFast(str(directory / "vocab.txt"), do_lower_case=True)
    assert tokenizer.tokenize("query: alpha") == ["query", ":", "alpha"]
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128, initializer_range=0.2)
    model_dir = directory / "tiny"
    BertModel(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(bi_encoder, "resolve_model", lambda spec, allowed: spec | {"license": "mit", "revision": None})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)        # CPU fp32 even on a GPU image
    monkeypatch.setattr(bi_encoder, "devices", lambda: [torch.device("cpu")] * 2)   # two workers share the queue
    torch.manual_seed(0)
    paths = {"train": tmp_path / "train", "test": tmp_path / "test", "pairs": tmp_path / "pairs",
             "folds": tmp_path / "folds.tsv", "out": tmp_path / "out", "cfg": tmp_path / "bi.json"}
    gold = make_train(paths["train"])
    countries = make_test(paths["test"])
    splits = make_pairs(paths["pairs"])
    make_folds(paths["folds"])
    cfg = json.loads(CONFIG.read_text()) | {
        "backbone": str(make_tiny_backbone(tmp_path)), "max_length": 48, "batch_size": 16, "lr": 1e-3, "log_every": 2,
        "embed_batch": 7, "top_k": K, "search_query_block": 3, "search_target_chunk": 11, "part_rows": 12}
    paths["cfg"].write_text(json.dumps(cfg))
    return paths, cfg, gold, countries, splits


def run(paths: dict, stage: str = "all") -> None:
    bi_encoder.main(["--config", str(paths["cfg"]), "--train-dir", str(paths["train"]),
                     "--test-dir", str(paths["test"]), "--folds", str(paths["folds"]),
                     "--pairs-root", str(paths["pairs"]), "--out", str(paths["out"]), "--stage", stage])


def read_split(folder: Path) -> dict:
    parts = sorted(folder.glob("part-*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    assert table.schema.field("s1_id").type == pa.string() and table.schema.field("t_id").type == pa.string()
    assert table.schema.field("dense_score").type == pa.float32()
    assert table.schema.field("dense_rank").type == pa.int16()
    return table.to_pydict()


def test_training_pairs_skip_fold_0_and_never_repeat_an_s1_in_a_batch(corpus):
    paths, cfg, gold, _, splits = corpus
    data = bi_encoder.training_pairs(cfg, paths["train"], paths["folds"], paths["pairs"])
    trained = [s for s, t in gold.items() if t and fold(int(s[3:])) != 0]
    assert all(fold(int(s[3:])) != 0 for s in data["s1"]) and data["folds_seen"] == [1, 2, 3, 4]
    assert len(data["s1"]) == 2 * len(trained) and set(data["s1"]) == set(trained)
    assert sorted(np.concatenate(data["batches"]).tolist()) == list(range(len(data["s1"])))
    for batch in data["batches"]:
        assert len(set(data["s1"][batch])) == len(batch)                 # in-batch negatives are never true
    for row in range(len(data["s1"])):                                   # every pair is a gold link, with its texts
        s1, target = data["s1"][row], data["b"][data["ib"][row]]
        assert data["a"][data["ia"][row]].startswith(WORDS[int(s1[3:]) % len(WORDS)].title())
        assert target  # non-empty target text
    few = bi_encoder.training_pairs(cfg | {"max_pairs": 20}, paths["train"], paths["folds"], paths["pairs"])
    assert len(few["s1"]) == 20 and len(set(few["s1"])) == 20             # one gold target per S1 before a second
    with pytest.raises(ValueError, match="fold 0"):
        bi_encoder.training_pairs(cfg | {"train_folds": [0, 1]}, paths["train"], paths["folds"], paths["pairs"])
    leaky = paths["folds"].read_text().replace("S1-5\t0", "S1-5\t1")      # a validation S1 put in a training fold
    paths["folds"].write_text(leaky)
    with pytest.raises(ValueError, match="validation/holdout"):
        bi_encoder.training_pairs(cfg, paths["train"], paths["folds"], paths["pairs"])


def test_search_matches_brute_force(monkeypatch):
    import torch

    monkeypatch.setattr(bi_encoder, "devices", lambda: [torch.device("cpu")] * 3)
    rng = np.random.default_rng(0)
    queries, targets = rng.normal(size=(37, 8)).astype(np.float32), rng.normal(size=(101, 8)).astype(np.float32)
    index, score = bi_encoder.search(queries, targets, 7, {"search_query_block": 4, "search_target_chunk": 10})
    full = queries @ targets.T
    expected = np.argsort(-full, axis=1)[:, :7]
    assert index.dtype == np.int32 and score.dtype == np.float32
    assert (index == expected).all() and np.allclose(score, np.take_along_axis(full, expected, 1), atol=1e-5)


def test_stage_all_end_to_end(corpus):
    paths, cfg, gold, countries, splits = corpus
    run(paths)
    out = paths["out"]

    log = json.loads((out / "train_log.json").read_text())
    trained = [s for s, t in gold.items() if t and fold(int(s[3:])) != 0]
    assert log["pairs"] == 2 * len(trained) and log["s1"] == len(trained) and log["train_folds_seen"] == [1, 2, 3, 4]
    assert log["steps"] == log["planned_steps"] > 0 and not log["stopped_early"] and log["throughput"] > 0
    assert log["license"] == "mit" and log["backbone"] == cfg["backbone"] and "revision" in log
    assert np.isfinite(log["loss_first"]) and np.isfinite(log["loss_last"]) and log["loss_curve"]
    assert (out / "model" / "config.json").is_file()

    stores = {name: bi_encoder.open_store(out / "emb" / name) for name in
              ("train_query", "train_target", "test_query", "test_target")}
    assert sorted(to_list(stores["train_query"]["ids"])) == sorted(splits["validation"] + splits["holdout"])
    assert len(stores["test_query"]["ids"]) == len(countries) and stores["test_target"]["vectors"].dtype == np.float16
    texts = {}
    for directory, kind in ((paths["train"], "train"), (paths["test"], "test")):
        for source in (1, 2, 3):
            for line in (directory / f"{kind}_source{source}.tsv").read_text(encoding="utf-8").splitlines()[1:]:
                entity, name, address, country = line.split("\t")
                texts[entity] = (" ".join(name.split()), " ".join(address.split()), country)

    for split, expected_s1 in (("validation", splits["validation"]), ("holdout", splits["holdout"]),
                               ("test", [s for s, c in countries.items() if c != "Atlantis"])):
        rows = read_split(out / "dense" / split)
        kind = "test" if split == "test" else "train"
        q, t = stores[f"{kind}_query"], stores[f"{kind}_target"]
        runs = [s for i, s in enumerate(rows["s1_id"]) if i == 0 or rows["s1_id"][i - 1] != s]
        assert runs == expected_s1                                        # grouped, in split / file order
        t_pos = {v: i for i, v in enumerate(to_list(t["ids"]))}
        q_pos = {v: i for i, v in enumerate(to_list(q["ids"]))}
        for s1 in expected_s1:
            mine = [i for i, s in enumerate(rows["s1_id"]) if s == s1]
            n = 2 if texts[s1][2] == "Monaco" else K
            assert [rows["dense_rank"][i] for i in mine] == list(range(n))
            got = [rows["t_id"][i] for i in mine]
            assert all(texts[x][2] == texts[s1][2] for x in got)          # same country label only
            # Scores are the cosine of the stored vectors and the best of the country, best first.
            qv = q["vectors"][q_pos[s1]].astype(np.float32)
            same = [x for x in t_pos if texts[x][2] == texts[s1][2]]
            best = np.sort(np.asarray([qv @ t["vectors"][t_pos[x]].astype(np.float32) for x in same]))[::-1][:n]
            scores = np.asarray([rows["dense_score"][i] for i in mine])
            assert np.allclose(scores, best, atol=1e-5) and (np.diff(scores) <= 1e-7).all()
            for x, value in zip(got, scores):
                assert abs(qv @ t["vectors"][t_pos[x]].astype(np.float32) - value) < 1e-5
            twin = [x for x in same if texts[x][:2] == texts[s1][:2]]
            if twin:                                                     # identical text: cosine 1, rank 0
                assert got[0] == twin[0] and scores[0] > 0.99
        manifest = json.loads((out / "dense" / split / bi_encoder.MANIFEST).read_text())
        assert manifest["s1_written"] == len(expected_s1) and manifest["rows"] == len(rows["s1_id"])
    twins = [s for s in splits["validation"] + splits["holdout"] if int(s[3:]) % 7 == 0]
    assert twins == ["S1-35", "S1-70"]                                    # the identical-text case is exercised
    assert json.loads((out / "dense" / "test" / bi_encoder.MANIFEST).read_text())["countries"]["Atlantis"] == {
        "s1": 1, "targets": 0}

    report = json.loads((out / "dense_report.json").read_text())
    for split in ("validation", "holdout"):
        r = report[split]
        links = [(s, x) for s in splits[split] for x in gold[s]]
        missing = [(s, x) for s, x in links if x not in filter_list(int(s[3:]))]
        assert r["gold_links"] == len(links) and r["missing_from_filter"] == len(missing) > 0
        assert r["filter_recall"] == pytest.approx(1 - len(missing) / len(links))
        assert set(r["by_country"]) == {"US", "India"} and r["s1"] == len(splits[split])
        assert sum(v["gold_links"] for v in r["by_country"].values()) == len(links)
        for k in (5,):
            assert 0 <= r[f"dense_recall@{k}"] <= r[f"union_recall@{k}"] <= 1
            assert r[f"union_recall@{k}"] >= r["filter_recall"]
            assert r[f"union_recall@{k}"] == pytest.approx(r["filter_recall"] + r[f"new_links@{k}"] / len(links))
        assert "dense_recall@10" not in r                                 # k above top_k is not reported
        assert sum(v["links"] for v in r["missing_by_target_script"].values()) == len(missing)
    # Recompute validation recall@5 straight from the files.
    rows = read_split(out / "dense" / "validation")
    dense = {(s, x) for s, x, rank in zip(rows["s1_id"], rows["t_id"], rows["dense_rank"]) if rank < 5}
    links = [(s, x) for s in splits["validation"] for x in gold[s]]
    assert report["validation"]["dense_recall@5"] == pytest.approx(np.mean([link in dense for link in links]))

    # Resumable: a second run finds every output and redoes nothing.
    stamps = {p: p.stat().st_mtime_ns for p in [out / "train_log.json", *sorted((out / "dense").rglob("*.parquet")),
                                                  out / "emb" / "test_target" / "vectors.npy"]}
    run(paths)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())

    # A new session (vectors in /tmp gone, earlier outputs attached): "all" reuses the finished dense lists and
    # embeds nothing.
    shutil.rmtree(out / "emb")
    run(paths)
    assert not (out / "emb").exists()
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items() if p.suffix == ".parquet")

    # Retrained model (new stamp): a lone retrieve refuses the older vectors instead of keeping the older lists;
    # "all" re-embeds and re-retrieves every split.
    run(paths, "embed")
    log["model_stamp"] = "retrained"
    (out / "train_log.json").write_text(json.dumps(log))
    with pytest.raises(RuntimeError, match="another model"):
        run(paths, "retrieve")
    run(paths)
    for split in ("validation", "holdout", "test"):
        assert json.loads((out / "dense" / split / bi_encoder.MANIFEST).read_text())["model"] == "retrained"
    assert json.loads((out / "emb" / "test_target" / "done.json").read_text())["model"] == "retrained"


def to_list(column) -> list:
    return column.to_pylist()


def test_kernel_finds_code_pairs_folds_and_data(tmp_path):
    spec = importlib.util.spec_from_file_location("bi_kernel", CODE_DIR / "kaggle_ce" / "bi_kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    inputs = tmp_path / "input"
    repo = inputs / "ber-ce-code" / "business-entity-resolution"
    (repo / "code" / "business_entity_resolution" / "src").mkdir(parents=True)
    (repo / "code" / "business_entity_resolution" / "src" / "bi_encoder.py").write_text("")
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "folds.tsv").write_text("source1_entity_id\tfold\n")
    for dataset in ("ber-ce-pairs-cascade", "ber-ce-pairs-fold0"):
        for split in ("train", "validation", "holdout"):
            (inputs / dataset / split).mkdir(parents=True)
            (inputs / dataset / split / "s1.parquet").write_text("")
    for kind in ("train", "test"):
        (inputs / "tsvs" / kind).mkdir(parents=True)
        (inputs / "tsvs" / kind / f"{kind}_source2.tsv").write_text("")
    assert kernel.find_repo(inputs) == repo
    assert kernel.find_pairs_root(inputs) == inputs / "ber-ce-pairs-fold0"          # preferred over other roots
    assert kernel.find_folds(repo, inputs, tmp_path / "scratch") == repo / "artifacts" / "folds.tsv"
    assert kernel.find_dir(inputs, "test_source2.tsv") == inputs / "tsvs" / "test"
    cmd = kernel.bi_command(Path("/w/ber"), Path("/p"), Path("/w/bi"), Path("/e"), Path("/d/train"), Path("/d/test"),
                            Path("/f.tsv"))
    assert cmd[1:3] == ["-m", "src.bi_encoder"] and cmd[cmd.index("--stage") + 1] == "all"
    assert cmd[cmd.index("--emb-dir") + 1] == "/e" and cmd[cmd.index("--folds") + 1] == "/f.tsv"
    assert cmd[cmd.index("--config") + 1] == "/w/ber/configs/bi.json"


def test_kernel_reuses_an_attached_earlier_output(tmp_path):
    spec = importlib.util.spec_from_file_location("bi_kernel", CODE_DIR / "kaggle_ce" / "bi_kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    inputs, out = tmp_path / "input", tmp_path / "working" / "bi"
    ce = inputs / "ber-ce-out" / "ce"                                  # a cross-encoder output is not a bi output
    (ce / "model").mkdir(parents=True)
    (ce / "model" / "config.json").write_text("{}")
    (ce / "train_log.json").write_text("{}")
    assert kernel.reuse_previous(inputs, out) is None and not out.exists()
    previous = inputs / "ber-bi-out" / "bi"
    (previous / "model").mkdir(parents=True)
    (previous / "model" / "config.json").write_text("{}")
    (previous / "train_log.json").write_text(json.dumps({"model_stamp": "1"}))
    for split in ("validation", "test.tmp"):
        (previous / "dense" / split).mkdir(parents=True)
        (previous / "dense" / split / "part-00000.parquet").write_text("")
        (previous / "dense" / split / bi_encoder.MANIFEST).write_text(json.dumps({"model": "1"}))
    (previous / "dense" / "holdout").mkdir()                           # unfinished: no manifest
    assert kernel.reuse_previous(inputs, out) == previous
    assert (out / "model" / "config.json").is_file() and json.loads((out / "train_log.json").read_text())
    assert sorted(p.name for p in (out / "dense").iterdir()) == ["validation"]
    assert (out / "dense" / "validation" / "part-00000.parquet").is_file()


def test_kernel_writes_folds_when_none_is_attached(tmp_path):
    spec = importlib.util.spec_from_file_location("bi_kernel", CODE_DIR / "kaggle_ce" / "bi_kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    from src.make_folds import stable_fold

    inputs, repo = tmp_path / "input", tmp_path / "repo"
    (inputs / "tsvs" / "train").mkdir(parents=True)
    (inputs / "tsvs" / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5\nS1-2\t\n")
    repo.mkdir()
    path = kernel.find_folds(repo, inputs, tmp_path / "scratch")
    assert path.read_text().splitlines() == ["source1_entity_id\tfold", f"S1-1\t{stable_fold('S1-1', 5)}",
                                             f"S1-2\t{stable_fold('S1-2', 5)}"]


def test_split_config_adds_the_training_split(tmp_path):
    from src import bi_encoder

    for split in ("validation", "train"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "s1.parquet").write_text("")
    try:
        assert bi_encoder.apply_split_config({}) == ("validation", "holdout")
        assert bi_encoder.labelled_splits(tmp_path) == ["validation"]
        bi_encoder.apply_split_config({"labelled_splits": ["validation", "holdout", "train"]})
        assert bi_encoder.labelled_splits(tmp_path) == ["validation", "train"]
    finally:
        bi_encoder.apply_split_config({})
