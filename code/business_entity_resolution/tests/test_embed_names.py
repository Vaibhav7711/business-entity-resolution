import json
import zlib
from pathlib import Path

import numpy as np

from src import embed_names
from src.embed_names import record_key


class FakeEncoder:
    """Deterministic character-trigram hashing encoder (no downloads, CPU only)."""

    dim = 16

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        out = np.full((len(sentences), self.dim), 1e-3, np.float32)
        for i, text in enumerate(sentences):
            text = text.lower()
            for j in range(max(len(text) - 2, 0)):
                out[i, zlib.crc32(text[j:j + 3].encode()) % self.dim] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "jasper",
         "kestrel", "lotus", "maple", "nimbus", "onyx", "pepper", "quartz", "raven", "sierra", "tango",
         "umber", "violet", "willow", "xenon", "yarrow", "zephyr", "amber", "birch", "coral", "dune"]


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def make_dataset(root: Path, test_suffix: str = "") -> None:
    head = ["entity_id", "business_name", "business_address", "country"]
    s1, s2, s3, gold, folds = [], [], [], [], []
    for i, word in enumerate(WORDS, start=1):
        country = "US" if i % 2 else "India"
        address = f"{i} Main Road, Town{i % 4}"
        s1.append([f"S1-{i}", f"{word.title()} Traders", address, country])
        s2.append([f"S2-{1000 + i}", f"{word.title()} Traders Pvt", address, country])
        t3_name = "शर्मा ट्रेडर्स" if i == 7 else f"{word.upper()} TRADING CO"
        s3.append([f"S3-{2000 + i}", t3_name, address, country])
        s3.append([f"S3-{3000 + i}", f"Other Shop {word}", address, country])   # same address, different business
        gold.append([f"S1-{i}", f"S2-{1000 + i},S3-{2000 + i}"])
        folds.append([f"S1-{i}", str(0 if i <= 6 else i % 4 + 1)])
    s2.append(["S2-9999", "Alpha Traders Pvt", "99 Side Street", "US"])            # duplicate name, other record
    s1.append(["S1-99", "Lonely Singleton", "1 Nowhere Lane", "US"])
    gold.append(["S1-99", ""])
    folds.append(["S1-99", "2"])
    train = root / "student_resource/dataset/train"
    write_tsv(train / "train_source1.tsv", head, s1)
    write_tsv(train / "train_source2.tsv", head, s2)
    write_tsv(train / "train_source3.tsv", head, s3)
    write_tsv(train / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gold)
    write_tsv(root / "artifacts/folds.tsv", ["source1_entity_id", "fold"], folds)
    test = root / "student_resource/dataset/test"
    write_tsv(test / "test_source1.tsv", head, [[f"S1-{500 + i}", f"Test Query {w}{test_suffix}", "5 Rue X", "France"]
                                                for i, w in enumerate(WORDS[:8])])
    write_tsv(test / "test_source2.tsv", head, [[f"S2-{7000 + i}", f"Test Target {w}{test_suffix}", "5 Rue X", "France"]
                                                for i, w in enumerate(WORDS[:6])])
    write_tsv(test / "test_source3.tsv", head, [[f"S3-{8000 + i}", f"Cible {w}{test_suffix}", "", "France"]
                                                for i, w in enumerate(WORDS[:5])])


def make_config(root: Path) -> Path:
    cfg = json.loads((Path(__file__).resolve().parents[3] / "configs/r3_embeddings.json").read_text())
    cfg["candidates"] = [{"name": "fake-b", "repo": "fake/b", "prefix": "", "cost": 3},
                         {"name": "fake-a", "repo": "fake/a", "prefix": "query: ", "cost": 1}]
    cfg["diagnostic"]["s1_sample"] = 1000
    cfg["encode"]["chunk_texts"] = 7
    cfg["pca"]["dims"] = 8
    path = root / "r3.json"
    path.write_text(json.dumps(cfg))
    return path


def patch(monkeypatch) -> None:
    monkeypatch.setattr(embed_names, "resolve_model", lambda spec, allowed: spec | {"license": "mit", "revision": "rev"})
    monkeypatch.setattr(embed_names, "load_encoder", lambda spec, device, enc: FakeEncoder())
    monkeypatch.setattr(embed_names, "devices", lambda: ["cpu"])


def run(root: Path, name: str) -> Path:
    out = root / f"out_{name}"
    embed_names.main(["--config", str(make_config(root)), "--work-dir", str(root / f"work_{name}"),
                      "--output-dir", str(out), "--stage", "all", "--root", str(root)])
    return out


def test_r3_end_to_end_alignment_keys_and_selection(tmp_path, monkeypatch):
    patch(monkeypatch)
    make_dataset(tmp_path)
    out = run(tmp_path, "a")
    manifest = json.loads((out / "manifest.json").read_text())
    diagnostic = json.loads((out / "diagnostic.json").read_text())
    assert diagnostic["selected"]["name"] == "fake-a"          # tie on AUC -> cheaper model
    assert diagnostic["pairs"]["same_address_negative"] > 0 and diagnostic["pairs"]["cross_script_true"] == 1
    assert manifest["model"]["license"] == "mit" and manifest["vector"]["dims"] == 8
    for split, source in [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]:
        label = f"{split}_s{source}"
        vectors = np.load(out / f"names_{label}.npy")
        keys = np.load(out / f"keys_{label}.npy")
        rows = (tmp_path / f"student_resource/dataset/{split}/{split}_source{source}.tsv").read_text().splitlines()[1:]
        assert vectors.dtype == np.float16 and vectors.shape == (len(rows), 8)
        assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1, atol=1e-2)
        assert keys.tolist() == [record_key(r.split("\t")[0]) for r in rows]
        assert manifest["files"][label]["rows"] == len(rows)
    s2 = np.load(out / "names_train_s2.npy")
    assert np.array_equal(s2[0], s2[-1])                       # same name string -> same vector, separate rows
    assert "EMB_REPORT.md" in {p.name for p in out.iterdir()}


def test_r3_pca_ignores_test_names_and_diagnostic_skips_fold0(tmp_path, monkeypatch):
    patch(monkeypatch)
    first, second = tmp_path / "one", tmp_path / "two"
    make_dataset(first)
    make_dataset(second, test_suffix=" Zzq")
    with np.load(run(first, "x") / "pca.npz") as a, np.load(run(second, "y") / "pca.npz") as b:
        assert np.array_equal(a["components"], b["components"])
    cfg = json.loads(make_config(first).read_text())
    pairs = embed_names.diagnostic_pairs(cfg, first)
    fold0_names = {f"{w.title()} Traders" for w in WORDS[:6]}
    assert not fold0_names & set(pairs["s1_name"].tolist())
    assert pairs["label"].sum() > 0 and (~pairs["label"]).sum() > 0


def test_select_prefers_cheaper_within_margin():
    results = [{"name": "big", "cost": 3, "metrics": {"auc_combined_oof": 0.9015}},
               {"name": "small", "cost": 1, "metrics": {"auc_combined_oof": 0.9000}},
               {"name": "none", "cost": 1, "metrics": {"auc_combined_oof": None}}]
    assert embed_names.select(results, 0.002)["name"] == "small"
    assert embed_names.select(results, 0.001)["name"] == "big"
