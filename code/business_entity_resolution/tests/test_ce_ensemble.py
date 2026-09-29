import json

import numpy as np

from src.ce_ensemble import main


def test_ensemble_averages_logits_and_keeps_unscored_rows_unscored(tmp_path):
    a, b = tmp_path / "a" / "scores", tmp_path / "b" / "scores"
    for d, shift, k in ((a, 0.0, 40), (b, 2.0, 30)):
        d.mkdir(parents=True)
        np.save(d / "validation.npy", np.arange(4, dtype=np.float32) + shift)
        test = np.arange(3, dtype=np.float32) + shift
        test[2] = np.nan if d == b else test[2]
        np.save(d / "test.npy", test)
        (d.parent / "test_k.json").write_text(json.dumps({"k": k}))
    out = tmp_path / "ens"
    main(["--scores-dirs", str(a), str(b), "--out", str(out), "--splits", "validation", "test"])
    assert np.allclose(np.load(out / "scores" / "validation.npy"), np.arange(4) + 1.0)
    test = np.load(out / "scores" / "test.npy")
    assert np.allclose(test[:2], [1.0, 2.0]) and np.isnan(test[2])
    assert json.loads((out / "test_k.json").read_text())["k"] == 30
