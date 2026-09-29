"""End-of-run acceptance checks for round 1c (account 2, ber-r1c) after its output is downloaded to ~/data/r1c_out.

Round 1c must be round 1b warm-started: the uploaded parent's train_log is round 1b's (all 331,000 fold-0 S1, seed
20260928), the child records that parent's backbone and revision, the kernel log shows the warm start from
ber-r1b-model and the intended pairs, training used nonzero pairs and positives, and the validation/holdout scores
have exactly the dense-pair row counts. Prints PASS/FAIL per check and R1C_END_OK or R1C_END_FAIL.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

D = Path.home() / "data"


def main() -> int:
    log = json.loads(next((D / "r1c_out").rglob("train_log.json")).read_text())
    parent = json.loads(next((D / "r1b_model_dl").rglob("train_log.json")).read_text())
    rows = json.loads((D / "dense_new" / "dense_merge_new.json").read_text())
    eval_path = next((D / "r1c_out").rglob("eval.json"), None)
    evaluation = json.loads(eval_path.read_text()) if eval_path else {}
    text = "\n".join(p.read_text(errors="replace") for p in (D / "r1c_out").rglob("*.log"))
    checks = {
        "parent is round 1b (331,000 S1, seed 20260928)": parent.get("s1") == 331000 and parent.get("config", {}).get("seed") == 20260928,
        "warm_start_from = round-1b backbone": log.get("warm_start_from") == parent.get("backbone") == "intfloat/multilingual-e5-small",
        "warm_start_revision = round-1b revision": log.get("warm_start_revision") == parent.get("revision"),
        "backbone is the copied warm-start model": str(log.get("backbone", "")).endswith("/warm/model"),
        "license mit": log.get("license") == "mit",
        "training pairs > 0": (log.get("pairs") or 0) > 0 and (log.get("pairs_seen") or 0) > 0,
        "0 < positives < pairs": 0 < (log.get("positives") or 0) < (log.get("pairs") or 0),
        "kernel log: pairs train -> ber-union-train": bool(re.search(r"pairs: train -> \S*ber-union-train/train", text)),
        "kernel log: pairs validation -> ber-dense-new": bool(re.search(r"pairs: validation -> \S*ber-dense-new/validation", text)),
        "kernel log: pairs holdout -> ber-dense-new": bool(re.search(r"pairs: holdout -> \S*ber-dense-new/holdout", text)),
        "kernel log: warm start from ber-r1b-model": bool(re.search(r"warm start from \S*ber-r1b-model", text)),
    }
    for split in ("validation", "holdout"):
        want = rows[split]["rows"]
        scores = np.load(D / "r1c_new" / "scores" / f"{split}.npy")
        checks[f"{split}.npy = {want:,} finite rows"] = len(scores) == want and bool(np.isfinite(scores).all())
        checks[f"eval.json {split} rows = {want:,}"] = evaluation.get(split, {}).get("rows") == want
    print({k: log.get(k) for k in ("pairs", "positives", "s1", "pairs_seen", "steps", "throughput", "stopped_early",
                                   "warm_start_from", "license")})
    print("eval:", {s: {k: v for k, v in evaluation.get(s, {}).items() if k in ("rows", "positives", "auc", "top1_hit")}
                    for s in ("validation", "holdout")})
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    print("R1C_END_OK" if all(checks.values()) else "R1C_END_FAIL")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
