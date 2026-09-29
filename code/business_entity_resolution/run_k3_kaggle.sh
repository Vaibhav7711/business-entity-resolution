#!/usr/bin/env bash
# K3 on Kaggle CPU: test inference -> output/matching_results.tsv + output/candidate_pairs.tsv + validator.
# Inputs attached: this code, the test TSVs (satwiksps dataset), the three test-blocking notebook outputs,
# the K1 results dataset, and the K2 notebook output. No labels are read.
set -euo pipefail
export PHASE2A_THREADS=1
CONFIG=../../configs/k3_inference.json
TEST=$(dirname "$(find /kaggle/input -name test_source2.tsv | head -1)")
K1=$(dirname "$(find /kaggle/input -name k1_results.json | head -1)")
K2=${K2:-$(dirname "$(dirname "$(find /kaggle/input -name champion_policy.json | head -1)")")}
WORK=${WORK:-/tmp/k3_work}
OUT=${OUT:-/kaggle/working/k3}
echo "test: $TEST"; echo "k1: $K1"; echo "k2: $K2"
find /kaggle/input -name "test_range_*_manifest.json"
cd "$TEST" && sha256sum -c - <<'SHA'
3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5  test_source1.tsv
79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd  test_source2.tsv
850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035  test_source3.tsv
SHA
cd - > /dev/null
python3 -m pip install -q -r requirements.txt
python3 -m pytest -q
mkdir -p "$OUT"
python3 -m src.k3_inference --config "$CONFIG" --test-dir "$TEST" --blocking-root /kaggle/input --k1-dir "$K1" \
  --k2-dir "$K2" --work-dir "$WORK" --output-dir "$OUT" --stage all 2>&1 | tee "$OUT/k3_run.log" \
  | grep --line-buffered -E "K3:|stores:|score: shard|write:|PASS|FAIL|WARNING|Error|Traceback"
ls -la "$OUT/output"
