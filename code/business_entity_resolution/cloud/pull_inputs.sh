#!/usr/bin/env bash
# Pull official training/test TSVs (hash-verified) and the Kaggle Phase 1C fold-0 shards onto this node.
# Needs KAGGLE_API_TOKEN in the environment. Usage: bash cloud/pull_inputs.sh (from code/business_entity_resolution)
set -euo pipefail
. "$HOME/venv/bin/activate"
REPO=$(cd ../.. && pwd)
DATA=${DATA:-$HOME/data}
mkdir -p "$DATA/kaggle_ds" "$REPO/student_resource/dataset/train" "$REPO/student_resource/dataset/test"
if [ ! -f "$DATA/kaggle_ds/.done" ]; then
  kaggle datasets download -d satwiksps/amazon-ml-challenge-2026 -p "$DATA/kaggle_ds" --unzip && touch "$DATA/kaggle_ds/.done"
fi
for f in train_source1 train_source2 train_source3 train_ground_truth; do
  cp -n "$(find "$DATA/kaggle_ds" -name "$f.tsv" | head -1)" "$REPO/student_resource/dataset/train/"
done
for f in test_source1 test_source2 test_source3; do
  cp -n "$(find "$DATA/kaggle_ds" -name "$f.tsv" | head -1)" "$REPO/student_resource/dataset/test/"
done
cd "$REPO/student_resource/dataset/train" && sha256sum -c - <<'SHA'
70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037  train_ground_truth.tsv
591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973  train_source1.tsv
6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed  train_source2.tsv
67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9  train_source3.tsv
SHA
cd "$REPO/student_resource/dataset/test" && sha256sum -c - <<'SHA'
3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5  test_source1.tsv
79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd  test_source2.tsv
850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035  test_source3.tsv
SHA
if [ ! -f "$DATA/phase1c_kaggle/.done" ]; then
  mkdir -p "$DATA/phase1c_kaggle"
  kaggle kernels output vaibhav0383/notebookd36610ca97 -p "$DATA/phase1c_kaggle" && touch "$DATA/phase1c_kaggle/.done"
fi
P1C=$(dirname "$(find "$DATA/phase1c_kaggle" -name state.json -path '*phase1c/work*' | head -1)")
echo "fold-0 Phase 1C work dir: $P1C"; test -f "$P1C/state.json"
ln -sfn "$P1C" "$DATA/fold0_work"
