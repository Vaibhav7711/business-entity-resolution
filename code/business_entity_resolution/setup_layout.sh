#!/usr/bin/env bash
# Recreate the directory layout the pipeline expects, from this folder plus the official challenge data.
#
# Configs and scripts address files relative to the repository root, two levels above this folder:
#   ../../configs/  ../../artifacts/  ../../student_resource/{dataset/{train,test}, utils/validate_submission.py}
# This copies configs/ and artifacts/ from here to ../../ (copies, not links: cloud/kaggle_ce.sh archives them for
# the GPU kernels and tar stores a link as a link), links the official student_resource folder, and checks the data
# against the sha256 sums of the files the submission was built from.
#
# Usage (from code/business_entity_resolution): bash setup_layout.sh /path/to/student_resource
set -euo pipefail
[ -f src/ce_stack.py ] || { echo "run from code/business_entity_resolution"; exit 1; }
SR=$(cd "${1:?usage: bash setup_layout.sh /path/to/student_resource}" && pwd)
ROOT=$(cd ../.. && pwd)
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
while read -r expected file; do
  [ -f "$SR/$file" ] || { echo "missing $SR/$file"; exit 1; }
  [ "$(sha "$SR/$file")" = "$expected" ] || { echo "sha256 mismatch: $file (not the official data?)"; exit 1; }
done <<'SHA'
70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037 dataset/train/train_ground_truth.tsv
591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973 dataset/train/train_source1.tsv
6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed dataset/train/train_source2.tsv
67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9 dataset/train/train_source3.tsv
3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5 dataset/test/test_source1.tsv
79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd dataset/test/test_source2.tsv
850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035 dataset/test/test_source3.tsv
SHA
[ -f "$SR/utils/validate_submission.py" ] || { echo "missing $SR/utils/validate_submission.py"; exit 1; }
mkdir -p "$ROOT/configs" "$ROOT/artifacts"
cp -R configs/. "$ROOT/configs/"
cp -R artifacts/. "$ROOT/artifacts/"
if [ ! -e "$ROOT/student_resource" ]; then ln -s "$SR" "$ROOT/student_resource"; fi
echo "layout ready under $ROOT: configs/, artifacts/, student_resource -> $SR (data sha256 verified)"
