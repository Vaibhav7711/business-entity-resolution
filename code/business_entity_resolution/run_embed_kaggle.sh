#!/usr/bin/env bash
# R3 on a Kaggle GPU notebook (Accelerator "GPU T4 x2", Internet on): multilingual name vectors for every
# train and test record. Inputs attached: this code and the training/test TSVs (satwiksps dataset).
# Output (/kaggle/working/emb, about 6 GB): names_<split>_s<k>.npy (float16, file row order), keys_*.npy,
# pca.npz, manifest.json, diagnostic.json, EMB_REPORT.md.
set -euo pipefail
export PHASE2A_THREADS=4
CONFIG=../../configs/r3_embeddings.json
WORK=${WORK:-/tmp/emb_work}
OUT=${OUT:-/kaggle/working/emb}
mkdir -p "$OUT"
python3 -m pip install -q -r requirements.txt
python3 -c "import sentence_transformers" 2>/dev/null || python3 -m pip install -q -r requirements-gpu.txt
python3 -c "import sys, torch; n = torch.cuda.device_count(); print('GPUs:', n); sys.exit(0 if n else 'No GPU: set Accelerator to GPU T4 x2')"
python3 -m pytest -q
python3 -m src.embed_names --config "$CONFIG" --work-dir "$WORK" --output-dir "$OUT" --stage all 2>&1 | tee "$OUT/r3_run.log" \
  | grep --line-buffered -E "diagnose:|vocab:|pca:|encode part|finalize:|R3 done|Error|Traceback"
ls -la "$OUT"
