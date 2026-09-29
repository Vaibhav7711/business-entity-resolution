#!/usr/bin/env bash
# Drive the cross-encoder GPU run on Kaggle from the node through the Kaggle API (token in ~/.kaggle_api_token).
#   bash cloud/kaggle_ce.sh code|fold0|test   # upload one private dataset (new version if it exists); 'ready' shows status
#   bash cloud/kaggle_ce.sh push [slug config stage splits datasets kernels]   # push + start a GPU kernel
#   bash cloud/kaggle_ce.sh status     # kernel status
#   bash cloud/kaggle_ce.sh fetch      # download the kernel output (scores, model, logs) to ~/data/ce_out
# Run from code/business_entity_resolution. Only creates/updates the ber-ce* datasets and the ber-ce kernel.
set -euo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "${KAGGLE_TOKEN_FILE:-$HOME/.kaggle_api_token}")   # KAGGLE_TOKEN_FILE + KAGGLE_OWNER: a second account
USER_SLUG=${KAGGLE_OWNER:-vaibhav0383}
DATA=${DATA:-$HOME/data}
STAGE=${KAGGLE_STAGE:-$DATA/kaggle_stage}
REPO=$(cd ../.. && pwd)
MACHINE=${MACHINE:-NvidiaTeslaT4}

upload() {  # upload <slug> <title> <dir>
  local slug=$1 title=$2 dir=$3
  cat > "$dir/dataset-metadata.json" <<EOF
{"title": "$title", "id": "$USER_SLUG/$slug", "licenses": [{"name": "other"}]}
EOF
  if kaggle datasets status "$USER_SLUG/$slug" >/dev/null 2>&1; then
    kaggle datasets version -p "$dir" -m "update $(date -u +%FT%TZ)" --dir-mode zip -q
  else
    kaggle datasets create -p "$dir" --dir-mode zip -q
  fi
}

case "${1:?stage}" in
  code)
    rm -rf "$STAGE/code" && mkdir -p "$STAGE/code/business-entity-resolution"
    (cd "$REPO" && tar --exclude='__pycache__' --exclude='.pytest_cache' -cf - code configs artifacts/folds.tsv \
        artifacts/k1_drive student_resource/utils) | tar -xf - -C "$STAGE/code/business-entity-resolution"
    upload ber-ce-code "ber-ce-code" "$STAGE/code"
    ;;
  fold0)
    rm -rf "$STAGE/pairs_fold0" && mkdir -p "$STAGE/pairs_fold0"
    for s in train validation holdout; do cp -al "$DATA/pairs_fold0/$s" "$STAGE/pairs_fold0/$s"; done   # hard links
    upload ber-ce-pairs-fold0 "ber-ce-pairs-fold0" "$STAGE/pairs_fold0"
    ;;
  test)
    rm -rf "$STAGE/pairs_test" && mkdir -p "$STAGE/pairs_test/notes"
    cp -al "$DATA/pairs_test/test" "$STAGE/pairs_test/test"
    # A dataset holding a single folder is flattened by Kaggle; a second folder keeps test/ intact.
    echo "Top-40 test candidate lists per S1. Split directory: test/." > "$STAGE/pairs_test/notes/README.txt"
    upload ber-ce-pairs-test "ber-ce-pairs-test" "$STAGE/pairs_test"
    ;;
  cascade)
    rm -rf "$STAGE/pairs_cascade" && mkdir -p "$STAGE/pairs_cascade"
    for s in validation holdout test; do cp -al "$DATA/pairs_cascade/$s" "$STAGE/pairs_cascade/$s"; done
    upload ber-ce-pairs-cascade "ber-ce-pairs-cascade" "$STAGE/pairs_cascade"
    ;;
  ready)
    for d in ber-ce-code ber-ce-pairs-fold0 ber-ce-pairs-test; do echo "$d: $(kaggle datasets status "$USER_SLUG/$d" 2>&1)"; done
    ;;
  push)   # push [slug] [config] [stage] [splits|-] [datasets,comma,separated] [kernel-sources|-]
    SLUG=${2:-ber-ce}; CFG=${3:-ce.json}; KSTAGE=${4:-all}; KSPLITS=${5:--}
    DS=${6:-$USER_SLUG/ber-ce-code,$USER_SLUG/ber-ce-pairs-fold0,$USER_SLUG/ber-ce-pairs-test}; KS=${7:--}
    K=$STAGE/kernel-$SLUG && rm -rf "$K" && mkdir -p "$K"
    python3 - "$K" "$SLUG" "$CFG" "$KSTAGE" "$KSPLITS" "$DS" "$KS" "$USER_SLUG" "$MACHINE" "${KERNEL:-ce_kernel.py}" <<'PY'
import json, os, re, sys
from pathlib import Path
out, slug, cfg, stage, splits, ds, ks, owner, machine, kernel = sys.argv[1:11]   # KERNEL=bi_kernel.py for the dense route
code = Path("kaggle_ce", kernel).read_text()
code = code.replace('STAGE = "all"', f'STAGE = "{stage}"', 1)
code = re.sub(r'^CONFIG = "[^"]*"', f'CONFIG = "{cfg}"', code, count=1, flags=re.M)
extra_env = json.loads(os.environ.get("KERNEL_ENV") or "{}")             # e.g. '{"CE_TEST_ONLY": "3,4,5"}'
if os.environ.get("WARM_START") == "1":                                  # train from the attached earlier output
    code = code.replace("WARM_START = False", "WARM_START = True", 1)
if extra_env:
    code = code.replace("ENV: dict[str, str] = {}", f"ENV: dict[str, str] = {extra_env!r}", 1)
if splits != "-":
    code = code.replace("SPLITS: list[str] = []", f"SPLITS: list[str] = {splits.split(',')!r}", 1)
Path(out, kernel).write_text(code)
meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": kernel, "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": machine,
        "dataset_sources": ds.split(",") + ["satwiksps/amazon-ml-challenge-2026"],
        "competition_sources": [], "kernel_sources": [] if ks == "-" else ks.split(","), "model_sources": []}
Path(out, "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
PY
    out=$(kaggle kernels push -p "$K" --accelerator "$MACHINE" 2>&1) || true   # the CLI exits 0 on a rejected push
    echo "$out"
    grep -q "successfully pushed" <<< "$out" || { echo "kernel push of $SLUG failed"; exit 1; }
    ;;
  status)
    kaggle kernels status "$USER_SLUG/${2:-ber-ce}"
    ;;
  fetch)   # fetch [slug] [dest]
    DEST=${3:-$DATA/ce_out}; mkdir -p "$DEST" && kaggle kernels output "$USER_SLUG/${2:-ber-ce}" -p "$DEST" -o
    ;;
esac
