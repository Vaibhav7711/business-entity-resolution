"""Kaggle GPU script kernel: bi-encoder dense retrieval route (``src.bi_encoder``) on the attached inputs.

Attach (any dataset names; everything is found by content under /kaggle/input):

* the code dataset: this repository, with ``code/business_entity_resolution/src/bi_encoder.py``, ``configs/bi.json``
  and ``artifacts/folds.tsv`` (without it the folds are rebuilt from the ground truth exactly as ``src.make_folds``);
* the fold-0 top-40 pairs (ber-ce-pairs-fold0): a directory holding ``validation/`` and ``holdout/`` with
  ``s1.parquet`` and ``part-*.parquet``;
* the challenge TSVs (``train_source*.tsv``, ``train_ground_truth.tsv`` and ``test_source*.tsv``).

Accelerator: GPU (T4 x2: training on cuda:0, embedding and search on both GPUs); Internet on (pip, model download).
Outputs in /kaggle/working/bi: dense/<split>/part-*.parquet, dense_report.json, train_log.json, model/, bi_run.log.
The ~17 GB of float16 vectors stay in /tmp/bi_emb (not saved), so they never survive into a new session. To resume,
attach an earlier output of this kernel (a directory holding train_log.json with a model_stamp and model/) and keep
``STAGE = "all"``: its model (training is skipped) and its finished dense splits are reused, and only the missing
splits are embedded and retrieved. ``STAGE = "report"`` recomputes dense_report.json from the attached dense lists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
STAGE = "all"                     # "all" | "train" | "embed" | "retrieve" | "report"
CONFIG = "bi.json"                # file under configs/
PAIRS_HINT = "fold0"              # preferred pairs root when several are attached
RUN_TESTS = True                  # the CPU unit tests catch library incompatibilities in a minute
KEEP_MODEL = True                 # keep model/ (~0.5 GB) in the output for re-embedding later
SKIP_DIRS = {"__MACOSX", ".git", "__pycache__"}
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "__MACOSX", "student_resource", "artifacts",
                                     "outputs")


def say(message: str) -> None:
    print(f"[bi_kernel] {message}", flush=True)


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        yield Path(dirpath), dirnames, filenames


def find_repo(input_dir: Path) -> Path:
    for path, _, filenames in walk(input_dir):
        if "bi_encoder.py" in filenames and path.name == "src":
            repo = path.parents[2]
            if (repo / "code" / "business_entity_resolution").is_dir():
                return repo
    raise SystemExit(f"no code dataset (code/business_entity_resolution/src/bi_encoder.py) under {input_dir}")


def find_pairs_root(input_dir: Path) -> Path:
    roots = [path for path, _, _ in walk(input_dir) if (path / "validation" / "s1.parquet").is_file()]
    if not roots:
        raise SystemExit(f"no pairs root (a directory with validation/s1.parquet) under {input_dir}")
    root = next((r for r in roots if PAIRS_HINT in str(r)), roots[0])
    for other in roots:
        if other != root:
            say(f"pairs: ignoring {other}")
    say(f"pairs: {root} ({', '.join(s for s in ('validation', 'holdout') if (root / s / 's1.parquet').is_file())})")
    return root


def find_dir(input_dir: Path, filename: str) -> Path | None:
    return next((path for path, _, filenames in walk(input_dir) if filename in filenames), None)


def find_folds(repo_src: Path, input_dir: Path, scratch: Path) -> Path:
    """The repository's artifacts/folds.tsv, else any attached folds.tsv, else rebuilt from the ground truth."""
    if (repo_src / "artifacts" / "folds.tsv").is_file():
        return repo_src / "artifacts" / "folds.tsv"
    found = find_dir(input_dir, "folds.tsv")
    if found is not None:
        return found / "folds.tsv"
    truth = find_dir(input_dir, "train_ground_truth.tsv")
    if truth is None:
        raise SystemExit("no folds.tsv and no train_ground_truth.tsv to rebuild it from")
    sys.path.insert(0, str(repo_src / "code" / "business_entity_resolution"))
    from src.make_folds import stable_fold

    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "folds.tsv"
    lines = (truth / "train_ground_truth.tsv").read_text(encoding="utf-8").splitlines()[1:]
    ids = [line.split("\t", 1)[0].strip() for line in lines if line]
    path.write_text("source1_entity_id\tfold\n" + "".join(f"{s}\t{stable_fold(s, 5)}\n" for s in ids))
    say(f"folds: rebuilt {len(ids):,} rows from {truth / 'train_ground_truth.tsv'} -> {path}")
    return path


def find_previous_output(input_dir: Path) -> Path | None:
    for path, _, filenames in walk(input_dir):
        if "train_log.json" in filenames and (path / "model" / "config.json").is_file():
            if "model_stamp" in json.loads((path / "train_log.json").read_text()):
                return path
    return None


def reuse_previous(input_dir: Path, out: Path) -> Path | None:
    """Copy an attached earlier output's model/, train_log.json and finished dense splits into ``out``."""
    previous = find_previous_output(input_dir)
    if previous is None:
        return None
    shutil.copytree(previous / "model", out / "model", dirs_exist_ok=True)
    shutil.copy2(previous / "train_log.json", out / "train_log.json")
    for split in sorted((previous / "dense").glob("*/_manifest.json")):
        if split.parent.name.endswith(".tmp"):                   # a split whose final rename never happened
            continue
        shutil.copytree(split.parent, out / "dense" / split.parent.name, dirs_exist_ok=True)
        say(f"dense: reusing {split.parent}")
    say(f"model: {previous / 'model'}")
    return previous


def gpu_count() -> int:
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return 0
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def stream(cmd: list[str], cwd: Path, log_path: Path | None = None, env: dict | None = None) -> int:
    say("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, errors="replace")
    log_file = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_file:
                log_file.write(line)
                log_file.flush()
    finally:
        if log_file:
            log_file.close()
    return proc.wait()


def bi_command(repo: Path, pairs: Path, out: Path, emb: Path, train_dir: Path, test_dir: Path,
               folds: Path) -> list[str]:
    return [sys.executable, "-m", "src.bi_encoder", "--config", str(repo / "configs" / CONFIG),
            "--train-dir", str(train_dir), "--test-dir", str(test_dir), "--folds", str(folds),
            "--pairs-root", str(pairs), "--out", str(out), "--emb-dir", str(emb), "--stage", STAGE]


def main(input_dir: Path = INPUT, working: Path = WORKING, scratch: Path = Path("/tmp")) -> int:
    repo_src = find_repo(input_dir)
    # Code copy and vectors live outside /kaggle/working so they are not saved as output.
    repo, out, emb = scratch / "ber", working / "bi", scratch / "bi_emb"
    shutil.copytree(repo_src, repo, ignore=COPY_IGNORE, dirs_exist_ok=True)
    code = repo / "code" / "business_entity_resolution"
    say(f"code: {repo_src} -> {repo}")
    out.mkdir(parents=True, exist_ok=True)
    for requirements in ("requirements.txt", "requirements-gpu.txt"):
        if (code / requirements).is_file():
            if stream([sys.executable, "-m", "pip", "install", "-q", "-r", requirements], code) != 0:
                raise SystemExit(f"pip install -r {requirements} failed")
    pairs = find_pairs_root(input_dir)
    train_dir = find_dir(input_dir, "train_source2.tsv")
    test_dir = find_dir(input_dir, "test_source2.tsv")
    say(f"train TSVs: {train_dir}; test TSVs: {test_dir}")
    if train_dir is None or not (train_dir / "train_ground_truth.tsv").is_file():
        raise SystemExit("no train_source2.tsv with train_ground_truth.tsv attached")
    if test_dir is None:
        raise SystemExit("no test_source2.tsv attached")
    folds = find_folds(repo_src, input_dir, scratch)
    say(f"folds: {folds}")
    if STAGE != "train" and reuse_previous(input_dir, out) is None and STAGE != "all":
        raise SystemExit(f"STAGE={STAGE} needs an attached earlier output (train_log.json and model/)")
    for path in (scratch, working):
        usage = shutil.disk_usage(path)
        say(f"disk {path}: {usage.free / 2**30:,.1f} GiB free of {usage.total / 2**30:,.1f} GiB")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    if gpu_count() == 0:
        raise SystemExit("no GPU visible: set the notebook Accelerator to a GPU")
    env = os.environ | {"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "true", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cpu_env = env | {"CUDA_VISIBLE_DEVICES": ""}
    if RUN_TESTS and stream([sys.executable, "-m", "pytest", "-q", "tests/test_bi_encoder.py"], code, env=cpu_env) != 0:
        raise SystemExit("tests/test_bi_encoder.py failed on this image")
    rc = stream(bi_command(repo, pairs, out, emb, train_dir, test_dir, folds), code, out / "bi_run.log", env)
    if rc == 0 and not KEEP_MODEL:
        shutil.rmtree(out / "model", ignore_errors=True)
    if (out / "dense_report.json").is_file():
        print((out / "dense_report.json").read_text(), flush=True)
    subprocess.run(["du", "-sh", str(out), str(emb)], check=False)
    subprocess.run(["ls", "-la", str(out), *[str(p) for p in sorted((out / "dense").glob("*"))]], check=False)
    say(f"bi_encoder exited with {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
