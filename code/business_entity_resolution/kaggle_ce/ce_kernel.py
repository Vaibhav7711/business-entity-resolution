"""Kaggle GPU script kernel: cross-encoder train, score and evaluate (``src.ce_model``) on the attached inputs.

Attach (any dataset names; everything is found by content under /kaggle/input):

* the code dataset: this repository, with ``code/business_entity_resolution/src/ce_model.py`` and ``configs/ce.json``;
* the top-K pairs: one or two datasets whose split directories (train/validation/holdout and/or test) each hold
  ``s1.parquet`` and ``part-*.parquet``; every split found is symlinked under /tmp/pairs;
* the challenge TSVs (``train_source*.tsv`` and ``test_source*.tsv``).

Accelerator: GPU (T4 x2 trains and scores on both GPUs through torchrun); Internet on (pip and the model download).
Outputs in /kaggle/working/ce: model/, scores/<split>.npy, train_log.json, score_log.json, eval.json, ce_run.log.
Set ``STAGE = "score"`` to rescore with an earlier output attached (a directory holding train_log.json and model/)
or a model dataset (config.json and train_log.json side by side, as cloud/lightning_ce_large.sh uploads it).
``ENV`` is added to ce_model's environment, e.g. {"CE_TEST_SHARDS": "6", "CE_TEST_ONLY": "3,4,5"} to score only some
test shards (see src/ce_model.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
STAGE = "all"                     # "all" | "train" | "score"
CONFIG = "ce.json"                # file under configs/ (e.g. ce_large.json for the round-2 model)
SPLITS: list[str] = []            # empty: validation, holdout and test, whichever are attached
RUN_TESTS = True                  # the CPU unit test catches library incompatibilities in a minute
ENV: dict[str, str] = {}          # extra environment for ce_model (test shards)
WARM_START = False                # train from an attached earlier output's model instead of the config backbone
SPLIT_NAMES = ("train", "validation", "holdout", "test")
SKIP_DIRS = {"__MACOSX", ".git", "__pycache__"}
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "__MACOSX", "student_resource", "artifacts",
                                     "outputs")


def say(message: str) -> None:
    print(f"[ce_kernel] {message}", flush=True)


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        yield Path(dirpath), dirnames, filenames


def find_repo(input_dir: Path) -> Path:
    for path, _, filenames in walk(input_dir):
        if "ce_model.py" in filenames and path.name == "src":
            repo = path.parents[2]
            if (repo / "code" / "business_entity_resolution").is_dir():
                return repo
    raise SystemExit(f"no code dataset (code/business_entity_resolution/src/ce_model.py) under {input_dir}")


def link_pairs(input_dir: Path, pairs_dir: Path) -> dict[str, Path]:
    """Symlink every split directory of every pairs root found into ``pairs_dir``; the first root wins a clash."""
    found: dict[str, Path] = {}
    for path, _, _ in walk(input_dir):
        if not any((path / s / "s1.parquet").is_file() for s in SPLIT_NAMES):   # a train-only root counts too
            continue
        for split in SPLIT_NAMES:
            if (path / split / "s1.parquet").is_file():
                if split in found:
                    say(f"pairs: ignoring {path / split} (already have {found[split]})")
                else:
                    found[split] = path / split
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for split, source in found.items():
        link = pairs_dir / split
        if link.is_symlink():
            link.unlink()
        link.symlink_to(source, target_is_directory=True)
        say(f"pairs: {split} -> {source} ({len(list(source.glob('part-*.parquet')))} parts)")
    return found


def find_dir(input_dir: Path, filename: str) -> Path | None:
    return next((path for path, _, filenames in walk(input_dir) if filename in filenames), None)


def find_previous_output(input_dir: Path) -> tuple[Path, Path] | None:
    """(model directory, train_log.json) of an earlier output (<dir>/model/) or of a model dataset (flat)."""
    for path, _, filenames in walk(input_dir):
        if "train_log.json" in filenames and (path / "model" / "config.json").is_file():
            return path / "model", path / "train_log.json"
        if "train_log.json" in filenames and "config.json" in filenames:
            return path, path / "train_log.json"
    return None


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


def ce_command(repo: Path, pairs: Path, out: Path, train_dir: Path | None, test_dir: Path | None,
               gpus: int, backbone: Path | None = None) -> list[str]:
    args = ["-m", "src.ce_model", "--config", str(repo / "configs" / CONFIG), "--pairs-root", str(pairs),
            "--out", str(out), "--stage", STAGE]
    if backbone:
        args += ["--backbone", str(backbone)]
    if train_dir:
        args += ["--train-dir", str(train_dir)]
    if test_dir:
        args += ["--test-dir", str(test_dir)]
    if SPLITS:
        args += ["--splits", *SPLITS]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", *args]


def main(input_dir: Path = INPUT, working: Path = WORKING, scratch: Path = Path("/tmp")) -> int:
    repo_src = find_repo(input_dir)
    # Code copy and pair links live outside /kaggle/working so they are not saved (or followed) as output.
    repo, pairs, out = scratch / "ber", scratch / "pairs", working / "ce"
    shutil.copytree(repo_src, repo, ignore=COPY_IGNORE, dirs_exist_ok=True)
    code = repo / "code" / "business_entity_resolution"
    say(f"code: {repo_src} -> {repo}")
    out.mkdir(parents=True, exist_ok=True)
    for requirements in ("requirements.txt", "requirements-gpu.txt"):
        if (code / requirements).is_file():
            if stream([sys.executable, "-m", "pip", "install", "-q", "-r", requirements], code) != 0:
                raise SystemExit(f"pip install -r {requirements} failed")
    found = link_pairs(input_dir, pairs)
    train_dir = find_dir(input_dir, "train_source2.tsv")
    test_dir = find_dir(input_dir, "test_source2.tsv")
    say(f"train TSVs: {train_dir}; test TSVs: {test_dir}")
    if STAGE in ("all", "train") and "train" not in found:
        raise SystemExit("no train split attached (needed to train)")
    if any(s != "test" for s in found) and train_dir is None:
        raise SystemExit("labelled splits attached but no train_source2.tsv")
    if "test" in found and test_dir is None:
        raise SystemExit("test split attached but no test_source2.tsv")
    if STAGE == "score":
        previous = find_previous_output(input_dir)
        if previous is None:
            raise SystemExit("STAGE=score needs an attached earlier output (train_log.json and model/)")
        model_dir, train_log = previous
        shutil.copytree(model_dir, out / "model", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("train_log.json", "dataset-metadata.json"))
        shutil.copy2(train_log, out / "train_log.json")
        say(f"model: {model_dir}")
    warm = None
    if WARM_START and STAGE in ("all", "train"):
        previous = find_previous_output(input_dir)
        if previous is None:
            raise SystemExit("WARM_START needs an attached earlier output (train_log.json and model/)")
        model_dir, train_log = previous
        warm = scratch / "warm" / "model"
        shutil.copytree(model_dir, warm, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("train_log.json", "dataset-metadata.json"))
        shutil.copy2(train_log, warm.parent / "train_log.json")
        say(f"warm start from {model_dir}")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    gpus = gpu_count()
    if gpus == 0:
        raise SystemExit("no GPU visible: set the notebook Accelerator to a GPU")
    env = os.environ | {"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "true", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} | ENV
    if ENV:
        say(f"extra environment: {ENV}")
    cpu_env = env | {"CUDA_VISIBLE_DEVICES": ""}
    if RUN_TESTS and stream([sys.executable, "-m", "pytest", "-q", "tests/test_ce_model.py"], code, env=cpu_env) != 0:
        raise SystemExit("tests/test_ce_model.py failed on this image")
    rc = stream(ce_command(repo, pairs, out, train_dir, test_dir, gpus, warm), code, out / "ce_run.log", env)
    subprocess.run(["ls", "-laR", str(out)], check=False)
    say(f"ce_model exited with {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
