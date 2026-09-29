"""Phase 2A local resource limits. Import first, before numpy/scipy/sklearn/lightgbm.

The local machine is an 8 GB MacBook Air shared with other work, so every
Phase 2A entry point caps BLAS/OpenMP threads at 2 and checks memory and swap
between chunks, stopping cleanly (with a resume command) instead of pushing
the system into swap or an out-of-memory crash.
"""

from __future__ import annotations

import os

THREADS = os.environ.get("PHASE2A_THREADS", "2")  # local default 2; cloud script sets 4
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = THREADS

WORKERS = int(THREADS)

import time  # noqa: E402

import psutil  # noqa: E402


class ResourceStop(Exception):
    """Raised when memory or swap limits are exceeded; completed chunks remain on disk."""


class ResourceGuard:
    def __init__(self, min_available_gib: float, max_swap_growth_gib: float, resume_command: str):
        self.min_available = min_available_gib * 2**30
        self.max_swap_growth = max_swap_growth_gib * 2**30
        self.swap_start = psutil.swap_memory().used
        self.resume_command = resume_command
        self.peak_rss = 0
        self.min_seen_available = float("inf")

    def check(self, label: str) -> None:
        available = psutil.virtual_memory().available
        swap_growth = psutil.swap_memory().used - self.swap_start
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        self.min_seen_available = min(self.min_seen_available, available)
        if available < self.min_available or swap_growth > self.max_swap_growth:
            raise ResourceStop(
                f"{label}: stopping cleanly (available {available/2**30:.2f} GiB, swap growth "
                f"{swap_growth/2**30:.2f} GiB). Completed chunks are kept. Resume with:\n  {self.resume_command}")

    def summary(self) -> dict:
        return {"peak_process_rss_bytes": self.peak_rss,
                "min_available_memory_bytes": None if self.min_seen_available == float("inf") else self.min_seen_available,
                "swap_growth_bytes": psutil.swap_memory().used - self.swap_start,
                "thread_limit": int(THREADS)}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
