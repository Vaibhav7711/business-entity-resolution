"""Compare the integrated Phase 1A rerun with the preserved original pilot."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .evaluate_blocking import ROOT


PACKAGES = ("numpy", "scipy", "scikit-learn", "psutil", "sparse-dot-topn")


def compare(original: dict, reproduced: dict) -> dict:
    checks = {}
    for key in ("size", "validation_fold", "country_counts", "true_links", "singleton_count"):
        checks[f"sample/{key}"] = original["sample"][key] == reproduced["sample"][key]
    for route in original["routes"]:
        for key in (
            "positive_link_candidate_recall", "retrieved_true_links", "true_links",
            "entities_with_every_true_match_pct", "positive_entities_with_every_true_match_pct",
            "recall_at", "candidate_count_per_s1", "zero_candidate_rate",
            "candidate_recall_by_source", "candidate_recall_by_country",
            "non_ascii_records", "target_address_missing",
        ):
            checks[f"{route}/{key}"] = original["routes"][route][key] == reproduced["routes"][route][key]
    checks["unique_additional_true_links"] = (
        original["unique_additional_true_links"] == reproduced["unique_additional_true_links"]
    )
    return {
        "all_core_metrics_equal": all(checks.values()),
        "checks": checks,
        "package_versions": {package: importlib.metadata.version(package) for package in PACKAGES},
        "original_runtime_seconds": original["runtime_seconds"],
        "reproduced_runtime_seconds": reproduced["runtime_seconds"],
        "original_peak_ram_bytes": original["peak_ram_bytes"],
        "reproduced_peak_ram_bytes": reproduced["peak_ram_bytes"],
    }


def main() -> None:
    original = json.loads((ROOT / "artifacts/phase1b/phase1a_original_metrics.json").read_text())
    reproduced = json.loads((ROOT / "artifacts/phase1a/pilot_metrics.json").read_text())
    result = compare(original, reproduced)
    output = ROOT / "artifacts/phase1b/reproduction_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_core_metrics_equal"]:
        failures = [key for key, okay in result["checks"].items() if not okay]
        raise RuntimeError(f"Phase 1A reproduction differs: {failures}")
    print(f"All {len(result['checks'])} Phase 1A core metric comparisons matched exactly.")


if __name__ == "__main__":
    main()
