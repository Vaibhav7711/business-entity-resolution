"""Write artifacts/phase2a/README.md from the Phase 2A QA and model metrics.

    python3 -m src.phase2a_report --config ../../configs/phase2a_benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_blocking import ROOT
from .evaluate_phase1c import sha256_file


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def model_rows(runs: dict) -> list[str]:
    rows = ["| Model | Validation F0.5 | Holdout F0.5 | Holdout oracle | Predicted-match rate | Pair precision | "
            "Recall (all gold) | Singleton accuracy | Holdout FP pairs | Matcher FN | Blocking FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in runs.items():
        h = m["holdout"]
        rows.append(f"| {name} | {m['validation']['macro_f05']:.4f} | {h['macro_f05']:.4f} | "
                    f"{m['oracle_ceiling']['holdout']['macro_f05']:.4f} | {pct(h['predicted_match_rate'])} | "
                    f"{pct(h['pair_precision'])} | {pct(h['link_recall_vs_all_gold'])} | {pct(h['singleton_accuracy'])} | "
                    f"{h['errors']['false_positive_pairs']:,} | {h['errors']['false_negative_links_matcher']:,} | "
                    f"{h['errors']['false_negative_links_blocking']:,} |")
    return rows


def make_readme(config: dict, qa: dict, runs: dict, config_path: Path, config_sha256: str) -> str:
    lines = ["# Phase 2A — candidate-only pair-scoring baseline", "",
             f"Run `{config['run_id']}`. {config['description']}", "",
             "## Data", "",
             "| Split | S1 | Candidate pairs | Gold links | Retrieved gold | Blocker recall ceiling | Singletons | Feature rows |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, t in qa["splits"].items():
        lines.append(f"| {split} | {t['s1']:,} | {t['pairs']:,} | {t['truth_links']:,} | {t['retrieved_truth']:,} | "
                     f"{pct(t['blocker_recall_ceiling'])} | {t['singletons']:,} | {t['feature_rows']:,} |")
    lines += ["", "The blocker recall ceiling is a hard limit: a gold link absent from the frozen candidates cannot be recovered by any matcher.",
              "", f"QA checks: {', '.join(k for k, v in qa['checks'].items() if v)}. "
              f"Unavailable fields: {', '.join(qa['unavailable_fields'])}. Country consistency is constant by construction (exact-country blocking).",
              "", f"Validation definition: {config['split']['unit']}. Train {config['split']['train']}, validation "
              f"{config['split']['validation']}, holdout {config['split']['holdout']} (fold-0 rank positions). "
              f"Train pairs: all positives + {config['sampling']['hard_negatives_per_s1']} top-RRF + "
              f"{config['sampling']['random_negatives_per_s1']} seeded random negatives per S1; validation/holdout: every candidate pair.",
              "", "## Results (official entity-level macro F0.5)", ""]
    lines += model_rows(runs)
    lines += ["", "Thresholds (and the singleton empty-set guard) were swept on validation only; holdout is reported untouched. "
              "Oracle = perfect matcher on the same candidates (blocker-limited ceiling).", ""]
    for name, m in runs.items():
        h = m["holdout"]
        lines += [f"### {name}", ""]
        if "policy" in m:
            lines.append(f"- Policy: threshold {m['policy']['threshold']:.3f}, empty guard {m['policy']['empty_threshold']}; "
                         f"no-guard plateau (±0.001) {m['threshold_plateau_no_guard']}; validation ROC-AUC "
                         f"{m['pair_diagnostics']['validation_roc_auc']:.4f} (diagnostic only).")
        if "selected" in m:
            lines.append(f"- Selected rule: `{m['selected']}` (validation). Empty-prediction baseline holdout F0.5: "
                         f"{m['empty_prediction_baseline']['holdout']['macro_f05']:.4f}.")
        if "reproducible_refit_identical" in m:
            lines.append(f"- Reproducible refit: {m['reproducible_refit_identical']}; C = {m['hyperparameters']['C']}.")
        if "feature_importance_gain" in m:
            top = list(m["feature_importance_gain"].items())[:8]
            lines.append("- Top features by gain: " + ", ".join(f"{k}" for k, _ in top) + ".")
        lines.append("- Holdout slices: " + "; ".join(f"{k} {v['macro_f05']:.4f} (n={v['entities']:,})"
                                                     for k, v in h["slices_macro_f05"].items() if v["macro_f05"] is not None) + ".")
        lines.append(f"- Runtime {m['runtime_seconds']/60:.1f} min; peak process RSS {m['resources']['peak_process_rss_bytes']/2**30:.2f} GiB.")
        lines += ["", "Holdout false positives:", ""]
        lines += [f"  - {e['s1_name']} | {e['s1_address']}  →  {e['candidate']}: {e['cand_name']} | {e['cand_address']} (score {e['score']})"
                  for e in m["holdout_examples"]["false_positive"]]
        lines += ["", "Holdout matcher false negatives:", ""]
        lines += [f"  - {e['s1_name']} | {e['s1_address']}  →  {e['candidate']}: {e['cand_name']} | {e['cand_address']} (score {e['score']})"
                  for e in m["holdout_examples"]["false_negative"]]
        lines.append("")
    lines += ["## Reproduce", "", "From `code/business_entity_resolution`:", "", "```bash",
              f"python3 -m src.phase2a_pairs --config {config_path} --stage all",
              *[f"python3 -m src.phase2a_train --config {config_path} --model {m}" for m in ("rules", "logistic", "lightgbm")],
              f"python3 -m src.phase2a_report --config {config_path}", "```", "",
              f"Config SHA-256: `{config_sha256}`. Inputs: Phase 1C route shards listed with hashes in `pairs/input_hashes.json`; "
              "feature definitions in `src/phase2a_pairs.py` (`FEATURES`, `compute_features`).", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    out = ROOT / config["paths"]["output_dir"]
    qa = json.loads((out / "qa/data_qa.json").read_text())
    runs = {name: json.loads((out / "runs" / name / "metrics.json").read_text())
            for name in ("rules", "logistic", "lightgbm") if (out / "runs" / name / "metrics.json").exists()}
    (out / "README.md").write_text(make_readme(config, qa, runs, config_path.relative_to(ROOT), sha256_file(config_path)))
    print((out / "README.md").read_text())


if __name__ == "__main__":
    main()
