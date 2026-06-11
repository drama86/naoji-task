"""Batch preprocessing ablations for time-window and trial-filter studies."""

import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from machine_learning_config import PREPROCESSING_CONFIG
from machine_learning_process import parse_args, run_experiment
from machine_learning_reporting import write_csv, write_json
from results_layout import PREPROCESSED_TUNING_ROOT, timestamped_name


EXPERIMENTS = (
    {
        "name": "tw_full_keep",
        "description": "完整 0-10 s，不剔除异常 trial",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "time_window_seconds": None,
            "drop_flagged_trials": False,
        },
    },
    {
        "name": "tw_1p0_5p0_keep",
        "description": "1.0-5.0 s，不剔除异常 trial",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "time_window_seconds": (1.0, 5.0),
            "drop_flagged_trials": False,
        },
    },
    {
        "name": "tw_1p5_5p5_keep",
        "description": "1.5-5.5 s，不剔除异常 trial",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
    },
    {
        "name": "tw_2p0_6p0_keep",
        "description": "2.0-6.0 s，不剔除异常 trial",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "time_window_seconds": (2.0, 6.0),
            "drop_flagged_trials": False,
        },
    },
    {
        "name": "tw_1p5_5p5_drop_either",
        "description": "1.5-5.5 s，按 peak-to-peak 或 RMS 异常标记剔除 trial",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": True,
            "drop_flagged_trials_rule": "either",
        },
    },
)


def _read_summary_metrics(summary_path):
    """Read summary metrics CSV into a list of dictionaries."""
    with Path(summary_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_dropped_trials(dropped_path):
    """Read dropped-trial JSON and return the total number of removed samples."""
    import json

    with Path(dropped_path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return int(sum(record["dropped_count"] for record in records))


def _build_args(output_root, run_name):
    """Create experiment args based on the default ML process settings."""
    args = parse_args()
    args.output_root = Path(output_root)
    args.run_name = run_name
    return args


def run_preprocessing_experiments():
    """Run all predefined preprocessing ablations and save comparison tables."""
    original_preprocessing = deepcopy(PREPROCESSING_CONFIG)
    batch_name = timestamped_name("preprocessing_tuning")
    batch_root = PREPROCESSED_TUNING_ROOT / "preprocessing_tuning" / batch_name
    batch_root.mkdir(parents=True, exist_ok=False)

    all_summary_records = []
    best_records = []
    experiment_manifest = []

    try:
        for experiment in EXPERIMENTS:
            PREPROCESSING_CONFIG.clear()
            PREPROCESSING_CONFIG.update(deepcopy(original_preprocessing))
            PREPROCESSING_CONFIG.update(experiment["preprocessing_overrides"])
            run_name = experiment["name"]
            print(f"运行预处理实验：{run_name}")
            run_dir = run_experiment(_build_args(batch_root, run_name))
            summary_records = _read_summary_metrics(run_dir / "summary_metrics.csv")
            dropped_count = _read_dropped_trials(run_dir / "dropped_trials.json")
            manifest_record = {
                "experiment": run_name,
                "description": experiment["description"],
                "run_dir": str(run_dir),
                "dropped_trial_count": dropped_count,
                "preprocessing": deepcopy(PREPROCESSING_CONFIG),
            }
            experiment_manifest.append(manifest_record)

            for row in summary_records:
                enriched = dict(row)
                enriched.update(
                    {
                        "experiment": run_name,
                        "description": experiment["description"],
                        "dropped_trial_count": dropped_count,
                        "time_window_seconds": str(
                            PREPROCESSING_CONFIG["time_window_seconds"]
                        ),
                        "drop_flagged_trials": str(
                            PREPROCESSING_CONFIG["drop_flagged_trials"]
                        ),
                        "drop_flagged_trials_rule": PREPROCESSING_CONFIG[
                            "drop_flagged_trials_rule"
                        ],
                        "spatial_reference": str(
                            PREPROCESSING_CONFIG["spatial_reference"]
                        ),
                    }
                )
                all_summary_records.append(enriched)

            for task_name in sorted({row["task"] for row in summary_records}):
                task_rows = [row for row in summary_records if row["task"] == task_name]
                best = max(task_rows, key=lambda row: float(row["balanced_accuracy"]))
                best_records.append(
                    {
                        "experiment": run_name,
                        "description": experiment["description"],
                        "task": task_name,
                        "best_classifier": best["classifier"],
                        "balanced_accuracy": best["balanced_accuracy"],
                        "macro_f1": best["macro_f1"],
                        "dropped_trial_count": dropped_count,
                        "time_window_seconds": str(
                            PREPROCESSING_CONFIG["time_window_seconds"]
                        ),
                        "drop_flagged_trials": str(
                            PREPROCESSING_CONFIG["drop_flagged_trials"]
                        ),
                        "spatial_reference": str(
                            PREPROCESSING_CONFIG["spatial_reference"]
                        ),
                    }
                )
    finally:
        PREPROCESSING_CONFIG.clear()
        PREPROCESSING_CONFIG.update(original_preprocessing)

    write_json(batch_root / "experiment_manifest.json", experiment_manifest)
    write_csv(
        batch_root / "all_summary_metrics.csv",
        all_summary_records,
        list(all_summary_records[0]),
    )
    write_csv(
        batch_root / "best_by_task.csv",
        best_records,
        list(best_records[0]),
    )
    print(f"预处理对比实验完成：{batch_root}")
    return batch_root


def main():
    """Run the fixed preprocessing ablation batch."""
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_preprocessing_experiments()


if __name__ == "__main__":
    main()
