"""Batch feature-ablation experiments for MI EEG classification."""

import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from machine_learning_config import (
    CLASSIFIER_NAMES,
    FEATURE_CONFIG,
    PREPROCESSING_CONFIG,
    RANDOM_SEED,
    SAMPLING_RATE,
)
from machine_learning_process import build_signal_task, load_preprocessed_data
from machine_learning_reporting import write_csv, write_json
from model_evaluation import evaluate_classifier
from results_layout import FEATURE_COMPARISONS_ROOT, PROJECT_ROOT, timestamped_name


EXPERIMENTS = (
    {
        "name": "basic_8_30_tw1p5_5p5",
        "description": "基础时域+PSD特征，8-30 Hz 预处理，1.5-5.5 s 时间窗",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "low_cut_hz": 8.0,
            "high_cut_hz": 30.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "basic",
            "selection": {
                "enabled": False,
                "method": "mibif",
                "k_best": 24,
            },
        },
    },
    {
        "name": "fbcsp_ovr_4_40_none",
        "description": "多带 OVR-FBCSP，4-40 Hz 预处理，无 CAR，无特征选择",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "low_cut_hz": 4.0,
            "high_cut_hz": 40.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "fbcsp",
            "selection": {
                "enabled": False,
                "method": "mibif",
                "k_best": 24,
            },
        },
    },
    {
        "name": "fbcsp_ovr_4_40_car",
        "description": "多带 OVR-FBCSP，4-40 Hz 预处理，CAR，无特征选择",
        "preprocessing_overrides": {
            "spatial_reference": "car",
            "low_cut_hz": 4.0,
            "high_cut_hz": 40.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "fbcsp",
            "selection": {
                "enabled": False,
                "method": "mibif",
                "k_best": 24,
            },
        },
    },
    {
        "name": "fbcsp_ovr_4_40_none_mibif24",
        "description": "多带 OVR-FBCSP，4-40 Hz 预处理，无 CAR，MIBIF 24 维",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "low_cut_hz": 4.0,
            "high_cut_hz": 40.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "fbcsp",
            "selection": {
                "enabled": True,
                "method": "mibif",
                "k_best": 24,
            },
        },
    },
    {
        "name": "fbcsp_ovr_4_40_none_mibif36",
        "description": "多带 OVR-FBCSP，4-40 Hz 预处理，无 CAR，MIBIF 36 维",
        "preprocessing_overrides": {
            "spatial_reference": None,
            "low_cut_hz": 4.0,
            "high_cut_hz": 40.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "fbcsp",
            "selection": {
                "enabled": True,
                "method": "mibif",
                "k_best": 36,
            },
        },
    },
    {
        "name": "fbcsp_ovr_4_40_car_mibif24",
        "description": "多带 OVR-FBCSP，4-40 Hz 预处理，CAR，MIBIF 24 维",
        "preprocessing_overrides": {
            "spatial_reference": "car",
            "low_cut_hz": 4.0,
            "high_cut_hz": 40.0,
            "time_window_seconds": (1.5, 5.5),
            "drop_flagged_trials": False,
        },
        "feature_overrides": {
            "mode": "fbcsp",
            "selection": {
                "enabled": True,
                "method": "mibif",
                "k_best": 24,
            },
        },
    },
)

TASKS = ("2class", "4class", "6class")
CLASSIFIERS = ("lda", "logistic_regression", "svm")


def _flatten_result(experiment_name, experiment_description, task_name, classifier_name, result):
    metrics = result["out_of_fold_metrics"]
    return {
        "experiment": experiment_name,
        "description": experiment_description,
        "task": task_name,
        "classifier": classifier_name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "fold_count": result["fold_count"],
    }


def run_feature_experiments():
    """Run predefined feature-engineering experiments and save comparisons."""
    original_preprocessing = deepcopy(PREPROCESSING_CONFIG)
    original_feature = deepcopy(FEATURE_CONFIG)
    batch_name = timestamped_name("feature_tuning")
    batch_root = FEATURE_COMPARISONS_ROOT / "feature_tuning" / batch_name
    batch_root.mkdir(parents=True, exist_ok=False)

    all_summary_records = []
    best_records = []
    experiment_manifest = []

    try:
        for experiment in EXPERIMENTS:
            PREPROCESSING_CONFIG.clear()
            PREPROCESSING_CONFIG.update(deepcopy(original_preprocessing))
            PREPROCESSING_CONFIG.update(experiment["preprocessing_overrides"])

            FEATURE_CONFIG.clear()
            FEATURE_CONFIG.update(deepcopy(original_feature))
            FEATURE_CONFIG.update(experiment["feature_overrides"])

            print(f"运行特征实验：{experiment['name']}")
            signals, y, subject_ids, processed_shape, quality, dropped = (
                load_preprocessed_data(PROJECT_ROOT / "Data")
            )

            experiment_manifest.append(
                {
                    "experiment": experiment["name"],
                    "description": experiment["description"],
                    "preprocessing": deepcopy(PREPROCESSING_CONFIG),
                    "feature_config": deepcopy(FEATURE_CONFIG),
                    "processed_shape": processed_shape,
                }
            )

            task_rows = []
            for task_name in TASKS:
                task_signals, task_y, task_subject_ids, metadata = build_signal_task(
                    signals,
                    y,
                    subject_ids,
                    task_name,
                )
                for classifier_name in CLASSIFIERS:
                    result = evaluate_classifier(
                        task_signals,
                        task_y,
                        task_subject_ids,
                        classifier_name,
                        "within_subject_5fold",
                        feature_config=FEATURE_CONFIG,
                        sampling_rate=SAMPLING_RATE,
                        random_seed=RANDOM_SEED,
                    )
                    row = _flatten_result(
                        experiment["name"],
                        experiment["description"],
                        task_name,
                        classifier_name,
                        result,
                    )
                    all_summary_records.append(row)
                    task_rows.append(row)

            for task_name in TASKS:
                best = max(
                    [row for row in task_rows if row["task"] == task_name],
                    key=lambda row: row["balanced_accuracy"],
                )
                best_records.append(best)
    finally:
        PREPROCESSING_CONFIG.clear()
        PREPROCESSING_CONFIG.update(original_preprocessing)
        FEATURE_CONFIG.clear()
        FEATURE_CONFIG.update(original_feature)

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
    print(f"特征对比实验完成：{batch_root}")
    return batch_root


def main():
    """Run the fixed feature-ablation batch."""
    run_feature_experiments()


if __name__ == "__main__":
    main()
