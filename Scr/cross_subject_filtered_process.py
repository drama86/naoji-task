"""Exploratory LOSO experiment with poor training subjects removed."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from classifiers import create_model_pipeline, classifier_parameters
from feature_extraction import expected_feature_names, summarize_feature_config
from machine_learning_config import (
    CHANNEL_COUNT,
    CLASSIFICATION_TASKS,
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_NAMES,
    FEATURE_CONFIG,
    PREPROCESSING_CONFIG,
    RANDOM_SEED,
    SAMPLING_RATE,
    VALIDATION_STRATEGIES,
)
from machine_learning_process import build_signal_task, load_preprocessed_data
from machine_learning_reporting import (
    flatten_summary_records,
    flatten_subject_records,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_subject_metric_comparison,
    write_csv,
    write_json,
    write_markdown_report,
)
from model_evaluation import SUMMARY_METRICS, _classification_metrics
from results_layout import COURSE_REPORTS_ROOT, ensure_directory


WITHIN_SUBJECT_REFERENCE = (
    COURSE_REPORTS_ROOT / "course_report_20260613_120032" / "subject_metrics.csv"
)
REMOVED_TRAIN_SUBJECTS_PER_FOLD = 1


def _create_run_directory():
    run_name = datetime.now().strftime(
        "course_report_loso_filtered_%Y%m%d_%H%M%S"
    )
    run_dir = COURSE_REPORTS_ROOT / run_name
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    ensure_directory(run_dir)
    return run_dir


def _load_within_subject_ranking(subject_metrics_path):
    """Rank subjects by their best within-subject accuracy for each task."""
    rows = list(
        csv.DictReader(
            Path(subject_metrics_path).open("r", encoding="utf-8-sig")
        )
    )
    task_scores = defaultdict(dict)
    for row in rows:
        task = row["task"]
        subject_id = int(row["subject_id"])
        accuracy = float(row["accuracy"])
        best = task_scores[task].get(subject_id)
        if best is None or accuracy > best:
            task_scores[task][subject_id] = accuracy
    return {
        task: [
            subject_id
            for subject_id, _ in sorted(
                subject_scores.items(),
                key=lambda item: (item[1], item[0]),
            )
        ]
        for task, subject_scores in task_scores.items()
    }


def _build_filtered_loso_splits(subject_ids, ranked_subjects, remove_count):
    """Create LOSO splits after removing worst training subjects per fold."""
    subject_ids = np.asarray(subject_ids)
    splits = []
    for fold_index, held_out_subject in enumerate(np.unique(subject_ids), start=1):
        test_indices = np.flatnonzero(subject_ids == held_out_subject)
        training_subjects = [
            subject_id
            for subject_id in ranked_subjects
            if subject_id != int(held_out_subject)
        ]
        removed_subjects = training_subjects[:remove_count]
        kept_subjects = training_subjects[remove_count:]
        train_mask = np.isin(subject_ids, kept_subjects)
        train_indices = np.flatnonzero(train_mask)
        if train_indices.size == 0:
            raise RuntimeError("No training samples remain after subject filtering")
        splits.append(
            {
                "fold": fold_index,
                "held_out_subject": int(held_out_subject),
                "train_indices": train_indices,
                "test_indices": test_indices,
                "removed_training_subject_ids": removed_subjects,
                "kept_training_subject_ids": kept_subjects,
            }
        )
    return splits


def _evaluate_filtered_classifier(
    task_name,
    signals,
    y,
    subject_ids,
    classifier_name,
    ranked_subjects,
):
    labels = np.unique(y)
    base_estimator = create_model_pipeline(
        classifier_name,
        FEATURE_CONFIG,
        sampling_rate=SAMPLING_RATE,
        random_seed=RANDOM_SEED,
    )
    splits = _build_filtered_loso_splits(
        subject_ids,
        ranked_subjects=ranked_subjects,
        remove_count=REMOVED_TRAIN_SUBJECTS_PER_FOLD,
    )
    predictions = np.full(y.shape, -1, dtype=np.int64)
    prediction_counts = np.zeros(y.shape, dtype=np.int64)
    fold_records = []

    for split in splits:
        estimator = clone(base_estimator)
        estimator.fit(signals[split["train_indices"]], y[split["train_indices"]])
        fold_predictions = estimator.predict(signals[split["test_indices"]])
        predictions[split["test_indices"]] = fold_predictions
        prediction_counts[split["test_indices"]] += 1
        overall, per_class = _classification_metrics(
            y[split["test_indices"]],
            fold_predictions,
            labels,
        )
        fold_records.append(
            {
                "fold": split["fold"],
                "held_out_subject": split["held_out_subject"],
                "removed_training_subject_ids": split[
                    "removed_training_subject_ids"
                ],
                "kept_training_subject_ids": split["kept_training_subject_ids"],
                "classifier": classifier_name,
                "train_sample_count": int(split["train_indices"].size),
                "test_sample_count": int(split["test_indices"].size),
                **overall,
                "per_class": per_class,
            }
        )

    if not np.all(prediction_counts == 1):
        raise RuntimeError("Each sample must receive exactly one prediction")

    overall, per_class = _classification_metrics(y, predictions, labels)
    matrix = confusion_matrix(y, predictions, labels=labels)
    normalized_matrix = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    metric_values = defaultdict(list)
    for record in fold_records:
        for metric in SUMMARY_METRICS:
            metric_values[metric].append(record[metric])
    fold_summary = {
        f"{metric}_{statistic}": float(function(metric_values[metric]))
        for metric in SUMMARY_METRICS
        for statistic, function in (("mean", np.mean), ("std", np.std))
    }
    subject_records = []
    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_overall, subject_per_class = _classification_metrics(
            y[mask],
            predictions[mask],
            labels,
        )
        subject_matrix = confusion_matrix(
            y[mask],
            predictions[mask],
            labels=labels,
        )
        subject_normalized = np.divide(
            subject_matrix,
            subject_matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(subject_matrix, dtype=np.float64),
            where=subject_matrix.sum(axis=1, keepdims=True) > 0,
        )
        subject_records.append(
            {
                "subject_id": int(subject_id),
                "sample_count": int(mask.sum()),
                **subject_overall,
                "per_class_metrics": subject_per_class,
                "confusion_matrix": subject_matrix.astype(int).tolist(),
                "normalized_confusion_matrix": subject_normalized.tolist(),
            }
        )
    return {
        "classifier": classifier_name,
        "validation_strategy": "loso_filtered",
        "fold_count": len(fold_records),
        "labels": labels.astype(int).tolist(),
        "fold_records": fold_records,
        "fold_summary": fold_summary,
        "out_of_fold_metrics": overall,
        "per_class_metrics": per_class,
        "subject_records": subject_records,
        "confusion_matrix": matrix.astype(int).tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
        "predictions": predictions.astype(int).tolist(),
        "exploratory_note": (
            "Training folds remove the worst within-subject performers among "
            "available training subjects. This is exploratory and should not "
            "replace the main LOSO result."
        ),
    }


def run_filtered_loso_experiment():
    """Run exploratory filtered LOSO experiments with fixed subject removal."""
    run_dir = _create_run_directory()
    figures_dir = run_dir / "figures"
    (
        signals,
        y,
        subject_ids,
        processed_shape_note,
        trial_quality_summary,
        dropped_trial_summary,
    ) = load_preprocessed_data(Path("Data"))
    ranked_subjects_by_task = _load_within_subject_ranking(
        WITHIN_SUBJECT_REFERENCE
    )

    experiment_config = {
        "random_seed": RANDOM_SEED,
        "sampling_rate_hz": SAMPLING_RATE,
        "channel_count": CHANNEL_COUNT,
        "preprocessing": PREPROCESSING_CONFIG,
        "processed_signal_shape": processed_shape_note,
        "feature_extraction": summarize_feature_config(
            FEATURE_CONFIG,
            sampling_rate=SAMPLING_RATE,
        ),
        "tasks": list(CLASSIFICATION_TASKS),
        "classifiers": list(CLASSIFIER_NAMES),
        "classifier_display_names": CLASSIFIER_DISPLAY_NAMES,
        "classifier_parameters": {
            name: classifier_parameters(name)
            for name in CLASSIFIER_NAMES
        },
        "validation_strategy": "loso_filtered",
        "validation_strategy_display": (
            "Exploratory LOSO after removing poor training subjects"
        ),
        "subject_filtering": {
            "reference_within_subject_metrics": str(WITHIN_SUBJECT_REFERENCE),
            "remove_training_subjects_per_fold": REMOVED_TRAIN_SUBJECTS_PER_FOLD,
            "ranking_rule": (
                "Rank subjects by best within-subject accuracy per task; "
                "remove the lowest-ranked training subject in each LOSO fold."
            ),
            "exploratory_warning": (
                "This filtering uses previous within-subject performance and "
                "is only for exploratory analysis."
            ),
        },
    }
    write_json(run_dir / "experiment_config.json", experiment_config)
    write_json(run_dir / "trial_quality_summary.json", trial_quality_summary)
    write_json(run_dir / "dropped_trials.json", dropped_trial_summary)

    all_summary_records = []
    all_subject_records = []
    complete_results = {}
    feature_names_by_task = {}
    task_reports = {}

    for task_name in ("2class", "4class", "6class"):
        task_signals, task_y, task_subject_ids, metadata = build_signal_task(
            signals,
            y,
            subject_ids,
            task_name,
        )
        results = {
            classifier_name: _evaluate_filtered_classifier(
                task_name,
                task_signals,
                task_y,
                task_subject_ids,
                classifier_name,
                ranked_subjects_by_task[task_name],
            )
            for classifier_name in CLASSIFIER_NAMES
        }
        complete_results[task_name] = results
        summary_records = flatten_summary_records(task_name, results)
        subject_summary_records = flatten_subject_records(task_name, results)
        all_summary_records.extend(summary_records)
        all_subject_records.extend(subject_summary_records)
        feature_names_by_task[task_name] = expected_feature_names(
            FEATURE_CONFIG,
            class_count=len(metadata["class_names"]),
            channel_count=CHANNEL_COUNT,
        )

        task_figures_dir = figures_dir / task_name
        comparison_path = plot_model_comparison(
            results,
            task_figures_dir / "model_comparison.png",
            f"{task_name} 模型指标对比",
        )
        subject_metric_path = plot_subject_metric_comparison(
            results,
            task_figures_dir / "subject_balanced_accuracy.png",
            f"{task_name} 各被试 Balanced Accuracy",
        )
        confusion_figures = {}
        subject_confusion_figures = {}
        for classifier_name, result in results.items():
            normalized_path = task_figures_dir / (
                f"{classifier_name}_confusion_normalized.png"
            )
            plot_confusion_matrix(
                result["normalized_confusion_matrix"],
                metadata["class_names"],
                normalized_path,
                (
                    f"{task_name} | "
                    f"{CLASSIFIER_DISPLAY_NAMES[classifier_name]} | 归一化"
                ),
                normalized=True,
            )
            confusion_figures[classifier_name] = (
                normalized_path.relative_to(run_dir).as_posix()
            )
            classifier_subject_figures = {}
            for subject_record in result["subject_records"]:
                subject_id = subject_record["subject_id"]
                subject_path = task_figures_dir / "subjects" / classifier_name / (
                    f"subject_{subject_id:02d}_confusion_normalized.png"
                )
                plot_confusion_matrix(
                    subject_record["normalized_confusion_matrix"],
                    metadata["class_names"],
                    subject_path,
                    (
                        f"{task_name} | {CLASSIFIER_DISPLAY_NAMES[classifier_name]} "
                        f"| 被试 {subject_id} | 归一化"
                    ),
                    normalized=True,
                )
                classifier_subject_figures[str(subject_id)] = (
                    subject_path.relative_to(run_dir).as_posix()
                )
            subject_confusion_figures[classifier_name] = classifier_subject_figures
        task_reports[task_name] = {
            "task_display_name": metadata["task_name"],
            "sample_count": int(task_signals.shape[0]),
            "class_names": metadata["class_names"],
            "feature_count": len(feature_names_by_task[task_name]),
            "summary_records": summary_records,
            "subject_summary_records": sorted(
                subject_summary_records,
                key=lambda item: (item["subject_id"], item["classifier"]),
            ),
            "comparison_figure": comparison_path.relative_to(run_dir).as_posix(),
            "fold_figure": comparison_path.relative_to(run_dir).as_posix(),
            "subject_metric_figure": subject_metric_path.relative_to(
                run_dir
            ).as_posix(),
            "confusion_figures": confusion_figures,
            "subject_confusion_figures": subject_confusion_figures,
        }

    write_csv(
        run_dir / "summary_metrics.csv",
        all_summary_records,
        list(all_summary_records[0]),
    )
    write_csv(
        run_dir / "subject_metrics.csv",
        all_subject_records,
        list(all_subject_records[0]),
    )
    write_json(run_dir / "complete_results.json", complete_results)
    write_json(run_dir / "feature_names.json", feature_names_by_task)
    write_markdown_report(run_dir / "report.md", experiment_config, task_reports)
    return run_dir


def main():
    """Run the exploratory filtered LOSO experiment."""
    run_dir = run_filtered_loso_experiment()
    print(f"探索性筛除训练被试的 LOSO 实验完成：{run_dir}")


if __name__ == "__main__":
    main()
