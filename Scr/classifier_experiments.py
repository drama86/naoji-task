"""Classifier-focused tuning experiments on the current best front-end."""

from copy import deepcopy

from machine_learning_config import FEATURE_CONFIG, PREPROCESSING_CONFIG, RANDOM_SEED, SAMPLING_RATE
from machine_learning_process import build_signal_task, load_preprocessed_data
from machine_learning_reporting import write_csv, write_json
from model_evaluation import evaluate_classifier
from results_layout import CLASSIFIER_TUNING_ROOT, PROJECT_ROOT, timestamped_name


TASK_CANDIDATES = {
    "2class": {
        "lda": (
            {"name": "lda_default", "overrides": {}},
            {"name": "lda_svd", "overrides": {"solver": "svd", "shrinkage": None}},
            {"name": "lda_eigen_auto", "overrides": {"solver": "eigen", "shrinkage": "auto"}},
            {"name": "lda_lsqr_shrink_0p3", "overrides": {"solver": "lsqr", "shrinkage": 0.3}},
        ),
        "logistic_regression": (
            {"name": "logreg_c0p03_bal", "overrides": {"C": 0.03, "class_weight": "balanced"}},
            {"name": "logreg_c0p1_bal", "overrides": {"C": 0.1, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_bal", "overrides": {"C": 0.3, "class_weight": "balanced"}},
            {"name": "logreg_c1_bal", "overrides": {"C": 1.0, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_none", "overrides": {"C": 0.3, "class_weight": None}},
            {"name": "logreg_c1_none", "overrides": {"C": 1.0, "class_weight": None}},
        ),
        "svm": (
            {"name": "svm_linear_c0p3", "overrides": {"kernel": "linear", "C": 0.3, "class_weight": None}},
            {"name": "svm_linear_c1", "overrides": {"kernel": "linear", "C": 1.0, "class_weight": None}},
            {"name": "svm_linear_c3", "overrides": {"kernel": "linear", "C": 3.0, "class_weight": None}},
            {"name": "svm_rbf_c1_scale", "overrides": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}},
            {"name": "svm_rbf_c3_scale", "overrides": {"kernel": "rbf", "C": 3.0, "gamma": "scale", "class_weight": None}},
            {"name": "svm_rbf_c10_scale", "overrides": {"kernel": "rbf", "C": 10.0, "gamma": "scale", "class_weight": "balanced"}},
        ),
        "random_forest": (
            {"name": "rf_200_sqrt", "overrides": {"n_estimators": 200, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_sqrt", "overrides": {"n_estimators": 500, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_log2", "overrides": {"n_estimators": 500, "max_features": "log2", "class_weight": None}},
            {"name": "rf_800_sqrt", "overrides": {"n_estimators": 800, "max_features": "sqrt", "class_weight": None}},
        ),
    },
    "4class": {
        "lda": (
            {"name": "lda_default", "overrides": {}},
            {"name": "lda_svd", "overrides": {"solver": "svd", "shrinkage": None}},
            {"name": "lda_eigen_auto", "overrides": {"solver": "eigen", "shrinkage": "auto"}},
            {"name": "lda_lsqr_shrink_0p5", "overrides": {"solver": "lsqr", "shrinkage": 0.5}},
        ),
        "logistic_regression": (
            {"name": "logreg_c0p1_bal", "overrides": {"C": 0.1, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_bal", "overrides": {"C": 0.3, "class_weight": "balanced"}},
            {"name": "logreg_c1_bal", "overrides": {"C": 1.0, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_none", "overrides": {"C": 0.3, "class_weight": None}},
        ),
        "svm": (
            {"name": "svm_linear_c0p3", "overrides": {"kernel": "linear", "C": 0.3, "class_weight": None}},
            {"name": "svm_linear_c1", "overrides": {"kernel": "linear", "C": 1.0, "class_weight": None}},
            {"name": "svm_rbf_c1_scale", "overrides": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}},
            {"name": "svm_rbf_c3_scale", "overrides": {"kernel": "rbf", "C": 3.0, "gamma": "scale", "class_weight": None}},
        ),
        "random_forest": (
            {"name": "rf_200_sqrt", "overrides": {"n_estimators": 200, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_sqrt", "overrides": {"n_estimators": 500, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_800_sqrt", "overrides": {"n_estimators": 800, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_log2", "overrides": {"n_estimators": 500, "max_features": "log2", "class_weight": None}},
            {"name": "rf_500_sqrt_bal", "overrides": {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced"}},
            {"name": "rf_800_log2", "overrides": {"n_estimators": 800, "max_features": "log2", "class_weight": None}},
        ),
    },
    "6class": {
        "lda": (
            {"name": "lda_default", "overrides": {}},
            {"name": "lda_svd", "overrides": {"solver": "svd", "shrinkage": None}},
            {"name": "lda_eigen_auto", "overrides": {"solver": "eigen", "shrinkage": "auto"}},
            {"name": "lda_lsqr_shrink_0p5", "overrides": {"solver": "lsqr", "shrinkage": 0.5}},
        ),
        "logistic_regression": (
            {"name": "logreg_c0p1_bal", "overrides": {"C": 0.1, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_bal", "overrides": {"C": 0.3, "class_weight": "balanced"}},
            {"name": "logreg_c1_bal", "overrides": {"C": 1.0, "class_weight": "balanced"}},
            {"name": "logreg_c0p3_none", "overrides": {"C": 0.3, "class_weight": None}},
        ),
        "svm": (
            {"name": "svm_linear_c0p3", "overrides": {"kernel": "linear", "C": 0.3, "class_weight": None}},
            {"name": "svm_linear_c1", "overrides": {"kernel": "linear", "C": 1.0, "class_weight": None}},
            {"name": "svm_rbf_c1_scale", "overrides": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}},
            {"name": "svm_rbf_c3_scale", "overrides": {"kernel": "rbf", "C": 3.0, "gamma": "scale", "class_weight": None}},
        ),
        "random_forest": (
            {"name": "rf_200_sqrt", "overrides": {"n_estimators": 200, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_sqrt", "overrides": {"n_estimators": 500, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_800_sqrt", "overrides": {"n_estimators": 800, "max_features": "sqrt", "class_weight": None}},
            {"name": "rf_500_log2", "overrides": {"n_estimators": 500, "max_features": "log2", "class_weight": None}},
        ),
    },
}


def _recommended_frontend():
    """Return the current most stable preprocessing + feature configuration."""
    preprocessing = {
        "spatial_reference": None,
        "low_cut_hz": 8.0,
        "high_cut_hz": 30.0,
        "time_window_seconds": (1.5, 5.5),
        "drop_flagged_trials": False,
    }
    feature = {
        "mode": "basic",
        "selection": {
            "enabled": False,
            "method": "mibif",
            "k_best": 24,
        },
    }
    return preprocessing, feature


def _flatten_result(task_name, classifier_name, candidate_name, overrides, result):
    metrics = result["out_of_fold_metrics"]
    return {
        "task": task_name,
        "classifier": classifier_name,
        "candidate": candidate_name,
        "classifier_overrides": str(overrides),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "fold_count": result["fold_count"],
    }


def run_classifier_experiments():
    """Tune classifier hyperparameters on the current best front-end."""
    original_preprocessing = deepcopy(PREPROCESSING_CONFIG)
    original_feature = deepcopy(FEATURE_CONFIG)
    batch_name = timestamped_name("classifier_tuning")
    batch_root = CLASSIFIER_TUNING_ROOT / batch_name
    batch_root.mkdir(parents=True, exist_ok=False)

    preprocessing_overrides, feature_overrides = _recommended_frontend()
    all_summary_records = []
    best_records = []
    task_best_records = []

    try:
        PREPROCESSING_CONFIG.clear()
        PREPROCESSING_CONFIG.update(deepcopy(original_preprocessing))
        PREPROCESSING_CONFIG.update(preprocessing_overrides)

        FEATURE_CONFIG.clear()
        FEATURE_CONFIG.update(deepcopy(original_feature))
        FEATURE_CONFIG.update(feature_overrides)

        signals, y, subject_ids, processed_shape, quality, dropped = (
            load_preprocessed_data(PROJECT_ROOT / "Data")
        )

        manifest = {
            "preprocessing": deepcopy(PREPROCESSING_CONFIG),
            "feature_config": deepcopy(FEATURE_CONFIG),
            "processed_shape": processed_shape,
            "task_candidates": TASK_CANDIDATES,
            "selection_note": (
                "This is a development-stage CV tuning sweep on the same "
                "outer validation protocol; confirm final report numbers "
                "with a nested-CV rerun if needed."
            ),
        }

        for task_name, classifier_candidates in TASK_CANDIDATES.items():
            task_signals, task_y, task_subject_ids, metadata = build_signal_task(
                signals,
                y,
                subject_ids,
                task_name,
            )
            print(f"任务 {task_name} 开始调参：样本 {task_signals.shape[0]}")
            task_rows = []
            for classifier_name, candidates in classifier_candidates.items():
                for candidate in candidates:
                    print(
                        f"  {classifier_name} -> {candidate['name']}",
                        flush=True,
                    )
                    result = evaluate_classifier(
                        task_signals,
                        task_y,
                        task_subject_ids,
                        classifier_name,
                        "within_subject_5fold",
                        feature_config=FEATURE_CONFIG,
                        sampling_rate=SAMPLING_RATE,
                        random_seed=RANDOM_SEED,
                        classifier_overrides=candidate["overrides"],
                    )
                    row = _flatten_result(
                        task_name,
                        classifier_name,
                        candidate["name"],
                        candidate["overrides"],
                        result,
                    )
                    all_summary_records.append(row)
                    task_rows.append(row)

            best_overall = max(
                task_rows,
                key=lambda item: item["balanced_accuracy"],
            )
            task_best_records.append(best_overall)

            for classifier_name in classifier_candidates:
                best_in_classifier = max(
                    [row for row in task_rows if row["classifier"] == classifier_name],
                    key=lambda item: item["balanced_accuracy"],
                )
                best_records.append(best_in_classifier)
    finally:
        PREPROCESSING_CONFIG.clear()
        PREPROCESSING_CONFIG.update(original_preprocessing)
        FEATURE_CONFIG.clear()
        FEATURE_CONFIG.update(original_feature)

    write_json(batch_root / "experiment_manifest.json", manifest)
    write_csv(
        batch_root / "all_summary_metrics.csv",
        all_summary_records,
        list(all_summary_records[0]),
    )
    write_csv(
        batch_root / "best_by_classifier.csv",
        best_records,
        list(best_records[0]),
    )
    write_csv(
        batch_root / "best_by_task.csv",
        task_best_records,
        list(task_best_records[0]),
    )
    print(f"分类器调参实验完成：{batch_root}")
    return batch_root


def main():
    """Run the fixed classifier-tuning batch."""
    run_classifier_experiments()


if __name__ == "__main__":
    main()
