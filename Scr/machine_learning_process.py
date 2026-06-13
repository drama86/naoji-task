"""Debuggable orchestration entry for the classical ML course report."""

import argparse
from datetime import datetime
import os
from pathlib import Path

import numpy as np

# Avoid joblib's deprecated WMIC-based physical-core probe on newer Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from classifiers import classifier_parameters
from data_loader import load_subject_data
from feature_extraction import expected_feature_names, summarize_feature_config
from machine_learning_config import (
    CHANNEL_COUNT,
    CLASSIFICATION_TASKS,
    CLASS_NAMES,
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_NAMES,
    FEATURE_CONFIG,
    PREPROCESSING_CONFIG,
    RANDOM_SEED,
    SAMPLING_RATE,
    VALIDATION_STRATEGIES,
)
from machine_learning_reporting import (
    flatten_fold_records,
    flatten_summary_records,
    flatten_subject_records,
    plot_confusion_matrix,
    plot_fold_accuracy,
    plot_model_comparison,
    plot_subject_metric_comparison,
    write_csv,
    write_json,
    write_markdown_report,
)
from model_evaluation import evaluate_classifiers
from preprocessing import preprocess_eeg, summarize_trial_quality
from results_layout import (
    COURSE_REPORTS_ROOT,
    PROJECT_ROOT,
    ensure_directory,
)


def parse_args():
    """Return fixed parameters suitable for direct VS Code F5 debugging.

    Edit values in this function to switch tasks, classifiers, or validation.
    The default runs 2-, 4-, and 6-class within-subject 5-fold experiments.
    """
    return argparse.Namespace(
        # 六个 sub_XX_MI.mat 文件所在目录，一般不需要修改。
        data_dir=PROJECT_ROOT / "Data",
        # 实验结果根目录；每次运行会在其下新建时间戳子目录。
        output_root=COURSE_REPORTS_ROOT.parent,
        # 分类任务及执行顺序。
        # 可改为 ("2class",) 只跑二分类，或任意组合/顺序。
        tasks=("2class", "4class", "6class"),
        # 本次要运行的分类器。
        # 可改为 ("svm", "lda") 等；合法名称见 CLASSIFIER_NAMES。
        classifiers=CLASSIFIER_NAMES,
        # "loso"：跨被试留一验证；更严格，通常分数较低。
        # "within_subject_5fold"：每个被试内部做分层 5 折。
        validation_strategy="within_subject_5fold",
        # 结果目录名称。None 使用 course_report_日期_时间，避免覆盖。
        # 可设为 "time_window_1_5s" 等有意义名称，但名称不能已存在。
        run_name=None,
    )


def preprocess_subject_file(subject_file, preprocessing_config=None):
    """Load and preprocess one subject file.

    Returns:
        X_processed: Shape (samples, channels, time_points).
        y_processed: Shape (samples,).
        processed_shape: Shared signal shape excluding sample axis.
        quality_summary: Trial-quality diagnostics for the subject.
        dropped_summary: Dropped-trial summary for the subject.
    """
    if preprocessing_config is None:
        preprocessing_config = PREPROCESSING_CONFIG
    subject_file = Path(subject_file)
    subject_id = int(subject_file.stem.split("_")[1])
    X, y = load_subject_data(subject_file, channels=CHANNEL_COUNT)
    X_processed = preprocess_eeg(
        X,
        sampling_rate=SAMPLING_RATE,
        low_cut=preprocessing_config["low_cut_hz"],
        high_cut=preprocessing_config["high_cut_hz"],
        filter_order=preprocessing_config["filter_order"],
        time_window=preprocessing_config["time_window_seconds"],
        notch_hz=preprocessing_config["notch_hz"],
        notch_quality_factor=preprocessing_config["notch_quality_factor"],
        spatial_reference=preprocessing_config["spatial_reference"],
        normalize_mode=preprocessing_config["normalize_mode"],
    )
    quality_summary = summarize_trial_quality(
        X_processed,
        robust_z_threshold=preprocessing_config[
            "trial_quality_robust_z_threshold"
        ],
    )
    quality_summary.update(
        {
            "subject_id": subject_id,
            "source_file": subject_file.name,
        }
    )

    if preprocessing_config["drop_flagged_trials"]:
        flagged_rms = set(
            quality_summary["trial_rms"]["flagged_trial_indices_zero_based"]
        )
        flagged_peak_to_peak = set(
            quality_summary["trial_peak_to_peak"][
                "flagged_trial_indices_zero_based"
            ]
        )
        rule = preprocessing_config["drop_flagged_trials_rule"]
        if rule == "rms":
            dropped_indices = sorted(flagged_rms)
        elif rule == "peak_to_peak":
            dropped_indices = sorted(flagged_peak_to_peak)
        elif rule == "either":
            dropped_indices = sorted(flagged_rms | flagged_peak_to_peak)
        else:
            raise ValueError(
                "drop_flagged_trials_rule must be 'peak_to_peak', "
                "'rms', or 'either'"
            )
        keep_mask = np.ones(X_processed.shape[0], dtype=bool)
        keep_mask[dropped_indices] = False
        X_processed = X_processed[keep_mask]
        y = y[keep_mask]
    else:
        dropped_indices = []

    dropped_summary = {
        "subject_id": subject_id,
        "source_file": subject_file.name,
        "dropped_trial_indices_zero_based": dropped_indices,
        "dropped_count": len(dropped_indices),
        "remaining_sample_count": int(y.shape[0]),
    }
    return (
        X_processed,
        y,
        list(X_processed.shape[1:]),
        quality_summary,
        dropped_summary,
    )


def load_preprocessed_data(data_dir, preprocessing_config=None):
    """Load each subject and return preprocessed trial signals.

    Returns:
        signals: Shape (samples, channels, time_points).
        y: Shape (720,).
        subject_ids: Shape (720,).
        processed_shape_note: Shared processed signal shape excluding samples.
    """
    subject_files = sorted(Path(data_dir).glob("sub_*_MI.mat"))
    if len(subject_files) != 6:
        raise ValueError(
            f"Expected 6 subject files, found {len(subject_files)}"
        )
    signal_blocks = []
    label_blocks = []
    subject_blocks = []
    processed_shape_note = None
    subject_quality_summaries = []
    dropped_trial_summary = []

    for subject_file in subject_files:
        subject_id = int(subject_file.stem.split("_")[1])
        (
            X_processed,
            y,
            subject_processed_shape,
            quality_summary,
            dropped_summary,
        ) = preprocess_subject_file(
            subject_file,
            preprocessing_config=preprocessing_config,
        )
        original_sample_count = int(y.shape[0]) + int(
            dropped_summary["dropped_count"]
        )
        subject_quality_summaries.append(quality_summary)
        dropped_trial_summary.append(dropped_summary)

        if processed_shape_note is None:
            processed_shape_note = subject_processed_shape
        elif subject_processed_shape != processed_shape_note:
            raise RuntimeError("Processed signal shapes differ between subjects")
        signal_blocks.append(X_processed)
        label_blocks.append(y)
        subject_blocks.append(
            np.full(y.shape, subject_id, dtype=np.int64)
        )
        print(
            f"被试 {subject_id}：原始样本 {original_sample_count}，"
            f"处理后 {X_processed.shape}"
        )

    return (
        np.concatenate(signal_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
        np.concatenate(subject_blocks, axis=0),
        processed_shape_note,
        subject_quality_summaries,
        dropped_trial_summary,
    )


def build_signal_task(signals, y, subject_ids, task_name):
    """Filter preprocessed trial signals for one explicit classification task."""
    if task_name not in CLASSIFICATION_TASKS:
        raise KeyError(f"Unknown task: {task_name}")
    task = CLASSIFICATION_TASKS[task_name]
    class_ids = task["class_ids"]
    mask = np.isin(y, class_ids)
    mapping = {
        original_label: new_label
        for new_label, original_label in enumerate(class_ids)
    }
    remapped_y = np.fromiter(
        (mapping[int(label)] for label in y[mask]),
        dtype=np.int64,
        count=int(mask.sum()),
    )
    metadata = {
        "task_key": task_name,
        "task_name": task["name"],
        "original_class_ids": list(class_ids),
        "class_names": [CLASS_NAMES[class_id] for class_id in class_ids],
        "label_mapping": mapping,
    }
    return (
        signals[mask].copy(),
        remapped_y,
        subject_ids[mask].copy(),
        metadata,
    )


def _create_run_directory(output_root, run_name):
    """Create a non-overwriting experiment directory."""
    if run_name is None:
        run_name = datetime.now().strftime("course_report_%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "course_reports" / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {run_dir}"
        )
    ensure_directory(run_dir)
    return run_dir


def run_experiment(args):
    """Run the configured end-to-end classical ML experiment."""
    for task_name in args.tasks:
        if task_name not in CLASSIFICATION_TASKS:
            raise KeyError(f"Unknown task: {task_name}")
    for classifier_name in args.classifiers:
        if classifier_name not in CLASSIFIER_NAMES:
            raise KeyError(f"Unknown classifier: {classifier_name}")
    if args.validation_strategy not in VALIDATION_STRATEGIES:
        raise KeyError(
            f"Unknown validation strategy: {args.validation_strategy}"
        )

    run_dir = _create_run_directory(args.output_root, args.run_name)
    figures_dir = run_dir / "figures"
    print(f"实验输出目录：{run_dir}")
    print("阶段 1/4：读取并预处理全部被试信号")
    (
        signals,
        y,
        subject_ids,
        processed_shape_note,
        trial_quality_summary,
        dropped_trial_summary,
    ) = load_preprocessed_data(args.data_dir)

    experiment_config = {
        "random_seed": RANDOM_SEED,
        "subject_count": int(np.unique(subject_ids).size),
        "sampling_rate_hz": SAMPLING_RATE,
        "channel_count": CHANNEL_COUNT,
        "preprocessing": PREPROCESSING_CONFIG,
        "processed_signal_shape": processed_shape_note,
        "feature_extraction": summarize_feature_config(
            FEATURE_CONFIG,
            sampling_rate=SAMPLING_RATE,
        ),
        "tasks": list(args.tasks),
        "classifiers": list(args.classifiers),
        "classifier_display_names": {
            name: CLASSIFIER_DISPLAY_NAMES[name]
            for name in args.classifiers
        },
        "classifier_parameters": {
            name: classifier_parameters(name)
            for name in args.classifiers
        },
        "validation_strategy": args.validation_strategy,
        "validation_strategy_display": VALIDATION_STRATEGIES[
            args.validation_strategy
        ],
        "leakage_control": (
            "For fold-fitted feature extractors such as CSP/FBCSP, "
            "the feature extractor, StandardScaler, and classifier are "
            "all fitted inside each training fold only."
        ),
    }
    write_json(run_dir / "experiment_config.json", experiment_config)
    write_json(run_dir / "trial_quality_summary.json", trial_quality_summary)
    write_json(run_dir / "dropped_trials.json", dropped_trial_summary)

    all_fold_records = []
    all_summary_records = []
    all_subject_records = []
    task_reports = {}
    complete_results = {}
    feature_names_by_task = {}
    print("阶段 2/4：执行分类任务和交叉验证")
    for task_name in args.tasks:
        task_signals, task_y, task_subject_ids, metadata = (
            build_signal_task(signals, y, subject_ids, task_name)
        )
        task_feature_names = expected_feature_names(
            FEATURE_CONFIG,
            class_count=len(metadata["class_names"]),
            channel_count=CHANNEL_COUNT,
        )
        feature_names_by_task[task_name] = task_feature_names
        print(
            f"{task_name}：样本 {task_signals.shape[0]}，"
            f"信号形状 {task_signals.shape[1:]}"
        )
        results = evaluate_classifiers(
            task_signals,
            task_y,
            task_subject_ids,
            args.classifiers,
            args.validation_strategy,
            feature_config=FEATURE_CONFIG,
            sampling_rate=SAMPLING_RATE,
            random_seed=RANDOM_SEED,
        )
        complete_results[task_name] = results
        fold_records = flatten_fold_records(task_name, results)
        summary_records = flatten_summary_records(task_name, results)
        subject_summary_records = flatten_subject_records(task_name, results)
        all_fold_records.extend(fold_records)
        all_summary_records.extend(summary_records)
        all_subject_records.extend(subject_summary_records)

        print("阶段 3/4：保存指标和绘制结果图")
        task_figures_dir = figures_dir / task_name
        comparison_path = plot_model_comparison(
            results,
            task_figures_dir / "model_comparison.png",
            f"{task_name} 模型指标对比",
        )
        fold_path = plot_fold_accuracy(
            results,
            task_figures_dir / "fold_balanced_accuracy.png",
            f"{task_name} 各折 Balanced Accuracy",
        )
        subject_metric_path = plot_subject_metric_comparison(
            results,
            task_figures_dir / "subject_balanced_accuracy.png",
            f"{task_name} 各被试 Balanced Accuracy",
            metric="balanced_accuracy",
        )
        confusion_figures = {}
        subject_confusion_figures = {}
        for classifier_name, result in results.items():
            count_path = task_figures_dir / (
                f"{classifier_name}_confusion_counts.png"
            )
            normalized_path = task_figures_dir / (
                f"{classifier_name}_confusion_normalized.png"
            )
            plot_confusion_matrix(
                result["confusion_matrix"],
                metadata["class_names"],
                count_path,
                f"{task_name} | {CLASSIFIER_DISPLAY_NAMES[classifier_name]}",
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
            for subject_record in result.get("subject_records", []):
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
            subject_confusion_figures[classifier_name] = (
                classifier_subject_figures
            )
        task_reports[task_name] = {
            "task_display_name": metadata["task_name"],
            "sample_count": int(task_signals.shape[0]),
            "class_names": metadata["class_names"],
            "feature_count": len(task_feature_names),
            "summary_records": summary_records,
            "subject_summary_records": sorted(
                subject_summary_records,
                key=lambda item: (item["subject_id"], item["classifier"]),
            ),
            "comparison_figure": comparison_path.relative_to(
                run_dir
            ).as_posix(),
            "fold_figure": fold_path.relative_to(run_dir).as_posix(),
            "subject_metric_figure": subject_metric_path.relative_to(
                run_dir
            ).as_posix(),
            "confusion_figures": confusion_figures,
            "subject_confusion_figures": subject_confusion_figures,
        }

    fold_fields = [
        "task",
        "classifier",
        "validation_strategy",
        "fold",
        "subject_id",
        "subject_fold",
        "train_sample_count",
        "test_sample_count",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    summary_fields = list(all_summary_records[0])
    write_csv(run_dir / "fold_metrics.csv", all_fold_records, fold_fields)
    write_csv(
        run_dir / "summary_metrics.csv",
        all_summary_records,
        summary_fields,
    )
    subject_fields = list(all_subject_records[0])
    write_csv(
        run_dir / "subject_metrics.csv",
        all_subject_records,
        subject_fields,
    )
    write_json(run_dir / "complete_results.json", complete_results)
    write_json(run_dir / "feature_names.json", feature_names_by_task)

    print("阶段 4/4：生成 Markdown 课程汇报")
    report_path = write_markdown_report(
        run_dir / "report.md",
        experiment_config,
        task_reports,
    )
    print(f"实验完成：{report_path}")
    return run_dir


def main():
    """Run the fixed debug configuration."""
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
