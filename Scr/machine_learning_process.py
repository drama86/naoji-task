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
from feature_extraction import extract_basic_features
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
    plot_confusion_matrix,
    plot_fold_accuracy,
    plot_model_comparison,
    write_csv,
    write_json,
    write_markdown_report,
)
from model_evaluation import evaluate_classifiers
from preprocessing import preprocess_eeg


def parse_args():
    """Return fixed parameters suitable for direct VS Code F5 debugging.

    Edit values in this function to switch tasks, classifiers, or validation.
    The default runs 2-, 4-, and 6-class LOSO experiments in that order.
    """
    project_root = Path(__file__).resolve().parents[1]
    return argparse.Namespace(
        # 六个 sub_XX_MI.mat 文件所在目录，一般不需要修改。
        data_dir=project_root / "Data",
        # 实验结果根目录；每次运行会在其下新建时间戳子目录。
        output_root=project_root / "Results" / "machine_learning",
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


def load_preprocessed_features(data_dir):
    """Load each subject, preprocess it, and immediately extract features.

    Returns:
        features: Shape (720, feature_count).
        y: Shape (720,).
        subject_ids: Shape (720,).
        feature_names: Stable feature-column names.
        feature_config: JSON-serializable extraction configuration.
    """
    subject_files = sorted(Path(data_dir).glob("sub_*_MI.mat"))
    if len(subject_files) != 6:
        raise ValueError(
            f"Expected 6 subject files, found {len(subject_files)}"
        )
    feature_blocks = []
    label_blocks = []
    subject_blocks = []
    reference_names = None
    reference_config = None

    for subject_file in subject_files:
        subject_id = int(subject_file.stem.split("_")[1])
        X, y = load_subject_data(subject_file, channels=CHANNEL_COUNT)
        X_processed = preprocess_eeg(
            X,
            sampling_rate=SAMPLING_RATE,
            low_cut=PREPROCESSING_CONFIG["low_cut_hz"],
            high_cut=PREPROCESSING_CONFIG["high_cut_hz"],
            filter_order=PREPROCESSING_CONFIG["filter_order"],
            time_window=PREPROCESSING_CONFIG["time_window_seconds"],
        )
        features, feature_names, feature_config = extract_basic_features(
            X_processed,
            feature_sets=FEATURE_CONFIG["feature_sets"],
            sampling_rate=SAMPLING_RATE,
            frequency_bands=FEATURE_CONFIG["frequency_bands"],
            total_power_band=FEATURE_CONFIG["total_power_band"],
            nperseg=FEATURE_CONFIG["nperseg"],
            noverlap=FEATURE_CONFIG["noverlap"],
        )
        if reference_names is None:
            reference_names = feature_names
            reference_config = feature_config
        elif feature_names != reference_names:
            raise RuntimeError("Feature columns differ between subjects")
        feature_blocks.append(features)
        label_blocks.append(y)
        subject_blocks.append(
            np.full(y.shape, subject_id, dtype=np.int64)
        )
        print(
            f"被试 {subject_id}：原始 {X.shape}，"
            f"处理后 {X_processed.shape}，特征 {features.shape}"
        )

    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
        np.concatenate(subject_blocks, axis=0),
        reference_names,
        reference_config,
    )


def build_feature_task(features, y, subject_ids, task_name):
    """Filter already extracted per-trial features for one explicit task."""
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
        features[mask].copy(),
        remapped_y,
        subject_ids[mask].copy(),
        metadata,
    )


def _create_run_directory(output_root, run_name):
    """Create a non-overwriting experiment directory."""
    if run_name is None:
        run_name = datetime.now().strftime("course_report_%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "course_report" / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)
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
    print("阶段 1/4：读取、预处理并提取全部被试特征")
    features, y, subject_ids, feature_names, feature_config = (
        load_preprocessed_features(args.data_dir)
    )

    experiment_config = {
        "random_seed": RANDOM_SEED,
        "subject_count": int(np.unique(subject_ids).size),
        "sampling_rate_hz": SAMPLING_RATE,
        "channel_count": CHANNEL_COUNT,
        "preprocessing": PREPROCESSING_CONFIG,
        "feature_extraction": feature_config,
        "feature_count": len(feature_names),
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
            "Per-trial features use no cross-sample fitted statistics. "
            "StandardScaler and classifier are fitted inside each training fold."
        ),
    }
    write_json(run_dir / "experiment_config.json", experiment_config)
    write_json(run_dir / "feature_names.json", feature_names)

    all_fold_records = []
    all_summary_records = []
    task_reports = {}
    complete_results = {}
    print("阶段 2/4：执行分类任务和交叉验证")
    for task_name in args.tasks:
        task_features, task_y, task_subject_ids, metadata = (
            build_feature_task(features, y, subject_ids, task_name)
        )
        print(
            f"{task_name}：样本 {task_features.shape[0]}，"
            f"特征 {task_features.shape[1]}"
        )
        results = evaluate_classifiers(
            task_features,
            task_y,
            task_subject_ids,
            args.classifiers,
            args.validation_strategy,
            random_seed=RANDOM_SEED,
        )
        complete_results[task_name] = results
        fold_records = flatten_fold_records(task_name, results)
        summary_records = flatten_summary_records(task_name, results)
        all_fold_records.extend(fold_records)
        all_summary_records.extend(summary_records)

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
        confusion_figures = {}
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
        task_reports[task_name] = {
            "task_display_name": metadata["task_name"],
            "sample_count": int(task_features.shape[0]),
            "class_names": metadata["class_names"],
            "summary_records": summary_records,
            "comparison_figure": comparison_path.relative_to(
                run_dir
            ).as_posix(),
            "fold_figure": fold_path.relative_to(run_dir).as_posix(),
            "confusion_figures": confusion_figures,
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
    write_json(run_dir / "complete_results.json", complete_results)

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
