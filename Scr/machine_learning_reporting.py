"""Save machine-learning metrics, figures, configuration, and Markdown."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from machine_learning_config import CLASSIFIER_DISPLAY_NAMES
from model_evaluation import SUMMARY_METRICS


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def write_json(file_path, value):
    """Write a JSON-serializable value using UTF-8."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_csv(file_path, records, fieldnames):
    """Write records to a UTF-8 CSV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def flatten_fold_records(task_name, results):
    """Flatten fold metrics from all classifiers for CSV output."""
    records = []
    for classifier_name, result in results.items():
        for fold_record in result["fold_records"]:
            records.append(
                {
                    "task": task_name,
                    "classifier": classifier_name,
                    "validation_strategy": result["validation_strategy"],
                    "fold": fold_record["fold"],
                    "subject_id": fold_record.get(
                        "held_out_subject",
                        fold_record.get("subject_id", ""),
                    ),
                    "subject_fold": fold_record.get("subject_fold", ""),
                    "train_sample_count": fold_record["train_sample_count"],
                    "test_sample_count": fold_record["test_sample_count"],
                    **{
                        metric: fold_record[metric]
                        for metric in SUMMARY_METRICS
                    },
                }
            )
    return records


def flatten_summary_records(task_name, results):
    """Flatten classifier summaries for CSV output."""
    records = []
    for classifier_name, result in results.items():
        record = {
            "task": task_name,
            "classifier": classifier_name,
            "validation_strategy": result["validation_strategy"],
            "fold_count": result["fold_count"],
        }
        record.update(result["out_of_fold_metrics"])
        record.update(result["fold_summary"])
        records.append(record)
    return records


def plot_confusion_matrix(
    matrix,
    class_names,
    output_path,
    title,
    normalized=False,
):
    """Plot a count or row-normalized confusion matrix."""
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=(1 if normalized else None),
    )
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="预测类别",
        ylabel="真实类别",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = (matrix.max() / 2.0) if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_model_comparison(results, output_path, title):
    """Plot main out-of-fold metrics for all classifiers."""
    classifier_names = list(results)
    metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    positions = np.arange(len(classifier_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for metric_index, metric in enumerate(metrics):
        values = [
            results[name]["out_of_fold_metrics"][metric]
            for name in classifier_names
        ]
        axis.bar(
            positions + (metric_index - 1) * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [CLASSIFIER_DISPLAY_NAMES[name] for name in classifier_names],
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("得分")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_fold_accuracy(results, output_path, title):
    """Plot each classifier's accuracy over validation folds."""
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for classifier_name, result in results.items():
        folds = [record["fold"] for record in result["fold_records"]]
        values = [
            record["balanced_accuracy"]
            for record in result["fold_records"]
        ]
        axis.plot(
            folds,
            values,
            marker="o",
            linewidth=1.2,
            markersize=4,
            label=CLASSIFIER_DISPLAY_NAMES[classifier_name],
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("交叉验证折")
    axis.set_ylabel("Balanced Accuracy")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def write_markdown_report(
    output_path,
    experiment_config,
    task_reports,
):
    """Create a self-contained Markdown report referencing generated figures."""
    output_path = Path(output_path)
    lines = [
        "# 运动想象 EEG 经典机器学习实验报告",
        "",
        "## 实验目的",
        "",
        "使用前 30 个 EEG 通道完成运动想象分类，并比较 LDA、SVM、"
        "逻辑回归、KNN 和随机森林等经典分类器。",
        "",
        "## 数据与预处理",
        "",
        f"- 被试数量：{experiment_config['subject_count']}",
        f"- 采样率：{experiment_config['sampling_rate_hz']} Hz",
        f"- 使用通道：前 {experiment_config['channel_count']} 个通道",
        (
            "- 预处理：逐 trial、逐通道去均值，"
            f"{experiment_config['preprocessing']['notch_hz']:g} Hz 陷波，"
            f"{experiment_config['preprocessing']['low_cut_hz']:g}–"
            f"{experiment_config['preprocessing']['high_cut_hz']:g} Hz "
            "Butterworth 零相位带通滤波"
            if experiment_config["preprocessing"]["notch_hz"] is not None
            else (
                "- 预处理：逐 trial、逐通道去均值，"
                f"{experiment_config['preprocessing']['low_cut_hz']:g}–"
                f"{experiment_config['preprocessing']['high_cut_hz']:g} Hz "
                "Butterworth 零相位带通滤波"
            )
        ),
        (
            "- 信号归一化：关闭"
            if experiment_config["preprocessing"]["normalize_mode"] is None
            else (
                "- 信号归一化："
                f"{experiment_config['preprocessing']['normalize_mode']}"
            )
        ),
        (
            "- 时间窗：完整 0–10 s"
            if experiment_config["preprocessing"]["time_window_seconds"] is None
            else (
                "- 时间窗："
                f"{experiment_config['preprocessing']['time_window_seconds']}"
            )
        ),
        "",
        "## 特征与验证方法",
        "",
        (
            "- 特征："
            f"{experiment_config['feature_extraction']['report_description']}"
        ),
        (
            "- 特征选择：关闭"
            if experiment_config["feature_extraction"].get("selection") is None
            else (
                "- 特征选择："
                f"{experiment_config['feature_extraction']['selection']['method']}"
                f"，保留前 "
                f"{experiment_config['feature_extraction']['selection']['k_best']} 维，"
                "并在每个训练折内部拟合。"
            )
        ),
        (
            "- 标准化：在每个训练折内部通过 `StandardScaler` 拟合；"
            "随机森林不执行标准化。"
        ),
        (
            "- 验证策略："
            f"{experiment_config['validation_strategy_display']}。"
        ),
        (
            "- 数据泄漏控制：测试折不参与特征提取器、标准化器"
            "和分类器拟合。"
        ),
        "",
    ]
    for task_name, task_report in task_reports.items():
        lines.extend(
            [
                f"## {task_name}：{task_report['task_display_name']}",
                "",
                (
                    f"样本数：{task_report['sample_count']}；"
                    f"类别：{'、'.join(task_report['class_names'])}；"
                    f"特征维数：{task_report['feature_count']}。"
                ),
                "",
                "| 模型 | Accuracy | Balanced Accuracy | Macro F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for summary in task_report["summary_records"]:
            lines.append(
                "| "
                f"{CLASSIFIER_DISPLAY_NAMES[summary['classifier']]} | "
                f"{summary['accuracy']:.4f} | "
                f"{summary['balanced_accuracy']:.4f} | "
                f"{summary['macro_f1']:.4f} |"
            )
        best = max(
            task_report["summary_records"],
            key=lambda item: item["balanced_accuracy"],
        )
        lines.extend(
            [
                "",
                (
                    "按 Balanced Accuracy，本次实验表现最佳的模型为 "
                    f"**{CLASSIFIER_DISPLAY_NAMES[best['classifier']]}**"
                    f"（{best['balanced_accuracy']:.4f}）。"
                ),
                "",
                "### 模型对比",
                "",
                f"![模型指标对比]({task_report['comparison_figure']})",
                "",
                "### 各折结果",
                "",
                f"![各折结果]({task_report['fold_figure']})",
                "",
                "### 混淆矩阵",
                "",
            ]
        )
        for classifier_name, figure_path in task_report[
            "confusion_figures"
        ].items():
            lines.extend(
                [
                    f"#### {CLASSIFIER_DISPLAY_NAMES[classifier_name]}",
                    "",
                    f"![{classifier_name} 归一化混淆矩阵]({figure_path})",
                    "",
                ]
            )
    lines.extend(
        [
            "## 结果说明",
            "",
            "- 本报告中的指标均来自交叉验证的折外预测。",
            "- 结果用于第一组经典机器学习分析，不包含深度学习模型。",
            "- 时间窗并非数据说明给出的提示区间，而是当前实验配置。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
