# 运动想象 EEG 经典机器学习分析

本仓库当前只服务第一组任务：数据预处理与经典机器学习分析。我们保留 `Data/` 中原始 `.mat` 不动，把所有中间结果、对比实验和最终报告统一整理到 `Results/` 的三层结果区中，方便后续复现实验和写报告。

## 项目结构

```text
Data/          解压后的原始 .mat EEG 数据
SourceFile/    原始压缩包及其他未经处理的源文件
Scr/           Python 源代码
Results/
  01_preprocessed_data/   预处理后数据、诊断图与预处理研究 README
  02_feature_results/     特征导出、特征对比结果与研究 README
  03_ml_classification/   经典机器学习报告、图像与分类器调参 README
  archive/                旧布局归档与一次性历史结果
AGENTS.md
README.md
EXPERIMENT_PARAMETERS.md  当前有效实验参数与调参入口说明
PREPROCESSING_README.md   预处理与特征提取小结报告
requirements.txt
数据说明.txt
```

## 数据约定

每个被试文件应包含变量 `eeg`，形状为：

```text
6 x 44 x 5000 x 20
```

含义依次为 6 个类别、44 个通道、5000 个采样点和每类 20 个 trial。当前分析固定使用前 30 个 EEG 通道，采样率为 `500 Hz`。

当前类别顺序依据 `数据说明.txt` 暂定为：

| 标签 | 类别 |
|---:|---|
| 0 | 左手运动想象 |
| 1 | 右手运动想象 |
| 2 | 左腕运动想象 |
| 3 | 右腕运动想象 |
| 4 | 左臂运动想象 |
| 5 | 右臂运动想象 |

## Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前依赖如下：

| 包 | 用途 | 是否必需 |
|---|---|---|
| `numpy` | 数组计算、标签生成、特征矩阵导出 | 必需 |
| `scipy` | 读取 MATLAB `.mat` 文件、滤波、Welch PSD | 必需 |
| `matplotlib` | 绘制混淆矩阵、模型对比图与折间结果图 | 必需 |
| `scikit-learn` | Pipeline、经典分类器、交叉验证、评价指标、MIBIF | 必需 |

## 代码文件说明

| 文件 | 功能 |
|---|---|
| `Scr/data_loader.py` | 读取单个被试 `.mat` 文件，检查 `eeg` 变量与 4 维形状，截取前 30 个通道，并转换为 `samples x channels x time_points` 的 trial 级输入。 |
| `Scr/preprocessing.py` | EEG 预处理函数库：输入校验、逐 trial 去均值、可选 CAR、可选 50 Hz 陷波、零相位带通、时间窗截取、单 trial z-score 与 trial 质量摘要。 |
| `Scr/feature_extraction.py` | 提供基础时频特征、CSP、FBCSP 与折内 MIBIF 选择器，并封装为可放入 scikit-learn `Pipeline` 的变换器。 |
| `Scr/classifiers.py` | 创建 LDA、SVM、逻辑回归、KNN、随机森林，并自动组装需要折内标准化的 `Pipeline`。 |
| `Scr/model_evaluation.py` | 构建 LOSO 或被试内 5 折验证，计算折内指标、总体指标、被试级指标和混淆矩阵。 |
| `Scr/machine_learning_reporting.py` | 写出 JSON、CSV、Markdown 报告，并保存总体/被试级混淆矩阵、模型对比图和折间曲线。 |
| `Scr/machine_learning_config.py` | 集中保存类别定义、预处理、特征、分类器和验证策略配置。 |
| `Scr/results_layout.py` | 统一定义新 `Results/` 目录布局和常用路径常量。 |
| `Scr/machine_learning_process.py` | 默认主流程入口：读取 6 个被试、完成推荐预处理、执行 2/4/6 分类实验并生成课程报告。 |
| `Scr/cross_subject_process.py` | 跨被试 LOSO 主流程入口：复用默认流程，但改为留一被试测试的跨被试验证。 |
| `Scr/preprocessing_experiments.py` | 预处理消融实验入口，比较不同时间窗和异常 trial 剔除策略。 |
| `Scr/feature_experiments.py` | 特征工程对比入口，比较基础特征、CSP、FBCSP 与 OVR-FBCSP + MIBIF 方案。 |
| `Scr/classifier_experiments.py` | 分类器专项调参入口，在当前最稳前端上比较多组分类器超参数。 |
| `Scr/export_preprocessed_data.py` | 导出真实可复用的预处理后 trial 数据，每个被试一个 `.npz`。 |
| `Scr/export_basic_features.py` | 导出全样本基础特征矩阵；只导出基础特征，不导出全局 CSP/FBCSP 矩阵以避免泄漏概念混乱。 |

## 当前主线方法

默认配置位于 `Scr/machine_learning_config.py`。当前推荐主线为：

| 项目 | 当前设置 |
|---|---|
| 通道 | 前 30 个通道 |
| 采样率 | `500 Hz` |
| 预处理 | 去均值、`50 Hz` 陷波、`8-30 Hz` 零相位带通、时间窗 `(1.5, 5.5)` 秒、单 trial-单通道 `z-score`、不自动剔除异常 trial |
| 特征 | 默认主线为基础频域 + 时域特征；同时保留 CSP/FBCSP/MIBIF 作为对照研究 |
| 任务 | `2class`、`4class`、`6class` |
| 默认验证 | 被试内分层 5 折 `within_subject_5fold` |

预处理与特征提取结论见 [PREPROCESSING_README.md](/home/epilogue/智能i信息/PREPROCESSING_README.md)。

此外，项目现支持两种验证口径：

| 验证策略 | 含义 | 入口 |
|---|---|---|
| `within_subject_5fold` | 每个被试内部独立做分层 5 折，衡量被试内分类能力 | `Scr/machine_learning_process.py` |
| `loso` | 每次留出 1 个完整被试测试，剩余 5 个被试训练，衡量跨被试泛化能力 | `Scr/cross_subject_process.py` |

## 常用运行命令

运行默认经典机器学习实验：

```powershell
python Scr\machine_learning_process.py
```

运行跨被试 LOSO 经典机器学习实验：

```powershell
python Scr\cross_subject_process.py
```

运行预处理消融实验：

```powershell
python Scr\preprocessing_experiments.py
```

运行特征工程对比实验：

```powershell
python Scr\feature_experiments.py
```

运行分类器专项调参实验：

```powershell
python Scr\classifier_experiments.py
```

导出预处理后中间数据：

```powershell
python Scr\export_preprocessed_data.py
```

导出基础特征矩阵：

```powershell
python Scr\export_basic_features.py
```

运行前请确认：

1. `Data/` 中存在 `sub_01_MI.mat` 到 `sub_06_MI.mat` 六个文件。
2. 每个 `.mat` 文件包含形状为 `(6, 44, 5000, 20)` 的 `eeg`。
3. 已安装 `requirements.txt` 中列出的依赖。
4. 从项目根目录运行命令，确保 `Scr/` 中模块可以正常导入。

## Results 结构与输出

重构后的 `Results/` 结构为：

```text
Results/
  01_preprocessed_data/
    README.md
    dataset_exports/
    diagnostics/
    tuning/
  02_feature_results/
    README.md
    basic_feature_exports/
    comparisons/
  03_ml_classification/
    README.md
    course_reports/
    classifier_tuning/
  archive/
```

默认主流程每次运行会创建：

```text
Results/03_ml_classification/course_reports/course_report_YYYYMMDD_HHMMSS/
```

跨被试 LOSO 主流程每次运行会创建：

```text
Results/03_ml_classification/course_reports/course_report_loso_YYYYMMDD_HHMMSS/
```

主要输出包括：

| 文件或目录 | 内容 |
|---|---|
| `experiment_config.json` | 本次实验参数、验证策略和泄漏控制说明 |
| `feature_names.json` | 各任务的特征名称 |
| `trial_quality_summary.json` | 各被试预处理后 trial 质量诊断摘要 |
| `dropped_trials.json` | 异常 trial 剔除记录；默认主线不删样本 |
| `fold_metrics.csv` | 每个任务、分类器、验证折的指标 |
| `summary_metrics.csv` | 每个任务与分类器的汇总指标 |
| `subject_metrics.csv` | 每个任务、分类器、被试的单独汇总指标 |
| `complete_results.json` | 完整折级结果、预测和混淆矩阵 |
| `figures/` | 模型对比图、折间曲线、总体混淆矩阵、被试级对比图与各被试归一化混淆矩阵 |
| `report.md` | 自动生成的 Markdown 课程报告 |

说明：在 `within_subject_5fold` 下，`subject_metrics.csv` 表示“该被试内部 5 折汇总结果”；在 `loso` 下，表示“该被试作为完整测试集被留出时的跨被试结果”。

预处理、特征和分类器调参输出分别位于：

| 目录 | 内容 |
|---|---|
| `Results/01_preprocessed_data/tuning/preprocessing_tuning/...` | 预处理消融实验结果 |
| `Results/02_feature_results/comparisons/feature_tuning/...` | 特征工程对比实验结果 |
| `Results/03_ml_classification/classifier_tuning/...` | 分类器专项调参结果 |

## 数据泄漏控制

基础特征按单个 trial 独立计算，不使用跨样本统计量。预处理中的单 trial-单通道 `z-score` 只使用该 trial 自身时间轴统计量。对于 `CSP/FBCSP/MIBIF` 这类需要训练标签或训练集分布拟合的步骤，项目统一放入 scikit-learn `Pipeline`，并在每个训练折内部拟合，测试折不参与特征提取、标准化或分类器训练。

当前不在默认流程中启用重叠滑动窗口增强。原因是同一原始 trial 的重叠片段若被拆进不同折，容易造成结果偏乐观。若后续要做滑窗增强，必须同时引入基于原始 trial 的分组划分。
