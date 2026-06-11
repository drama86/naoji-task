# 经典机器学习实验参数说明

本文档只说明当前第一组工程中仍然有效、且确实会影响实验结果的主要参数。它对应的是现在这套仓库结构与脚本，而不是早期旧版本。

## 1. 当前主线默认值

默认主流程由 [Scr/machine_learning_process.py](/home/epilogue/智能i信息/Scr/machine_learning_process.py) 调用，核心配置集中在 [Scr/machine_learning_config.py](/home/epilogue/智能i信息/Scr/machine_learning_config.py)。

当前推荐主线默认值为：

```text
任务：2class + 4class + 6class
验证：within_subject_5fold
预处理：无 CAR + 50 Hz 陷波 + 8-30 Hz 带通 + 1.5-5.5 s 时间窗 + 单 trial z-score
特征：basic 模式，frequency + time
分类器：LDA、SVM、Logistic Regression、KNN、Random Forest
```

## 2. 运行入口与对应输出

| 脚本 | 用途 | 默认输出位置 |
|---|---|---|
| `python Scr\\machine_learning_process.py` | 默认主流程 | `Results/03_ml_classification/course_reports/` |
| `python Scr\\preprocessing_experiments.py` | 预处理消融 | `Results/01_preprocessed_data/tuning/preprocessing_tuning/` |
| `python Scr\\feature_experiments.py` | 特征对比 | `Results/02_feature_results/comparisons/feature_tuning/` |
| `python Scr\\classifier_experiments.py` | 分类器专项调参 | `Results/03_ml_classification/classifier_tuning/` |
| `python Scr\\export_preprocessed_data.py` | 导出预处理后数值数据 | `Results/01_preprocessed_data/dataset_exports/` |
| `python Scr\\export_basic_features.py` | 导出基础特征矩阵 | `Results/02_feature_results/basic_feature_exports/` |

## 3. 主流程可直接改的参数

这部分参数在 [Scr/machine_learning_process.py](/home/epilogue/智能i信息/Scr/machine_learning_process.py) 的 `parse_args()` 中控制。

### 3.1 `tasks`

默认值：

```python
tasks=("2class", "4class", "6class")
```

可选值：

```text
2class：左手 vs 右手
4class：左手、右手、左腕、右腕
6class：全部六类
```

影响：

- 类别越多，任务越难。
- 只跑单个任务可以明显缩短实验时间。

### 3.2 `classifiers`

默认值：

```python
classifiers=CLASSIFIER_NAMES
```

等价于：

```python
("lda", "svm", "logistic_regression", "knn", "random_forest")
```

影响：

- 决定主流程里比较哪些模型。
- 减少模型数量可以缩短运行时间。

### 3.3 `validation_strategy`

默认值：

```python
validation_strategy="within_subject_5fold"
```

可选值：

```text
within_subject_5fold
loso
```

影响：

- `within_subject_5fold`：衡量同一被试内部可分性，通常分数更高，是当前默认主线。
- `loso`：衡量跨被试泛化能力，更严格，通常分数更低。
- 两者回答的问题不同，不应直接混成一个结论。

### 3.4 `run_name`

默认值：

```python
run_name=None
```

影响：

- `None` 时会自动生成时间戳目录。
- 手动指定时适合做对照实验，但目录名不能与已有结果冲突。

## 4. 全局配置参数

这部分参数在 [Scr/machine_learning_config.py](/home/epilogue/智能i信息/Scr/machine_learning_config.py) 中集中定义。

### 4.1 数据与任务基础参数

| 参数 | 当前值 | 说明 |
|---|---|---|
| `RANDOM_SEED` | `42` | 控制随机过程，比较实验时应保持一致 |
| `SAMPLING_RATE` | `500` | 原始数据采样率，不作为调优参数修改 |
| `CHANNEL_COUNT` | `30` | 课程要求分析前 30 个通道，通常不改 |

### 4.2 预处理参数

| 参数 | 当前值 | 作用 |
|---|---|---|
| `low_cut_hz` | `8.0` | 带通下截止频率 |
| `high_cut_hz` | `30.0` | 带通上截止频率 |
| `filter_order` | `4` | Butterworth 阶数 |
| `spatial_reference` | `None` | 空间重参考，当前默认不启用 CAR |
| `notch_hz` | `50.0` | 工频陷波中心 |
| `notch_quality_factor` | `30.0` | 陷波带宽控制 |
| `time_window_seconds` | `(1.5, 5.5)` | 当前推荐时间窗 |
| `normalize_mode` | `"zscore_per_trial_channel"` | 单 trial-单通道标准化 |
| `drop_flagged_trials` | `False` | 默认不自动剔除异常 trial |
| `drop_flagged_trials_rule` | `"either"` | 若启用剔除，按 RMS 或 peak-to-peak 任一异常删除 |

说明：

- 当前综合实验不推荐把 `CAR` 作为默认主线。
- 当前也不推荐默认自动剔除异常 trial。
- 若改动预处理参数，建议同步重跑 `preprocessing_experiments.py` 做确认。

### 4.3 特征参数

`FEATURE_CONFIG["mode"]` 可选：

```text
basic
csp
fbcsp
```

当前默认：

```text
mode = "basic"
basic.feature_sets = ("frequency", "time")
```

说明：

- `basic`：当前统一主线，稳定性最好。
- `csp` / `fbcsp`：作为空间判别特征对照方案保留。
- `selection.enabled=True` 时会在训练折内部启用 `MIBIF` 特征选择。

常用基础特征参数：

| 参数 | 当前值 | 说明 |
|---|---|---|
| `feature_sets` | `("frequency", "time")` | 当前默认先频域后时域 |
| `frequency_bands` | `alpha 8-13, beta 13-30` | Welch PSD 频带定义 |
| `total_power_band` | `(8.0, 30.0)` | 相对功率、谱质心、谱熵使用的总频段 |
| `nperseg` | `500` | Welch 每段 1 秒 |
| `noverlap` | `250` | Welch 50% 重叠 |

### 4.4 分类器参数来源

主流程默认分类器参数定义在 [Scr/classifiers.py](/home/epilogue/智能i信息/Scr/classifiers.py)。

如果目标是专项调参，不建议直接在主流程里手改默认参数，而应优先运行：

```powershell
python Scr\classifier_experiments.py
```

因为这个脚本已经把不同任务下的候选参数组组织好了，结果也会自动落到新的分类器结果目录。

## 5. 什么时候改哪个文件

| 目标 | 优先改动位置 |
|---|---|
| 只改一次主流程任务、分类器或验证方式 | `Scr/machine_learning_process.py` 的 `parse_args()` |
| 改默认预处理、特征、类别定义 | `Scr/machine_learning_config.py` |
| 做预处理对比 | `Scr/preprocessing_experiments.py` 中 `EXPERIMENTS` |
| 做特征对比 | `Scr/feature_experiments.py` 中 `EXPERIMENTS` |
| 做分类器调参 | `Scr/classifier_experiments.py` 中候选参数表 |
| 导出可复用数据 | `Scr/export_preprocessed_data.py` / `Scr/export_basic_features.py` |

## 6. 调参时的注意事项

- 不要在报告里把 `within_subject_5fold` 和 `loso` 结果混为一谈。
- 不要把全局固定的 `CSP/FBCSP` 特征矩阵当成可复用中间数据导出，否则容易造成泄漏理解混乱。
- 修改参数后，最好保留新的时间戳结果目录，不覆盖旧实验。
- 若只是写课程报告，优先引用 `Results/03_ml_classification/` 下的正式结果，而不是历史归档目录。
