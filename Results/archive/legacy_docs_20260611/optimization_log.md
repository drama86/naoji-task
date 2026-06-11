# 机器学习优化实验日志与后续方向

本文档记录截至目前围绕第一组任务开展的经典机器学习优化实验，重点关注数据预处理、时间窗、分类器参数、验证策略和信号特征设置对结果的影响。实验均围绕运动想象 EEG 的传统机器学习分析展开，不涉及深度学习部署。

## 当前基础流程

- 数据：6 名被试，每名被试数据形状为 `6 x 44 x 5000 x 20`。
- 通道：选取前 30 个 EEG 通道。
- 采样率：500 Hz。
- 预处理：逐 trial、逐通道去均值，8-30 Hz 带通滤波，按实验设置截取时间窗。
- 特征：时域统计特征和 Welch PSD 频域特征。
- 分类器：LDA、SVM、Logistic Regression、KNN、Random Forest。
- 评价指标：accuracy、balanced accuracy、macro precision、macro recall、macro F1、混淆矩阵。
- 数据泄漏控制：标准化和分类器训练均在交叉验证训练折内完成。

## 一、时间窗优化

### 实验设置

固定其他参数，比较不同时间窗对分类结果的影响。主要比较过的时间窗包括：

```text
0-10s/full
1.0-5.0s
1.5-5.5s
2.0-6.0s
2.5-6.5s
3.0-7.0s
3.5-7.5s
4.0-8.0s
4.5-8.5s
5.0-9.0s
5.5-9.5s
6.0-10.0s
```

### 主要结果

二分类任务中，`1.5-5.5s` 时间窗表现最好：

| 时间窗 | 最优模型 | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|
| `1.5-5.5s` | Random Forest | 0.5625 | 0.5623 |
| `1.0-5.0s` | Random Forest | 0.5458 | 0.5430 |
| `2.0-6.0s` | Random Forest | 0.5458 | 0.5400 |
| `2.5-6.5s` | KNN | 0.5458 | 0.5308 |
| `0-10s/full` | Logistic Regression | 0.5125 | 0.5087 |

### 分析结论

- 完整 `0-10s` 并不是最优，可能包含较多非任务相关片段和噪声。
- `1.5-5.5s` 对二分类最有利，说明有效判别信息更集中在 trial 的前中段。
- 后续二分类优化可优先固定 `1.5-5.5s` 作为时间窗基线。
- 若综合 2/4/6 分类全部任务，`1.0-5.0s` 和 `1.5-5.5s` 非常接近，均可作为候选。

## 二、分类器参数优化

### 实验设置

在较优时间窗 `1.5-5.5s` 下，对分类器参数进行比较。

| 参数组 | SVM | Logistic Regression | KNN | Random Forest |
|---|---|---|---|---|
| 默认参数 | `C=1` | `C=1` | `k=3` | `n_estimators=200` |
| 保守参数 | `C=0.1` | `C=0.1` | `k=3` | `n_estimators=100` |
| 复杂参数 | `C=10` | `C=10` | `k=5` | `n_estimators=500` |

### 整体结果

| 参数组 | 平均 Balanced Accuracy | 平均 Macro F1 |
|---|---:|---:|
| 默认参数 | 0.3283 | 0.3229 |
| 复杂参数 | 0.3260 | 0.3212 |
| 保守参数 | 0.3237 | 0.3149 |

### 按任务最优结果

| 任务 | 最优方法 | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|
| 2 分类 | Random Forest，默认参数 | 0.5625 | 0.5623 |
| 4 分类 | SVM，`C=10` | 0.3021 | 0.2998 |
| 6 分类 | Random Forest，`n_estimators=500` | 0.2028 | 0.1975 |

### 分析结论

- 分类器参数不是越复杂越好。
- 二分类下 Random Forest 默认参数表现最好。
- 综合 2/4/6 分类，SVM 在 `C=10` 时整体表现较好。
- Logistic Regression 使用较强正则化，即 `C=0.1`，相对更稳定。
- KNN 从 `k=3` 改为 `k=5` 后整体表现下降，不建议作为主要优化方向。

## 三、验证策略对照

### LOSO 与被试内 5 折

已比较两种验证策略：

| 验证策略 | 含义 | 衡量目标 |
|---|---|---|
| `loso` | Leave-One-Subject-Out，每次留出 1 名完整被试作为测试集 | 跨被试泛化能力 |
| `within_subject_5fold` | 每名被试单独做分层 5 折，再汇总 6 名被试结果 | 同一被试内部识别能力 |

被试内 5 折的折数为：

```text
6 名被试 x 每名被试 5 折 = 30 折
```

### 对照结果

在相同复杂参数设置下：

| 验证策略 | 平均 Balanced Accuracy | 平均 Macro F1 |
|---|---:|---:|
| LOSO | 0.3260 | 0.3212 |
| 被试内 5 折 | 0.3772 | 0.3764 |

### 被试内 5 折最优结果

| 任务 | 最优模型 | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|
| 2 分类 | Logistic Regression | 0.5708 | 0.5702 |
| 4 分类 | KNN | 0.3313 | 0.3265 |
| 6 分类 | LDA | 0.2722 | 0.2732 |

### 分析结论

- 被试内 5 折显著高于 LOSO，说明当前特征和模型能学习同一被试内部的运动想象模式。
- LOSO 较低说明不同被试之间 EEG 分布差异较大，跨被试泛化是当前难点。
- 报告中建议同时保留两种验证方式：被试内 5 折展示个体内可识别性，LOSO 展示跨被试泛化能力。

## 四、信号特征优化

### 实验设置

固定以下条件：

```text
validation_strategy = within_subject_5fold
time_window = 1.5-5.5s
bandpass = 8-30 Hz
Welch nperseg = 500
Welch noverlap = 250
SVM C = 10
Logistic Regression C = 0.1
KNN k = 5
Random Forest n_estimators = 500
```

比较不同特征集合：

| 特征设置 | 特征数 |
|---|---:|
| `time + frequency` | 480 |
| `time` | 300 |
| `frequency` | 180 |
| `frequency + time` | 480 |

### 整体结果

| 特征设置 | 平均 Balanced Accuracy | 平均 Macro F1 |
|---|---:|---:|
| `frequency` | 0.3896 | 0.3891 |
| `frequency + time` | 0.3775 | 0.3767 |
| `time + frequency` | 0.3769 | 0.3761 |
| `time` | 0.3526 | 0.3519 |

### 各任务最优结果

| 任务 | 最优特征 | 最优模型 | Balanced Accuracy | Macro F1 |
|---|---|---|---:|---:|
| 2 分类 | `frequency` | LDA | 0.5792 | 0.5788 |
| 4 分类 | `frequency` | Logistic Regression / Random Forest | 0.3646 | 约 0.365 |
| 6 分类 | `frequency` | LDA | 0.2806 | 0.2813 |

### 分析结论

- 仅频域特征 `frequency` 是当前表现最好的特征方案。
- 频域特征维度仅 180，低于组合特征 480，但性能更好，说明时域特征可能引入噪声或冗余。
- 当前运动想象分类的有效信息主要集中在 8-30 Hz 频域节律中，尤其与 alpha/mu 和 beta 节律相关。
- 后续应优先沿频域特征和空间滤波方向优化，而不是继续堆叠基础时域统计特征。

## 当前推荐方案

### 被试内识别任务

```text
time_window = 1.5-5.5s
feature_sets = frequency
validation_strategy = within_subject_5fold
```

推荐模型：

| 任务 | 推荐模型 |
|---|---|
| 2 分类 | LDA 或 Logistic Regression |
| 4 分类 | Logistic Regression 或 Random Forest |
| 6 分类 | LDA |

### 跨被试泛化任务

```text
time_window = 1.5-5.5s
feature_sets = frequency
validation_strategy = loso
```

推荐模型：

| 任务 | 推荐模型 |
|---|---|
| 2 分类 | Random Forest |
| 4 分类 | SVM |
| 6 分类 | Random Forest 或 LDA |

## 下一步优化方向

### 1. 频带细分

当前频域特征只使用两个频带：

```text
alpha: 8-13 Hz
beta: 13-30 Hz
```

建议进一步细分为：

```text
8-12 Hz
12-16 Hz
16-20 Hz
20-24 Hz
24-30 Hz
```

或：

```text
8-10 Hz
10-12 Hz
12-15 Hz
15-20 Hz
20-25 Hz
25-30 Hz
```

目标是检查运动想象差异是否集中在更窄的频段中。

### 2. 引入 CSP / FBCSP

运动想象 EEG 常用 CSP 和 FBCSP 提取空间模式。建议下一步重点实现：

```text
带通滤波
-> 截取 1.5-5.5s
-> CSP 或 FBCSP
-> LDA / SVM / Logistic Regression
-> 交叉验证
```

注意：

- CSP/FBCSP 必须在每个训练折内拟合。
- 不能先在全数据上拟合 CSP/FBCSP，再交叉验证。
- 二分类优先尝试 CSP，4/6 分类可考虑 one-vs-rest CSP 或 FBCSP 方案。

### 3. 保留双验证体系

后续报告建议同时保留：

```text
within_subject_5fold
loso
```

被试内 5 折用于说明模型能否识别同一被试内部模式；LOSO 用于说明跨被试泛化能力。

### 4. 小范围调参

在 `frequency` 特征方案下，小范围比较以下参数：

```text
LDA:
solver = lsqr
shrinkage = auto

Logistic Regression:
C = 0.01, 0.1, 1

SVM:
C = 1, 3, 10
gamma = scale, 0.01
kernel = rbf

Random Forest:
n_estimators = 200, 500
max_features = sqrt, log2
min_samples_leaf = 1, 2, 4
```

### 5. 结果记录规范

每次后续实验建议记录：

- 时间窗。
- 验证策略。
- 特征集合。
- 频带定义。
- Welch 参数。
- 分类器参数。
- 2/4/6 分类的 accuracy、balanced accuracy、macro F1。
- 最优模型和混淆矩阵。

## 阶段性结论

截至目前，最明确的优化结论是：

1. 时间窗方面，二分类优先选择 `1.5-5.5s`。
2. 验证策略方面，被试内 5 折明显高于 LOSO，说明被试差异是主要难点。
3. 特征方面，`frequency` 仅频域特征优于 `time` 和 `time+frequency`。
4. 模型方面，LDA、Logistic Regression、Random Forest 在当前最佳频域特征下表现较稳；KNN 不建议作为主要优化方向。
5. 下一步最值得投入的是频带细分和 CSP/FBCSP，而不是继续堆叠基础时域特征。
