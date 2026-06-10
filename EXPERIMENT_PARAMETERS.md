# 经典机器学习实验参数说明

本文档集中列出会显著影响实验结果的可调参数。修改参数后应重新运行：

```powershell
python Scr\machine_learning_process.py
```

每次实验会创建新的结果目录，不会覆盖此前结果。

## 一、实验任务与验证方式

### 1. 分类任务 `tasks`

修改位置：[Scr/machine_learning_process.py](Scr/machine_learning_process.py:56)

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

示例：

```python
tasks=("2class",)
tasks=("4class", "6class")
```

影响：

- 类别越多，分类通常越困难。
- 不同分类任务的随机猜测水平分别约为 `50%`、`25%` 和 `16.7%`。
- 任务顺序只影响运行和报告顺序，不影响模型结果。

### 2. 分类器列表 `classifiers`

修改位置：[Scr/machine_learning_process.py](Scr/machine_learning_process.py:59)

默认值：

```python
classifiers=CLASSIFIER_NAMES
```

等价于：

```python
classifiers=(
    "lda",
    "svm",
    "logistic_regression",
    "knn",
    "random_forest",
)
```

示例：

```python
classifiers=("svm", "lda")
```

影响：

- 决定运行哪些模型以及报告中的展示顺序。
- 减少模型可以缩短运行时间。
- 不同模型对特征尺度、样本量和非线性关系的适应能力不同。

### 3. 验证策略 `validation_strategy`

修改位置：[Scr/machine_learning_process.py](Scr/machine_learning_process.py:62)

默认值：

```python
validation_strategy="loso"
```

可选值：

```text
loso
within_subject_5fold
```

影响：

- `loso`：每次使用 5 个被试训练、1 个完整被试测试，衡量跨被试泛化，
  通常更严格、分数更低。
- `within_subject_5fold`：每个被试内部进行分层 5 折，衡量同一被试内
  分类能力，通常分数更高。
- 两种结果代表不同问题，不能直接混合平均或只保留分数较高者。


## 二、数据与预处理参数

### 1. 随机种子 `RANDOM_SEED`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:5)

默认值：

```python
RANDOM_SEED = 42
```

影响：

- 控制随机森林和被试内交叉验证洗牌等随机过程。
- 更换随机种子可能使结果产生小幅变化。
- 公平比较参数时应固定同一个随机种子。

### 2. 分析通道数 `CHANNEL_COUNT`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:12)

默认值：

```python
CHANNEL_COUNT = 30
```

影响：

- 决定从原始 44 通道中保留多少个前序通道。
- 通道越多，包含的信息和特征维度越多，也可能增加噪声和过拟合风险。
- 当前课程明确要求分析前 30 个通道，通常不应修改。

### 3. 带通下截止频率 `low_cut_hz`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:41)

默认值：

```python
"low_cut_hz": 8.0
```

影响：

- 决定保留的最低频率。
- 提高该值会去除更多低频成分，也可能损失 mu/alpha 节律。
- 必须满足 `0 < low_cut_hz < high_cut_hz`。

### 4. 带通上截止频率 `high_cut_hz`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:44)

默认值：

```python
"high_cut_hz": 30.0
```

影响：

- 决定保留的最高频率。
- 提高该值可以引入更高频信息，也可能引入更多肌电和噪声。
- 必须小于 Nyquist 频率，本数据中为 `250 Hz`。

### 5. 滤波器阶数 `filter_order`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:47)

默认值：

```python
"filter_order": 4
```

建议对比：

```text
2、4、6
```

影响：

- 阶数越高，截止频率附近的过渡越陡。
- 过高阶数可能产生更明显的边缘效应或数值问题。

### 6. 时间窗 `time_window_seconds`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:50)

默认值：

```python
"time_window_seconds": None
```

候选值：

```python
None        # 完整 0-10 s
(1.0, 5.0)  # 1-5 s
(2.0, 6.0)  # 2-6 s
```

影响：

- 时间窗可能是当前实验中影响结果最明显的参数之一。
- 较短窗口可能减少无关信号，但也会减少可用信息。
- 数据说明没有给出正式任务提示时刻，因此候选窗口属于实验假设。
- 应完整报告尝试过的时间窗，不能只展示效果最好的一个。

### 7. 采样率 `SAMPLING_RATE`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:8)

默认值：

```python
SAMPLING_RATE = 500
```

影响：

- 采样率由原始数据给定，会影响滤波和频率轴计算。
- 当前代码没有实现重采样，因此不应将其作为调优参数修改。

## 三、基础特征参数

### 1. 特征集合 `feature_sets`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:60)

默认值：

```python
"feature_sets": ("time", "frequency")
```

可选值：

```python
("time",)
("frequency",)
("time", "frequency")
```

影响：

- `time`：使用时域统计量。
- `frequency`：使用 Welch PSD 频域特征。
- 组合特征信息更完整，但维度更高，可能增加过拟合风险。
- 当前组合产生每通道 16 个、共 480 个特征。

### 2. 频带定义 `frequency_bands`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:63)

默认值：

```python
"frequency_bands": (
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)
```

可选细分示例：

```python
"frequency_bands": (
    ("mu_low", 8.0, 10.0),
    ("mu_high", 10.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
)
```

影响：

- 频带边界决定功率特征关注的脑电节律。
- 增加频带会提高特征维度，也可能引入冗余。
- 频带应位于滤波保留范围和 `0-250 Hz` 之间。

### 3. 总功率范围 `total_power_band`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:69)

默认值：

```python
"total_power_band": (8.0, 30.0)
```

影响：

- 用作相对频带功率的分母。
- 同时用于计算谱质心和归一化谱熵。
- 改变范围会改变所有相对功率与谱统计特征。
- 应覆盖希望比较的全部 `frequency_bands`。

### 4. Welch 分段长度 `nperseg`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:72)

默认值：

```python
"nperseg": 500
```

在 500 Hz 采样率下：

```text
250 点 = 0.5 秒
500 点 = 1 秒
1000 点 = 2 秒
```

影响：

- 值越大，频率分辨率越高。
- 值越大，可用于平均的分段数量越少，PSD 估计可能更不稳定。
- 不得超过实际时间窗的采样点数。

### 5. Welch 重叠点数 `noverlap`

修改位置：[Scr/machine_learning_config.py](Scr/machine_learning_config.py:75)

默认值：

```python
"noverlap": 250
```

影响：

- 当前相当于 `nperseg=500` 时的 50% 重叠。
- 增加重叠可获得更多分段，但会提高计算量和分段相关性。
- 必须满足 `0 <= noverlap < nperseg`。

## 四、分类器参数

修改分类器参数后，必须重新运行完整交叉验证。

### 1. LDA

默认值：

```python
LinearDiscriminantAnalysis(
    solver="lsqr",
    shrinkage="auto",
)
```

重要参数：

| 参数 | 默认值 | 修改位置 | 意义 |
|---|---|---|---|
| `solver` | `"lsqr"` | [Scr/classifiers.py](Scr/classifiers.py:31) | 求解方法；支持协方差收缩 |
| `shrinkage` | `"auto"` | [Scr/classifiers.py](Scr/classifiers.py:33) | 自动稳定高维、小样本条件下的协方差估计 |

注意：

- 若使用 `solver="svd"`，必须将 `shrinkage=None`。
- 当前 480 维特征相对样本量较高，收缩通常有助于稳定。

### 2. SVM

默认值：

```python
SVC(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
)
```

重要参数：

| 参数 | 建议候选 | 修改位置 | 意义 |
|---|---|---|---|
| `C` | `0.1、1、10` | [Scr/classifiers.py](Scr/classifiers.py:45) | 越大越强调训练集拟合；越小正则化越强 |
| `kernel` | `"linear"`、`"rbf"` | [Scr/classifiers.py](Scr/classifiers.py:47) | 线性或非线性分类边界 |
| `gamma` | `"scale"`、`"auto"` | [Scr/classifiers.py](Scr/classifiers.py:50) | RBF 核影响范围 |
| `class_weight` | `None`、`"balanced"` | [Scr/classifiers.py](Scr/classifiers.py:53) | 是否补偿类别数量不平衡 |

### 3. 逻辑回归

默认值：

```python
LogisticRegression(
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)
```

重要参数：

| 参数 | 建议候选 | 修改位置 | 意义 |
|---|---|---|---|
| `C` | `0.1、1、10` | [Scr/classifiers.py](Scr/classifiers.py:64) | 越小正则化越强 |
| `max_iter` | `1000、2000、5000` | [Scr/classifiers.py](Scr/classifiers.py:66) | 最大迭代次数；主要用于解决未收敛 |
| `class_weight` | `None`、`"balanced"` | [Scr/classifiers.py](Scr/classifiers.py:68) | 类别权重 |

### 4. KNN

默认值：

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    p=2,
)
```

重要参数：

| 参数 | 建议候选 | 修改位置 | 意义 |
|---|---|---|---|
| `n_neighbors` | `3、5、7、9` | [Scr/classifiers.py](Scr/classifiers.py:82) | 邻居数；越大决策边界越平滑 |
| `weights` | `"uniform"`、`"distance"` | [Scr/classifiers.py](Scr/classifiers.py:84) | 等权或按距离加权投票 |
| `metric` | `"minkowski"` | [Scr/classifiers.py](Scr/classifiers.py:86) | 距离度量名称 |
| `p` | `1、2` | [Scr/classifiers.py](Scr/classifiers.py:88) | `1` 为曼哈顿距离，`2` 为欧氏距离 |

### 5. 随机森林

默认值：

```python
RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=None,
)
```

重要参数：

| 参数 | 建议候选 | 修改位置 | 意义 |
|---|---|---|---|
| `n_estimators` | `100、200、500` | [Scr/classifiers.py](Scr/classifiers.py:99) | 决策树数量；越多通常越稳定但越慢 |
| `max_features` | `"sqrt"`、`"log2"` | [Scr/classifiers.py](Scr/classifiers.py:102) | 每次分裂考虑的特征数量 |
| `class_weight` | `None`、`"balanced"` | [Scr/classifiers.py](Scr/classifiers.py:104) | 类别权重 |
| `n_jobs` | `None`、`-1` | [Scr/classifiers.py](Scr/classifiers.py:109) | 串行或使用全部逻辑核心 |

注意：

- `n_jobs=-1` 可能加快运行，但部分 Windows 环境会显示 joblib/WMIC
  核心探测警告。
- 随机森林当前不使用 `StandardScaler`，这是树模型的正常设置。

## 五、参数实验规范

### 1. 一次只改变一个主要因素

例如比较时间窗时，应固定：

```text
分类任务、特征、分类器、验证策略、随机种子
```

否则无法判断结果变化来自哪个参数。

### 2. 不要使用 LOSO 测试被试反复调参

LOSO 的留出被试相当于测试集。若根据 LOSO 结果不断修改参数并只保留
最佳配置，会产生测试信息泄漏。

较严格的调参方式是：

```text
外层 LOSO：评估最终跨被试效果
内层交叉验证：仅在外层训练被试中选择参数
```

当前代码尚未实现嵌套交叉验证，因此建议只比较少量、事先声明的参数。

### 3. 保留全部实验记录

每次实验应保留：

```text
experiment_config.json
summary_metrics.csv
fold_metrics.csv
report.md
```

课程汇报中应说明尝试过的主要参数，而不是只报告最优结果。

## 六、推荐优先级

建议按以下顺序开展参数对比：

1. 比较 `loso` 与 `within_subject_5fold`，明确两者评价目标不同。
2. 比较完整 `0-10 s`、`1-5 s`、`2-6 s` 时间窗。
3. 比较时域、频域和组合特征。
4. 比较 SVM 的 `linear/rbf` 以及少量 `C` 值。
5. 比较 KNN 邻居数和随机森林树数量。
6. 最后再考虑细分频带和滤波器参数。
