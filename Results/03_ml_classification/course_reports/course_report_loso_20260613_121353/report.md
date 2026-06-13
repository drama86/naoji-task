# 运动想象 EEG 经典机器学习实验报告

## 实验目的

使用前 30 个 EEG 通道完成运动想象分类，并比较 LDA、SVM、逻辑回归、KNN 和随机森林等经典分类器。

## 数据与预处理

- 被试数量：6
- 采样率：500 Hz
- 使用通道：前 30 个通道
- 预处理：逐 trial、逐通道去均值，50 Hz 陷波，8–30 Hz Butterworth 零相位带通滤波
- 信号归一化：zscore_per_trial_channel
- 时间窗：(1.5, 5.5)

## 特征与验证方法

- 特征：逐通道时域统计特征与 Welch PSD 频域特征。
- 特征选择：关闭
- 标准化：在每个训练折内部通过 `StandardScaler` 拟合；随机森林不执行标准化。
- 验证策略：Leave-One-Subject-Out。
- 数据泄漏控制：测试折不参与特征提取器、标准化器和分类器拟合。

## 2class：左手 vs 右手

样本数：240；类别：左手运动想象、右手运动想象；特征维数：480。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.5208 | 0.5208 | 0.5184 |
| SVM | 0.4958 | 0.4958 | 0.4912 |
| Logistic Regression | 0.5417 | 0.5417 | 0.5396 |
| KNN | 0.5375 | 0.5375 | 0.5371 |
| Random Forest | 0.5333 | 0.5333 | 0.5328 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **Logistic Regression**（0.5417）。

### 模型对比

![模型指标对比](figures/2class/model_comparison.png)

### 各折结果

![各折结果](figures/2class/fold_balanced_accuracy.png)

### 混淆矩阵

#### LDA

![lda 归一化混淆矩阵](figures/2class/lda_confusion_normalized.png)

#### SVM

![svm 归一化混淆矩阵](figures/2class/svm_confusion_normalized.png)

#### Logistic Regression

![logistic_regression 归一化混淆矩阵](figures/2class/logistic_regression_confusion_normalized.png)

#### KNN

![knn 归一化混淆矩阵](figures/2class/knn_confusion_normalized.png)

#### Random Forest

![random_forest 归一化混淆矩阵](figures/2class/random_forest_confusion_normalized.png)

### 被试级结果

| 被试 | 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| 1 | KNN | 0.6000 | 0.6000 | 0.5733 |
| 1 | LDA | 0.6000 | 0.6000 | 0.5442 |
| 1 | Logistic Regression | 0.6500 | 0.6500 | 0.6154 |
| 1 | Random Forest | 0.5250 | 0.5250 | 0.4689 |
| 1 | SVM | 0.5000 | 0.5000 | 0.4302 |
| 2 | KNN | 0.5250 | 0.5250 | 0.4689 |
| 2 | LDA | 0.4500 | 0.4500 | 0.3732 |
| 2 | Logistic Regression | 0.5000 | 0.5000 | 0.4048 |
| 2 | Random Forest | 0.5250 | 0.5250 | 0.3866 |
| 2 | SVM | 0.5250 | 0.5250 | 0.3866 |
| 3 | KNN | 0.6250 | 0.6250 | 0.6248 |
| 3 | LDA | 0.5000 | 0.5000 | 0.4505 |
| 3 | Logistic Regression | 0.4750 | 0.4750 | 0.4320 |
| 3 | Random Forest | 0.5750 | 0.5750 | 0.5747 |
| 3 | SVM | 0.5500 | 0.5500 | 0.5200 |
| 4 | KNN | 0.5250 | 0.5250 | 0.5223 |
| 4 | LDA | 0.5250 | 0.5250 | 0.5223 |
| 4 | Logistic Regression | 0.5750 | 0.5750 | 0.5747 |
| 4 | Random Forest | 0.4750 | 0.4750 | 0.4667 |
| 4 | SVM | 0.4000 | 0.4000 | 0.3750 |
| 5 | KNN | 0.4500 | 0.4500 | 0.4486 |
| 5 | LDA | 0.5000 | 0.5000 | 0.3730 |
| 5 | Logistic Regression | 0.5000 | 0.5000 | 0.3730 |
| 5 | Random Forest | 0.5500 | 0.5500 | 0.5312 |
| 5 | SVM | 0.5250 | 0.5250 | 0.4473 |
| 6 | KNN | 0.5000 | 0.5000 | 0.4792 |
| 6 | LDA | 0.5500 | 0.5500 | 0.5200 |
| 6 | Logistic Regression | 0.5500 | 0.5500 | 0.5200 |
| 6 | Random Forest | 0.5500 | 0.5500 | 0.5489 |
| 6 | SVM | 0.4750 | 0.4750 | 0.4667 |

### 被试级 Balanced Accuracy 对比

![被试级结果对比](figures/2class/subject_balanced_accuracy.png)

#### LDA 各被试归一化混淆矩阵

被试 1

![subject_1_lda](figures/2class/subjects/lda/subject_01_confusion_normalized.png)

被试 2

![subject_2_lda](figures/2class/subjects/lda/subject_02_confusion_normalized.png)

被试 3

![subject_3_lda](figures/2class/subjects/lda/subject_03_confusion_normalized.png)

被试 4

![subject_4_lda](figures/2class/subjects/lda/subject_04_confusion_normalized.png)

被试 5

![subject_5_lda](figures/2class/subjects/lda/subject_05_confusion_normalized.png)

被试 6

![subject_6_lda](figures/2class/subjects/lda/subject_06_confusion_normalized.png)

#### SVM 各被试归一化混淆矩阵

被试 1

![subject_1_svm](figures/2class/subjects/svm/subject_01_confusion_normalized.png)

被试 2

![subject_2_svm](figures/2class/subjects/svm/subject_02_confusion_normalized.png)

被试 3

![subject_3_svm](figures/2class/subjects/svm/subject_03_confusion_normalized.png)

被试 4

![subject_4_svm](figures/2class/subjects/svm/subject_04_confusion_normalized.png)

被试 5

![subject_5_svm](figures/2class/subjects/svm/subject_05_confusion_normalized.png)

被试 6

![subject_6_svm](figures/2class/subjects/svm/subject_06_confusion_normalized.png)

#### Logistic Regression 各被试归一化混淆矩阵

被试 1

![subject_1_logistic_regression](figures/2class/subjects/logistic_regression/subject_01_confusion_normalized.png)

被试 2

![subject_2_logistic_regression](figures/2class/subjects/logistic_regression/subject_02_confusion_normalized.png)

被试 3

![subject_3_logistic_regression](figures/2class/subjects/logistic_regression/subject_03_confusion_normalized.png)

被试 4

![subject_4_logistic_regression](figures/2class/subjects/logistic_regression/subject_04_confusion_normalized.png)

被试 5

![subject_5_logistic_regression](figures/2class/subjects/logistic_regression/subject_05_confusion_normalized.png)

被试 6

![subject_6_logistic_regression](figures/2class/subjects/logistic_regression/subject_06_confusion_normalized.png)

#### KNN 各被试归一化混淆矩阵

被试 1

![subject_1_knn](figures/2class/subjects/knn/subject_01_confusion_normalized.png)

被试 2

![subject_2_knn](figures/2class/subjects/knn/subject_02_confusion_normalized.png)

被试 3

![subject_3_knn](figures/2class/subjects/knn/subject_03_confusion_normalized.png)

被试 4

![subject_4_knn](figures/2class/subjects/knn/subject_04_confusion_normalized.png)

被试 5

![subject_5_knn](figures/2class/subjects/knn/subject_05_confusion_normalized.png)

被试 6

![subject_6_knn](figures/2class/subjects/knn/subject_06_confusion_normalized.png)

#### Random Forest 各被试归一化混淆矩阵

被试 1

![subject_1_random_forest](figures/2class/subjects/random_forest/subject_01_confusion_normalized.png)

被试 2

![subject_2_random_forest](figures/2class/subjects/random_forest/subject_02_confusion_normalized.png)

被试 3

![subject_3_random_forest](figures/2class/subjects/random_forest/subject_03_confusion_normalized.png)

被试 4

![subject_4_random_forest](figures/2class/subjects/random_forest/subject_04_confusion_normalized.png)

被试 5

![subject_5_random_forest](figures/2class/subjects/random_forest/subject_05_confusion_normalized.png)

被试 6

![subject_6_random_forest](figures/2class/subjects/random_forest/subject_06_confusion_normalized.png)

## 4class：左手、右手、左腕、右腕

样本数：480；类别：左手运动想象、右手运动想象、左腕运动想象、右腕运动想象；特征维数：480。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.2708 | 0.2708 | 0.2713 |
| SVM | 0.2729 | 0.2729 | 0.2687 |
| Logistic Regression | 0.2708 | 0.2708 | 0.2706 |
| KNN | 0.2875 | 0.2875 | 0.2866 |
| Random Forest | 0.2875 | 0.2875 | 0.2869 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **KNN**（0.2875）。

### 模型对比

![模型指标对比](figures/4class/model_comparison.png)

### 各折结果

![各折结果](figures/4class/fold_balanced_accuracy.png)

### 混淆矩阵

#### LDA

![lda 归一化混淆矩阵](figures/4class/lda_confusion_normalized.png)

#### SVM

![svm 归一化混淆矩阵](figures/4class/svm_confusion_normalized.png)

#### Logistic Regression

![logistic_regression 归一化混淆矩阵](figures/4class/logistic_regression_confusion_normalized.png)

#### KNN

![knn 归一化混淆矩阵](figures/4class/knn_confusion_normalized.png)

#### Random Forest

![random_forest 归一化混淆矩阵](figures/4class/random_forest_confusion_normalized.png)

### 被试级结果

| 被试 | 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| 1 | KNN | 0.3125 | 0.3125 | 0.2625 |
| 1 | LDA | 0.2500 | 0.2500 | 0.1831 |
| 1 | Logistic Regression | 0.2250 | 0.2250 | 0.1669 |
| 1 | Random Forest | 0.2625 | 0.2625 | 0.1960 |
| 1 | SVM | 0.2875 | 0.2875 | 0.1745 |
| 2 | KNN | 0.3375 | 0.3375 | 0.2936 |
| 2 | LDA | 0.2625 | 0.2625 | 0.1681 |
| 2 | Logistic Regression | 0.2500 | 0.2500 | 0.1357 |
| 2 | Random Forest | 0.2625 | 0.2625 | 0.1669 |
| 2 | SVM | 0.2750 | 0.2750 | 0.1852 |
| 3 | KNN | 0.2875 | 0.2875 | 0.2891 |
| 3 | LDA | 0.3000 | 0.3000 | 0.2878 |
| 3 | Logistic Regression | 0.2875 | 0.2875 | 0.2807 |
| 3 | Random Forest | 0.3500 | 0.3500 | 0.3466 |
| 3 | SVM | 0.3000 | 0.3000 | 0.2891 |
| 4 | KNN | 0.2500 | 0.2500 | 0.2475 |
| 4 | LDA | 0.3000 | 0.3000 | 0.2923 |
| 4 | Logistic Regression | 0.3000 | 0.3000 | 0.2873 |
| 4 | Random Forest | 0.2250 | 0.2250 | 0.2288 |
| 4 | SVM | 0.2375 | 0.2375 | 0.2245 |
| 5 | KNN | 0.2875 | 0.2875 | 0.2840 |
| 5 | LDA | 0.2750 | 0.2750 | 0.2000 |
| 5 | Logistic Regression | 0.3125 | 0.3125 | 0.2174 |
| 5 | Random Forest | 0.2875 | 0.2875 | 0.2658 |
| 5 | SVM | 0.2250 | 0.2250 | 0.2028 |
| 6 | KNN | 0.2500 | 0.2500 | 0.2271 |
| 6 | LDA | 0.2375 | 0.2375 | 0.1737 |
| 6 | Logistic Regression | 0.2500 | 0.2500 | 0.2034 |
| 6 | Random Forest | 0.3375 | 0.3375 | 0.3235 |
| 6 | SVM | 0.3125 | 0.3125 | 0.2988 |

### 被试级 Balanced Accuracy 对比

![被试级结果对比](figures/4class/subject_balanced_accuracy.png)

#### LDA 各被试归一化混淆矩阵

被试 1

![subject_1_lda](figures/4class/subjects/lda/subject_01_confusion_normalized.png)

被试 2

![subject_2_lda](figures/4class/subjects/lda/subject_02_confusion_normalized.png)

被试 3

![subject_3_lda](figures/4class/subjects/lda/subject_03_confusion_normalized.png)

被试 4

![subject_4_lda](figures/4class/subjects/lda/subject_04_confusion_normalized.png)

被试 5

![subject_5_lda](figures/4class/subjects/lda/subject_05_confusion_normalized.png)

被试 6

![subject_6_lda](figures/4class/subjects/lda/subject_06_confusion_normalized.png)

#### SVM 各被试归一化混淆矩阵

被试 1

![subject_1_svm](figures/4class/subjects/svm/subject_01_confusion_normalized.png)

被试 2

![subject_2_svm](figures/4class/subjects/svm/subject_02_confusion_normalized.png)

被试 3

![subject_3_svm](figures/4class/subjects/svm/subject_03_confusion_normalized.png)

被试 4

![subject_4_svm](figures/4class/subjects/svm/subject_04_confusion_normalized.png)

被试 5

![subject_5_svm](figures/4class/subjects/svm/subject_05_confusion_normalized.png)

被试 6

![subject_6_svm](figures/4class/subjects/svm/subject_06_confusion_normalized.png)

#### Logistic Regression 各被试归一化混淆矩阵

被试 1

![subject_1_logistic_regression](figures/4class/subjects/logistic_regression/subject_01_confusion_normalized.png)

被试 2

![subject_2_logistic_regression](figures/4class/subjects/logistic_regression/subject_02_confusion_normalized.png)

被试 3

![subject_3_logistic_regression](figures/4class/subjects/logistic_regression/subject_03_confusion_normalized.png)

被试 4

![subject_4_logistic_regression](figures/4class/subjects/logistic_regression/subject_04_confusion_normalized.png)

被试 5

![subject_5_logistic_regression](figures/4class/subjects/logistic_regression/subject_05_confusion_normalized.png)

被试 6

![subject_6_logistic_regression](figures/4class/subjects/logistic_regression/subject_06_confusion_normalized.png)

#### KNN 各被试归一化混淆矩阵

被试 1

![subject_1_knn](figures/4class/subjects/knn/subject_01_confusion_normalized.png)

被试 2

![subject_2_knn](figures/4class/subjects/knn/subject_02_confusion_normalized.png)

被试 3

![subject_3_knn](figures/4class/subjects/knn/subject_03_confusion_normalized.png)

被试 4

![subject_4_knn](figures/4class/subjects/knn/subject_04_confusion_normalized.png)

被试 5

![subject_5_knn](figures/4class/subjects/knn/subject_05_confusion_normalized.png)

被试 6

![subject_6_knn](figures/4class/subjects/knn/subject_06_confusion_normalized.png)

#### Random Forest 各被试归一化混淆矩阵

被试 1

![subject_1_random_forest](figures/4class/subjects/random_forest/subject_01_confusion_normalized.png)

被试 2

![subject_2_random_forest](figures/4class/subjects/random_forest/subject_02_confusion_normalized.png)

被试 3

![subject_3_random_forest](figures/4class/subjects/random_forest/subject_03_confusion_normalized.png)

被试 4

![subject_4_random_forest](figures/4class/subjects/random_forest/subject_04_confusion_normalized.png)

被试 5

![subject_5_random_forest](figures/4class/subjects/random_forest/subject_05_confusion_normalized.png)

被试 6

![subject_6_random_forest](figures/4class/subjects/random_forest/subject_06_confusion_normalized.png)

## 6class：全部六类运动想象

样本数：720；类别：左手运动想象、右手运动想象、左腕运动想象、右腕运动想象、左臂运动想象、右臂运动想象；特征维数：480。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.2069 | 0.2069 | 0.2045 |
| SVM | 0.1917 | 0.1917 | 0.1908 |
| Logistic Regression | 0.1972 | 0.1972 | 0.1974 |
| KNN | 0.1986 | 0.1986 | 0.1964 |
| Random Forest | 0.2014 | 0.2014 | 0.2009 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.2069）。

### 模型对比

![模型指标对比](figures/6class/model_comparison.png)

### 各折结果

![各折结果](figures/6class/fold_balanced_accuracy.png)

### 混淆矩阵

#### LDA

![lda 归一化混淆矩阵](figures/6class/lda_confusion_normalized.png)

#### SVM

![svm 归一化混淆矩阵](figures/6class/svm_confusion_normalized.png)

#### Logistic Regression

![logistic_regression 归一化混淆矩阵](figures/6class/logistic_regression_confusion_normalized.png)

#### KNN

![knn 归一化混淆矩阵](figures/6class/knn_confusion_normalized.png)

#### Random Forest

![random_forest 归一化混淆矩阵](figures/6class/random_forest_confusion_normalized.png)

### 被试级结果

| 被试 | 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| 1 | KNN | 0.2000 | 0.2000 | 0.1691 |
| 1 | LDA | 0.1667 | 0.1667 | 0.1227 |
| 1 | Logistic Regression | 0.1500 | 0.1500 | 0.1075 |
| 1 | Random Forest | 0.1750 | 0.1750 | 0.1410 |
| 1 | SVM | 0.2167 | 0.2167 | 0.1614 |
| 2 | KNN | 0.2250 | 0.2250 | 0.1771 |
| 2 | LDA | 0.2000 | 0.2000 | 0.1281 |
| 2 | Logistic Regression | 0.1833 | 0.1833 | 0.1219 |
| 2 | Random Forest | 0.1917 | 0.1917 | 0.1336 |
| 2 | SVM | 0.2000 | 0.2000 | 0.1301 |
| 3 | KNN | 0.2083 | 0.2083 | 0.2101 |
| 3 | LDA | 0.2417 | 0.2417 | 0.2137 |
| 3 | Logistic Regression | 0.2500 | 0.2500 | 0.2273 |
| 3 | Random Forest | 0.2333 | 0.2333 | 0.2163 |
| 3 | SVM | 0.2083 | 0.2083 | 0.2023 |
| 4 | KNN | 0.2000 | 0.2000 | 0.1932 |
| 4 | LDA | 0.2333 | 0.2333 | 0.2226 |
| 4 | Logistic Regression | 0.2000 | 0.2000 | 0.1937 |
| 4 | Random Forest | 0.1667 | 0.1667 | 0.1557 |
| 4 | SVM | 0.1833 | 0.1833 | 0.1720 |
| 5 | KNN | 0.1750 | 0.1750 | 0.1720 |
| 5 | LDA | 0.2000 | 0.2000 | 0.1530 |
| 5 | Logistic Regression | 0.2083 | 0.2083 | 0.1558 |
| 5 | Random Forest | 0.2000 | 0.2000 | 0.1871 |
| 5 | SVM | 0.1833 | 0.1833 | 0.1709 |
| 6 | KNN | 0.1833 | 0.1833 | 0.1649 |
| 6 | LDA | 0.2000 | 0.2000 | 0.1792 |
| 6 | Logistic Regression | 0.1917 | 0.1917 | 0.1706 |
| 6 | Random Forest | 0.2417 | 0.2417 | 0.2319 |
| 6 | SVM | 0.1583 | 0.1583 | 0.1454 |

### 被试级 Balanced Accuracy 对比

![被试级结果对比](figures/6class/subject_balanced_accuracy.png)

#### LDA 各被试归一化混淆矩阵

被试 1

![subject_1_lda](figures/6class/subjects/lda/subject_01_confusion_normalized.png)

被试 2

![subject_2_lda](figures/6class/subjects/lda/subject_02_confusion_normalized.png)

被试 3

![subject_3_lda](figures/6class/subjects/lda/subject_03_confusion_normalized.png)

被试 4

![subject_4_lda](figures/6class/subjects/lda/subject_04_confusion_normalized.png)

被试 5

![subject_5_lda](figures/6class/subjects/lda/subject_05_confusion_normalized.png)

被试 6

![subject_6_lda](figures/6class/subjects/lda/subject_06_confusion_normalized.png)

#### SVM 各被试归一化混淆矩阵

被试 1

![subject_1_svm](figures/6class/subjects/svm/subject_01_confusion_normalized.png)

被试 2

![subject_2_svm](figures/6class/subjects/svm/subject_02_confusion_normalized.png)

被试 3

![subject_3_svm](figures/6class/subjects/svm/subject_03_confusion_normalized.png)

被试 4

![subject_4_svm](figures/6class/subjects/svm/subject_04_confusion_normalized.png)

被试 5

![subject_5_svm](figures/6class/subjects/svm/subject_05_confusion_normalized.png)

被试 6

![subject_6_svm](figures/6class/subjects/svm/subject_06_confusion_normalized.png)

#### Logistic Regression 各被试归一化混淆矩阵

被试 1

![subject_1_logistic_regression](figures/6class/subjects/logistic_regression/subject_01_confusion_normalized.png)

被试 2

![subject_2_logistic_regression](figures/6class/subjects/logistic_regression/subject_02_confusion_normalized.png)

被试 3

![subject_3_logistic_regression](figures/6class/subjects/logistic_regression/subject_03_confusion_normalized.png)

被试 4

![subject_4_logistic_regression](figures/6class/subjects/logistic_regression/subject_04_confusion_normalized.png)

被试 5

![subject_5_logistic_regression](figures/6class/subjects/logistic_regression/subject_05_confusion_normalized.png)

被试 6

![subject_6_logistic_regression](figures/6class/subjects/logistic_regression/subject_06_confusion_normalized.png)

#### KNN 各被试归一化混淆矩阵

被试 1

![subject_1_knn](figures/6class/subjects/knn/subject_01_confusion_normalized.png)

被试 2

![subject_2_knn](figures/6class/subjects/knn/subject_02_confusion_normalized.png)

被试 3

![subject_3_knn](figures/6class/subjects/knn/subject_03_confusion_normalized.png)

被试 4

![subject_4_knn](figures/6class/subjects/knn/subject_04_confusion_normalized.png)

被试 5

![subject_5_knn](figures/6class/subjects/knn/subject_05_confusion_normalized.png)

被试 6

![subject_6_knn](figures/6class/subjects/knn/subject_06_confusion_normalized.png)

#### Random Forest 各被试归一化混淆矩阵

被试 1

![subject_1_random_forest](figures/6class/subjects/random_forest/subject_01_confusion_normalized.png)

被试 2

![subject_2_random_forest](figures/6class/subjects/random_forest/subject_02_confusion_normalized.png)

被试 3

![subject_3_random_forest](figures/6class/subjects/random_forest/subject_03_confusion_normalized.png)

被试 4

![subject_4_random_forest](figures/6class/subjects/random_forest/subject_04_confusion_normalized.png)

被试 5

![subject_5_random_forest](figures/6class/subjects/random_forest/subject_05_confusion_normalized.png)

被试 6

![subject_6_random_forest](figures/6class/subjects/random_forest/subject_06_confusion_normalized.png)

## 结果说明

- 本报告中的指标均来自交叉验证的折外预测。
- 除总体结果外，报告额外给出每个被试的单独汇总指标与混淆矩阵。
- 结果用于第一组经典机器学习分析，不包含深度学习模型。
- 时间窗并非数据说明给出的提示区间，而是当前实验配置。
