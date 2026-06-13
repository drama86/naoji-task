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
- 验证策略：Per-subject Stratified 5-Fold。
- 数据泄漏控制：测试折不参与特征提取器、标准化器和分类器拟合。

## 2class：左手 vs 右手

样本数：240；类别：左手运动想象、右手运动想象；特征维数：480。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.6250 | 0.6250 | 0.6241 |
| SVM | 0.5875 | 0.5875 | 0.5873 |
| Logistic Regression | 0.6292 | 0.6292 | 0.6286 |
| KNN | 0.5417 | 0.5417 | 0.5401 |
| Random Forest | 0.6042 | 0.6042 | 0.6042 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **Logistic Regression**（0.6292）。

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
| 1 | KNN | 0.6500 | 0.6500 | 0.6419 |
| 1 | LDA | 0.5750 | 0.5750 | 0.5726 |
| 1 | Logistic Regression | 0.6000 | 0.6000 | 0.5960 |
| 1 | Random Forest | 0.6250 | 0.6250 | 0.6248 |
| 1 | SVM | 0.6750 | 0.6750 | 0.6732 |
| 2 | KNN | 0.4500 | 0.4500 | 0.4486 |
| 2 | LDA | 0.6750 | 0.6750 | 0.6748 |
| 2 | Logistic Regression | 0.6500 | 0.6500 | 0.6500 |
| 2 | Random Forest | 0.5000 | 0.5000 | 0.4987 |
| 2 | SVM | 0.5750 | 0.5750 | 0.5747 |
| 3 | KNN | 0.5750 | 0.5750 | 0.5747 |
| 3 | LDA | 0.8250 | 0.8250 | 0.8249 |
| 3 | Logistic Regression | 0.7750 | 0.7750 | 0.7749 |
| 3 | Random Forest | 0.7750 | 0.7750 | 0.7749 |
| 3 | SVM | 0.6250 | 0.6250 | 0.6248 |
| 4 | KNN | 0.4750 | 0.4750 | 0.4720 |
| 4 | LDA | 0.4500 | 0.4500 | 0.4444 |
| 4 | Logistic Regression | 0.5250 | 0.5250 | 0.5247 |
| 4 | Random Forest | 0.5750 | 0.5750 | 0.5726 |
| 4 | SVM | 0.4750 | 0.4750 | 0.4720 |
| 5 | KNN | 0.6000 | 0.6000 | 0.5960 |
| 5 | LDA | 0.6250 | 0.6250 | 0.6190 |
| 5 | Logistic Regression | 0.6750 | 0.6750 | 0.6698 |
| 5 | Random Forest | 0.6500 | 0.6500 | 0.6491 |
| 5 | SVM | 0.6500 | 0.6500 | 0.6491 |
| 6 | KNN | 0.5000 | 0.5000 | 0.4987 |
| 6 | LDA | 0.6000 | 0.6000 | 0.5990 |
| 6 | Logistic Regression | 0.5500 | 0.5500 | 0.5489 |
| 6 | Random Forest | 0.5000 | 0.5000 | 0.4949 |
| 6 | SVM | 0.5250 | 0.5250 | 0.5247 |

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
| LDA | 0.3375 | 0.3375 | 0.3370 |
| SVM | 0.3104 | 0.3104 | 0.3103 |
| Logistic Regression | 0.3229 | 0.3229 | 0.3219 |
| KNN | 0.2729 | 0.2729 | 0.2707 |
| Random Forest | 0.3458 | 0.3458 | 0.3457 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **Random Forest**（0.3458）。

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
| 1 | KNN | 0.1875 | 0.1875 | 0.1880 |
| 1 | LDA | 0.3125 | 0.3125 | 0.3100 |
| 1 | Logistic Regression | 0.3125 | 0.3125 | 0.3055 |
| 1 | Random Forest | 0.2500 | 0.2500 | 0.2522 |
| 1 | SVM | 0.2750 | 0.2750 | 0.2673 |
| 2 | KNN | 0.2375 | 0.2375 | 0.2431 |
| 2 | LDA | 0.3375 | 0.3375 | 0.3528 |
| 2 | Logistic Regression | 0.3500 | 0.3500 | 0.3576 |
| 2 | Random Forest | 0.4125 | 0.4125 | 0.4138 |
| 2 | SVM | 0.2875 | 0.2875 | 0.2859 |
| 3 | KNN | 0.3625 | 0.3625 | 0.3511 |
| 3 | LDA | 0.3875 | 0.3875 | 0.3860 |
| 3 | Logistic Regression | 0.3000 | 0.3000 | 0.2913 |
| 3 | Random Forest | 0.3875 | 0.3875 | 0.3777 |
| 3 | SVM | 0.4125 | 0.4125 | 0.4153 |
| 4 | KNN | 0.3000 | 0.3000 | 0.2908 |
| 4 | LDA | 0.3000 | 0.3000 | 0.2958 |
| 4 | Logistic Regression | 0.3250 | 0.3250 | 0.3226 |
| 4 | Random Forest | 0.3875 | 0.3875 | 0.3782 |
| 4 | SVM | 0.3000 | 0.3000 | 0.3047 |
| 5 | KNN | 0.3375 | 0.3375 | 0.3230 |
| 5 | LDA | 0.3750 | 0.3750 | 0.3742 |
| 5 | Logistic Regression | 0.3250 | 0.3250 | 0.3248 |
| 5 | Random Forest | 0.3125 | 0.3125 | 0.3129 |
| 5 | SVM | 0.3500 | 0.3500 | 0.3508 |
| 6 | KNN | 0.2125 | 0.2125 | 0.1934 |
| 6 | LDA | 0.3125 | 0.3125 | 0.3114 |
| 6 | Logistic Regression | 0.3250 | 0.3250 | 0.3242 |
| 6 | Random Forest | 0.3250 | 0.3250 | 0.3228 |
| 6 | SVM | 0.2375 | 0.2375 | 0.2381 |

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
| LDA | 0.2639 | 0.2639 | 0.2651 |
| SVM | 0.2417 | 0.2417 | 0.2427 |
| Logistic Regression | 0.2611 | 0.2611 | 0.2619 |
| KNN | 0.2222 | 0.2222 | 0.2216 |
| Random Forest | 0.2389 | 0.2389 | 0.2396 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.2639）。

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
| 1 | KNN | 0.1583 | 0.1583 | 0.1564 |
| 1 | LDA | 0.1833 | 0.1833 | 0.1820 |
| 1 | Logistic Regression | 0.2000 | 0.2000 | 0.1944 |
| 1 | Random Forest | 0.1583 | 0.1583 | 0.1568 |
| 1 | SVM | 0.1833 | 0.1833 | 0.1701 |
| 2 | KNN | 0.2083 | 0.2083 | 0.2112 |
| 2 | LDA | 0.2667 | 0.2667 | 0.2745 |
| 2 | Logistic Regression | 0.2500 | 0.2500 | 0.2592 |
| 2 | Random Forest | 0.2417 | 0.2417 | 0.2429 |
| 2 | SVM | 0.2167 | 0.2167 | 0.2154 |
| 3 | KNN | 0.3417 | 0.3417 | 0.3412 |
| 3 | LDA | 0.3083 | 0.3083 | 0.3099 |
| 3 | Logistic Regression | 0.3167 | 0.3167 | 0.3115 |
| 3 | Random Forest | 0.2917 | 0.2917 | 0.2818 |
| 3 | SVM | 0.3333 | 0.3333 | 0.3350 |
| 4 | KNN | 0.2250 | 0.2250 | 0.2207 |
| 4 | LDA | 0.2417 | 0.2417 | 0.2415 |
| 4 | Logistic Regression | 0.2250 | 0.2250 | 0.2255 |
| 4 | Random Forest | 0.2083 | 0.2083 | 0.2084 |
| 4 | SVM | 0.2250 | 0.2250 | 0.2262 |
| 5 | KNN | 0.2417 | 0.2417 | 0.2365 |
| 5 | LDA | 0.3000 | 0.3000 | 0.2991 |
| 5 | Logistic Regression | 0.2583 | 0.2583 | 0.2574 |
| 5 | Random Forest | 0.2750 | 0.2750 | 0.2759 |
| 5 | SVM | 0.2667 | 0.2667 | 0.2675 |
| 6 | KNN | 0.1583 | 0.1583 | 0.1579 |
| 6 | LDA | 0.2833 | 0.2833 | 0.2956 |
| 6 | Logistic Regression | 0.3167 | 0.3167 | 0.3177 |
| 6 | Random Forest | 0.2583 | 0.2583 | 0.2621 |
| 6 | SVM | 0.2250 | 0.2250 | 0.2302 |

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
