# 运动想象 EEG 经典机器学习实验报告

## 实验目的

使用前 30 个 EEG 通道完成运动想象分类，并比较 LDA、SVM、逻辑回归、KNN 和随机森林等经典分类器。

## 数据与预处理

- 被试数量：6
- 采样率：500 Hz
- 使用通道：前 30 个通道
- 预处理：逐 trial、逐通道去均值，8–30 Hz Butterworth 零相位带通滤波
- 时间窗：(4, 8)

## 特征与验证方法

- 特征：逐通道时域统计量与 Welch PSD 频域特征，共 480 维。
- 标准化：在每个训练折内部通过 `StandardScaler` 拟合；随机森林不执行标准化。
- 验证策略：Leave-One-Subject-Out。
- 数据泄漏控制：测试折不参与标准化器和分类器拟合。

## 2class：左手 vs 右手

样本数：240；类别：左手运动想象、右手运动想象。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.5042 | 0.5042 | 0.4877 |
| SVM | 0.4792 | 0.4792 | 0.4691 |
| Logistic Regression | 0.4917 | 0.4917 | 0.4881 |
| KNN | 0.4375 | 0.4375 | 0.4353 |
| Random Forest | 0.4833 | 0.4833 | 0.4751 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.5042）。

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

## 4class：左手、右手、左腕、右腕

样本数：480；类别：左手运动想象、右手运动想象、左腕运动想象、右腕运动想象。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.2979 | 0.2979 | 0.2929 |
| SVM | 0.2917 | 0.2917 | 0.2766 |
| Logistic Regression | 0.2771 | 0.2771 | 0.2759 |
| KNN | 0.2417 | 0.2417 | 0.2399 |
| Random Forest | 0.2708 | 0.2708 | 0.2547 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.2979）。

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

## 6class：全部六类运动想象

样本数：720；类别：左手运动想象、右手运动想象、左腕运动想象、右腕运动想象、左臂运动想象、右臂运动想象。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.2181 | 0.2181 | 0.2161 |
| SVM | 0.1875 | 0.1875 | 0.1773 |
| Logistic Regression | 0.2181 | 0.2181 | 0.2185 |
| KNN | 0.1681 | 0.1681 | 0.1651 |
| Random Forest | 0.1903 | 0.1903 | 0.1818 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.2181）。

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

## 结果说明

- 本报告中的指标均来自交叉验证的折外预测。
- 结果用于第一组经典机器学习分析，不包含深度学习模型。
- 时间窗并非数据说明给出的提示区间，而是当前实验配置。
