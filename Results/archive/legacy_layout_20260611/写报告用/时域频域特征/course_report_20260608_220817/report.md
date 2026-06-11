# 运动想象 EEG 经典机器学习实验报告

## 实验目的

使用前 30 个 EEG 通道完成运动想象分类，并比较 LDA、SVM、逻辑回归、KNN 和随机森林等经典分类器。

## 数据与预处理

- 被试数量：6
- 采样率：500 Hz
- 使用通道：前 30 个通道
- 预处理：逐 trial、逐通道去均值，8–30 Hz Butterworth 零相位带通滤波
- 时间窗：(1.5, 5.5)

## 特征与验证方法

- 特征：逐通道时域统计量与 Welch PSD 频域特征，共 480 维。
- 标准化：在每个训练折内部通过 `StandardScaler` 拟合；随机森林不执行标准化。
- 验证策略：Per-subject Stratified 5-Fold。
- 数据泄漏控制：测试折不参与标准化器和分类器拟合。

## 2class：左手 vs 右手

样本数：240；类别：左手运动想象、右手运动想象。

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| LDA | 0.5542 | 0.5542 | 0.5540 |
| SVM | 0.5208 | 0.5208 | 0.5204 |
| Logistic Regression | 0.5625 | 0.5625 | 0.5619 |
| KNN | 0.5542 | 0.5542 | 0.5485 |
| Random Forest | 0.5458 | 0.5458 | 0.5456 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **Logistic Regression**（0.5625）。

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
| LDA | 0.3229 | 0.3229 | 0.3234 |
| SVM | 0.3229 | 0.3229 | 0.3240 |
| Logistic Regression | 0.3271 | 0.3271 | 0.3275 |
| KNN | 0.3312 | 0.3312 | 0.3265 |
| Random Forest | 0.3438 | 0.3438 | 0.3421 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **Random Forest**（0.3438）。

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
| LDA | 0.2722 | 0.2722 | 0.2732 |
| SVM | 0.2472 | 0.2472 | 0.2478 |
| Logistic Regression | 0.2528 | 0.2528 | 0.2540 |
| KNN | 0.2347 | 0.2347 | 0.2341 |
| Random Forest | 0.2708 | 0.2708 | 0.2673 |

按 Balanced Accuracy，本次实验表现最佳的模型为 **LDA**（0.2722）。

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
