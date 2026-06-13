# 经典机器学习结果区研究说明

## 1. 当前主线

本目录保存第一组最终最关心的经典机器学习结果，包括默认课程报告和分类器专项调参结果。当前统一前端采用：

```text
预处理：无 CAR + 50 Hz 陷波 + 8-30 Hz 带通 + 1.5-5.5 s 时间窗 + 单 trial z-score
特征：基础频域 + 时域特征
验证：被试内分层 5 折
```

除默认被试内 5 折外，本目录也支持保存跨被试 `LOSO` 结果，两者共用同一套 CSV、JSON、图像和报告格式，便于直接横向比较。

## 2. 三个任务的当前推荐模型

依据当前分类器专项调参结果，三种任务的最佳 `Balanced Accuracy` 为：

| 任务 | 当前最佳模型 | Balanced Accuracy |
|---|---|---:|
| `2class` | Logistic Regression, `C=0.03`, `class_weight='balanced'` | `0.6417` |
| `4class` | Random Forest, `n_estimators=500`, `max_features='sqrt'` | `0.3458` |
| `6class` | LDA 默认配置 | `0.2639` |

这说明三种任务暂时还没有一个单一分类器能全线最优，因此当前策略是针对任务分别给出推荐模型。

## 3. 子目录含义

```text
03_ml_classification/
  course_reports/      每次默认实验的完整课程报告
  classifier_tuning/   分类器专项调参结果
```

- `course_reports/`：保存默认主流程跑出的完整结果目录，含 `report.md`、总体/被试级 CSV、JSON 和图像。
- `course_reports/`：同时保存被试内 5 折与跨被试 LOSO 的完整结果目录；目录名会标明是否为 `loso`。
- `classifier_tuning/`：保存固定前端下的大批量分类器比较结果，重点看 `best_by_task.csv` 和 `best_by_classifier.csv`。

## 4. 图像、CSV、JSON 的用途

- 图像：更适合直接放入课程报告，展示总体与被试级混淆矩阵、模型对比。
- CSV：适合快速筛选每个任务的最佳模型，并单独查看每个被试的汇总指标。
- JSON：保留最完整的结构化实验信息，便于复查配置和二次分析。

## 5. 如何运行默认实验与分类器专项调参

运行默认课程报告：

```powershell
python Scr\machine_learning_process.py
```

输出路径：

```text
Results/03_ml_classification/course_reports/course_report_YYYYMMDD_HHMMSS/
```

运行跨被试 LOSO 课程报告：

```powershell
python Scr\cross_subject_process.py
```

输出路径：

```text
Results/03_ml_classification/course_reports/course_report_loso_YYYYMMDD_HHMMSS/
```

运行分类器专项调参：

```powershell
python Scr\classifier_experiments.py
```

输出路径：

```text
Results/03_ml_classification/classifier_tuning/classifier_tuning_YYYYMMDD_HHMMSS/
```
