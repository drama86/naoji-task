# 特征结果区研究说明

## 1. 目录定位

本目录保存特征提取阶段的两类产物：

- 可复用的基础特征矩阵导出；
- 特征方案对比实验与结论文件。

```text
02_feature_results/
  basic_feature_exports/   基础特征矩阵导出
  comparisons/            特征对比实验、阶段性 JSON 与汇总表
```

## 2. 基础特征、CSP、FBCSP、MIBIF 的作用与边界

- 基础特征：逐 trial 独立提取时域统计量和 Welch PSD 频域特征，稳定、可解释、对小样本更友好。
- CSP：学习最能区分类别的空间滤波器，再用投影信号的对数方差作为特征。
- FBCSP：先分多个子频带，再在每个子频带上做 CSP，适合捕捉更细的运动想象频带差异。
- MIBIF：基于互信息对高维特征做筛选，只保留与标签最相关的前 `k` 维。

边界很重要：`CSP/FBCSP/MIBIF` 都必须在每个训练折内部拟合，因此它们适合作为“实验流程中的折内步骤”，不适合在整个数据集上导出一份全局固定矩阵。

## 3. 为什么只导出基础特征矩阵

本目录只导出“基础特征主线”的全局矩阵，原因是：

- 基础特征不依赖训练标签拟合；
- 不会产生“这份矩阵是不是已经偷看过测试集”的理解歧义；
- 更适合作为后续独立分析、可视化和统计检验的统一输入。

因此，`CSP/FBCSP/MIBIF` 在这里保留的是比较结果与研究结论，而不是一个可能误导的全局特征文件。

## 4. 当前对比结果文件位置

- `comparisons/feature_stage_compare/`：阶段性核心对比 JSON，例如 `lda_feature_mode_compare.json`、`fbcsp_mibif36_lda_only.json`
- `comparisons/feature_tuning/`：批量特征实验输出目录，包含 `experiment_manifest.json`、`all_summary_metrics.csv`、`best_by_task.csv`

## 5. 当前综合结论

- 统一主线仍然推荐基础时频特征，因为它在三种任务上的综合稳定性最好。
- FBCSP 在 `4class` 上能带来局部提升，因此保留为重要对照方案。
- OVR-FBCSP + MIBIF 没有形成对三类任务都明显更优的统一方案，因此目前更适合作为补充分析，而不是默认主线。

## 6. 如何重新导出基础特征与重跑对比

导出基础特征矩阵：

```powershell
python Scr\export_basic_features.py
```

输出路径：

```text
Results/02_feature_results/basic_feature_exports/basic_features_YYYYMMDD_HHMMSS/
```

重跑特征对比实验：

```powershell
python Scr\feature_experiments.py
```
