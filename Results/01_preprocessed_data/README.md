# 预处理结果区研究说明

## 1. 目录定位

本目录只保存第一组工作中的预处理相关产物，不改动 `Data/` 中原始 `.mat` 数据。这里既包括可复用的数值中间数据，也包括预处理诊断结果和预处理调参实验。

```text
01_preprocessed_data/
  dataset_exports/   真实预处理后数值数据导出
  diagnostics/       数据检查、处理预览和诊断类结果
  tuning/            预处理消融实验结果
```

## 2. 当前推荐预处理链路

当前推荐主线为：

```text
逐 trial 逐通道去均值
-> 50 Hz 陷波
-> 8-30 Hz 零相位带通
-> 时间窗 1.5-5.5 s
-> 单 trial-单通道 z-score
-> 只做异常 trial 标记，不自动剔除
```

对应关键配置：

```text
spatial_reference = None
notch_hz = 50.0
low_cut_hz = 8.0
high_cut_hz = 30.0
time_window_seconds = (1.5, 5.5)
normalize_mode = "zscore_per_trial_channel"
drop_flagged_trials = False
```

## 3. 导出文件格式

`dataset_exports/preprocessed_dataset_时间戳/` 下每次导出都会生成：

| 文件 | 说明 |
|---|---|
| `sub_01_preprocessed.npz` ... `sub_06_preprocessed.npz` | 每个被试一个压缩文件，包含 `signals(float32)`、`labels(int64)`、`subject_id` |
| `preprocessing_config.json` | 本次导出的固定预处理参数 |
| `trial_quality_summary.json` | 各被试 trial 的 RMS 与 peak-to-peak 稳健异常标记摘要 |
| `dropped_trials.json` | 当前配置下的异常 trial 删除记录；推荐主线默认不删样本 |
| `dataset_index.csv` | 每个被试导出文件名、样本数、通道数、时间点数与包含类别 |
| `class_mapping.json` | 标签到动作类别的映射 |

推荐时间窗为 4 秒，因此每个 trial 导出后的 `signals` 形状固定为 `(samples, 30, 2000)`。

## 4. 预处理诊断文件说明

`diagnostics/` 中主要保存早期数据结构检查、预处理预览图和处理过程核查结果。它们不作为最终可复用数据集，但有助于回答以下问题：

- 原始 `.mat` 的结构是否符合预期；
- 截取前 30 通道后信号形状是否正确；
- 滤波和时间窗裁剪是否按预期生效；
- 是否存在幅值异常明显的 trial。

## 5. 关键结论摘要

- 时间窗：从完整 `0-10 s` 改为中段窗是最有效的预处理改进；`1.5-5.5 s` 在 2/4/6 分类之间最均衡。
- CAR：在当前数据和特征体系下没有带来稳定增益，因此默认不启用。
- 异常 trial 剔除：基于 RMS 与 peak-to-peak 的粗粒度剔除对 `6class` 略有帮助，但会拖累 `2class` 和 `4class`，因此不纳入默认流程。

## 6. 如何重新导出预处理后数据

在项目根目录运行：

```powershell
python Scr\export_preprocessed_data.py
```

输出会写到：

```text
Results/01_preprocessed_data/dataset_exports/preprocessed_dataset_YYYYMMDD_HHMMSS/
```

若要重新跑预处理消融实验：

```powershell
python Scr\preprocessing_experiments.py
```
