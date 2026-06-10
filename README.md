# 运动想象 EEG 经典机器学习分析

本项目当前只服务第一组任务：数据预处理和经典机器学习算法分析。代码围绕 `.mat` EEG 数据读取、前 30 个通道预处理、基础特征提取、经典分类器评估以及结果保存展开，不包含深度学习部署和 PPT 制作内容。

## 项目结构

```text
Data/          解压后的 .mat EEG 数据
SourceFile/    原始压缩包及其他未经处理的源文件
Scr/           Python 源代码
Results/       实验指标、日志、表格和结果图片
AGENTS.md      项目协作规范
README.md      项目说明和代码文件说明
requirements.txt Python 依赖版本
数据说明.txt   课程数据说明
```

## 数据约定

每个被试文件应包含变量 `eeg`，形状为：

```text
6 x 44 x 5000 x 20
```

含义依次为 6 个类别、44 个通道、5000 个采样点和每类 20 个 trial。当前分析使用前 30 个 EEG 通道，采样率为 500 Hz。

当前类别顺序依据 `数据说明.txt` 暂定为：

| 标签 | 类别 |
|---:|---|
| 0 | 左手运动想象 |
| 1 | 右手运动想象 |
| 2 | 左腕运动想象 |
| 3 | 右腕运动想象 |
| 4 | 左臂运动想象 |
| 5 | 右臂运动想象 |

## Python 环境

建议从项目根目录创建并启用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前依赖如下：

| 包 | 用途 | 是否必需 |
|---|---|---|
| `numpy` | 数组转换、标签生成、特征矩阵拼接和数值计算 | 必需 |
| `scipy` | 读取 MATLAB `.mat` 文件、滤波、Welch PSD | 必需 |
| `matplotlib` | 保存混淆矩阵、模型对比图和折间结果图 | 必需 |
| `scikit-learn` | Pipeline、标准化、经典分类器、交叉验证和评价指标 | 必需 |

## 代码文件说明

| 文件 | 功能 |
|---|---|
| `Scr/data_loader.py` | 读取单个被试 `.mat` 文件，检查 `eeg` 变量和 4 维形状，截取前 30 个通道；将 `classes x channels x time_points x trials` 转换为机器学习使用的 `X, y`，其中 `X` 为 `samples x channels x time_points`，`y` 为类别标签。 |
| `Scr/preprocessing.py` | 提供 EEG 预处理函数：输入校验、逐 trial 逐通道去均值、Butterworth 零相位带通滤波、可选时间窗截取，以及统一入口 `preprocess_eeg()`。输入输出主形状为 `samples x channels x time_points`。 |
| `Scr/feature_extraction.py` | 提取基础时域和频域特征。时域特征包括均值、标准差、方差、RMS、峰峰值、平均绝对值、偏度、峰度、波形长度和过零率；频域特征基于 Welch PSD，计算 alpha/beta 频带绝对功率、相对功率、谱质心和归一化谱熵。统一入口 `extract_basic_features()` 输出二维特征矩阵、特征名和可保存配置。 |
| `Scr/machine_learning_config.py` | 集中保存实验配置：随机种子、采样率、通道数、类别名称、2/4/6 分类任务、预处理参数、特征参数、分类器列表和验证策略名称。 |
| `Scr/classifiers.py` | 创建 LDA、SVM、逻辑回归、KNN 和随机森林分类器。需要标准化的模型使用 `StandardScaler + classifier` 的 `Pipeline`，保证标准化只在每个训练折内部拟合；随机森林使用 passthrough scaler。 |
| `Scr/model_evaluation.py` | 构建验证划分并计算指标。支持 LOSO 跨被试留一验证和被试内分层 5 折验证；对每个分类器生成折内指标、折外总体指标、每类 precision/recall/F1、计数混淆矩阵和归一化混淆矩阵。 |
| `Scr/machine_learning_reporting.py` | 保存实验输出。负责写入 UTF-8 JSON、CSV 指标表，绘制混淆矩阵、模型指标对比图、各折 balanced accuracy 曲线，并生成包含关键结果和图片引用的 Markdown 报告。 |
| `Scr/machine_learning_process.py` | 端到端实验入口。按配置读取 6 个被试数据，完成预处理和特征提取，分别构建 2 分类、4 分类、6 分类任务，运行多个经典分类器和交叉验证，最后把配置、特征名、完整结果、CSV、图片和 `report.md` 保存到 `Results/`。 |

## 当前实验流程

运行默认经典机器学习实验：

```powershell
python Scr\machine_learning_process.py
```

运行前请确认：

1. `Data/` 中存在 `sub_01_MI.mat` 到 `sub_06_MI.mat` 六个文件。
2. 每个 `.mat` 文件包含形状为 `(6, 44, 5000, 20)` 的 `eeg`。
3. 已安装 `requirements.txt` 中列出的依赖。
4. 从项目根目录运行命令，确保 `Scr/` 中模块可以正常导入。

默认配置位于 `Scr/machine_learning_config.py` 和 `Scr/machine_learning_process.py`：

| 项目 | 当前设置 |
|---|---|
| 通道 | 前 30 个通道 |
| 采样率 | 500 Hz |
| 预处理 | 去均值，8-30 Hz Butterworth 零相位带通滤波，时间窗 `(1.5, 5.5)` 秒 |
| 特征 | Welch PSD 频域特征 + 时域统计特征 |
| 任务 | `2class`、`4class`、`6class` |
| 分类器 | LDA、SVM、逻辑回归、KNN、随机森林 |
| 默认验证 | 被试内分层 5 折：`within_subject_5fold` |

每次运行会创建独立结果目录：

```text
Results/machine_learning/course_report/course_report_日期_时间/
```

主要输出包括：

| 文件或目录 | 内容 |
|---|---|
| `experiment_config.json` | 本次实验参数、分类器参数、验证策略和数据泄漏控制说明 |
| `feature_names.json` | 特征列名称 |
| `fold_metrics.csv` | 每个任务、分类器和验证折的指标 |
| `summary_metrics.csv` | 每个任务和分类器的汇总指标 |
| `complete_results.json` | 完整折内记录、折外预测、每类指标和混淆矩阵 |
| `figures/` | 模型对比图、折间 balanced accuracy 图、计数和归一化混淆矩阵 |
| `report.md` | 自动生成的 Markdown 实验报告 |

## 数据泄漏控制

当前基础特征按单个 trial 独立计算，不拟合跨样本统计量。`StandardScaler` 和分类器都放在 scikit-learn `Pipeline` 中，并在每个交叉验证训练折内拟合，测试折不参与标准化器或分类器训练。后续若加入 CSP、PCA 或特征选择，也必须放入交叉验证训练折内部拟合。
