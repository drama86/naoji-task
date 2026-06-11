"""Central configuration for classical machine-learning experiments."""

# 随机种子：控制随机森林、交叉验证洗牌等随机过程。
# 使用相同数值可复现实验；比较实验时不建议随意更改。
RANDOM_SEED = 42

# 原始数据采样率（Hz）：由数据说明给定，不应作为调优参数修改。
SAMPLING_RATE = 500

# 分析通道数量：课程要求使用原始 EEG 的前 30 个通道。
# 仅当课程要求改变时修改，不能超过原始数据的 44 个通道。
CHANNEL_COUNT = 30

CLASS_NAMES = {
    0: "左手运动想象",
    1: "右手运动想象",
    2: "左腕运动想象",
    3: "右腕运动想象",
    4: "左臂运动想象",
    5: "右臂运动想象",
}

CLASSIFICATION_TASKS = {
    "2class": {
        "name": "左手 vs 右手",
        "class_ids": (0, 1),
    },
    "4class": {
        "name": "左手、右手、左腕、右腕",
        "class_ids": (0, 1, 2, 3),
    },
    "6class": {
        "name": "全部六类运动想象",
        "class_ids": (0, 1, 2, 3, 4, 5),
    },
}

PREPROCESSING_CONFIG = {
    # 带通滤波下截止频率（Hz）：保留不低于该频率的成分。
    # 运动想象常关注 mu/alpha 和 beta 节律，当前从 8 Hz 开始。
    "low_cut_hz": 8.0,
    # 带通滤波上截止频率（Hz）：保留不高于该频率的成分。
    # 必须大于 low_cut_hz，且小于 Nyquist 频率 250 Hz。
    "high_cut_hz": 30.0,
    # Butterworth 滤波器阶数：越大过渡带越陡，但也可能增强边缘效应。
    # 建议先比较 2、4、6 阶，当前使用较常见的 4 阶。
    "filter_order": 4,
    # 空间重参考：None 或 "car"（common average reference）。
    # 当前综合实验更推荐不启用 CAR，作为默认主线。
    "spatial_reference": None,
    # 50 Hz 工频陷波中心频率；None 表示关闭陷波。
    "notch_hz": 50.0,
    # 陷波品质因数：越大带宽越窄，通常 20-35 是较常见起点。
    "notch_quality_factor": 30.0,
    # 时间窗（秒）：None 表示使用完整 0-10 s。
    # 可改为 (1.0, 5.0) 或 (2.0, 6.0) 做实验，但数据说明未给出提示时刻。
    "time_window_seconds": (1.5, 5.5),
    "time_window_note": (
        "The dataset documentation does not provide cue timing. "
        "The selected time window is an experimental choice."
    ),
    # 信号级归一化：None 或 "zscore_per_trial_channel"。
    "normalize_mode": "zscore_per_trial_channel",
    # 只做异常 trial 标记，不在当前阶段自动剔除样本。
    "trial_quality_robust_z_threshold": 3.5,
    # 是否按稳健异常标记剔除 trial；当前默认关闭，仅保存诊断。
    "drop_flagged_trials": False,
    # 剔除时的合并规则："peak_to_peak"、"rms" 或 "either"。
    "drop_flagged_trials_rule": "either",
}

FEATURE_CONFIG = {
    # 可选模式："basic"、"csp"、"fbcsp"。
    # basic：当前基础时域 + PSD 特征。
    # csp / fbcsp：需在每个训练折内部拟合，避免数据泄漏。
    "mode": "basic",
    "basic": {
        # 特征集合：可使用 ("time",)、("frequency",) 或二者组合。
        # 组合顺序同时决定输出特征列的排列顺序。
        "feature_sets": ("frequency", "time"),
        # Welch PSD 频带定义：(名称, 下边界Hz, 上边界Hz)。
        "frequency_bands": (
            ("alpha", 8.0, 13.0),
            ("beta", 13.0, 30.0),
        ),
        # 相对功率、谱质心和谱熵使用的总频率范围。
        "total_power_band": (8.0, 30.0),
        # Welch 每段采样点数：500 点在 500 Hz 下对应 1 秒。
        "nperseg": 500,
        # Welch 相邻分段重叠点数：250 表示 50% 重叠。
        "noverlap": 250,
    },
    "csp": {
        # CSP 特征数量：从最大/最小特征值两端对称选取。
        "n_components": 4,
        # 当前多分类使用 one-vs-rest 方式构造多个二分类 CSP。
        "multiclass_strategy": "ovr",
        # 协方差轻微对角正则，降低小样本数值不稳定。
        "regularization": 1e-6,
    },
    "fbcsp": {
        # 子频带设置：当前按参考方案提供 4-40 Hz 内 9 个无重叠 4 Hz 子带。
        "filter_bands": (
            ("band_4_8", 4.0, 8.0),
            ("band_8_12", 8.0, 12.0),
            ("band_12_16", 12.0, 16.0),
            ("band_16_20", 16.0, 20.0),
            ("band_20_24", 20.0, 24.0),
            ("band_24_28", 24.0, 28.0),
            ("band_28_32", 28.0, 32.0),
            ("band_32_36", 32.0, 36.0),
            ("band_36_40", 36.0, 40.0),
        ),
        "filter_order": 4,
        # 每个 OVR-CSP 子带提取两端各 2 个分量，共 4 个 log-variance 特征。
        "n_components": 4,
        "multiclass_strategy": "ovr",
        "regularization": 1e-6,
    },
    "selection": {
        # enabled=False 时不做特征选择；True 时在训练折内做 MIBIF。
        "enabled": False,
        "method": "mibif",
        # 参考方案常用 24 维；也可比较 12、24、36、48。
        "k_best": 24,
    },
}

# 可运行的分类器内部名称。删除某项可缩短默认实验时间；
# 顺序也决定报告中模型展示顺序。
CLASSIFIER_NAMES = (
    "lda",
    "svm",
    "logistic_regression",
    "knn",
    "random_forest",
)

CLASSIFIER_DISPLAY_NAMES = {
    "lda": "LDA",
    "svm": "SVM",
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "random_forest": "Random Forest",
}

VALIDATION_STRATEGIES = {
    # LOSO：每次留出一个完整被试测试，衡量跨被试泛化能力。
    "loso": "Leave-One-Subject-Out",
    # 被试内 5 折：每名被试单独分层 5 折，衡量同一被试内分类能力。
    "within_subject_5fold": "Per-subject Stratified 5-Fold",
}
