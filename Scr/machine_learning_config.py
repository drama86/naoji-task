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
    # 时间窗（秒）：None 表示使用完整 0-10 s。
    # 可改为 (1.0, 5.0) 或 (2.0, 6.0) 做实验，但数据说明未给出提示时刻。
    "time_window_seconds": (1.5, 5.5),
    "time_window_note": (
        "The dataset documentation does not provide cue timing. "
        "The full 0-10 s trial is used by default."
    ),
}

FEATURE_CONFIG = {
    # 特征集合：可使用 ("time",)、("frequency",) 或二者组合。
    # 组合顺序同时决定输出特征列的排列顺序。
    "feature_sets": ("frequency", "time"),
    # Welch PSD 频带定义：(名称, 下边界Hz, 上边界Hz)。
    # 可以继续细分频带，但频带必须位于 0-250 Hz 内。
    "frequency_bands": (
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    ),
    # 相对功率、谱质心和谱熵使用的总频率范围。
    # 应覆盖 frequency_bands 中希望比较的全部频带。
    "total_power_band": (8.0, 30.0),
    # Welch 每段采样点数：500 点在 500 Hz 下对应 1 秒。
    # 值越大频率分辨率越高，但可平均的分段数量越少。
    "nperseg": 500,
    # Welch 相邻分段重叠点数：250 表示 50% 重叠。
    # 必须满足 0 <= noverlap < nperseg。
    "noverlap": 250,
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
