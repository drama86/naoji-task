"""Classical classifier factory with leakage-safe preprocessing."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from machine_learning_config import CLASSIFIER_NAMES, RANDOM_SEED


def create_classifier(classifier_name, random_seed=RANDOM_SEED):
    """Create one configured classical classifier.

    The returned estimator contains StandardScaler inside its Pipeline when
    feature scaling is appropriate. Therefore each cross-validation training
    fold fits its own scaler without using test-fold statistics.
    """
    if classifier_name not in CLASSIFIER_NAMES:
        raise KeyError(
            f"Unknown classifier {classifier_name!r}; "
            f"choose from {CLASSIFIER_NAMES}"
        )

    if classifier_name == "lda":
        estimator = LinearDiscriminantAnalysis(
            # 求解器："lsqr" 支持 shrinkage，适合特征数较多的情况。
            # 若改为 "svd"，必须同时将 shrinkage 设为 None。
            solver="lsqr",
            # 协方差收缩："auto" 自动估计，有助于小样本高维数据稳定。
            shrinkage="auto",
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
    if classifier_name == "svm":
        estimator = SVC(
            # 正则化强度：C 越大越强调训练集拟合，越小正则化越强。
            # 可预先比较 0.1、1、10；正式调参应使用嵌套交叉验证。
            C=10,
            # 核函数："rbf" 适合非线性边界，也可比较 "linear"。
            kernel="rbf",
            # RBF 核宽度："scale" 根据训练折方差自动计算。
            # 也可使用 "auto" 或正浮点数，但不可根据测试折反复选择。
            gamma="scale",
            # 类别权重："balanced" 按训练折类别数量自动补偿不平衡。
            # 当前数据均衡，也可改为 None 作为对照。
            class_weight="balanced",
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
    if classifier_name == "logistic_regression":
        estimator = LogisticRegression(
            # 正则化参数：C 越小正则化越强，可预先比较 0.1、1、10。
            C=0.1,
            # 最大迭代次数：若出现未收敛警告可增加，不代表模型更复杂。
            max_iter=2000,
            # 按训练折类别频数自动设置权重；均衡数据可改为 None。
            class_weight="balanced",
            # 固定涉及随机性的求解过程，保证实验可复现。
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
    if classifier_name == "knn":
        estimator = KNeighborsClassifier(
            # 邻居数量：较小更灵活但易受噪声影响，较大更平滑。
            # 当前样本量可预先比较 3、5、7、9。
            n_neighbors=5,
            # "distance" 让较近邻居权重更高；"uniform" 为等权投票。
            weights="distance",
            # Minkowski 距离；与 p=2 组合时即欧氏距离。
            metric="minkowski",
            # p=1 为曼哈顿距离，p=2 为欧氏距离。
            p=2,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )

    estimator = RandomForestClassifier(
        # 决策树数量：更多通常更稳定但运行更慢，可比较 100、200、500。
        n_estimators=500,
        # 每次节点分裂考虑的特征数："sqrt" 可降低树之间相关性。
        # 也可比较 "log2" 或正整数。
        max_features="sqrt",
        # 按训练折类别数量自动设置权重；均衡数据可改为 None。
        class_weight="balanced",
        # 固定抽样和建树过程，保证重复实验结果一致。
        random_state=random_seed,
        # 并行任务数：None 使用串行默认行为；-1 使用全部逻辑核心。
        # Windows 环境若出现 joblib/WMIC 警告，建议保持 None。
        n_jobs=None,
    )
    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("classifier", estimator),
        ]
    )


def classifier_parameters(classifier_name, random_seed=RANDOM_SEED):
    """Return JSON-serializable parameters for one configured classifier."""
    pipeline = create_classifier(classifier_name, random_seed=random_seed)
    parameters = pipeline.get_params(deep=False)
    classifier = pipeline.named_steps["classifier"]
    return {
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "scaler": (
            "passthrough"
            if pipeline.named_steps["scaler"] == "passthrough"
            else "StandardScaler"
        ),
        "classifier_class": classifier.__class__.__name__,
        "classifier_parameters": {
            key: value
            for key, value in classifier.get_params().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "pipeline_parameters": {
            key: value
            for key, value in parameters.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }
