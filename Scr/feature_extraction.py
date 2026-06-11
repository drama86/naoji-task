import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif

TIME_FEATURES = (
    "mean",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "peak_to_peak",
    "mean_absolute_value",
    "skewness",
    "kurtosis",
    "waveform_length",
    "zero_crossing_rate",
)

DEFAULT_FREQUENCY_BANDS = (
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)

DEFAULT_FBCSP_BANDS = (
    ("band_8_12", 8.0, 12.0),
    ("band_12_16", 12.0, 16.0),
    ("band_16_20", 16.0, 20.0),
    ("band_20_24", 20.0, 24.0),
    ("band_24_28", 24.0, 28.0),
    ("band_26_30", 26.0, 30.0),
)


def _validate_eeg_features_input(X):
    """Validate EEG data with shape (samples, channels, time_points)."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            "Expected X with shape (samples, channels, time_points), "
            f"got {X.shape}"
        )
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] < 2:
        raise ValueError(
            "X must contain at least one sample, one channel, and "
            "two time points"
        )
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("EEG data must contain numeric values")
    if not np.isfinite(X).all():
        raise ValueError("EEG data contains NaN or infinite values")
    return X.astype(np.float64, copy=False)


def _flatten_feature_blocks(blocks):
    """Flatten channel feature blocks into (samples, features)."""
    return np.concatenate(blocks, axis=1)


def _channel_feature_names(channel_count, feature_names):
    """Return stable names ordered by feature and then channel."""
    return [
        f"{feature_name}_ch_{channel_number:02d}"
        for feature_name in feature_names
        for channel_number in range(1, channel_count + 1)
    ]


def extract_time_features(X):
    """Extract per-trial, per-channel time-domain features.

    Args:
        X: EEG data with shape (samples, channels, time_points).

    Returns:
        features: Shape (samples, channels * 10). Columns are ordered by
            feature type and then by one-based channel number.
        feature_names: Stable name for every feature column.
    """
    X = _validate_eeg_features_input(X)
    mean = X.mean(axis=-1)
    centered = X - mean[..., np.newaxis]
    variance = np.mean(centered**2, axis=-1)
    standard_deviation = np.sqrt(variance)
    safe_standard_deviation = np.where(
        standard_deviation > np.finfo(np.float64).eps,
        standard_deviation,
        1.0,
    )
    standardized = centered / safe_standard_deviation[..., np.newaxis]

    skewness = np.mean(standardized**3, axis=-1)
    kurtosis = np.mean(standardized**4, axis=-1) - 3.0
    constant_channels = standard_deviation <= np.finfo(np.float64).eps
    skewness[constant_channels] = 0.0
    kurtosis[constant_channels] = 0.0

    blocks = (
        mean,
        standard_deviation,
        variance,
        np.sqrt(np.mean(X**2, axis=-1)),
        np.ptp(X, axis=-1),
        np.mean(np.abs(X), axis=-1),
        skewness,
        kurtosis,
        np.sum(np.abs(np.diff(X, axis=-1)), axis=-1),
        np.mean(
            np.signbit(X[..., 1:]) != np.signbit(X[..., :-1]),
            axis=-1,
        ),
    )
    features = _flatten_feature_blocks(blocks)
    feature_names = _channel_feature_names(X.shape[1], TIME_FEATURES)

    if not np.isfinite(features).all():
        raise ValueError("Time-domain feature extraction produced non-finite values")
    return features, feature_names


def _validate_frequency_bands(frequency_bands, sampling_rate):
    """Validate and normalize named frequency bands."""
    nyquist = sampling_rate / 2.0
    normalized = []
    names = set()
    for band in frequency_bands:
        if len(band) != 3:
            raise ValueError(
                "Each frequency band must be (name, low_hz, high_hz)"
            )
        name, low_hz, high_hz = band
        name = str(name).strip()
        if not name:
            raise ValueError("Frequency-band names cannot be empty")
        if name in names:
            raise ValueError(f"Duplicate frequency-band name: {name}")
        low_hz = float(low_hz)
        high_hz = float(high_hz)
        if not 0 <= low_hz < high_hz <= nyquist:
            raise ValueError(
                f"Band {name!r} must satisfy "
                f"0 <= low_hz < high_hz <= {nyquist:g}"
            )
        names.add(name)
        normalized.append((name, low_hz, high_hz))
    if not normalized:
        raise ValueError("At least one frequency band is required")
    return tuple(normalized)


def _integrate_psd(psd, frequencies, low_hz, high_hz):
    """Integrate PSD over an inclusive frequency interval."""
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Welch frequency resolution is too coarse for "
            f"{low_hz:g}-{high_hz:g} Hz"
        )
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def extract_frequency_features(
    X,
    sampling_rate=500,
    frequency_bands=DEFAULT_FREQUENCY_BANDS,
    total_power_band=(8.0, 30.0),
    nperseg=500,
    noverlap=250,
):
    """Extract per-channel Welch PSD summary features.

    Absolute band power is the integral of the Welch PSD inside each band.
    Relative band power divides that value by total power in
    `total_power_band`. Spectral centroid and normalized spectral entropy
    are also calculated inside `total_power_band`.

    Returns:
        features: Shape (samples, channels * (2 * bands + 2)).
        feature_names: Stable name for every feature column.
    """
    X = _validate_eeg_features_input(X)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if isinstance(nperseg, bool) or int(nperseg) != nperseg or nperseg < 2:
        raise ValueError("nperseg must be an integer greater than 1")
    nperseg = min(int(nperseg), X.shape[-1])
    if noverlap is None:
        noverlap = nperseg // 2
    if (
        isinstance(noverlap, bool)
        or int(noverlap) != noverlap
        or not 0 <= noverlap < nperseg
    ):
        raise ValueError("noverlap must be an integer in [0, nperseg)")
    noverlap = int(noverlap)

    bands = _validate_frequency_bands(frequency_bands, sampling_rate)
    if len(total_power_band) != 2:
        raise ValueError("total_power_band must be (low_hz, high_hz)")
    total_low, total_high = map(float, total_power_band)
    nyquist = sampling_rate / 2.0
    if not 0 <= total_low < total_high <= nyquist:
        raise ValueError(
            "total_power_band must satisfy "
            f"0 <= low_hz < high_hz <= {nyquist:g}"
        )

    frequencies, psd = welch(
        X,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    total_power = _integrate_psd(
        psd,
        frequencies,
        total_low,
        total_high,
    )
    safe_total_power = np.maximum(total_power, np.finfo(np.float64).tiny)

    blocks = []
    block_names = []
    for band_name, low_hz, high_hz in bands:
        absolute_power = _integrate_psd(
            psd,
            frequencies,
            low_hz,
            high_hz,
        )
        blocks.append(absolute_power)
        block_names.append(f"{band_name}_absolute_power")

    for band_name, low_hz, high_hz in bands:
        absolute_power = _integrate_psd(
            psd,
            frequencies,
            low_hz,
            high_hz,
        )
        blocks.append(absolute_power / safe_total_power)
        block_names.append(f"{band_name}_relative_power")

    total_mask = (
        (frequencies >= total_low) & (frequencies <= total_high)
    )
    selected_frequencies = frequencies[total_mask]
    selected_psd = psd[..., total_mask]
    psd_sum = selected_psd.sum(axis=-1, keepdims=True)
    probability = np.divide(
        selected_psd,
        psd_sum,
        out=np.zeros_like(selected_psd),
        where=psd_sum > 0,
    )
    spectral_centroid = np.sum(
        probability * selected_frequencies,
        axis=-1,
    )
    positive_probability = np.where(probability > 0, probability, 1.0)
    entropy = -np.sum(
        probability * np.log(positive_probability),
        axis=-1,
    )
    if selected_frequencies.size > 1:
        entropy /= np.log(selected_frequencies.size)
    blocks.extend((spectral_centroid, entropy))
    block_names.extend(("spectral_centroid", "spectral_entropy"))

    features = _flatten_feature_blocks(blocks)
    feature_names = _channel_feature_names(X.shape[1], block_names)
    if not np.isfinite(features).all():
        raise ValueError("Frequency feature extraction produced non-finite values")
    return features, feature_names


def extract_basic_features(
    X,
    feature_sets=("time", "frequency"),
    sampling_rate=500,
    frequency_bands=DEFAULT_FREQUENCY_BANDS,
    total_power_band=(8.0, 30.0),
    nperseg=500,
    noverlap=250,
):
    """Extract selected basic EEG feature sets through one stable interface.

    Returns:
        features: Two-dimensional array (samples, features).
        feature_names: Names aligned with feature columns.
        config: JSON-serializable extraction configuration.
    """
    X = _validate_eeg_features_input(X)
    feature_sets = tuple(str(item).lower() for item in feature_sets)
    if not feature_sets:
        raise ValueError("feature_sets cannot be empty")
    if len(set(feature_sets)) != len(feature_sets):
        raise ValueError("feature_sets cannot contain duplicates")
    unknown_sets = set(feature_sets) - {"time", "frequency"}
    if unknown_sets:
        raise ValueError(f"Unknown feature sets: {sorted(unknown_sets)}")

    feature_blocks = []
    feature_names = []
    for feature_set in feature_sets:
        if feature_set == "time":
            block, names = extract_time_features(X)
        else:
            block, names = extract_frequency_features(
                X,
                sampling_rate=sampling_rate,
                frequency_bands=frequency_bands,
                total_power_band=total_power_band,
                nperseg=nperseg,
                noverlap=noverlap,
            )
        feature_blocks.append(block)
        feature_names.extend(names)

    features = np.concatenate(feature_blocks, axis=1)
    config = {
        "feature_sets": list(feature_sets),
        "input_shape": list(X.shape),
        "output_shape": list(features.shape),
        "feature_order": "feature_type_then_channel_number",
        "time_features": (
            list(TIME_FEATURES) if "time" in feature_sets else []
        ),
        "frequency": (
            {
                "method": "Welch PSD",
                "sampling_rate_hz": sampling_rate,
                "nperseg": min(int(nperseg), X.shape[-1]),
                "noverlap": (
                    min(int(nperseg), X.shape[-1]) // 2
                    if noverlap is None
                    else int(noverlap)
                ),
                "bands_hz": [
                    {"name": name, "low": low_hz, "high": high_hz}
                    for name, low_hz, high_hz in frequency_bands
                ],
                "total_power_band_hz": list(total_power_band),
                "features_per_channel": [
                    "absolute_band_power",
                    "relative_band_power",
                    "spectral_centroid",
                    "normalized_spectral_entropy",
                ],
            }
            if "frequency" in feature_sets
            else None
        ),
        "uses_dataset_statistics": False,
        "standardization_note": (
            "Feature standardization must be fitted inside each training fold."
        ),
    }
    if features.shape[1] != len(feature_names):
        raise RuntimeError("Feature columns and names are not aligned")
    return features, feature_names, config


def _validate_sample_aligned_inputs(X, y=None):
    """Validate sample-first EEG arrays for fold-fitted transformers."""
    X = _validate_eeg_features_input(X)
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    return X, y.astype(np.int64)


def _trace_normalized_covariances(X, regularization=1e-6):
    """Return mean per-trial covariance normalized by trace."""
    sample_count, channel_count, time_points = X.shape
    covariance = np.matmul(X, np.transpose(X, (0, 2, 1))) / time_points
    traces = np.trace(covariance, axis1=1, axis2=2)
    safe_traces = np.where(
        traces > np.finfo(np.float64).eps,
        traces,
        1.0,
    )
    covariance = covariance / safe_traces[:, np.newaxis, np.newaxis]
    mean_covariance = covariance.mean(axis=0)
    diagonal_scale = np.trace(mean_covariance) / max(channel_count, 1)
    regularized = mean_covariance + (
        regularization * diagonal_scale * np.eye(channel_count)
    )
    return 0.5 * (regularized + regularized.T)


def _csp_component_indices(total_components, requested_components):
    """Select symmetric largest/smallest CSP components."""
    requested_components = max(1, min(int(requested_components), total_components))
    top_count = int(np.ceil(requested_components / 2.0))
    bottom_count = requested_components - top_count
    selected = list(range(top_count))
    if bottom_count > 0:
        selected.extend(range(total_components - bottom_count, total_components))
    return np.asarray(selected, dtype=np.int64)


def _csp_log_variance_features(X_projected):
    """Convert projected signals into log-variance CSP features."""
    variance = np.mean(X_projected**2, axis=-1)
    total_variance = variance.sum(axis=1, keepdims=True)
    safe_total = np.where(
        total_variance > np.finfo(np.float64).tiny,
        total_variance,
        1.0,
    )
    normalized = variance / safe_total
    return np.log(np.maximum(normalized, np.finfo(np.float64).tiny))


class BasicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Stateless wrapper that keeps basic feature extraction inside a pipeline."""

    def __init__(
        self,
        feature_sets=("time", "frequency"),
        sampling_rate=500,
        frequency_bands=DEFAULT_FREQUENCY_BANDS,
        total_power_band=(8.0, 30.0),
        nperseg=500,
        noverlap=250,
    ):
        self.feature_sets = feature_sets
        self.sampling_rate = sampling_rate
        self.frequency_bands = frequency_bands
        self.total_power_band = total_power_band
        self.nperseg = nperseg
        self.noverlap = noverlap

    def fit(self, X, y=None):
        X, _ = _validate_sample_aligned_inputs(X, y)
        self.n_channels_in_ = int(X.shape[1])
        _, self.feature_names_out_, self.feature_config_ = extract_basic_features(
            X[:1],
            feature_sets=self.feature_sets,
            sampling_rate=self.sampling_rate,
            frequency_bands=self.frequency_bands,
            total_power_band=self.total_power_band,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        return self

    def transform(self, X):
        X, _ = _validate_sample_aligned_inputs(X, None)
        features, _, _ = extract_basic_features(
            X,
            feature_sets=self.feature_sets,
            sampling_rate=self.sampling_rate,
            frequency_bands=self.frequency_bands,
            total_power_band=self.total_power_band,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        return features

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class CSPFeatureTransformer(BaseEstimator, TransformerMixin):
    """Common Spatial Pattern feature extractor with one-vs-rest multiclass mode."""

    def __init__(
        self,
        n_components=4,
        multiclass_strategy="ovr",
        regularization=1e-6,
    ):
        self.n_components = n_components
        self.multiclass_strategy = multiclass_strategy
        self.regularization = regularization

    def fit(self, X, y):
        X, y = _validate_sample_aligned_inputs(X, y)
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("CSP requires at least two classes")
        if self.multiclass_strategy != "ovr":
            raise ValueError("multiclass_strategy must be 'ovr'")

        filters = []
        feature_names = []
        if classes.size == 2:
            class_pairs = [(classes[0], classes[1], "binary")]
        else:
            class_pairs = [
                (class_label, None, f"class_{int(class_label)}")
                for class_label in classes
            ]

        for positive_class, negative_class, prefix in class_pairs:
            if negative_class is None:
                positive_mask = y == positive_class
                negative_mask = ~positive_mask
            else:
                positive_mask = y == positive_class
                negative_mask = y == negative_class
            positive_trials = X[positive_mask]
            negative_trials = X[negative_mask]
            if positive_trials.shape[0] == 0 or negative_trials.shape[0] == 0:
                raise ValueError("CSP requires non-empty class partitions")

            positive_covariance = _trace_normalized_covariances(
                positive_trials,
                regularization=self.regularization,
            )
            negative_covariance = _trace_normalized_covariances(
                negative_trials,
                regularization=self.regularization,
            )
            eigenvalues, eigenvectors = eigh(
                positive_covariance,
                positive_covariance + negative_covariance,
            )
            order = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order]
            component_indices = _csp_component_indices(
                eigenvectors.shape[1],
                self.n_components,
            )
            selected_filters = eigenvectors[:, component_indices].T
            filters.append(
                {
                    "positive_class": int(positive_class),
                    "filters": selected_filters,
                }
            )
            for component_index in range(selected_filters.shape[0]):
                feature_names.append(
                    f"csp_{prefix}_comp_{component_index + 1:02d}"
                )

        self.classes_ = classes.astype(np.int64)
        self.filter_blocks_ = filters
        self.feature_names_out_ = feature_names
        return self

    def transform(self, X):
        X, _ = _validate_sample_aligned_inputs(X, None)
        feature_blocks = []
        for block in self.filter_blocks_:
            projected = np.einsum(
                "fc,nct->nft",
                block["filters"],
                X,
            )
            feature_blocks.append(_csp_log_variance_features(projected))
        return np.concatenate(feature_blocks, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class FBCSPFeatureTransformer(BaseEstimator, TransformerMixin):
    """Filter-bank CSP transformer with per-band CSP feature concatenation."""

    def __init__(
        self,
        sampling_rate=500,
        filter_bands=DEFAULT_FBCSP_BANDS,
        filter_order=4,
        n_components=2,
        multiclass_strategy="ovr",
        regularization=1e-6,
    ):
        self.sampling_rate = sampling_rate
        self.filter_bands = filter_bands
        self.filter_order = filter_order
        self.n_components = n_components
        self.multiclass_strategy = multiclass_strategy
        self.regularization = regularization

    def fit(self, X, y):
        from preprocessing import bandpass_filter

        X, y = _validate_sample_aligned_inputs(X, y)
        normalized_bands = _validate_frequency_bands(
            self.filter_bands,
            self.sampling_rate,
        )
        transformers = []
        feature_names = []
        for band_name, low_hz, high_hz in normalized_bands:
            X_band = bandpass_filter(
                X,
                sampling_rate=self.sampling_rate,
                low_cut=low_hz,
                high_cut=high_hz,
                filter_order=self.filter_order,
            )
            transformer = CSPFeatureTransformer(
                n_components=self.n_components,
                multiclass_strategy=self.multiclass_strategy,
                regularization=self.regularization,
            )
            transformer.fit(X_band, y)
            transformers.append(
                {
                    "name": band_name,
                    "low_hz": low_hz,
                    "high_hz": high_hz,
                    "transformer": transformer,
                }
            )
            feature_names.extend(
                [
                    f"{band_name}_{feature_name}"
                    for feature_name in transformer.get_feature_names_out()
                ]
            )

        self.band_transformers_ = transformers
        self.feature_names_out_ = feature_names
        return self

    def transform(self, X):
        from preprocessing import bandpass_filter

        X, _ = _validate_sample_aligned_inputs(X, None)
        feature_blocks = []
        for band in self.band_transformers_:
            X_band = bandpass_filter(
                X,
                sampling_rate=self.sampling_rate,
                low_cut=band["low_hz"],
                high_cut=band["high_hz"],
                filter_order=self.filter_order,
            )
            feature_blocks.append(
                band["transformer"].transform(X_band)
            )
        return np.concatenate(feature_blocks, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def build_feature_transformer(feature_config, sampling_rate=500):
    """Create one scikit-learn compatible feature transformer."""
    mode = str(feature_config["mode"]).lower()
    if mode == "basic":
        basic_config = feature_config["basic"]
        return BasicFeatureTransformer(
            feature_sets=basic_config["feature_sets"],
            sampling_rate=sampling_rate,
            frequency_bands=basic_config["frequency_bands"],
            total_power_band=basic_config["total_power_band"],
            nperseg=basic_config["nperseg"],
            noverlap=basic_config["noverlap"],
        )
    if mode == "csp":
        csp_config = feature_config["csp"]
        return CSPFeatureTransformer(
            n_components=csp_config["n_components"],
            multiclass_strategy=csp_config["multiclass_strategy"],
            regularization=csp_config["regularization"],
        )
    if mode == "fbcsp":
        fbcsp_config = feature_config["fbcsp"]
        return FBCSPFeatureTransformer(
            sampling_rate=sampling_rate,
            filter_bands=fbcsp_config["filter_bands"],
            filter_order=fbcsp_config["filter_order"],
            n_components=fbcsp_config["n_components"],
            multiclass_strategy=fbcsp_config["multiclass_strategy"],
            regularization=fbcsp_config["regularization"],
        )
    raise ValueError(f"Unknown feature mode: {mode}")


def build_feature_selector(feature_config):
    """Create an optional fold-fitted feature selector."""
    selection_config = feature_config.get("selection", {})
    if not selection_config or not selection_config.get("enabled", False):
        return "passthrough"

    method = str(selection_config["method"]).lower()
    if method != "mibif":
        raise ValueError(f"Unknown feature-selection method: {method}")

    k_best = int(selection_config["k_best"])
    if k_best <= 0:
        raise ValueError("selection.k_best must be positive")

    return SelectKBest(
        score_func=mutual_info_classif,
        k=k_best,
    )


def summarize_feature_config(feature_config, sampling_rate=500):
    """Return a JSON-serializable summary of the current feature setup."""
    mode = str(feature_config["mode"]).lower()
    selection_config = feature_config.get("selection", {})
    selection_summary = (
        None
        if not selection_config or not selection_config.get("enabled", False)
        else {
            "enabled": True,
            "method": str(selection_config["method"]).lower(),
            "k_best": int(selection_config["k_best"]),
            "fit_scope": "Fitted inside each training fold only.",
        }
    )

    if mode == "basic":
        basic_config = feature_config["basic"]
        return {
            "mode": "basic",
            "report_description": (
                "逐通道时域统计特征与 Welch PSD 频域特征。"
            ),
            "fit_scope": "No cross-sample fit inside feature extractor.",
            "selection": selection_summary,
            "basic": {
                "feature_sets": list(basic_config["feature_sets"]),
                "sampling_rate_hz": sampling_rate,
                "frequency_bands": [
                    {"name": name, "low_hz": low_hz, "high_hz": high_hz}
                    for name, low_hz, high_hz in basic_config["frequency_bands"]
                ],
                "total_power_band_hz": list(basic_config["total_power_band"]),
                "nperseg": basic_config["nperseg"],
                "noverlap": basic_config["noverlap"],
            },
        }
    if mode == "csp":
        csp_config = feature_config["csp"]
        return {
            "mode": "csp",
            "report_description": (
                "CSP 空间判别特征；每个训练折内部重新拟合空间滤波器。"
            ),
            "fit_scope": "Fitted inside each training fold only.",
            "selection": selection_summary,
            "csp": {
                "n_components": csp_config["n_components"],
                "multiclass_strategy": csp_config["multiclass_strategy"],
                "regularization": csp_config["regularization"],
            },
        }
    if mode == "fbcsp":
        fbcsp_config = feature_config["fbcsp"]
        return {
            "mode": "fbcsp",
            "report_description": (
                "Filter-Bank CSP 空间判别特征；每个训练折内部按子频带重新拟合。"
            ),
            "fit_scope": "Fitted inside each training fold only.",
            "selection": selection_summary,
            "fbcsp": {
                "sampling_rate_hz": sampling_rate,
                "filter_bands": [
                    {"name": name, "low_hz": low_hz, "high_hz": high_hz}
                    for name, low_hz, high_hz in fbcsp_config["filter_bands"]
                ],
                "filter_order": fbcsp_config["filter_order"],
                "n_components": fbcsp_config["n_components"],
                "multiclass_strategy": fbcsp_config["multiclass_strategy"],
                "regularization": fbcsp_config["regularization"],
            },
        }
    raise ValueError(f"Unknown feature mode: {mode}")


def expected_feature_names(
    feature_config,
    class_count,
    channel_count,
):
    """Return stable expected feature names for reporting/export."""
    selection_config = feature_config.get("selection", {})
    selection_enabled = bool(
        selection_config and selection_config.get("enabled", False)
    )

    mode = str(feature_config["mode"]).lower()
    if mode == "basic":
        basic_config = feature_config["basic"]
        names = []
        for feature_set in basic_config["feature_sets"]:
            if feature_set == "frequency":
                block_names = []
                for band_name, _, _ in basic_config["frequency_bands"]:
                    block_names.append(f"{band_name}_absolute_power")
                for band_name, _, _ in basic_config["frequency_bands"]:
                    block_names.append(f"{band_name}_relative_power")
                block_names.extend(("spectral_centroid", "spectral_entropy"))
                names.extend(_channel_feature_names(channel_count, block_names))
            elif feature_set == "time":
                names.extend(_channel_feature_names(channel_count, TIME_FEATURES))
        if selection_enabled:
            return [
                f"selected_feature_{index + 1:02d}"
                for index in range(
                    min(len(names), int(selection_config["k_best"]))
                )
            ]
        return names

    if mode == "csp":
        csp_config = feature_config["csp"]
        if class_count == 2:
            prefixes = ("binary",)
        else:
            prefixes = tuple(f"class_{index}" for index in range(class_count))
        names = []
        for prefix in prefixes:
            for component_index in range(int(csp_config["n_components"])):
                names.append(
                    f"csp_{prefix}_comp_{component_index + 1:02d}"
                )
        if selection_enabled:
            return [
                f"selected_feature_{index + 1:02d}"
                for index in range(
                    min(len(names), int(selection_config["k_best"]))
                )
            ]
        return names

    if mode == "fbcsp":
        fbcsp_config = feature_config["fbcsp"]
        if class_count == 2:
            prefixes = ("binary",)
        else:
            prefixes = tuple(f"class_{index}" for index in range(class_count))
        names = []
        for band_name, _, _ in fbcsp_config["filter_bands"]:
            for prefix in prefixes:
                for component_index in range(int(fbcsp_config["n_components"])):
                    names.append(
                        f"{band_name}_csp_{prefix}_comp_{component_index + 1:02d}"
                    )
        if selection_enabled:
            return [
                f"selected_feature_{index + 1:02d}"
                for index in range(
                    min(len(names), int(selection_config["k_best"]))
                )
            ]
        return names

    raise ValueError(f"Unknown feature mode: {mode}")
