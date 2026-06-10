import numpy as np
from scipy.signal import welch

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
