import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def _validate_eeg_array(X):
    """Validate EEG sample data with shape (samples, channels, time_points)."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            "Expected X with shape (samples, channels, time_points), "
            f"got {X.shape}"
        )
    if X.shape[-1] == 0:
        raise ValueError("EEG data must contain at least one time point")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("EEG data must contain numeric values")
    if not np.isfinite(X).all():
        raise ValueError("EEG data contains NaN or infinite values")
    return X


def remove_channel_mean(X):
    """Remove each trial-channel temporal mean.

    Args:
        X: EEG data with shape (samples, channels, time_points).

    Returns:
        A new floating-point array with the same shape as X.
    """
    X = _validate_eeg_array(X)
    X_float = X.astype(np.float64, copy=True)
    return X_float - X_float.mean(axis=-1, keepdims=True)


def common_average_reference(X):
    """Apply common average reference across channels at each time point.

    Args:
        X: EEG data with shape (samples, channels, time_points).

    Returns:
        Re-referenced array with the same shape as X.
    """
    X = _validate_eeg_array(X)
    X_float = X.astype(np.float64, copy=False)
    channel_mean = X_float.mean(axis=1, keepdims=True)
    referenced = X_float - channel_mean
    if not np.isfinite(referenced).all():
        raise ValueError("Common average reference produced NaN or infinite values")
    return referenced


def design_bandpass_sos(
    sampling_rate=500,
    low_cut=8.0,
    high_cut=30.0,
    filter_order=4,
):
    """Design a Butterworth band-pass filter in SOS form."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if filter_order <= 0 or isinstance(filter_order, bool):
        raise ValueError("filter_order must be a positive integer")
    if int(filter_order) != filter_order:
        raise ValueError("filter_order must be a positive integer")

    nyquist = sampling_rate / 2.0
    if not 0 < low_cut < high_cut < nyquist:
        raise ValueError(
            "Cutoff frequencies must satisfy "
            f"0 < low_cut < high_cut < {nyquist:g} Hz"
        )

    return butter(
        int(filter_order),
        [low_cut, high_cut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )


def design_notch_sos(
    sampling_rate=500,
    notch_hz=50.0,
    quality_factor=30.0,
):
    """Design a notch filter in SOS form for line-noise suppression."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    nyquist = sampling_rate / 2.0
    if not 0 < notch_hz < nyquist:
        raise ValueError(
            f"notch_hz must satisfy 0 < notch_hz < {nyquist:g} Hz"
        )
    if quality_factor <= 0:
        raise ValueError("quality_factor must be positive")

    b, a = iirnotch(notch_hz, quality_factor, fs=sampling_rate)
    return tf2sos(b, a)


def bandpass_filter(
    X,
    sampling_rate=500,
    low_cut=8.0,
    high_cut=30.0,
    filter_order=4,
):
    """Apply a zero-phase Butterworth band-pass filter along time.

    Input and output shape:
        (samples, channels, time_points)
    """
    X = _validate_eeg_array(X)
    sos = design_bandpass_sos(
        sampling_rate=sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
        filter_order=filter_order,
    )
    X_float = X.astype(np.float64, copy=False)

    try:
        filtered = sosfiltfilt(sos, X_float, axis=-1)
    except ValueError as error:
        raise ValueError(
            "EEG signal is too short for zero-phase filtering; "
            f"time_points={X.shape[-1]}, filter_order={filter_order}"
        ) from error

    if not np.isfinite(filtered).all():
        raise ValueError("Band-pass filtering produced NaN or infinite values")
    return filtered


def notch_filter(
    X,
    sampling_rate=500,
    notch_hz=50.0,
    quality_factor=30.0,
):
    """Apply a zero-phase notch filter along time.

    Input and output shape:
        (samples, channels, time_points)
    """
    X = _validate_eeg_array(X)
    sos = design_notch_sos(
        sampling_rate=sampling_rate,
        notch_hz=notch_hz,
        quality_factor=quality_factor,
    )
    X_float = X.astype(np.float64, copy=False)

    try:
        filtered = sosfiltfilt(sos, X_float, axis=-1)
    except ValueError as error:
        raise ValueError(
            "EEG signal is too short for zero-phase notch filtering; "
            f"time_points={X.shape[-1]}"
        ) from error

    if not np.isfinite(filtered).all():
        raise ValueError("Notch filtering produced NaN or infinite values")
    return filtered


def crop_time_window(X, start_seconds, end_seconds, sampling_rate=500):
    """Crop an EEG time window.

    Args:
        X: EEG data with shape (samples, channels, time_points).
        start_seconds: Inclusive start time in seconds.
        end_seconds: Exclusive end time in seconds.
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Cropped data with shape
        (samples, channels, (end_seconds - start_seconds) * sampling_rate).
    """
    X = _validate_eeg_array(X)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds cannot be negative")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    duration_seconds = X.shape[-1] / sampling_rate
    if end_seconds > duration_seconds:
        raise ValueError(
            f"Time window ends at {end_seconds:g} s, but signal duration "
            f"is only {duration_seconds:g} s"
        )

    start_index_float = start_seconds * sampling_rate
    end_index_float = end_seconds * sampling_rate
    start_index = int(round(start_index_float))
    end_index = int(round(end_index_float))
    if not np.isclose(start_index, start_index_float) or not np.isclose(
        end_index, end_index_float
    ):
        raise ValueError(
            "Time-window boundaries must align with sampling points"
        )

    return X[..., start_index:end_index].copy()


def zscore_per_trial_channel(X):
    """Z-score each trial-channel independently along the time axis.

    Args:
        X: EEG data with shape (samples, channels, time_points).

    Returns:
        Normalized array with the same shape as X.
    """
    X = _validate_eeg_array(X)
    X_float = X.astype(np.float64, copy=False)
    mean = X_float.mean(axis=-1, keepdims=True)
    standard_deviation = X_float.std(axis=-1, keepdims=True)
    safe_standard_deviation = np.where(
        standard_deviation > np.finfo(np.float64).eps,
        standard_deviation,
        1.0,
    )
    normalized = (X_float - mean) / safe_standard_deviation
    if not np.isfinite(normalized).all():
        raise ValueError("Signal normalization produced NaN or infinite values")
    return normalized


def summarize_trial_quality(X, robust_z_threshold=3.5):
    """Summarize trial-wise amplitude statistics without dropping samples.

    Args:
        X: EEG data with shape (samples, channels, time_points).
        robust_z_threshold: Threshold for modified z-score flagging.

    Returns:
        JSON-serializable dictionary describing RMS and peak-to-peak
        distributions across trials.
    """
    X = _validate_eeg_array(X)
    if robust_z_threshold <= 0:
        raise ValueError("robust_z_threshold must be positive")

    trial_rms = np.sqrt(np.mean(X.astype(np.float64, copy=False) ** 2, axis=(1, 2)))
    trial_peak_to_peak = np.ptp(X, axis=2).max(axis=1)

    def _robust_flags(values):
        values = np.asarray(values, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad <= np.finfo(np.float64).eps:
            modified_z = np.zeros_like(values)
        else:
            modified_z = 0.6745 * (values - median) / mad
        flagged = np.flatnonzero(np.abs(modified_z) > robust_z_threshold)
        return {
            "median": median,
            "mad": mad,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "flagged_trial_indices_zero_based": flagged.astype(int).tolist(),
            "flagged_count": int(flagged.size),
        }

    return {
        "sample_count": int(X.shape[0]),
        "robust_z_threshold": float(robust_z_threshold),
        "trial_rms": _robust_flags(trial_rms),
        "trial_peak_to_peak": _robust_flags(trial_peak_to_peak),
    }


def preprocess_eeg(
    X,
    sampling_rate,
    low_cut,
    high_cut,
    filter_order,
    time_window,
    notch_hz=None,
    notch_quality_factor=30.0,
    spatial_reference=None,
    normalize_mode=None,
):
    """Run the reusable EEG preprocessing pipeline.

    Processing order:
        per-trial channel demeaning -> optional spatial re-reference ->
        optional zero-phase notch -> zero-phase band-pass -> optional crop ->
        optional signal normalization.

    Args:
        X: EEG data with shape (samples, channels, time_points).
        sampling_rate: Sampling frequency in Hz.
        low_cut: Lower band-pass cutoff in Hz.
        high_cut: Upper band-pass cutoff in Hz.
        filter_order: Butterworth filter order.
        time_window: Optional (start_seconds, end_seconds). The dataset
            documentation does not provide cue timing, so this is an
            experimental choice rather than a known task interval.
        notch_hz: Optional notch center frequency in Hz. None disables it.
        notch_quality_factor: Positive notch quality factor.
        spatial_reference: None or "car".
        normalize_mode: None or "zscore_per_trial_channel".

    Returns:
        Processed floating-point EEG data. Shape is unchanged when
        time_window is None; otherwise the time dimension is cropped.
    """
    demeaned = remove_channel_mean(X)
    if spatial_reference is None:
        rereferenced = demeaned
    elif spatial_reference == "car":
        rereferenced = common_average_reference(demeaned)
    else:
        raise ValueError("spatial_reference must be None or 'car'")

    if notch_hz is None:
        denoised = rereferenced
    else:
        denoised = notch_filter(
            rereferenced,
            sampling_rate=sampling_rate,
            notch_hz=notch_hz,
            quality_factor=notch_quality_factor,
        )

    filtered = bandpass_filter(
        denoised,
        sampling_rate=sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
        filter_order=filter_order,
    )

    if time_window is None:
        processed = filtered
    else:
        if len(time_window) != 2:
            raise ValueError(
                "time_window must be None or (start_seconds, end_seconds)"
            )
        processed = crop_time_window(
            filtered,
            start_seconds=time_window[0],
            end_seconds=time_window[1],
            sampling_rate=sampling_rate,
        )

    if normalize_mode is None:
        normalized = processed
    elif normalize_mode == "zscore_per_trial_channel":
        normalized = zscore_per_trial_channel(processed)
    else:
        raise ValueError(
            "normalize_mode must be None or 'zscore_per_trial_channel'"
        )

    if not np.isfinite(normalized).all():
        raise ValueError("Preprocessing produced NaN or infinite values")
    return normalized
