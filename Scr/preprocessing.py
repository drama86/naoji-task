import numpy as np
from scipy.signal import butter, sosfiltfilt


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


def preprocess_eeg(
    X,
    sampling_rate,
    low_cut,
    high_cut,
    filter_order,
    time_window,
):
    """Run the reusable EEG preprocessing pipeline.

    Processing order:
        per-trial channel demeaning -> zero-phase band-pass -> optional crop.

    Args:
        X: EEG data with shape (samples, channels, time_points).
        sampling_rate: Sampling frequency in Hz.
        low_cut: Lower band-pass cutoff in Hz.
        high_cut: Upper band-pass cutoff in Hz.
        filter_order: Butterworth filter order.
        time_window: Optional (start_seconds, end_seconds). The dataset
            documentation does not provide cue timing, so this is an
            experimental choice rather than a known task interval.

    Returns:
        Processed floating-point EEG data. Shape is unchanged when
        time_window is None; otherwise the time dimension is cropped.
    """
    demeaned = remove_channel_mean(X)
    filtered = bandpass_filter(
        demeaned,
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

    if not np.isfinite(processed).all():
        raise ValueError("Preprocessing produced NaN or infinite values")
    return processed
