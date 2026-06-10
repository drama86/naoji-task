from pathlib import Path

import numpy as np
import scipy.io as sio


def load_eeg_mat(file_path, channels=30):
    """Load one MI EEG .mat file and keep the first `channels` EEG channels.

    The expected data shape is:
        classes x channels x time_points x trials
        6x44x5000x20 其中6代表6个类别 分别为被试执行左右手运动想象、左右腕运动想象、左右臂运动想象；
        44代表通道数【提取1~30通道分析】 5000代表样本点 采样率500 20代表20个trail
    """
    file_path = Path(file_path)
    mat_data = sio.loadmat(file_path)

    if "eeg" not in mat_data:
        raise KeyError(f"No variable named 'eeg' found in {file_path}")

    eeg = mat_data["eeg"]

    if eeg.ndim != 4:
        raise ValueError(f"Expected eeg to be 4D, got shape {eeg.shape}")

    if eeg.shape[1] < channels:
        raise ValueError(
            f"Expected at least {channels} channels, got {eeg.shape[1]}"
        )

    return eeg[:, :channels, :, :]


def eeg_to_samples(eeg):
    """Convert EEG data from 4D format to sample-label format.

    Input shape:
        classes x channels x time_points x trials

    Output:
        X shape: samples x channels x time_points
        y shape: samples
    """
    if eeg.ndim != 4:
        raise ValueError(f"Expected eeg to be 4D, got shape {eeg.shape}")

    class_count, channel_count, time_points, trial_count = eeg.shape

    X = eeg.transpose(0, 3, 1, 2)
    X = X.reshape(class_count * trial_count, channel_count, time_points)

    y = np.repeat(np.arange(class_count), trial_count)

    return X, y


def load_subject_data(file_path, channels=30):
    """Load one subject file and return X, y."""
    eeg = load_eeg_mat(file_path, channels=channels)
    return eeg_to_samples(eeg)
