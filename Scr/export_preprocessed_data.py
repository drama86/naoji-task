"""Export reusable preprocessed EEG trial data to the reorganized Results tree."""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from machine_learning_config import CHANNEL_COUNT, CLASS_NAMES, PREPROCESSING_CONFIG, SAMPLING_RATE
from machine_learning_process import preprocess_subject_file
from machine_learning_reporting import write_json
from results_layout import PREPROCESSED_EXPORTS_ROOT, PROJECT_ROOT, timestamped_name


RECOMMENDED_PREPROCESSING = {
    "spatial_reference": None,
    "notch_hz": 50.0,
    "low_cut_hz": 8.0,
    "high_cut_hz": 30.0,
    "time_window_seconds": (1.5, 5.5),
    "normalize_mode": "zscore_per_trial_channel",
    "drop_flagged_trials": False,
}


def _build_export_config():
    config = deepcopy(PREPROCESSING_CONFIG)
    config.update(RECOMMENDED_PREPROCESSING)
    return config


def _write_dataset_index(file_path, rows):
    fieldnames = [
        "subject_id",
        "source_file",
        "output_file",
        "sample_count",
        "channel_count",
        "time_points",
        "class_labels_present",
    ]
    with Path(file_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_preprocessed_dataset(data_dir, output_dir):
    """Export one `.npz` per subject plus dataset-level metadata files."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    subject_files = sorted(data_dir.glob("sub_*_MI.mat"))
    if len(subject_files) != 6:
        raise ValueError(f"Expected 6 subject files, found {len(subject_files)}")

    export_config = _build_export_config()
    dataset_index_rows = []
    trial_quality_summary = []
    dropped_trials = []

    for subject_file in subject_files:
        subject_id = int(subject_file.stem.split("_")[1])
        (
            X_processed,
            y_processed,
            processed_shape,
            quality_summary,
            dropped_summary,
        ) = preprocess_subject_file(
            subject_file,
            preprocessing_config=export_config,
        )
        output_name = f"sub_{subject_id:02d}_preprocessed.npz"
        np.savez_compressed(
            output_dir / output_name,
            signals=X_processed.astype(np.float32, copy=False),
            labels=y_processed.astype(np.int64, copy=False),
            subject_id=np.int64(subject_id),
        )
        dataset_index_rows.append(
            {
                "subject_id": subject_id,
                "source_file": subject_file.name,
                "output_file": output_name,
                "sample_count": int(y_processed.shape[0]),
                "channel_count": int(processed_shape[0]),
                "time_points": int(processed_shape[1]),
                "class_labels_present": ",".join(
                    str(label) for label in np.unique(y_processed)
                ),
            }
        )
        trial_quality_summary.append(quality_summary)
        dropped_trials.append(dropped_summary)

    write_json(output_dir / "preprocessing_config.json", export_config)
    write_json(output_dir / "trial_quality_summary.json", trial_quality_summary)
    write_json(output_dir / "dropped_trials.json", dropped_trials)
    write_json(output_dir / "class_mapping.json", CLASS_NAMES)
    _write_dataset_index(output_dir / "dataset_index.csv", dataset_index_rows)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "Data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREPROCESSED_EXPORTS_ROOT / timestamped_name("preprocessed_dataset"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = export_preprocessed_dataset(args.data_dir, args.output_dir)
    print(f"预处理后数据已导出：{output_dir}")
    print(
        "固定输出规格："
        f" 6 个被试文件，signals 形状为 (samples, {CHANNEL_COUNT}, 2000)，"
        f"采样率 {SAMPLING_RATE} Hz"
    )


if __name__ == "__main__":
    main()
