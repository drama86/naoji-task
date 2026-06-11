"""Export the global basic-feature matrix for reusable downstream analysis."""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from feature_extraction import extract_basic_features
from machine_learning_config import FEATURE_CONFIG, PREPROCESSING_CONFIG, SAMPLING_RATE
from machine_learning_process import load_preprocessed_data
from machine_learning_reporting import write_json
from results_layout import FEATURE_EXPORTS_ROOT, PROJECT_ROOT, timestamped_name


RECOMMENDED_PREPROCESSING = {
    "spatial_reference": None,
    "notch_hz": 50.0,
    "low_cut_hz": 8.0,
    "high_cut_hz": 30.0,
    "time_window_seconds": (1.5, 5.5),
    "normalize_mode": "zscore_per_trial_channel",
    "drop_flagged_trials": False,
}

RECOMMENDED_FEATURE_CONFIG = {
    "feature_sets": ("frequency", "time"),
}


def _build_preprocessing_config():
    config = deepcopy(PREPROCESSING_CONFIG)
    config.update(RECOMMENDED_PREPROCESSING)
    return config


def _build_basic_feature_config():
    config = deepcopy(FEATURE_CONFIG["basic"])
    config.update(RECOMMENDED_FEATURE_CONFIG)
    return config


def _write_dataset_index(file_path, y, subject_ids):
    fieldnames = ["sample_index", "subject_id", "label"]
    with Path(file_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, (subject_id, label) in enumerate(zip(subject_ids, y)):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "subject_id": int(subject_id),
                    "label": int(label),
                }
            )


def export_basic_features(data_dir, output_dir):
    """Export only the leakage-safe basic feature matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    preprocessing_config = _build_preprocessing_config()
    basic_feature_config = _build_basic_feature_config()
    signals, y, subject_ids, processed_shape, _, _ = load_preprocessed_data(
        Path(data_dir),
        preprocessing_config=preprocessing_config,
    )
    X, feature_names, extraction_config = extract_basic_features(
        signals,
        feature_sets=basic_feature_config["feature_sets"],
        sampling_rate=SAMPLING_RATE,
        frequency_bands=basic_feature_config["frequency_bands"],
        total_power_band=basic_feature_config["total_power_band"],
        nperseg=basic_feature_config["nperseg"],
        noverlap=basic_feature_config["noverlap"],
    )

    np.savez_compressed(
        output_dir / "all_subjects_basic_features.npz",
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        subject_ids=subject_ids.astype(np.int64, copy=False),
    )
    write_json(output_dir / "feature_names.json", feature_names)
    write_json(
        output_dir / "feature_config.json",
        {
            "preprocessing": preprocessing_config,
            "basic_feature_config": basic_feature_config,
            "extraction_summary": extraction_config,
            "processed_signal_shape": processed_shape,
            "export_note": (
                "Only basic features are exported globally. CSP/FBCSP/MIBIF "
                "remain comparison-only because they must be fit inside each "
                "training fold."
            ),
        },
    )
    _write_dataset_index(output_dir / "dataset_index.csv", y, subject_ids)
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
        default=FEATURE_EXPORTS_ROOT / timestamped_name("basic_features"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = export_basic_features(args.data_dir, args.output_dir)
    print(f"基础特征矩阵已导出：{output_dir}")


if __name__ == "__main__":
    main()
