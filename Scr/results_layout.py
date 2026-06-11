"""Centralized Results/ directory layout for the first-group workflow."""

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results"

PREPROCESSED_RESULTS_ROOT = RESULTS_ROOT / "01_preprocessed_data"
PREPROCESSED_EXPORTS_ROOT = PREPROCESSED_RESULTS_ROOT / "dataset_exports"
PREPROCESSED_DIAGNOSTICS_ROOT = PREPROCESSED_RESULTS_ROOT / "diagnostics"
PREPROCESSED_TUNING_ROOT = PREPROCESSED_RESULTS_ROOT / "tuning"

FEATURE_RESULTS_ROOT = RESULTS_ROOT / "02_feature_results"
FEATURE_EXPORTS_ROOT = FEATURE_RESULTS_ROOT / "basic_feature_exports"
FEATURE_COMPARISONS_ROOT = FEATURE_RESULTS_ROOT / "comparisons"

CLASSIFICATION_RESULTS_ROOT = RESULTS_ROOT / "03_ml_classification"
COURSE_REPORTS_ROOT = CLASSIFICATION_RESULTS_ROOT / "course_reports"
CLASSIFIER_TUNING_ROOT = CLASSIFICATION_RESULTS_ROOT / "classifier_tuning"

ARCHIVE_ROOT = RESULTS_ROOT / "archive"
LEGACY_ARCHIVE_ROOT = ARCHIVE_ROOT / "legacy_layout_20260611"


def timestamped_name(prefix):
    """Return `<prefix>_YYYYMMDD_HHMMSS`."""
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def ensure_directory(path):
    """Create one directory and return it as `Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_results_roots():
    """Create the primary result roots required by the reorganized layout."""
    for path in (
        PREPROCESSED_EXPORTS_ROOT,
        PREPROCESSED_DIAGNOSTICS_ROOT,
        PREPROCESSED_TUNING_ROOT,
        FEATURE_EXPORTS_ROOT,
        FEATURE_COMPARISONS_ROOT,
        CLASSIFICATION_RESULTS_ROOT,
        COURSE_REPORTS_ROOT,
        CLASSIFIER_TUNING_ROOT,
        ARCHIVE_ROOT,
    ):
        ensure_directory(path)

