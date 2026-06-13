"""Cross-validation and metrics for classical EEG classifiers."""

from collections import defaultdict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from classifiers import create_model_pipeline
from machine_learning_config import RANDOM_SEED


SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def _validate_feature_arrays(features, y, subject_ids):
    """Validate aligned sample-first arrays for model evaluation."""
    features = np.asarray(features)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    if features.ndim < 2:
        raise ValueError(
            "Expected sample-first input with shape "
            f"(samples, ...), got {features.shape}"
        )
    if y.ndim != 1 or subject_ids.ndim != 1:
        raise ValueError("y and subject_ids must be one-dimensional")
    if not (features.shape[0] == y.size == subject_ids.size):
        raise ValueError("features, y, and subject_ids are not aligned")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")
    return features, y.astype(np.int64), subject_ids.astype(np.int64)


def _classification_metrics(y_true, y_pred, labels):
    """Calculate overall and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    per_class = {
        int(label): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return overall, per_class


def _subject_level_results(y_true, y_pred, subject_ids, labels):
    """Summarize out-of-fold predictions for each subject separately."""
    subject_records = []
    for subject_id in np.unique(subject_ids):
        subject_mask = subject_ids == subject_id
        subject_y_true = y_true[subject_mask]
        subject_y_pred = y_pred[subject_mask]
        overall, per_class = _classification_metrics(
            subject_y_true,
            subject_y_pred,
            labels,
        )
        matrix = confusion_matrix(
            subject_y_true,
            subject_y_pred,
            labels=labels,
        )
        normalized_matrix = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix, dtype=np.float64),
            where=matrix.sum(axis=1, keepdims=True) > 0,
        )
        subject_records.append(
            {
                "subject_id": int(subject_id),
                "sample_count": int(subject_mask.sum()),
                **overall,
                "per_class_metrics": per_class,
                "confusion_matrix": matrix.astype(int).tolist(),
                "normalized_confusion_matrix": normalized_matrix.tolist(),
            }
        )
    return subject_records


def build_validation_splits(
    y,
    subject_ids,
    strategy,
    n_splits=5,
    random_seed=RANDOM_SEED,
):
    """Build LOSO or per-subject stratified cross-validation splits."""
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    if strategy == "loso":
        splitter = LeaveOneGroupOut()
        return [
            {
                "fold": fold_index,
                "held_out_subject": int(np.unique(subject_ids[test])[0]),
                "train_indices": train,
                "test_indices": test,
            }
            for fold_index, (train, test) in enumerate(
                splitter.split(np.zeros(y.size), y, groups=subject_ids),
                start=1,
            )
        ]
    if strategy != "within_subject_5fold":
        raise KeyError(f"Unknown validation strategy: {strategy}")

    splits = []
    fold_index = 1
    for subject_id in np.unique(subject_ids):
        subject_indices = np.flatnonzero(subject_ids == subject_id)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )
        for subject_fold, (local_train, local_test) in enumerate(
            splitter.split(subject_indices, y[subject_indices]),
            start=1,
        ):
            splits.append(
                {
                    "fold": fold_index,
                    "subject_id": int(subject_id),
                    "subject_fold": subject_fold,
                    "train_indices": subject_indices[local_train],
                    "test_indices": subject_indices[local_test],
                }
            )
            fold_index += 1
    return splits


def evaluate_classifier(
    features,
    y,
    subject_ids,
    classifier_name,
    strategy,
    feature_config,
    sampling_rate,
    random_seed=RANDOM_SEED,
    n_splits=5,
    classifier_overrides=None,
):
    """Evaluate one classifier without sharing fitted state across folds."""
    features, y, subject_ids = _validate_feature_arrays(
        features,
        y,
        subject_ids,
    )
    labels = np.unique(y)
    base_estimator = create_model_pipeline(
        classifier_name,
        feature_config,
        sampling_rate=sampling_rate,
        random_seed=random_seed,
        classifier_overrides=classifier_overrides,
    )
    splits = build_validation_splits(
        y,
        subject_ids,
        strategy,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    predictions = np.full(y.shape, -1, dtype=np.int64)
    prediction_counts = np.zeros(y.shape, dtype=np.int64)
    fold_records = []

    for split in splits:
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]
        estimator = clone(base_estimator)
        estimator.fit(features[train_indices], y[train_indices])
        fold_predictions = estimator.predict(features[test_indices])
        predictions[test_indices] = fold_predictions
        prediction_counts[test_indices] += 1
        overall, per_class = _classification_metrics(
            y[test_indices],
            fold_predictions,
            labels,
        )
        fold_record = {
            key: value
            for key, value in split.items()
            if key not in {"train_indices", "test_indices"}
        }
        fold_record.update(
            {
                "classifier": classifier_name,
                "train_sample_count": int(train_indices.size),
                "test_sample_count": int(test_indices.size),
                **overall,
                "per_class": per_class,
            }
        )
        fold_records.append(fold_record)

    if not np.all(prediction_counts == 1):
        raise RuntimeError(
            "Each sample must receive exactly one out-of-fold prediction"
        )
    overall, per_class = _classification_metrics(y, predictions, labels)
    matrix = confusion_matrix(y, predictions, labels=labels)
    normalized_matrix = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    metric_values = defaultdict(list)
    for record in fold_records:
        for metric in SUMMARY_METRICS:
            metric_values[metric].append(record[metric])
    fold_summary = {
        f"{metric}_{statistic}": float(function(metric_values[metric]))
        for metric in SUMMARY_METRICS
        for statistic, function in (("mean", np.mean), ("std", np.std))
    }
    subject_records = _subject_level_results(
        y,
        predictions,
        subject_ids,
        labels,
    )
    return {
        "classifier": classifier_name,
        "validation_strategy": strategy,
        "fold_count": len(fold_records),
        "labels": labels.astype(int).tolist(),
        "fold_records": fold_records,
        "fold_summary": fold_summary,
        "out_of_fold_metrics": overall,
        "per_class_metrics": per_class,
        "subject_records": subject_records,
        "confusion_matrix": matrix.astype(int).tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
        "predictions": predictions.astype(int).tolist(),
    }


def evaluate_classifiers(
    features,
    y,
    subject_ids,
    classifier_names,
    strategy,
    feature_config,
    sampling_rate,
    random_seed=RANDOM_SEED,
    n_splits=5,
    classifier_overrides_map=None,
):
    """Evaluate several classifiers with identical validation splits."""
    return {
        classifier_name: evaluate_classifier(
            features,
            y,
            subject_ids,
            classifier_name,
            strategy,
            feature_config,
            sampling_rate,
            random_seed=random_seed,
            n_splits=n_splits,
            classifier_overrides=(
                None
                if classifier_overrides_map is None
                else classifier_overrides_map.get(classifier_name)
            ),
        )
        for classifier_name in classifier_names
    }
