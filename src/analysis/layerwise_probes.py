from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeAnalysisResult:
    layer_summary: pd.DataFrame
    split_results: pd.DataFrame
    class_counts: pd.DataFrame
    best_layer_confusion_matrix: pd.DataFrame
    filtered_feature_index: pd.DataFrame
    best_layer: int


def _clean_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype(str).str.strip()
    return cleaned.mask(cleaned.eq("") | cleaned.str.lower().eq("nan"))


def filter_feature_index_for_probe(
    feature_index: pd.DataFrame,
    label_field: str,
    min_examples_per_label: int = 20,
    top_n_labels: int | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep examples with usable features and labels large enough for probing."""
    required_columns = {label_field, "feature_path"}
    missing_columns = required_columns.difference(feature_index.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Feature index is missing required column(s): {missing}")

    filtered = feature_index.dropna(subset=["feature_path"]).copy()
    filtered[label_field] = _clean_labels(filtered[label_field])
    filtered = filtered.dropna(subset=[label_field])

    if exclude_labels:
        excluded = {label.strip() for label in exclude_labels}
        filtered = filtered[~filtered[label_field].isin(excluded)].copy()

    all_counts = filtered[label_field].value_counts().rename_axis(label_field).reset_index(name="n_examples")
    all_counts["included"] = all_counts["n_examples"] >= min_examples_per_label

    included_labels = all_counts.loc[all_counts["included"], label_field].tolist()
    if top_n_labels is not None:
        included_labels = included_labels[:top_n_labels]
        all_counts["included"] = all_counts[label_field].isin(included_labels)

    filtered = filtered[filtered[label_field].isin(included_labels)].copy()
    filtered = filtered.reset_index(drop=True)
    class_counts = all_counts.sort_values(["included", "n_examples", label_field], ascending=[False, False, True])
    class_counts = class_counts.reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            "No probe examples remain after filtering. "
            "Lower --min-examples-per-label or inspect the requested label field."
        )
    if filtered[label_field].nunique() < 2:
        raise ValueError("At least two labels are required to train a probe.")

    smallest_class = int(filtered[label_field].value_counts().min())
    if smallest_class < 2:
        raise ValueError("Each included label needs at least two examples for stratified splitting.")

    return filtered, class_counts


def load_pooled_embeddings(feature_paths: Sequence[str]) -> np.ndarray:
    arrays = []
    for feature_path in feature_paths:
        with np.load(feature_path) as data:
            arrays.append(data["pooled_hidden_states"])
    return np.stack(arrays, axis=0)


def load_pooled_cnn_features(feature_paths: Sequence[str]) -> np.ndarray | None:
    """Load the pooled raw CNN feature-encoder output for each recording.

    Returns None if any cache predates the CNN baseline, so callers can simply
    skip the baseline rather than crash on legacy feature stores.
    """
    arrays = []
    for feature_path in feature_paths:
        with np.load(feature_path) as data:
            if "pooled_cnn_features" not in data.files:
                return None
            arrays.append(data["pooled_cnn_features"])
    return np.stack(arrays, axis=0)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _make_probe(class_weight: str | None, max_iter: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
        ),
    )


CNN_BASELINE_LAYER = -2


def probe_by_layer(
    embeddings: np.ndarray,
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    class_weight: str | None = "balanced",
    max_iter: int = 2000,
    cnn_features: np.ndarray | None = None,
) -> pd.DataFrame:
    """Train repeated linear probes for every layer on pooled utterance embeddings.

    embeddings shape: (n_samples, n_layers, hidden_size)

    When ``cnn_features`` (n_samples, cnn_dim) is given, an extra probe is trained
    on the raw CNN feature-encoder output as an input baseline. It is recorded
    with model ``"cnn_baseline"`` at layer ``CNN_BASELINE_LAYER`` so it stays
    separate from the per-transformer-layer ``"linear_probe"`` curve.
    """
    y = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, float | int | str]] = []
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        y_train = y[train_idx]
        y_test = y[test_idx]

        baselines = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "stratified": DummyClassifier(strategy="stratified", random_state=random_state + split_idx),
        }
        for baseline_name, baseline in baselines.items():
            baseline.fit(np.zeros((len(train_idx), 1)), y_train)
            baseline_preds = baseline.predict(np.zeros((len(test_idx), 1)))
            row = {
                "layer": -1,
                "split": split_idx,
                "model": baseline_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            row.update(_metric_dict(y_test, baseline_preds))
            results.append(row)

        if cnn_features is not None:
            clf = _make_probe(class_weight=class_weight, max_iter=max_iter)
            clf.fit(cnn_features[train_idx], y_train)
            cnn_preds = clf.predict(cnn_features[test_idx])
            row = {
                "layer": CNN_BASELINE_LAYER,
                "split": split_idx,
                "model": "cnn_baseline",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            row.update(_metric_dict(y_test, cnn_preds))
            results.append(row)

        for layer_idx in range(embeddings.shape[1]):
            layer_x = embeddings[:, layer_idx, :]
            clf = _make_probe(class_weight=class_weight, max_iter=max_iter)
            clf.fit(layer_x[train_idx], y_train)
            preds = clf.predict(layer_x[test_idx])
            row = {
                "layer": int(layer_idx),
                "split": split_idx,
                "model": "linear_probe",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            row.update(_metric_dict(y_test, preds))
            results.append(row)

    return pd.DataFrame(results)


def summarize_probe_splits(split_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    grouped = split_results.groupby(["model", "layer"], dropna=False)
    summary = grouped[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["n_splits"] = grouped.size().to_numpy(dtype=int)
    summary = summary.sort_values(
        ["model", "macro_f1_mean", "balanced_accuracy_mean", "layer"],
        ascending=[True, False, False, True],
    )
    return summary.reset_index(drop=True)


def best_layer_confusion_matrix(
    embeddings: np.ndarray,
    labels: Sequence[str],
    layer_idx: int,
    test_size: float,
    random_state: int,
    n_splits: int,
    class_weight: str | None,
    max_iter: int,
) -> pd.DataFrame:
    y = np.asarray(labels)
    label_order = pd.Series(y).value_counts().sort_values(ascending=False).index.to_list()
    matrix = np.zeros((len(label_order), len(label_order)), dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    layer_x = embeddings[:, layer_idx, :]
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        clf = _make_probe(class_weight=class_weight, max_iter=max_iter)
        clf.fit(layer_x[train_idx], y[train_idx])
        preds = clf.predict(layer_x[test_idx])
        matrix += confusion_matrix(y[test_idx], preds, labels=label_order)

    confusion = pd.DataFrame(matrix, index=label_order, columns=label_order)
    confusion.index.name = "true_label"
    return confusion.reset_index()


def run_probe_analysis(
    feature_index: pd.DataFrame,
    label_field: str,
    min_examples_per_label: int = 20,
    top_n_labels: int | None = None,
    exclude_labels: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    class_weight: str | None = "balanced",
    max_iter: int = 2000,
) -> ProbeAnalysisResult:
    filtered, class_counts = filter_feature_index_for_probe(
        feature_index=feature_index,
        label_field=label_field,
        min_examples_per_label=min_examples_per_label,
        top_n_labels=top_n_labels,
        exclude_labels=exclude_labels,
    )
    embeddings = load_pooled_embeddings(filtered["feature_path"].tolist())
    cnn_features = load_pooled_cnn_features(filtered["feature_path"].tolist())
    labels = filtered[label_field].astype(str).tolist()
    split_results = probe_by_layer(
        embeddings=embeddings,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
        n_splits=n_splits,
        class_weight=class_weight,
        max_iter=max_iter,
        cnn_features=cnn_features,
    )
    layer_summary = summarize_probe_splits(split_results)
    probe_summary = layer_summary[layer_summary["model"].eq("linear_probe")].copy()
    best_layer = int(
        probe_summary.sort_values(
            ["macro_f1_mean", "balanced_accuracy_mean", "layer"],
            ascending=[False, False, True],
        ).iloc[0]["layer"]
    )
    confusion = best_layer_confusion_matrix(
        embeddings=embeddings,
        labels=labels,
        layer_idx=best_layer,
        test_size=test_size,
        random_state=random_state,
        n_splits=n_splits,
        class_weight=class_weight,
        max_iter=max_iter,
    )
    return ProbeAnalysisResult(
        layer_summary=layer_summary,
        split_results=split_results,
        class_counts=class_counts,
        best_layer_confusion_matrix=confusion,
        filtered_feature_index=filtered,
        best_layer=best_layer,
    )
