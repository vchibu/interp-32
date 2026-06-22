from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def probe_by_layer(
    embeddings: np.ndarray,
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict[str, float]]:
    """Train one linear probe per layer on pooled utterance embeddings.

    embeddings shape: (n_samples, n_layers, hidden_size)
    """
    y = np.asarray(labels)
    results: list[dict[str, float]] = []

    for layer_idx in range(embeddings.shape[1]):
        layer_x = embeddings[:, layer_idx, :]
        x_train, x_test, y_train, y_test = train_test_split(
            layer_x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        clf = LogisticRegression(max_iter=2000)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        results.append(
            {
                "layer": float(layer_idx),
                "accuracy": float(accuracy_score(y_test, preds)),
            }
        )

    return results


def load_pooled_embeddings(feature_paths: Sequence[str]) -> np.ndarray:
    arrays = []
    for feature_path in feature_paths:
        with np.load(feature_path) as data:
            arrays.append(data["pooled_hidden_states"])
    return np.stack(arrays, axis=0)


def run_probe_analysis(
    feature_index: pd.DataFrame,
    label_field: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    filtered = feature_index.dropna(subset=[label_field, "feature_path"]).copy()
    embeddings = load_pooled_embeddings(filtered["feature_path"].tolist())
    results = probe_by_layer(
        embeddings=embeddings,
        labels=filtered[label_field].astype(str).tolist(),
        test_size=test_size,
        random_state=random_state,
    )
    return pd.DataFrame(results)
