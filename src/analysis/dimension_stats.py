from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def dimension_mutual_information(
    embeddings: np.ndarray,
    labels: Sequence[str],
    layer_idx: int,
) -> np.ndarray:
    """Score hidden dimensions for one layer against a discrete label."""
    x = embeddings[:, layer_idx, :]
    y = np.asarray(labels)
    return mutual_info_classif(x, y, discrete_features=False, random_state=42)


def rank_dimensions_by_mi(
    feature_index: pd.DataFrame,
    label_field: str,
    layer_idx: int,
) -> pd.DataFrame:
    filtered = feature_index.dropna(subset=[label_field, "feature_path"]).copy()
    arrays = []
    for feature_path in filtered["feature_path"]:
        with np.load(feature_path) as data:
            arrays.append(data["pooled_hidden_states"])
    embeddings = np.stack(arrays, axis=0)
    scores = dimension_mutual_information(
        embeddings=embeddings,
        labels=filtered[label_field].astype(str).tolist(),
        layer_idx=layer_idx,
    )
    ranking = pd.DataFrame(
        {
            "dimension": np.arange(scores.shape[0], dtype=int),
            "mutual_information": scores,
        }
    )
    return ranking.sort_values("mutual_information", ascending=False).reset_index(drop=True)
