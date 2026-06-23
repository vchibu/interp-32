from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class CentroidClusteringResult:
    layer: int
    centroids: pd.DataFrame
    group_counts: pd.DataFrame
    distance_matrix: pd.DataFrame
    distance_long: pd.DataFrame
    cluster_order: pd.DataFrame


def best_probe_layer(probe_results: pd.DataFrame, metric: str = "macro_f1_mean") -> int:
    required = {"model", "layer", metric}
    missing = required.difference(probe_results.columns)
    if missing:
        raise KeyError(f"Probe results are missing required column(s): {', '.join(sorted(missing))}")

    probes = probe_results[probe_results["model"].eq("linear_probe")].copy()
    if probes.empty:
        raise ValueError("Probe results do not contain linear_probe rows.")
    best = probes.sort_values([metric, "balanced_accuracy_mean", "layer"], ascending=[False, False, True]).iloc[0]
    return int(best["layer"])


def load_layer_embeddings(feature_paths: list[str], layer: int) -> np.ndarray:
    arrays = []
    for feature_path in feature_paths:
        with np.load(feature_path) as data:
            pooled = data["pooled_hidden_states"]
            if layer < 0 or layer >= pooled.shape[0]:
                raise ValueError(f"Requested layer {layer}, but {feature_path} has {pooled.shape[0]} layers.")
            arrays.append(pooled[layer])
    return np.stack(arrays, axis=0)


def compute_group_centroids(
    feature_index: pd.DataFrame,
    label_field: str,
    layer: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {label_field, "feature_path"}
    missing = required.difference(feature_index.columns)
    if missing:
        raise KeyError(f"Feature index is missing required column(s): {', '.join(sorted(missing))}")

    filtered = feature_index.dropna(subset=[label_field, "feature_path"]).copy()
    filtered[label_field] = filtered[label_field].astype(str).str.strip()
    filtered = filtered[filtered[label_field].ne("")]
    embeddings = load_layer_embeddings(filtered["feature_path"].tolist(), layer=layer)
    filtered["_row_idx"] = np.arange(len(filtered), dtype=int)

    centroid_rows = []
    count_rows = []
    for label, group in filtered.groupby(label_field, sort=True):
        indices = group["_row_idx"].to_numpy(dtype=int)
        centroid = embeddings[indices].mean(axis=0)
        centroid_rows.append({"label": label, **{f"dim_{idx}": value for idx, value in enumerate(centroid)}})
        count_row = {
            "label": label,
            "n_examples": int(len(group)),
        }
        if "probe_sex" in group.columns:
            count_row["n_female"] = int((group["probe_sex"] == "female").sum())
            count_row["n_male"] = int((group["probe_sex"] == "male").sum())
        count_rows.append(count_row)

    centroids = pd.DataFrame(centroid_rows)
    group_counts = pd.DataFrame(count_rows).sort_values("n_examples", ascending=False).reset_index(drop=True)
    return centroids, group_counts


def centroid_distance_matrix(centroids: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    labels = centroids["label"].tolist()
    values = centroids.drop(columns=["label"]).to_numpy(dtype=float)
    distances = pairwise_distances(values, metric=metric)
    return pd.DataFrame(distances, index=labels, columns=labels)


def distance_matrix_to_long(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = list(distance_matrix.index)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            rows.append(
                {
                    "label_a": left,
                    "label_b": right,
                    "distance": float(distance_matrix.loc[left, right]),
                }
            )
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def _linkage_from_distance_matrix(distance_matrix: pd.DataFrame, method: str) -> np.ndarray:
    condensed = squareform(distance_matrix.to_numpy(dtype=float), checks=False)
    return linkage(condensed, method=method)


def cluster_order(distance_matrix: pd.DataFrame, method: str = "average") -> pd.DataFrame:
    labels = list(distance_matrix.index)
    linkage_matrix = _linkage_from_distance_matrix(distance_matrix, method=method)
    order = leaves_list(linkage_matrix)
    return pd.DataFrame(
        {
            "order": np.arange(len(order), dtype=int),
            "label": [labels[idx] for idx in order],
        }
    )


def run_centroid_clustering(
    feature_index: pd.DataFrame,
    label_field: str,
    layer: int,
    distance_metric: str = "cosine",
    linkage_method: str = "average",
) -> CentroidClusteringResult:
    centroids, group_counts = compute_group_centroids(
        feature_index=feature_index,
        label_field=label_field,
        layer=layer,
    )
    distance_matrix = centroid_distance_matrix(centroids=centroids, metric=distance_metric)
    return CentroidClusteringResult(
        layer=layer,
        centroids=centroids,
        group_counts=group_counts,
        distance_matrix=distance_matrix,
        distance_long=distance_matrix_to_long(distance_matrix),
        cluster_order=cluster_order(distance_matrix, method=linkage_method),
    )


def plot_distance_heatmap(
    distance_matrix: pd.DataFrame,
    output_path: str | Path,
    linkage_method: str = "average",
    title: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(distance_matrix.index)
    linkage_matrix = _linkage_from_distance_matrix(distance_matrix, method=linkage_method)
    ordered_indices = leaves_list(linkage_matrix)
    ordered = distance_matrix.iloc[ordered_indices, ordered_indices]
    ordered_labels = [labels[idx] for idx in ordered_indices]

    size = max(6.0, 0.55 * len(ordered_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(ordered.to_numpy(dtype=float), cmap="viridis_r")
    ax.set_xticks(np.arange(len(ordered_labels)))
    ax.set_yticks(np.arange(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_title(title or "Centroid cosine distance")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dendrogram(
    distance_matrix: pd.DataFrame,
    output_path: str | Path,
    linkage_method: str = "average",
    title: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    linkage_matrix = _linkage_from_distance_matrix(distance_matrix, method=linkage_method)
    labels = list(distance_matrix.index)
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.35 * len(labels) + 2.5)))
    dendrogram(linkage_matrix, labels=labels, orientation="right", leaf_font_size=8, ax=ax)
    ax.set_xlabel("Cosine distance")
    ax.set_title(title or "Accent-country centroid clustering")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_centroid_pca(
    centroids: pd.DataFrame,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = centroids["label"].tolist()
    values = centroids.drop(columns=["label"]).to_numpy(dtype=float)
    coords = PCA(n_components=2, random_state=42).fit_transform(values)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=55, color="#1f77b4")
    for label, x_coord, y_coord in zip(labels, coords[:, 0], coords[:, 1], strict=True):
        ax.annotate(label, (x_coord, y_coord), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#dddddd", linewidth=1)
    ax.axvline(0, color="#dddddd", linewidth=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or "Centroid PCA")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
