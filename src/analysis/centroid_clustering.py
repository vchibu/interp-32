from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


TITLECASE_EXCEPTIONS = {
    "uk": "UK",
    "usa": "USA",
}

PLOT_LABEL_FONTSIZE = 14
PLOT_TITLE_FONTSIZE = 17
PLOT_AXIS_FONTSIZE = 15
PLOT_TICK_FONTSIZE = 14
HEATMAP_LABEL_FONTSIZE = 30
HEATMAP_TITLE_FONTSIZE = 36
HEATMAP_TICK_FONTSIZE = 30


@dataclass(frozen=True)
class CentroidClusteringResult:
    layer: int
    centroids: pd.DataFrame
    group_counts: pd.DataFrame
    distance_matrix: pd.DataFrame
    distance_long: pd.DataFrame
    cluster_order: pd.DataFrame


def _format_label_part(text: str) -> str:
    cleaned = text.replace("_", " ").strip().lower()
    if cleaned in TITLECASE_EXCEPTIONS:
        return TITLECASE_EXCEPTIONS[cleaned]
    return " ".join(TITLECASE_EXCEPTIONS.get(part, part.capitalize()) for part in cleaned.split())


def format_accent_country_label(label: str) -> str:
    if "__" not in label:
        return _format_label_part(label)
    accent, country = label.split("__", maxsplit=1)
    return f"{_format_label_part(accent)} ({_format_label_part(country)})"


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
    ordered_labels = [format_accent_country_label(labels[idx]) for idx in ordered_indices]

    size = max(19.0, 1.35 * len(ordered_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        ordered.to_numpy(dtype=float),
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=float(np.nanmax(ordered.to_numpy(dtype=float))),
    )
    ax.set_xticks(np.arange(len(ordered_labels)))
    ax.set_yticks(np.arange(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_yticklabels(ordered_labels, fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_xticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title or "Centroid cosine distance", fontsize=HEATMAP_TITLE_FONTSIZE)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine distance: blue = closer, red = farther", fontsize=HEATMAP_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_TICK_FONTSIZE)
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
    labels = [format_accent_country_label(label) for label in distance_matrix.index]
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.42 * len(labels) + 2.5)))
    dendrogram(linkage_matrix, labels=labels, orientation="right", leaf_font_size=10, ax=ax)
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

    labels = [format_accent_country_label(label) for label in centroids["label"].tolist()]
    values = centroids.drop(columns=["label"]).to_numpy(dtype=float)
    coords = PCA(n_components=2, random_state=42).fit_transform(values)

    fig, ax = plt.subplots(figsize=(13.5, 9.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=70, color="#1f77b4", zorder=3)
    _set_padded_limits(ax, coords)
    ax.axhline(0, color="#dddddd", linewidth=1)
    ax.axvline(0, color="#dddddd", linewidth=1)
    _place_pca_labels(ax, labels, coords)
    ax.set_xlabel("PC1", fontsize=PLOT_AXIS_FONTSIZE)
    ax.set_ylabel("PC2", fontsize=PLOT_AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.set_title(title or "Centroid PCA", fontsize=PLOT_TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _set_padded_limits(ax: plt.Axes, coords: np.ndarray) -> None:
    x_span = max(float(np.ptp(coords[:, 0])), 1e-6)
    y_span = max(float(np.ptp(coords[:, 1])), 1e-6)
    ax.set_xlim(float(np.min(coords[:, 0])) - 0.24 * x_span, float(np.max(coords[:, 0])) + 0.28 * x_span)
    ax.set_ylim(float(np.min(coords[:, 1])) - 0.20 * y_span, float(np.max(coords[:, 1])) + 0.20 * y_span)


def _place_pca_labels(ax: plt.Axes, labels: list[str], coords: np.ndarray) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    point_boxes = _point_bboxes(ax, coords)
    placed_boxes: list[Bbox] = []
    center = coords.mean(axis=0)
    nearest_neighbor = pairwise_distances(coords)
    np.fill_diagonal(nearest_neighbor, np.inf)
    placement_order = np.argsort(nearest_neighbor.min(axis=1))

    for idx in placement_order:
        label = labels[idx]
        point = coords[idx]
        candidates = _label_offset_candidates(point, center)
        best_annotation = None
        best_box = None
        best_score = np.inf

        for offset in candidates:
            annotation = _make_pca_annotation(ax, label, point, offset)
            fig.canvas.draw()
            box = annotation.get_window_extent(renderer).expanded(1.08, 1.18)
            score = _label_box_score(box, point_boxes, placed_boxes, ax.get_window_extent(renderer))
            if score < best_score:
                if best_annotation is not None:
                    best_annotation.remove()
                best_annotation = annotation
                best_box = box
                best_score = score
            else:
                annotation.remove()
            if score == 0:
                break

        if best_box is not None:
            placed_boxes.append(best_box)


def _label_offset_candidates(point: np.ndarray, center: np.ndarray) -> list[tuple[int, int]]:
    direction = point - center
    if np.linalg.norm(direction) == 0:
        direction = np.array([1.0, 0.0])
    x_sign = 1 if direction[0] >= 0 else -1
    y_sign = 1 if direction[1] >= 0 else -1
    outward = [
        (x_sign * 96, y_sign * 44),
        (x_sign * 120, 0),
        (x_sign * 96, -y_sign * 44),
        (0, y_sign * 58),
        (x_sign * 150, y_sign * 64),
        (x_sign * 150, -y_sign * 64),
        (-x_sign * 96, y_sign * 44),
        (-x_sign * 120, 0),
        (-x_sign * 96, -y_sign * 44),
        (0, -y_sign * 58),
    ]
    wider_search = [
        (170, 82),
        (170, -82),
        (-170, 82),
        (-170, -82),
        (0, 92),
        (0, -92),
        (210, 0),
        (-210, 0),
    ]
    return outward + wider_search


def _make_pca_annotation(
    ax: plt.Axes,
    label: str,
    point: np.ndarray,
    offset: tuple[int, int],
) -> plt.Annotation:
    return ax.annotate(
        label,
        xy=(point[0], point[1]),
        xytext=offset,
        textcoords="offset points",
        fontsize=PLOT_LABEL_FONTSIZE,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7, "shrinkA": 4, "shrinkB": 6},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
        zorder=4,
    )


def _point_bboxes(ax: plt.Axes, coords: np.ndarray, radius_px: float = 10.0) -> list[Bbox]:
    display_points = ax.transData.transform(coords)
    return [
        Bbox.from_extents(x - radius_px, y - radius_px, x + radius_px, y + radius_px)
        for x, y in display_points
    ]


def _label_box_score(box: Bbox, point_boxes: list[Bbox], placed_boxes: list[Bbox], axes_box: Bbox) -> float:
    point_overlap = sum(_bbox_overlap_area(box, point_box) for point_box in point_boxes)
    label_overlap = sum(_bbox_overlap_area(box, placed_box) for placed_box in placed_boxes)
    outside = (
        max(0.0, axes_box.x0 - box.x0)
        + max(0.0, box.x1 - axes_box.x1)
        + max(0.0, axes_box.y0 - box.y0)
        + max(0.0, box.y1 - axes_box.y1)
    )
    return 1000.0 * point_overlap + 250.0 * label_overlap + 25.0 * outside


def _bbox_overlap_area(left: Bbox, right: Bbox) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height
