from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def summarize_top_accent_groups(
    metadata: pd.DataFrame,
    group_field: str = "native_language",
    english_residence_field: str = "english_residence",
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the largest accent groups and how many have no English residence recorded."""
    required_columns = {group_field, english_residence_field}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    filtered = metadata.copy()
    filtered[group_field] = _normalize_text_series(filtered[group_field])
    filtered[english_residence_field] = _normalize_text_series(filtered[english_residence_field])
    filtered = filtered[filtered[group_field] != ""].copy()

    no_english_residence = filtered[english_residence_field].isin({"", "none", "nan", "null", "n/a"})
    filtered["no_english_residence"] = no_english_residence

    summary = (
        filtered.groupby(group_field, dropna=False)
        .agg(
            total_speakers=("speaker_id", "nunique"),
            no_english_residence_speakers=("no_english_residence", "sum"),
        )
        .reset_index()
        .rename(columns={group_field: "accent_group"})
        .sort_values(["total_speakers", "no_english_residence_speakers", "accent_group"], ascending=[False, False, True])
        .head(top_n)
        .reset_index(drop=True)
    )

    summary["no_english_residence_speakers"] = summary["no_english_residence_speakers"].astype(int)
    return summary


def save_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
