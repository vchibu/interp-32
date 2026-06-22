from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def _normalize_country_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_text_series(series)
    cleaned = cleaned.str.replace(r"\s*\(map\)\s*$", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def _extract_country_from_birth_place_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_text_series(series)
    cleaned = cleaned.str.replace(r"\s*\(map\)\s*$", "", regex=True)
    cleaned = cleaned.str.split(",").str[-1].fillna("")
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def summarize_accent_groups_by_country(
    metadata: pd.DataFrame,
    group_field: str = "native_language",
    country_field: str = "country",
    birth_place_field: str = "birth_place",
    allowed_groups: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Summarize each accent group by country of origin using existing metadata data."""
    required_columns = {group_field}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required manifest columns: {sorted(missing)}")

    filtered = metadata.copy()
    filtered[group_field] = _normalize_text_series(filtered[group_field])
    if country_field in filtered.columns:
        filtered[country_field] = _normalize_country_series(filtered[country_field])
    elif birth_place_field in filtered.columns:
        filtered[country_field] = _extract_country_from_birth_place_series(filtered[birth_place_field])
    else:
        raise KeyError(f"Missing required metadata columns: ['{country_field}' or '{birth_place_field}']")
    filtered = filtered[(filtered[group_field] != "") & (filtered[country_field] != "")].copy()

    if allowed_groups is not None:
        allowed_normalized = [_normalize_text_series(pd.Series([group])).iat[0] for group in allowed_groups]
        filtered = filtered[filtered[group_field].isin(allowed_normalized)].copy()

    summary = (
        filtered.groupby([group_field, country_field], dropna=False)
        .agg(total_speakers=("speaker_id", "nunique"))
        .reset_index()
        .rename(columns={group_field: "accent_group", country_field: "country_of_origin"})
    )

    group_totals = summary.groupby("accent_group", dropna=False)["total_speakers"].transform("sum")
    summary["share_within_accent_group"] = summary["total_speakers"] / group_totals

    summary = summary.sort_values(
        ["total_speakers", "accent_group", "country_of_origin"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return summary


def save_breakdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path