from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


DEFAULT_ENGLISH_SPEAKING_COUNTRIES = {
    "australia",
    "canada",
    "ireland",
    "new zealand",
    "uk",
    "usa",
}


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


def _normalize_sex_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_text_series(series)
    cleaned = cleaned.replace({"m": "male", "f": "female"})
    return cleaned


def _parse_residency_years(series: pd.Series) -> pd.Series:
    cleaned = _normalize_text_series(series)
    years = cleaned.str.extract(r"(?P<years>\d+(?:\.\d+)?)")["years"]
    return pd.to_numeric(years, errors="coerce")


def _resolve_label_conflicts(
    df: pd.DataFrame,
    speaker_id_field: str = "speaker_id",
    group_field: str = "native_language",
) -> pd.DataFrame:
    """When a speaker has multiple labels, keep only the rarest (most specific) label.

    The Speech Accent Archive assigns both umbrella labels (e.g., 'chinese') and
    specific dialect labels (e.g., 'mandarin', 'cantonese') to the same speaker.
    For accent comparison, the specific dialect is more informative. By preferring
    the rarest label, we automatically select the most specific variant and ensure
    each speaker appears exactly once in the dataset.
    """
    if df.empty:
        return df
    label_counts = df[group_field].value_counts()
    df = df.copy()
    df["_label_frequency"] = df[group_field].map(label_counts).fillna(0).astype(int)
    df = df.sort_values([speaker_id_field, "_label_frequency"], ascending=[True, True])
    df = df.drop_duplicates(subset=[speaker_id_field], keep="first")
    df = df.drop(columns=["_label_frequency"])
    return df


def summarize_dataset_makeup(
    metadata: pd.DataFrame,
    group_field: str = "native_language",
    country_field: str = "country",
    birth_place_field: str = "birth_place",
    sex_field: str = "sex",
    residency_field: str = "length_of_english_residence",
    english_speaking_countries: set[str] | frozenset[str] = DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
) -> pd.DataFrame:
    """Summarize the dataset by country, accent group, sex, and residency buckets."""
    required_columns = {"speaker_id", group_field, sex_field, residency_field}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    filtered = metadata.copy()
    filtered[group_field] = _normalize_text_series(filtered[group_field])
    filtered[sex_field] = _normalize_sex_series(filtered[sex_field])
    filtered[residency_field] = _parse_residency_years(filtered[residency_field])

    if country_field in filtered.columns:
        filtered[country_field] = _normalize_country_series(filtered[country_field])
    elif birth_place_field in filtered.columns:
        filtered[country_field] = _extract_country_from_birth_place_series(filtered[birth_place_field])
    else:
        raise KeyError(f"Missing required metadata columns: ['{country_field}' or '{birth_place_field}']")

    filtered = filtered[(filtered[group_field] != "") & (filtered[country_field] != "")].copy()
    filtered = _resolve_label_conflicts(filtered, group_field=group_field)
    filtered["is_english_speaking_country"] = filtered[country_field].isin(english_speaking_countries)
    filtered["is_female"] = filtered[sex_field] == "female"
    filtered["is_male"] = filtered[sex_field] == "male"
    filtered["is_unknown_sex"] = ~(filtered["is_female"] | filtered["is_male"])
    filtered["no_residency_speakers_flag"] = (filtered[residency_field] == 0) & ~filtered["is_english_speaking_country"]
    filtered["one_year_residency_speakers_flag"] = (filtered[residency_field] == 1) & ~filtered["is_english_speaking_country"]
    filtered["two_year_residency_speakers_flag"] = (filtered[residency_field] == 2) & ~filtered["is_english_speaking_country"]
    filtered["three_year_residency_speakers_flag"] = (filtered[residency_field] == 3) & ~filtered["is_english_speaking_country"]

    summary = (
        filtered.groupby([country_field, group_field], dropna=False)
        .agg(
            total_speakers=("speaker_id", "nunique"),
            female_speakers=("is_female", "sum"),
            male_speakers=("is_male", "sum"),
            unknown_sex_speakers=("is_unknown_sex", "sum"),
            is_english_speaking_country=("is_english_speaking_country", "first"),
            no_residency_speakers=("no_residency_speakers_flag", "sum"),
            one_year_residency_speakers=("one_year_residency_speakers_flag", "sum"),
            two_year_residency_speakers=("two_year_residency_speakers_flag", "sum"),
            three_year_residency_speakers=("three_year_residency_speakers_flag", "sum"),
        )
        .reset_index()
        .rename(columns={country_field: "country_of_origin", group_field: "accent_group"})
    )

    summary = summary.sort_values(
        ["total_speakers", "country_of_origin", "accent_group"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    for column in [
        "female_speakers",
        "male_speakers",
        "unknown_sex_speakers",
        "no_residency_speakers",
        "one_year_residency_speakers",
        "two_year_residency_speakers",
        "three_year_residency_speakers",
    ]:
        summary[column] = summary[column].astype(int)

    return summary


def save_dataset_makeup(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def summarize_balanced_pairs_by_residency_cutoff(
    metadata: pd.DataFrame,
    group_field: str = "native_language",
    country_field: str = "country",
    birth_place_field: str = "birth_place",
    sex_field: str = "sex",
    residency_field: str = "length_of_english_residence",
    english_speaking_countries: set[str] | frozenset[str] = DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
    cutoffs: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Return male/female speaker counts and pairable counts per accent group at residency cutoffs.

    For non-English-speaking countries, speakers are retained only if residency years <= cutoff.
    For English-speaking countries, speakers are always retained regardless of residency length.
    """
    required_columns = {"speaker_id", group_field, sex_field, residency_field}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    normalized = metadata.copy()
    normalized[group_field] = _normalize_text_series(normalized[group_field])
    normalized[sex_field] = _normalize_sex_series(normalized[sex_field])
    normalized[residency_field] = _parse_residency_years(normalized[residency_field])

    if country_field in normalized.columns:
        normalized[country_field] = _normalize_country_series(normalized[country_field])
    elif birth_place_field in normalized.columns:
        normalized[country_field] = _extract_country_from_birth_place_series(normalized[birth_place_field])
    else:
        raise KeyError(f"Missing required metadata columns: ['{country_field}' or '{birth_place_field}']")

    normalized = normalized[(normalized[group_field] != "") & (normalized[country_field] != "")].copy()
    normalized = _resolve_label_conflicts(normalized, group_field=group_field)
    normalized["is_english_speaking_country"] = normalized[country_field].isin(english_speaking_countries)

    # Keep only rows with binary sex labels for pair counting.
    normalized = normalized[normalized[sex_field].isin({"female", "male"})].copy()

    base_rows = []
    for accent_group, group_df in normalized.groupby(group_field, dropna=False):
        female_total = int((group_df[sex_field] == "female").sum())
        male_total = int((group_df[sex_field] == "male").sum())
        row: dict[str, object] = {
            "accent_group": accent_group,
            "female_total": female_total,
            "male_total": male_total,
            "pairs_total": min(female_total, male_total),
            "total_binary_sex_speakers": int(len(group_df)),
        }

        for cutoff in cutoffs:
            eligible = group_df[
                group_df["is_english_speaking_country"] | (group_df[residency_field] <= float(cutoff))
            ]
            female_count = int((eligible[sex_field] == "female").sum())
            male_count = int((eligible[sex_field] == "male").sum())
            row[f"female_cutoff_{cutoff}y"] = female_count
            row[f"male_cutoff_{cutoff}y"] = male_count
            row[f"pairs_cutoff_{cutoff}y"] = min(female_count, male_count)

        base_rows.append(row)

    summary = pd.DataFrame(base_rows)
    if summary.empty:
        return summary

    summary = summary.sort_values(
        ["pairs_cutoff_3y", "pairs_cutoff_2y", "pairs_cutoff_1y", "accent_group"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary


def save_balanced_pairs_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def summarize_balanced_pairs_by_country_accent_cutoff(
    metadata: pd.DataFrame,
    group_field: str = "native_language",
    country_field: str = "country",
    birth_place_field: str = "birth_place",
    sex_field: str = "sex",
    residency_field: str = "length_of_english_residence",
    english_speaking_countries: set[str] | frozenset[str] = DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
    cutoffs: tuple[int, ...] = (1, 2, 3),
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Summarize viable male/female pairs by country, accent group, and residency cutoff.

    For non-English-speaking countries, speakers are retained only if residency years <= cutoff.
    For English-speaking countries, speakers are always retained regardless of residency length.
    """
    required_columns = {"speaker_id", group_field, sex_field, residency_field}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    normalized = metadata.copy()
    normalized[group_field] = _normalize_text_series(normalized[group_field])
    normalized[sex_field] = _normalize_sex_series(normalized[sex_field])
    normalized[residency_field] = _parse_residency_years(normalized[residency_field])

    if country_field in normalized.columns:
        normalized[country_field] = _normalize_country_series(normalized[country_field])
    elif birth_place_field in normalized.columns:
        normalized[country_field] = _extract_country_from_birth_place_series(normalized[birth_place_field])
    else:
        raise KeyError(f"Missing required metadata columns: ['{country_field}' or '{birth_place_field}']")

    normalized = normalized[(normalized[group_field] != "") & (normalized[country_field] != "")].copy()
    normalized = _resolve_label_conflicts(normalized, group_field=group_field)
    normalized["is_english_speaking_country"] = normalized[country_field].isin(english_speaking_countries)
    normalized = normalized[normalized[sex_field].isin({"female", "male"})].copy()

    if not cutoffs:
        raise ValueError("cutoffs must contain at least one integer value")

    unique_cutoffs = tuple(sorted({int(c) for c in cutoffs}))
    if any(c < 0 for c in unique_cutoffs):
        raise ValueError("cutoffs must be non-negative integers")

    base = (
        normalized.groupby([country_field, group_field], dropna=False)
        .agg(total_binary_sex_speakers=("speaker_id", "nunique"))
        .reset_index()
        .rename(columns={country_field: "country_of_origin", group_field: "accent_group"})
    )

    rows: list[dict[str, object]] = []
    for cutoff in unique_cutoffs:
        eligible = normalized[
            normalized["is_english_speaking_country"] | (normalized[residency_field] <= float(cutoff))
        ].copy()

        female_counts = (
            eligible[eligible[sex_field] == "female"]
            .groupby([country_field, group_field], dropna=False)["speaker_id"]
            .nunique()
            .rename("female_speakers")
        )
        male_counts = (
            eligible[eligible[sex_field] == "male"]
            .groupby([country_field, group_field], dropna=False)["speaker_id"]
            .nunique()
            .rename("male_speakers")
        )

        merged = (
            pd.concat([female_counts, male_counts], axis=1)
            .fillna(0)
            .reset_index()
            .rename(columns={country_field: "country_of_origin", group_field: "accent_group"})
        )
        merged["female_speakers"] = merged["female_speakers"].astype(int)
        merged["male_speakers"] = merged["male_speakers"].astype(int)
        merged["pair_count"] = merged[["female_speakers", "male_speakers"]].min(axis=1).astype(int)
        merged["cutoff_years"] = int(cutoff)
        merged["enough_pairs"] = merged["pair_count"] >= int(min_pairs)
        rows.append(merged)

    summary = pd.concat(rows, ignore_index=True)
    summary = summary.merge(base, on=["country_of_origin", "accent_group"], how="left")
    summary = summary[
        [
            "cutoff_years",
            "country_of_origin",
            "accent_group",
            "female_speakers",
            "male_speakers",
            "pair_count",
            "enough_pairs",
            "total_binary_sex_speakers",
        ]
    ]
    summary = summary.sort_values(
        ["cutoff_years", "enough_pairs", "pair_count", "country_of_origin", "accent_group"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    return summary