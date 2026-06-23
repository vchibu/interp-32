from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_ENGLISH_SPEAKING_COUNTRIES = (
    "usa",
    "uk",
    "canada",
    "australia",
    "ireland",
    "new zealand",
    "south africa",
)


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text or None


def _clean_series(series: pd.Series) -> pd.Series:
    return series.map(_clean_text)


def _parse_years(value: object) -> float:
    text = _clean_text(value)
    if text is None:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return np.nan
    return float(match.group(0))


def add_accent_country_probe_filter(
    manifest: pd.DataFrame,
    accent_field: str = "native_language",
    country_field: str = "country_of_origin",
    sex_field: str = "sex",
    english_residence_years_field: str = "length_of_english_residence",
    english_speaking_countries: Sequence[str] = DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
    max_non_english_country_residence_years: float = 2.0,
) -> pd.DataFrame:
    required = {accent_field, country_field, sex_field, english_residence_years_field}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing required column(s): {', '.join(sorted(missing))}")

    prepared = manifest.copy()
    prepared["probe_accent"] = _clean_series(prepared[accent_field])
    prepared["probe_country_of_birth"] = _clean_series(prepared[country_field])
    prepared["probe_sex"] = _clean_series(prepared[sex_field])
    prepared["english_residence_years"] = prepared[english_residence_years_field].map(_parse_years)

    english_countries = {_clean_text(country) for country in english_speaking_countries}
    english_countries.discard(None)
    prepared["is_english_speaking_birth_country"] = prepared["probe_country_of_birth"].isin(english_countries)
    too_much_residence = prepared["english_residence_years"] > max_non_english_country_residence_years
    prepared["excluded_for_english_residence"] = (
        ~prepared["is_english_speaking_birth_country"] & too_much_residence
    )
    prepared["eligible_for_accent_country_probe"] = (
        prepared["probe_accent"].notna()
        & prepared["probe_country_of_birth"].notna()
        & prepared["probe_sex"].isin(["female", "male"])
        & ~prepared["excluded_for_english_residence"]
    )
    prepared["accent_country_label"] = (
        prepared["probe_accent"].fillna("")
        + "__"
        + prepared["probe_country_of_birth"].fillna("")
    )
    return prepared


def summarize_accent_country_probe_groups(
    manifest: pd.DataFrame,
    accent_field: str = "native_language",
    country_field: str = "country_of_origin",
    sex_field: str = "sex",
    english_residence_years_field: str = "length_of_english_residence",
    english_speaking_countries: Sequence[str] = DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
    max_non_english_country_residence_years: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = add_accent_country_probe_filter(
        manifest=manifest,
        accent_field=accent_field,
        country_field=country_field,
        sex_field=sex_field,
        english_residence_years_field=english_residence_years_field,
        english_speaking_countries=english_speaking_countries,
        max_non_english_country_residence_years=max_non_english_country_residence_years,
    )

    eligible = prepared[prepared["eligible_for_accent_country_probe"]].copy()
    summary = (
        eligible.groupby(["probe_accent", "probe_country_of_birth"], dropna=False)
        .agg(
            n_speakers=("speaker_id", "nunique"),
            n_recordings=("sample_name", "nunique"),
            n_female=("probe_sex", lambda values: int((values == "female").sum())),
            n_male=("probe_sex", lambda values: int((values == "male").sum())),
            n_unknown_residence_years=("english_residence_years", lambda values: int(values.isna().sum())),
            median_english_residence_years=("english_residence_years", "median"),
            max_english_residence_years=("english_residence_years", "max"),
            english_speaking_birth_country=("is_english_speaking_birth_country", "first"),
        )
        .reset_index()
        .rename(
            columns={
                "probe_accent": "accent",
                "probe_country_of_birth": "country_of_birth",
            }
        )
    )
    summary["n_balanced_female_male_pairs"] = summary[["n_female", "n_male"]].min(axis=1)
    summary["accent_country_label"] = summary["accent"] + "__" + summary["country_of_birth"]
    summary = summary[
        [
            "accent_country_label",
            "accent",
            "country_of_birth",
            "n_speakers",
            "n_recordings",
            "n_female",
            "n_male",
            "n_balanced_female_male_pairs",
            "english_speaking_birth_country",
            "n_unknown_residence_years",
            "median_english_residence_years",
            "max_english_residence_years",
        ]
    ]
    summary = summary.sort_values(
        ["n_balanced_female_male_pairs", "n_speakers", "accent", "country_of_birth"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    return summary, prepared


def keep_groups_with_min_balanced_sex_pairs(
    manifest: pd.DataFrame,
    label_field: str = "accent_country_label",
    sex_field: str = "probe_sex",
    min_balanced_sex_pairs: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {label_field, sex_field}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing required column(s): {', '.join(sorted(missing))}")

    counts = (
        manifest.groupby(label_field, dropna=False)
        .agg(
            n_speakers=("speaker_id", "nunique") if "speaker_id" in manifest.columns else (label_field, "size"),
            n_female=(sex_field, lambda values: int((values == "female").sum())),
            n_male=(sex_field, lambda values: int((values == "male").sum())),
        )
        .reset_index()
    )
    counts["n_balanced_female_male_pairs"] = counts[["n_female", "n_male"]].min(axis=1)
    counts["included_by_balanced_sex_pairs"] = (
        counts["n_balanced_female_male_pairs"] >= min_balanced_sex_pairs
    )
    included_labels = counts.loc[counts["included_by_balanced_sex_pairs"], label_field]
    filtered = manifest[manifest[label_field].isin(included_labels)].copy()
    counts = counts.sort_values(
        ["included_by_balanced_sex_pairs", "n_balanced_female_male_pairs", "n_speakers", label_field],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return filtered, counts


def save_probe_group_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
