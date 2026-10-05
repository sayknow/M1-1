"""Clean, align, merge, and engineer variables from ECOS raw data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FILES = {
    "exchange_rate": "exchange_rate.csv",
    "base_rate": "base_rate.csv",
    "cpi": "cpi.csv",
}
SELECTED_LAGS = (1, 3, 6, 12)


def _clean_one(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"TIME": "string", "DATA_VALUE": "string"})
    required = {"TIME", "DATA_VALUE"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    clean = frame.loc[:, ["TIME", "DATA_VALUE"]].copy()
    clean["date"] = pd.to_datetime(clean["TIME"], format="%Y%m", errors="coerce")
    clean[value_name] = pd.to_numeric(
        clean["DATA_VALUE"].str.replace(",", "", regex=False), errors="coerce"
    )
    invalid = clean["date"].isna() | clean[value_name].isna()
    if invalid.any():
        examples = clean.loc[invalid, ["TIME", "DATA_VALUE"]].head().to_dict("records")
        raise ValueError(f"Invalid date/value rows in {path}: {examples}")

    clean = clean.loc[~invalid, ["date", value_name]]
    clean = clean.sort_values("date").drop_duplicates("date", keep="last")
    return clean.reset_index(drop=True)


def build_processed_data(
    raw_dir: str | Path,
    processed_dir: str | Path,
    start_period: str = "2015-01-01",
    end_period: str = "2025-12-01",
) -> tuple[pd.DataFrame, Path]:
    """Create a validated monthly panel and all requested analysis variables."""
    source = Path(raw_dir)
    destination = Path(processed_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frames = [
        _clean_one(source / filename, value_name)
        for value_name, filename in RAW_FILES.items()
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer", validate="one_to_one")

    expected_dates = pd.date_range(start_period, end_period, freq="MS")
    merged = merged.set_index("date").reindex(expected_dates)
    merged.index.name = "date"

    missing = merged[list(RAW_FILES)].isna().sum()
    if missing.any():
        details = ", ".join(f"{name}={count}" for name, count in missing.items())
        raise ValueError(f"Missing monthly source observations after alignment: {details}")

    merged["exchange_yoy"] = (
        merged["exchange_rate"].pct_change(12, fill_method=None) * 100
    )
    merged["base_rate_change"] = merged["base_rate"].diff()
    merged["cpi_yoy"] = merged["cpi"].pct_change(12, fill_method=None) * 100

    # A lag-k value at month t is the predictor observed at t-k.
    for lag in SELECTED_LAGS:
        merged[f"exchange_lag_{lag}"] = merged["exchange_yoy"].shift(lag)
        merged[f"base_rate_lag_{lag}"] = merged["base_rate_change"].shift(lag)

    merged = merged.reset_index()
    path = destination / "economic_indicators_monthly.csv"
    merged.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return merged, path
