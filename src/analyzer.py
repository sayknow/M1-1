"""Descriptive, correlation, lead-lag, and visualization routines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

# Keep Matplotlib's font cache inside this project instead of relying on a
# potentially read-only user home directory.
_MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MAX_LAG = 12


def _save_line_plot(
    data: pd.DataFrame,
    columns: Iterable[str],
    labels: Iterable[str],
    title: str,
    ylabel: str,
    path: Path,
    *,
    zero_line: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for column, label in zip(columns, labels):
        ax.plot(data["date"], data[column], label=label, linewidth=1.8)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set(title=title, xlabel="Date", ylabel=ylabel)
    ax.grid(alpha=0.25)
    if len(list(columns)) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calculate_lag_correlations(data: pd.DataFrame, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """Correlate predictor(t-lag) with CPI YoY(t) for every monthly lag."""
    rows: list[dict[str, float | int]] = []
    for lag in range(max_lag + 1):
        exchange_pair = pd.concat(
            [data["exchange_yoy"].shift(lag), data["cpi_yoy"]], axis=1
        ).dropna()
        rate_pair = pd.concat(
            [data["base_rate_change"].shift(lag), data["cpi_yoy"]], axis=1
        ).dropna()
        rows.append(
            {
                "lag_months": lag,
                "exchange_cpi_correlation": exchange_pair.iloc[:, 0].corr(
                    exchange_pair.iloc[:, 1]
                ),
                "exchange_observations": len(exchange_pair),
                "base_rate_cpi_correlation": rate_pair.iloc[:, 0].corr(
                    rate_pair.iloc[:, 1]
                ),
                "base_rate_observations": len(rate_pair),
            }
        )
    return pd.DataFrame(rows)


def _save_lag_plot(
    correlations: pd.DataFrame, value_column: str, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d95f02" if value >= 0 else "#1b9e77" for value in correlations[value_column]]
    ax.bar(correlations["lag_months"], correlations[value_column], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=title,
        xlabel="Predictor lead over CPI (months)",
        ylabel="Pearson correlation with CPI YoY",
        xticks=correlations["lag_months"],
        ylim=(-1, 1),
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_rate_cpi_plot(data: pd.DataFrame, path: Path) -> None:
    """Compare differently scaled rate changes and CPI inflation clearly."""
    fig, left = plt.subplots(figsize=(12, 5.5))
    right = left.twinx()
    rate_line = left.plot(
        data["date"], data["base_rate_change"],
        color="#1f77b4", label="Base-rate monthly change", linewidth=1.5
    )[0]
    cpi_line = right.plot(
        data["date"], data["cpi_yoy"],
        color="#ff7f0e", label="CPI YoY", linewidth=1.8
    )[0]
    left.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    left.set(
        title="Base-Rate Change and CPI Inflation",
        xlabel="Date",
        ylabel="Base-rate monthly change (percentage points)",
    )
    right.set_ylabel("CPI year-on-year change (percent)")
    left.grid(alpha=0.25)
    left.legend(handles=[rate_line, cpi_line], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_analysis(
    processed_csv: str | Path,
    figures_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run all analyses, save tables/figures, and return headline findings."""
    data = pd.read_csv(processed_csv, parse_dates=["date"])
    figures = Path(figures_dir)
    output = Path(output_dir)
    figures.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    source_columns = ["exchange_rate", "base_rate", "cpi"]
    summary_stats = data[source_columns + ["exchange_yoy", "base_rate_change", "cpi_yoy"]].describe().T
    summary_stats.to_csv(output / "summary_statistics.csv", encoding="utf-8-sig")

    correlations = calculate_lag_correlations(data)
    correlations.to_csv(output / "lag_correlations.csv", index=False, encoding="utf-8-sig")

    _save_line_plot(
        data, ["exchange_rate"], ["USD/KRW monthly average"],
        "USD/KRW Exchange Rate", "KRW per USD", figures / "01_exchange_rate.png"
    )
    _save_line_plot(
        data, ["base_rate"], ["Bank of Korea base rate"],
        "Bank of Korea Base Rate", "Percent per year", figures / "02_base_rate.png"
    )
    _save_line_plot(
        data, ["cpi"], ["Consumer Price Index"],
        "Consumer Price Index (2020=100)", "Index", figures / "03_cpi.png"
    )
    _save_line_plot(
        data, ["exchange_yoy", "cpi_yoy"], ["Exchange rate YoY", "CPI YoY"],
        "Exchange-Rate Change and CPI Inflation", "Year-on-year percent", figures / "04_exchange_yoy_vs_cpi_yoy.png", zero_line=True
    )
    _save_rate_cpi_plot(data, figures / "05_base_rate_change_vs_cpi_yoy.png")
    _save_lag_plot(
        correlations, "exchange_cpi_correlation",
        "Lead-Lag Correlation: Exchange Rate YoY vs CPI YoY", figures / "06_exchange_cpi_lag_correlation.png"
    )
    _save_lag_plot(
        correlations, "base_rate_cpi_correlation",
        "Lead-Lag Correlation: Base-Rate Change vs CPI YoY", figures / "07_base_rate_cpi_lag_correlation.png"
    )

    exchange_best = correlations.loc[
        correlations["exchange_cpi_correlation"].abs().idxmax()
    ]
    rate_best = correlations.loc[
        correlations["base_rate_cpi_correlation"].abs().idxmax()
    ]
    findings: dict[str, object] = {
        "period": {
            "start": data["date"].min().strftime("%Y-%m"),
            "end": data["date"].max().strftime("%Y-%m"),
            "months": int(len(data)),
        },
        "contemporaneous_correlations": {
            "exchange_yoy_vs_cpi_yoy": float(
                correlations.loc[0, "exchange_cpi_correlation"]
            ),
            "base_rate_change_vs_cpi_yoy": float(
                correlations.loc[0, "base_rate_cpi_correlation"]
            ),
        },
        "selected_lag_correlations": [
            {
                "lag_months": int(row["lag_months"]),
                "exchange_yoy_vs_cpi_yoy": float(
                    row["exchange_cpi_correlation"]
                ),
                "base_rate_change_vs_cpi_yoy": float(
                    row["base_rate_cpi_correlation"]
                ),
            }
            for _, row in correlations[
                correlations["lag_months"].isin([0, 1, 3, 6, 12])
            ].iterrows()
        ],
        "strongest_absolute_lag_correlations": {
            "exchange_yoy_vs_cpi_yoy": {
                "lag_months": int(exchange_best["lag_months"]),
                "correlation": float(exchange_best["exchange_cpi_correlation"]),
                "observations": int(exchange_best["exchange_observations"]),
            },
            "base_rate_change_vs_cpi_yoy": {
                "lag_months": int(rate_best["lag_months"]),
                "correlation": float(rate_best["base_rate_cpi_correlation"]),
                "observations": int(rate_best["base_rate_observations"]),
            },
        },
        "interpretation_warning": (
            "Correlations are descriptive associations, not causal estimates. "
            "Interest rates may respond to inflation, and omitted common shocks may affect all series."
        ),
        "lag_definition": (
            "At lag k, predictor(t-k) is correlated with CPI YoY(t); "
            "positive k means the predictor leads CPI."
        ),
    }
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(findings, file, ensure_ascii=False, indent=2)
    return findings
