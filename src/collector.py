"""Collect monthly economic indicators from the Bank of Korea ECOS API."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


ECOS_BASE_URL = "https://ecos.bok.or.kr/api"


@dataclass(frozen=True)
class SeriesSpec:
    """Parameters needed for one ECOS StatisticSearch request."""

    name: str
    stat_code: str
    cycle: str
    item_codes: tuple[str, ...]
    raw_filename: str


SERIES_SPECS = (
    SeriesSpec(
        name="exchange_rate",
        stat_code="731Y004",
        cycle="M",
        item_codes=("0000001", "0000100"),
        raw_filename="exchange_rate.csv",
    ),
    SeriesSpec(
        name="base_rate",
        stat_code="722Y001",
        cycle="M",
        item_codes=("0101000",),
        raw_filename="base_rate.csv",
    ),
    SeriesSpec(
        name="cpi",
        stat_code="901Y009",
        cycle="M",
        item_codes=("0",),
        raw_filename="cpi.csv",
    ),
)


class EcosApiError(RuntimeError):
    """Raised when ECOS returns an API or response-format error."""


class EcosClient:
    """Small ECOS client with pagination, retries, and response validation."""

    def __init__(
        self,
        api_key: str,
        *,
        language: str = "kr",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("ECOS_API_KEY is empty. Set it in .env.")
        self.api_key = api_key
        self.language = language
        self.timeout = timeout
        self.max_retries = max_retries
        # ECOS limits the public sample key to ten rows per call.
        self.page_size = 10 if api_key.lower() == "sample" else 1_000
        self.session = requests.Session()

    def _request_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if "RESULT" in payload:
                    result = payload["RESULT"]
                    raise EcosApiError(
                        f"ECOS {result.get('CODE', 'error')}: "
                        f"{result.get('MESSAGE', 'unknown error')}"
                    )
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
        raise EcosApiError(f"ECOS request failed after retries: {last_error}")

    def fetch_series(
        self, spec: SeriesSpec, start_period: str, end_period: str
    ) -> pd.DataFrame:
        """Fetch every row for a series, following ECOS row-number pagination."""
        rows: list[dict[str, Any]] = []
        start_row = 1
        total_count: int | None = None
        item_path = "/".join(spec.item_codes)

        while total_count is None or start_row <= total_count:
            end_row = start_row + self.page_size - 1
            url = (
                f"{ECOS_BASE_URL}/StatisticSearch/{self.api_key}/json/"
                f"{self.language}/{start_row}/{end_row}/{spec.stat_code}/"
                f"{spec.cycle}/{start_period}/{end_period}/{item_path}/"
            )
            payload = self._request_json(url)
            result = payload.get("StatisticSearch")
            if not isinstance(result, dict):
                raise EcosApiError(
                    f"Unexpected response for {spec.name}: StatisticSearch missing"
                )
            total_count = int(result.get("list_total_count", 0))
            page_rows = result.get("row", [])
            if not isinstance(page_rows, list):
                raise EcosApiError(f"Unexpected row data for {spec.name}")
            rows.extend(page_rows)
            if not page_rows:
                break
            start_row += len(page_rows)

        if not rows:
            raise EcosApiError(
                f"No data returned for {spec.name} ({start_period}-{end_period})"
            )

        frame = pd.DataFrame(rows)
        if "TIME" not in frame or "DATA_VALUE" not in frame:
            raise EcosApiError(f"Required fields missing for {spec.name}")
        return frame


def collect_all(
    raw_dir: str | Path,
    start_period: str = "201501",
    end_period: str = "202512",
    env_file: str | Path = ".env",
) -> dict[str, Path]:
    """Collect and save all configured raw series as UTF-8 CSV files."""
    load_dotenv(env_file)
    api_key = os.getenv("ECOS_API_KEY", "").strip()
    client = EcosClient(api_key)
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for spec in SERIES_SPECS:
        frame = client.fetch_series(spec, start_period, end_period)
        path = destination / spec.raw_filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        saved[spec.name] = path
        print(f"  collected {spec.name}: {len(frame)} rows -> {path}")
    return saved

