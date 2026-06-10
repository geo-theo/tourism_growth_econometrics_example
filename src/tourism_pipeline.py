"""Download, validate, and combine NPS visitation and NASA POWER weather data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "western_parks_monthly_2005_2024.csv"
)

NPS_DATA_URL = (
    "https://irma.nps.gov/DataStore/DownloadFile/753817?Reference=2316688"
)
NPS_CATALOG_URL = (
    "https://catalog.data.gov/dataset/"
    "nps-visitor-use-statistics-data-package-2024"
)
POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
POWER_DOCS_URL = "https://power.larc.nasa.gov/docs/services/api/temporal/monthly/"

START_YEAR = 2005
END_YEAR = 2024

# Coordinates approximate a central visitor-use area rather than every part of
# each park. NASA POWER meteorology is gridded, so false coordinate precision
# would not improve the underlying weather measurement.
PARKS = pd.DataFrame(
    [
        ("GLAC", "Glacier", "MT", 48.7596, -113.7870),
        ("YELL", "Yellowstone", "WY/MT/ID", 44.4280, -110.5885),
        ("GRTE", "Grand Teton", "WY", 43.7904, -110.6818),
        ("ROMO", "Rocky Mountain", "CO", 40.3428, -105.6836),
        ("GRCA", "Grand Canyon", "AZ", 36.1069, -112.1129),
        ("ZION", "Zion", "UT", 37.2982, -113.0263),
        ("BRCA", "Bryce Canyon", "UT", 37.6283, -112.1677),
        ("ARCH", "Arches", "UT", 38.7331, -109.5925),
        ("CANY", "Canyonlands", "UT", 38.3269, -109.8783),
        ("CARE", "Capitol Reef", "UT", 38.2917, -111.2615),
        ("YOSE", "Yosemite", "CA", 37.8651, -119.5383),
        ("SEQU", "Sequoia & Kings Canyon", "CA", 36.4864, -118.5658),
        ("MORA", "Mount Rainier", "WA", 46.8523, -121.7603),
        ("OLYM", "Olympic", "WA", 47.8021, -123.6044),
        ("CRLA", "Crater Lake", "OR", 42.9446, -122.1090),
        ("LAVO", "Lassen Volcanic", "CA", 40.4977, -121.4207),
        ("DEVA", "Death Valley", "CA/NV", 36.5323, -116.9325),
        ("JOTR", "Joshua Tree", "CA", 33.8734, -115.9010),
    ],
    columns=["park_code", "park_name", "state", "latitude", "longitude"],
)


def _session() -> requests.Session:
    """Return a requests session with conservative retry behavior."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "tourism-weather-econometrics-portfolio/1.0"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_nps_visitation(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    force: bool = False,
) -> pd.DataFrame:
    """Download and filter monthly recreation visits for the selected parks."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "nps_vustats_1979_2024.csv"

    if force or not cache_path.exists():
        with _session().get(NPS_DATA_URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            with cache_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

    selected_codes = set(PARKS["park_code"])
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        cache_path,
        usecols=["UnitCode", "Year", "Month", "Statistic", "Value"],
        dtype={
            "UnitCode": "string",
            "Year": "int16",
            "Month": "int8",
            "Statistic": "string",
            "Value": "int64",
        },
        chunksize=250_000,
    ):
        keep = (
            chunk["UnitCode"].isin(selected_codes)
            & chunk["Year"].between(start_year, end_year)
            & chunk["Statistic"].eq("TRV")
        )
        chunks.append(chunk.loc[keep])

    visits = pd.concat(chunks, ignore_index=True)
    visits = visits.rename(
        columns={
            "UnitCode": "park_code",
            "Year": "year",
            "Month": "month",
            "Value": "recreation_visits",
        }
    ).drop(columns="Statistic")

    expected_rows = len(PARKS) * (end_year - start_year + 1) * 12
    if len(visits) != expected_rows:
        counts = visits.groupby("park_code").size().to_dict()
        raise ValueError(
            f"Expected {expected_rows:,} NPS rows but found {len(visits):,}. "
            f"Counts by park: {counts}"
        )
    if visits.duplicated(["park_code", "year", "month"]).any():
        raise ValueError("NPS data contain duplicate park-month observations.")

    return visits


def _parse_power_response(payload: dict[str, Any], park_code: str) -> pd.DataFrame:
    """Convert a NASA POWER monthly JSON response to tidy rows."""
    parameters = payload["properties"]["parameter"]
    temperature = parameters["T2M"]
    precipitation = parameters["PRECTOTCORR"]

    records = []
    for key, temp_c in temperature.items():
        # POWER includes month 13 as an annual summary in monthly responses.
        if key.endswith("13"):
            continue
        records.append(
            {
                "park_code": park_code,
                "year": int(key[:4]),
                "month": int(key[4:]),
                "temp_c": temp_c,
                "precip_mm_day": precipitation[key],
            }
        )

    weather = pd.DataFrame.from_records(records)
    weather[["temp_c", "precip_mm_day"]] = weather[
        ["temp_c", "precip_mm_day"]
    ].replace(-999, pd.NA)
    return weather


def download_power_weather(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    force: bool = False,
    request_pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Request monthly temperature and precipitation for each park."""
    raw_dir = Path(raw_dir)
    weather_dir = raw_dir / "nasa_power"
    weather_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    frames: list[pd.DataFrame] = []

    for park in PARKS.itertuples(index=False):
        cache_path = weather_dir / (
            f"{park.park_code}_{start_year}_{end_year}.json"
        )
        if force or not cache_path.exists():
            params = {
                "parameters": "T2M,PRECTOTCORR",
                "community": "RE",
                "longitude": park.longitude,
                "latitude": park.latitude,
                "start": start_year,
                "end": end_year,
                "format": "JSON",
            }
            response = session.get(POWER_API_URL, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(request_pause_seconds)
        else:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))

        frames.append(_parse_power_response(payload, park.park_code))

    weather = pd.concat(frames, ignore_index=True)
    expected_rows = len(PARKS) * (end_year - start_year + 1) * 12
    if len(weather) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} weather rows but found "
            f"{len(weather):,}."
        )
    if weather[["temp_c", "precip_mm_day"]].isna().any().any():
        raise ValueError("NASA POWER returned missing weather values.")
    if weather.duplicated(["park_code", "year", "month"]).any():
        raise ValueError("Weather data contain duplicate park-month observations.")

    return weather


def build_analysis_panel(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_path: Path | str = DEFAULT_PROCESSED_PATH,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    force: bool = False,
) -> pd.DataFrame:
    """Build and save the complete monthly park-weather panel."""
    visits = download_nps_visitation(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force=force,
    )
    weather = download_power_weather(
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        force=force,
    )

    panel = visits.merge(
        weather,
        on=["park_code", "year", "month"],
        how="inner",
        validate="one_to_one",
    ).merge(PARKS, on="park_code", how="left", validate="many_to_one")
    panel["date"] = pd.to_datetime(
        {"year": panel["year"], "month": panel["month"], "day": 1}
    )
    panel = panel.sort_values(["park_code", "date"]).reset_index(drop=True)

    expected_rows = len(PARKS) * (end_year - start_year + 1) * 12
    if len(panel) != expected_rows:
        raise ValueError(
            f"Merge should produce {expected_rows:,} rows, not {len(panel):,}."
        )
    if panel.isna().any().any():
        missing = panel.isna().sum()
        raise ValueError(f"Panel contains missing values:\n{missing[missing > 0]}")

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed_path, index=False)
    return panel
