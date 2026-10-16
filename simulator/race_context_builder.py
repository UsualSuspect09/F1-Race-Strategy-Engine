"""Forecast-mode Race Context Builder for the F1 tyre-strategy project.

Dependencies: pandas, requests
Reads RACE_LAPS from src.config/config, circuit_metadata.csv, and the corrected
2023-2025 master stint datasets. Open-Meteo provides forecast air temperature;
historical per-GP medians provide TrackTemp because Open-Meteo does not provide
F1 track-surface temperature. If the forecast is unavailable/out of range, both
AirTemp and TrackTemp use explicitly labelled historical medians.
"""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping, Optional
import importlib
import re
import sys
import unicodedata

import pandas as pd
import requests

PROJECT_ROOT = Path("C:/Users/VANSH/Formula1")
# Ensure project-level config.py / src.config can be imported when run as a script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CIRCUIT_METADATA_PATH = PROJECT_ROOT / "datasets" / "circuit_metadata.csv"
HISTORICAL_DATA_PATHS = [
    PROJECT_ROOT / "datasets" / "2023" / "master_stint_2023_v4_targetfix.csv",
    PROJECT_ROOT / "datasets" / "2024" / "master_stint_2024_v4_targetfix.csv",
    PROJECT_ROOT / "datasets" / "2025" / "master_stint_2025_v2_targetfix.csv",
]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Approximate circuit coordinates, used only for local weather lookup.
CIRCUIT_COORDINATES = {
    "bahrain": (26.0325, 50.5106), "saudi arabian": (21.6319, 39.1044),
    "jeddah": (21.6319, 39.1044), "australian": (-37.8497, 144.9680),
    "melbourne": (-37.8497, 144.9680), "japanese": (34.8431, 136.5410),
    "suzuka": (34.8431, 136.5410), "chinese": (31.3389, 121.2200),
    "shanghai": (31.3389, 121.2200), "miami": (25.9581, -80.2389),
    "emilia romagna": (44.3439, 11.7167), "imola": (44.3439, 11.7167),
    "monaco": (43.7347, 7.4206), "canadian": (45.5000, -73.5228),
    "montreal": (45.5000, -73.5228), "spanish": (41.5700, 2.2611),
    "barcelona": (41.5700, 2.2611), "austrian": (47.2197, 14.7647),
    "spielberg": (47.2197, 14.7647), "british": (52.0786, -1.0169),
    "silverstone": (52.0786, -1.0169), "hungarian": (47.5789, 19.2486),
    "hungaroring": (47.5789, 19.2486), "belgian": (50.4372, 5.9714),
    "spa": (50.4372, 5.9714), "dutch": (52.3888, 4.5409),
    "zandvoort": (52.3888, 4.5409), "italian": (45.6156, 9.2811),
    "monza": (45.6156, 9.2811), "azerbaijan": (40.3725, 49.8533),
    "baku": (40.3725, 49.8533), "singapore": (1.2914, 103.8640),
    "united states": (30.1328, -97.6411), "austin": (30.1328, -97.6411),
    "mexico": (19.4042, -99.0907), "sao paulo": (-23.7014, -46.6969),
    "brazilian": (-23.7014, -46.6969), "las vegas": (36.1147, -115.1728),
    "qatar": (25.4900, 51.4542), "lusail": (25.4900, 51.4542),
    "abu dhabi": (24.4672, 54.6031), "yas marina": (24.4672, 54.6031),
    "madrid": (40.4637, -3.7492),
}


def _norm(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"\b(grand prix|gp|circuit|formula 1|f1)\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _load_race_laps(explicit: Optional[Mapping[str, int]] = None) -> Mapping[str, int]:
    if explicit is not None:
        return explicit
    errors = []
    for module_name in ("src.config", "config", "simulator.config"):
        try:
            mapping = getattr(importlib.import_module(module_name), "RACE_LAPS")
            if isinstance(mapping, Mapping) and mapping:
                return mapping
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "Could not import RACE_LAPS. Expected it in src.config or config.py. "
        + " | ".join(errors)
    )


def _match_key(name: str, keys) -> str:
    target = _norm(name)
    keys = list(keys)
    exact = [key for key in keys if _norm(key) == target]
    if len(exact) == 1:
        return exact[0]
    candidates = [key for key in keys if target and (target in _norm(key) or _norm(key) in target)]
    if candidates:
        candidates.sort(key=lambda key: len(_norm(key)), reverse=True)
        if len(candidates) == 1 or len(_norm(candidates[0])) > len(_norm(candidates[1])):
            return candidates[0]
    raise ValueError(f"Could not uniquely match Grand Prix '{name}'. Available: {keys}")


def _metadata_row(gp_name: str, path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Circuit metadata file not found: {path}")
    df = pd.read_csv(path)
    required = ["GP", "CircuitType", "TrackLength", "TrackAbrasiveness", "AverageCornerSpeed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    target = _norm(gp_name)
    matches = df.loc[df["GP"].astype(str).map(_norm) == target]
    if matches.empty:
        matches = df.loc[df["GP"].astype(str).map(lambda x: target in _norm(x) or _norm(x) in target)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely match '{gp_name}' in metadata. Matches: {matches['GP'].tolist()}")
    return matches.iloc[0]


def _coordinates(gp_name: str) -> tuple[float, float]:
    target = _norm(gp_name)
    entries = sorted(CIRCUIT_COORDINATES.items(), key=lambda item: len(_norm(item[0])), reverse=True)
    for fragment, coords in entries:
        if _norm(fragment) in target:
            return coords
    raise ValueError(
        f"No coordinates configured for '{gp_name}'. Add its circuit to CIRCUIT_COORDINATES."
    )


def _parse_datetime(race_date, race_time) -> datetime:
    if isinstance(race_date, datetime):
        day = race_date.date()
    elif isinstance(race_date, date):
        day = race_date
    else:
        try:
            day = date.fromisoformat(str(race_date).strip())
        except ValueError as exc:
            raise ValueError("race_date must use YYYY-MM-DD format.") from exc
    if race_time is None or str(race_time).strip() == "":
        clock = time(15, 0)
    elif isinstance(race_time, time):
        clock = race_time
    else:
        try:
            clock = time.fromisoformat(str(race_time).strip())
        except ValueError as exc:
            raise ValueError("race_time must use HH:MM or HH:MM:SS local time.") from exc
    return datetime.combine(day, clock)


def _fetch_open_meteo(latitude: float, longitude: float, target_dt: datetime) -> dict:
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m",
            "timezone": "auto",
            "forecast_days": 16,
        },
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    same_day = [i for i, stamp in enumerate(times) if not pd.isna(stamp) and stamp.date() == target_dt.date()]
    if not same_day:
        raise ValueError("Requested race date is outside the available forecast window.")
    target = pd.Timestamp(target_dt)
    idx = min(same_day, key=lambda i: abs((times[i] - target).total_seconds()))

    def get_value(key):
        values = hourly.get(key, [])
        if idx >= len(values) or values[idx] is None:
            return None
        try:
            value = float(values[idx])
            return value if pd.notna(value) else None
        except (ValueError, TypeError):
            return None

    air = get_value("temperature_2m")
    if air is None:
        raise ValueError("Weather API returned no air temperature for the selected time.")
    return {
        "air_temp": air,
        "humidity": get_value("relative_humidity_2m"),
        "pressure": get_value("surface_pressure"),
        "precipitation": get_value("precipitation"),
        "wind_speed": get_value("wind_speed_10m"),
        "forecast_timestamp": str(times[idx]),
        "weather_timezone": payload.get("timezone"),
        "weather_source": "open_meteo_forecast",
    }


def _historical_medians(gp_name: str, paths: list[Path]) -> dict:
    frames = []
    unreadable = []
    for path in paths:
        if not path.exists():
            unreadable.append(str(path))
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns
            if {"GP", "AirTemp", "TrackTemp"}.issubset(header):
                frames.append(pd.read_csv(path, usecols=["GP", "AirTemp", "TrackTemp"]))
        except Exception:
            unreadable.append(str(path))
    if not frames:
        raise FileNotFoundError(
            "No historical master dataset with GP, AirTemp and TrackTemp could be read. "
            f"Paths checked: {paths}"
        )
    all_data = pd.concat(frames, ignore_index=True)
    target = _norm(gp_name)
    data = all_data.loc[all_data["GP"].astype(str).map(_norm) == target]
    if data.empty:
        names = all_data["GP"].dropna().astype(str).unique().tolist()
        candidates = [name for name in names if target in _norm(name) or _norm(name) in target]
        if len(candidates) == 1:
            data = all_data.loc[all_data["GP"].astype(str) == candidates[0]]
    if data.empty:
        raise ValueError(f"No historical weather rows found for '{gp_name}'.")
    air = pd.to_numeric(data["AirTemp"], errors="coerce").dropna()
    track = pd.to_numeric(data["TrackTemp"], errors="coerce").dropna()
    if air.empty or track.empty:
        raise ValueError(f"Historical weather rows for '{gp_name}' lack valid AirTemp/TrackTemp.")
    return {
        "air_temp": round(float(air.median()), 2),
        "track_temp": round(float(track.median()), 2),
        "weather_source": "historical_gp_median_fallback",
        "historical_rows": int(len(data)),
        "fallback_note": "Historical GP medians are estimates, not a race-day weather forecast.",
        "unavailable_historical_files": unreadable,
    }


class RaceContextBuilder:
    """Build a forecast-mode context compatible with the existing RaceSimulator."""

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        race_laps: Optional[Mapping[str, int]] = None,
        circuit_metadata_path: str | Path | None = None,
        historical_data_paths: Optional[list[str | Path]] = None,
        forecast_fetcher=None,
    ):
        self.project_root = Path(project_root)
        self.race_laps = _load_race_laps(race_laps)
        self.metadata_path = Path(circuit_metadata_path or self.project_root / "datasets" / "circuit_metadata.csv")
        self.historical_paths = [Path(p) for p in (historical_data_paths or HISTORICAL_DATA_PATHS)]
        self.forecast_fetcher = forecast_fetcher or _fetch_open_meteo

    def build_context(
        self,
        gp_name: str,
        season: int,
        race_date: str | date,
        race_time: str | time | None = None,
        *,
        allow_historical_weather_fallback: bool = True,
        yellow_laps: int = 0,
        sc_laps: int = 0,
        vsc_laps: int = 0,
        redflag_laps: int = 0,
    ) -> dict[str, Any]:
        if not isinstance(season, int) or season < 1950:
            raise ValueError("season must be an integer year (1950 or later).")
        for key, value in {
            "yellow_laps": yellow_laps, "sc_laps": sc_laps,
            "vsc_laps": vsc_laps, "redflag_laps": redflag_laps,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer.")

        laps_key = _match_key(gp_name, self.race_laps)
        race_laps = int(self.race_laps[laps_key])
        if race_laps <= 0:
            raise ValueError(f"Invalid race distance configured for {laps_key}.")

        row = _metadata_row(gp_name, self.metadata_path)
        circuit = {
            "circuit_type": row["CircuitType"],
            "track_length": float(row["TrackLength"]),
            "track_abrasiveness": row["TrackAbrasiveness"],
            "average_corner_speed": float(row["AverageCornerSpeed"]),
        }
        if any(pd.isna(value) for value in circuit.values()):
            raise ValueError(f"Incomplete circuit metadata for {row['GP']}.")

        race_dt = _parse_datetime(race_date, race_time)
        latitude, longitude = _coordinates(str(row["GP"]))
        try:
            weather = self.forecast_fetcher(latitude, longitude, race_dt)
            # Track-surface temperature is not returned by the Open-Meteo forecast.
            historical = _historical_medians(str(row["GP"]), self.historical_paths)
            weather["track_temp"] = historical["track_temp"]
            weather["track_temp_source"] = "historical_gp_median_fallback"
            weather["air_temp_source"] = "weather_api_forecast"
            weather["track_temp_note"] = "Historical GP median; Open-Meteo does not provide track-surface temperature."
            weather["weather_fallback_used"] = False
        except Exception as exc:
            if not allow_historical_weather_fallback:
                raise RuntimeError(f"Forecast unavailable for {race_dt.isoformat()}: {exc}") from exc
            weather = _historical_medians(str(row["GP"]), self.historical_paths)
            weather.update({
                "track_temp_source": "historical_gp_median_fallback",
                "air_temp_source": "historical_gp_median_fallback",
                "weather_fallback_used": True,
                "forecast_error": str(exc),
                "humidity": None, "pressure": None, "precipitation": None, "wind_speed": None,
                "forecast_timestamp": None, "weather_timezone": None,
            })

        context = {
            # Required RaceSimulator inputs:
            "gp_name": str(row["GP"]),
            "air_temp": float(weather["air_temp"]),
            "track_temp": float(weather["track_temp"]),
            "season": int(season),
            "circuit_type": circuit["circuit_type"],
            "track_length": circuit["track_length"],
            "track_abrasiveness": circuit["track_abrasiveness"],
            "average_corner_speed": circuit["average_corner_speed"],
            "race_laps": race_laps,
            # Optional model inputs/scenario assumptions:
            "yellow_laps": yellow_laps, "sc_laps": sc_laps,
            "vsc_laps": vsc_laps, "redflag_laps": redflag_laps,
            # Provenance/diagnostics for logs and later SLM explanations:
            "race_date": race_dt.date().isoformat(),
            "race_time": race_dt.strftime("%H:%M"),
            "weather_source": weather["weather_source"],
            "weather_fallback_used": bool(weather.get("weather_fallback_used", False)),
            "air_temp_source": weather.get("air_temp_source", weather["weather_source"]),
            "track_temp_source": weather.get("track_temp_source", "historical_gp_median_fallback"),
            "humidity": weather.get("humidity"),
            "pressure": weather.get("pressure"),
            "precipitation": weather.get("precipitation"),
            "wind_speed": weather.get("wind_speed"),
            "forecast_timestamp": weather.get("forecast_timestamp"),
            "weather_timezone": weather.get("weather_timezone"),
            "weather_latitude": latitude, "weather_longitude": longitude,
            "fallback_note": weather.get("fallback_note"),
            "forecast_error": weather.get("forecast_error"),
        }
        self._validate_context(context)
        return context

    @staticmethod
    def _validate_context(context: Mapping[str, Any]) -> None:
        required = {
            "gp_name", "air_temp", "track_temp", "season", "circuit_type",
            "track_length", "track_abrasiveness", "average_corner_speed", "race_laps",
        }
        missing = required.difference(context)
        if missing:
            raise ValueError(f"Context missing required keys: {sorted(missing)}")
        for key in ("air_temp", "track_temp", "track_length", "average_corner_speed"):
            if not isinstance(context[key], (int, float)) or not pd.notna(context[key]):
                raise ValueError(f"Context field {key} must be numeric.")
        if int(context["race_laps"]) <= 0:
            raise ValueError("race_laps must be positive.")


def get_user_race_context() -> dict[str, Any]:
    gp = input("Grand Prix (use name from config/metadata): ").strip()
    season = int(input("Season (e.g. 2027): ").strip())
    race_date = input("Scheduled race date (YYYY-MM-DD): ").strip()
    race_time = input("Scheduled local start time (HH:MM; default 15:00): ").strip()
    return RaceContextBuilder().build_context(gp, season, race_date, race_time or None)


if __name__ == "__main__":
    try:
        context = get_user_race_context()
        print("\nRace context built successfully:")
        for key, value in context.items():
            print(f"{key}: {value}")
    except Exception as exc:
        raise SystemExit(f"Race Context Builder failed: {exc}")
