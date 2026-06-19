"""Open-Meteo weather-source integration for influenza forecasting workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import re
from typing import Any, Mapping

import pandas as pd

try:  # Network access is required only for weather-backed tools.
    import requests
except ImportError:  # pragma: no cover - exercised only in incorrectly installed envs.
    requests = None  # type: ignore[assignment]

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_FIELDS = "temperature_2m,relative_humidity_2m"


class WeatherSourceError(RuntimeError):
    """Base error for weather-source access and normalization."""


class WeatherLocationError(WeatherSourceError):
    """Raised when a city cannot be resolved to coordinates."""


class WeatherApiError(WeatherSourceError):
    """Raised when Open-Meteo data access fails."""


@dataclass(frozen=True)
class WeatherLocation:
    """Resolved weather-source location."""

    city: str
    query: str
    latitude: float
    longitude: float
    timezone: str
    source: str = "preset"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeatherApiSettings:
    """Open-Meteo API settings."""

    geocoding_url: str = OPEN_METEO_GEOCODING_URL
    archive_url: str = OPEN_METEO_ARCHIVE_URL
    hourly_fields: str = OPEN_METEO_HOURLY_FIELDS
    timeout_seconds: float = 60.0
    geocoding_language: str = "ru"
    geocoding_count: int = 1
    chunk_by_year: bool = True


@dataclass(frozen=True)
class WeatherFrameBundle:
    """Normalized weather-source result."""

    location: WeatherLocation
    hourly: pd.DataFrame
    weekly: pd.DataFrame
    source_url: str
    request: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "source": "Open-Meteo Archive API",
            "source_url": self.source_url,
            "location": self.location.to_dict(),
            "request": self.request,
            "hourly_rows": int(len(self.hourly)),
            "weekly_rows": int(len(self.weekly)),
            "date_range": {
                "start": self.weekly["week_start"].min().date().isoformat() if not self.weekly.empty else None,
                "end": self.weekly["week_start"].max().date().isoformat() if not self.weekly.empty else None,
            },
        }


CITY_WEATHER_PRESETS: dict[str, WeatherLocation] = {
    "spb": WeatherLocation("spb", "Saint Petersburg", 59.9311, 30.3609, "Europe/Moscow"),
    "moscow": WeatherLocation("moscow", "Moscow", 55.7558, 37.6173, "Europe/Moscow"),
    "novosibirsk": WeatherLocation("novosibirsk", "Novosibirsk", 54.9833, 82.8964, "Asia/Novosibirsk"),
    "yekaterinburg": WeatherLocation("yekaterinburg", "Yekaterinburg", 56.8389, 60.6057, "Asia/Yekaterinburg"),
    "krasnodar": WeatherLocation("krasnodar", "Krasnodar", 45.0355, 38.9753, "Europe/Moscow"),
}

_CITY_ALIASES: dict[str, str] = {
    "санкт-петербург": "spb",
    "санкт петербург": "spb",
    "петербург": "spb",
    "спб": "spb",
    "москва": "moscow",
    "новосибирск": "novosibirsk",
    "екатеринбург": "yekaterinburg",
    "краснодар": "krasnodar",
}


def _require_requests() -> None:
    if requests is None:
        raise WeatherApiError("Package requests is required for Open-Meteo weather-source access.")


def _coerce_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def normalize_weather_city_slug(city: str) -> str:
    """Normalize user-supplied city text to a safe slug."""
    raw = str(city).strip().lower().replace("ё", "е")
    if not raw:
        raise WeatherLocationError("city must not be empty.")
    if raw in _CITY_ALIASES:
        return _CITY_ALIASES[raw]
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^0-9a-zа-я_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raise WeatherLocationError("Could not build a weather city slug.")
    return raw


def resolve_weather_location(
    city: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> WeatherLocation:
    """Resolve a city to coordinates and timezone using presets or Open-Meteo geocoding."""
    slug = normalize_weather_city_slug(city)
    query = str(city).strip()
    if latitude is not None or longitude is not None or timezone is not None:
        if latitude is None or longitude is None or timezone is None:
            raise WeatherLocationError("latitude, longitude and timezone must be provided together.")
        return WeatherLocation(slug, query or slug, float(latitude), float(longitude), str(timezone), source="explicit")
    if slug in CITY_WEATHER_PRESETS:
        return CITY_WEATHER_PRESETS[slug]
    return geocode_weather_location(query or slug, city_slug=slug, settings=settings, session=session)


def geocode_weather_location(
    query: str,
    *,
    city_slug: str | None = None,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> WeatherLocation:
    """Resolve coordinates through Open-Meteo Geocoding API."""
    _require_requests()
    cfg = settings or WeatherApiSettings()
    http = session or requests
    response = http.get(
        cfg.geocoding_url,
        params={
            "name": query,
            "count": cfg.geocoding_count,
            "language": cfg.geocoding_language,
            "format": "json",
        },
        timeout=cfg.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise WeatherLocationError(f"Could not resolve coordinates for city: {query!r}.")
    best = results[0]
    if "latitude" not in best or "longitude" not in best:
        raise WeatherLocationError(f"Geocoding response has no coordinates for city: {query!r}.")
    return WeatherLocation(
        city=city_slug or normalize_weather_city_slug(query),
        query=query,
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
        timezone=str(best.get("timezone") or "UTC"),
        source="open_meteo_geocoding",
    )


def normalize_hourly_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize hourly weather to time/temp/rh."""
    columns = {str(col).strip().lower(): str(col) for col in frame.columns}
    aliases = {
        "time": ("time", "datetime", "date", "timestamp"),
        "temp": ("temp", "temperature", "temperature_2m", "t2m", "temp_c"),
        "rh": ("rh", "relative_humidity", "relative_humidity_2m", "humidity"),
    }
    resolved: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in columns:
                resolved[target] = columns[candidate]
                break
        else:
            raise WeatherApiError(f"Could not infer required weather column {target!r}.")
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(frame[resolved["time"]], errors="coerce"),
            "temp": pd.to_numeric(frame[resolved["temp"]], errors="coerce"),
            "rh": pd.to_numeric(frame[resolved["rh"]], errors="coerce"),
        }
    )
    if out["time"].isna().any():
        raise WeatherApiError("Hourly weather contains invalid timestamps.")
    if out[["temp", "rh"]].isna().any().any():
        raise WeatherApiError("Hourly weather contains missing temp/rh values.")
    if out["time"].dt.tz is not None:
        out["time"] = out["time"].dt.tz_localize(None)
    return out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)


def fetch_open_meteo_hourly_chunk(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> pd.DataFrame:
    """Fetch one hourly weather chunk from Open-Meteo Archive API."""
    _require_requests()
    cfg = settings or WeatherApiSettings()
    if end_date < start_date:
        raise WeatherApiError("end_date cannot be earlier than start_date.")
    http = session or requests
    response = http.get(
        cfg.archive_url,
        params={
            "latitude": float(latitude),
            "longitude": float(longitude),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": cfg.hourly_fields,
            "timezone": timezone,
        },
        timeout=cfg.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise WeatherApiError("Open-Meteo response does not contain an hourly object.")
    try:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly["time"]),
                "temp": hourly["temperature_2m"],
                "rh": hourly["relative_humidity_2m"],
            }
        )
    except KeyError as exc:
        raise WeatherApiError(f"Open-Meteo response is missing hourly field: {exc}.") from exc
    return normalize_hourly_weather_frame(frame)


def fetch_open_meteo_hourly(
    *,
    location: WeatherLocation,
    start_date: date,
    end_date: date,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> pd.DataFrame:
    """Fetch hourly weather; by default split requests by calendar year."""
    cfg = settings or WeatherApiSettings()
    if end_date < start_date:
        raise WeatherApiError("end_date cannot be earlier than start_date.")
    if not cfg.chunk_by_year:
        return fetch_open_meteo_hourly_chunk(
            latitude=location.latitude,
            longitude=location.longitude,
            start_date=start_date,
            end_date=end_date,
            timezone=location.timezone,
            settings=cfg,
            session=session,
        )
    parts: list[pd.DataFrame] = []
    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, date(year, 1, 1))
        chunk_end = min(end_date, date(year, 12, 31))
        if chunk_start > chunk_end:
            continue
        parts.append(
            fetch_open_meteo_hourly_chunk(
                latitude=location.latitude,
                longitude=location.longitude,
                start_date=chunk_start,
                end_date=chunk_end,
                timezone=location.timezone,
                settings=cfg,
                session=session,
            )
        )
    if not parts:
        raise WeatherApiError("No weather chunks were fetched.")
    out = pd.concat(parts, ignore_index=True)
    return normalize_hourly_weather_frame(out)


def aggregate_hourly_weather_to_weekly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized hourly weather to Monday-anchored weekly features."""
    df = normalize_hourly_weather_frame(hourly)
    # W-SUN periods start on Monday and end on Sunday; this matches influenza week anchors.
    df["week_start"] = df["time"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        df.groupby("week_start", as_index=False)
        .agg(
            temp_mean=("temp", "mean"),
            temp_max=("temp", "max"),
            temp_min=("temp", "min"),
            rh_mean=("rh", "mean"),
            rh_max=("rh", "max"),
            rh_min=("rh", "min"),
            n_hours=("time", "count"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    return weekly


def load_weather_until_date(
    city: str,
    *,
    start_date: date | datetime | pd.Timestamp | str,
    end_date: date | datetime | pd.Timestamp | str,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> WeatherFrameBundle:
    """Fetch and aggregate Open-Meteo weather for a date interval."""
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    location = resolve_weather_location(
        city,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        settings=settings,
        session=session,
    )
    hourly = fetch_open_meteo_hourly(location=location, start_date=start, end_date=end, settings=settings, session=session)
    weekly = aggregate_hourly_weather_to_weekly(hourly)
    return WeatherFrameBundle(
        location=location,
        hourly=hourly,
        weekly=weekly,
        source_url=(settings or WeatherApiSettings()).archive_url,
        request={
            "city": city,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
        },
    )


def load_weather_aligned_to_influenza(
    city: str,
    influenza_weekly: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    extend_days_after_last_week: int = 6,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    settings: WeatherApiSettings | None = None,
    session: Any | None = None,
) -> WeatherFrameBundle:
    """Fetch Open-Meteo weather covering the influenza weekly table."""
    if datetime_col not in influenza_weekly.columns:
        raise WeatherApiError(f"influenza_weekly is missing {datetime_col!r}.")
    dates = pd.to_datetime(influenza_weekly[datetime_col], errors="coerce")
    if dates.isna().any() or dates.empty:
        raise WeatherApiError("influenza_weekly contains invalid or empty dates.")
    start = dates.min().date()
    end = (dates.max() + pd.Timedelta(days=int(extend_days_after_last_week))).date()
    return load_weather_until_date(
        city,
        start_date=start,
        end_date=end,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        settings=settings,
        session=session,
    )


def merge_influenza_weather_weekly(influenza_weekly: pd.DataFrame, weather_weekly: pd.DataFrame) -> pd.DataFrame:
    """Merge influenza weekly incidence with Monday-anchored weekly weather features."""
    left = influenza_weekly.copy()
    right = weather_weekly.copy()
    left["datetime"] = pd.to_datetime(left["datetime"], errors="coerce").dt.normalize()
    right["week_start"] = pd.to_datetime(right["week_start"], errors="coerce").dt.normalize()
    merged = left.merge(right, left_on="datetime", right_on="week_start", how="left")
    merged = merged.drop(columns=["week_start"], errors="ignore")
    required_weather = ["temp_mean", "temp_max", "temp_min", "rh_mean", "rh_max", "rh_min", "n_hours"]
    missing_values = merged[required_weather].isna().any(axis=1)
    if missing_values.any():
        bad_dates = merged.loc[missing_values, "datetime"].dt.date.astype(str).head(10).tolist()
        raise WeatherApiError(f"Weather data is missing for {int(missing_values.sum())} influenza weeks; examples: {bad_dates}.")
    return merged.sort_values("datetime").reset_index(drop=True)
