"""Influenza DB access and age-group analytical utilities.

The module talks to the NII influenza CSV report endpoint using a runtime
authentication token. No token value is stored in source code or returned in MCP
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
import os
import re
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

try:  # Network access is required only for DB-backed tools.
    import requests
except ImportError:  # pragma: no cover - exercised only in incorrectly installed envs.
    requests = None  # type: ignore[assignment]


INFLUENZA_DB_BASE_URL = "https://db.influenza.spb.ru/scripts/report/rmancgi.exe"
INFLUENZA_DB_REPORT_NAME = "get_csv"
INFLUENZA_DB_REPORT_ID = "aripcr"

RAW_INFLUENZA_COLUMNS: dict[str, str] = {
    "YEAR": "datetime",
    "REGION_NAME": "region_name",
    "DISTRICT_NAME": "district_name",
    "ARI_TOTAL": "sars_total_cases",
    "ARI_0_2": "sars_cases_age_group_0",
    "ARI_3_6": "sars_cases_age_group_1",
    "ARI_7_14": "sars_cases_age_group_2",
    "ARI_15_64": "sars_cases_age_group_4",
    "ARI_65": "sars_cases_age_group_5",
    "POP_TOTAL": "total_population",
    "POP_0_2": "population_age_group_0",
    "POP_3_6": "population_age_group_1",
    "POP_7_14": "population_age_group_2",
    "POP_15_64": "population_age_group_4",
    "POP_65": "population_age_group_5",
    "SWB_TOTAL": "tested_total",
    "A_TOTAL": "tested_strain_0",
    "PDM_TOTAL": "tested_strain_1",
    "H3_TOTAL": "tested_strain_2",
    "B_TOTAL": "tested_strain_3",
}

AGE_GROUP_COLUMNS: dict[str, tuple[str, str]] = {
    "0-2": ("sars_cases_age_group_0", "population_age_group_0"),
    "3-6": ("sars_cases_age_group_1", "population_age_group_1"),
    "7-14": ("sars_cases_age_group_2", "population_age_group_2"),
    "15-64": ("sars_cases_age_group_4", "population_age_group_4"),
    "65+": ("sars_cases_age_group_5", "population_age_group_5"),
}

STRAIN_INDICES: tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True)
class CityInfo:
    """Supported influenza DB city descriptor."""

    slug: str
    api_id: int
    name_ru: str


CITY_REGISTRY: dict[str, CityInfo] = {
    "russia": CityInfo("russia", 0, "Россия"),
    "birobidzhan": CityInfo("birobidzhan", 7, "Биробиджан"),
    "arkhangelsk": CityInfo("arkhangelsk", 9, "Архангельск"),
    "astrakhan": CityInfo("astrakhan", 10, "Астрахань"),
    "barnaul": CityInfo("barnaul", 11, "Барнаул"),
    "orenburg": CityInfo("orenburg", 12, "Оренбург"),
    "vladivostok": CityInfo("vladivostok", 13, "Владивосток"),
    "volgograd": CityInfo("volgograd", 14, "Волгоград"),
    "voronezh": CityInfo("voronezh", 15, "Воронеж"),
    "nizhny_novgorod": CityInfo("nizhny_novgorod", 16, "Нижний Новгород"),
    "irkutsk": CityInfo("irkutsk", 19, "Иркутск"),
    "kaliningrad": CityInfo("kaliningrad", 20, "Калининград"),
    "murmansk": CityInfo("murmansk", 21, "Мурманск"),
    "novosibirsk": CityInfo("novosibirsk", 22, "Новосибирск"),
    "saratov": CityInfo("saratov", 24, "Саратов"),
    "khabarovsk": CityInfo("khabarovsk", 26, "Хабаровск"),
    "moscow": CityInfo("moscow", 32, "Москва"),
    "tomsk": CityInfo("tomsk", 34, "Томск"),
    "vladimir": CityInfo("vladimir", 36, "Владимир"),
    "spb": CityInfo("spb", 38, "Санкт-Петербург"),
    "yaroslavl": CityInfo("yaroslavl", 40, "Ярославль"),
    "kazan": CityInfo("kazan", 41, "Казань"),
    "kemerovo": CityInfo("kemerovo", 43, "Кемерово"),
    "kirov": CityInfo("kirov", 44, "Киров"),
    "cheboksary": CityInfo("cheboksary", 45, "Чебоксары"),
    "magadan": CityInfo("magadan", 46, "Магадан"),
    "norilsk": CityInfo("norilsk", 47, "Норильск"),
    "vladikavkaz": CityInfo("vladikavkaz", 48, "Владикавказ"),
    "perm": CityInfo("perm", 49, "Пермь"),
    "petropavlovsk": CityInfo("petropavlovsk", 50, "Петропавловск"),
    "rostov_na_donu": CityInfo("rostov_na_donu", 51, "Ростов-на-Дону"),
    "smolensk": CityInfo("smolensk", 53, "Смоленск"),
    "stavropol": CityInfo("stavropol", 54, "Ставрополь"),
    "ulan_ude": CityInfo("ulan_ude", 55, "Улан-Удэ"),
    "ufa": CityInfo("ufa", 56, "Уфа"),
    "chelyabinsk": CityInfo("chelyabinsk", 57, "Челябинск"),
    "yakutsk": CityInfo("yakutsk", 58, "Якутск"),
    "chita": CityInfo("chita", 59, "Чита"),
    "yuzhno_sakhalinsk": CityInfo("yuzhno_sakhalinsk", 60, "Южно-Сахалинск"),
    "krasnodar": CityInfo("krasnodar", 61, "Краснодар"),
    "krasnoyarsk": CityInfo("krasnoyarsk", 62, "Красноярск"),
    "samara": CityInfo("samara", 63, "Самара"),
    "omsk": CityInfo("omsk", 64, "Омск"),
    "yekaterinburg": CityInfo("yekaterinburg", 68, "Екатеринбург"),
    "pskov": CityInfo("pskov", 69, "Псков"),
    "petrozavodsk": CityInfo("petrozavodsk", 70, "Петрозаводск"),
    "lipetsk": CityInfo("lipetsk", 71, "Липецк"),
    "izhevsk": CityInfo("izhevsk", 72, "Ижевск"),
    "tula": CityInfo("tula", 73, "Тула"),
    "ulyanovsk": CityInfo("ulyanovsk", 74, "Ульяновск"),
    "bryansk": CityInfo("bryansk", 75, "Брянск"),
    "vologda": CityInfo("vologda", 76, "Вологда"),
    "syktyvkar": CityInfo("syktyvkar", 77, "Сыктывкар"),
    "orel": CityInfo("orel", 78, "Орёл"),
    "ryazan": CityInfo("ryazan", 79, "Рязань"),
    "tver": CityInfo("tver", 80, "Тверь"),
    "belgorod": CityInfo("belgorod", 81, "Белгород"),
    "kursk": CityInfo("kursk", 82, "Курск"),
    "cherepovets": CityInfo("cherepovets", 83, "Череповец"),
    "penza": CityInfo("penza", 84, "Пенза"),
    "veliky_novgorod": CityInfo("veliky_novgorod", 85, "Великий Новгород"),
    "simferopol": CityInfo("simferopol", 91, "Симферополь"),
    "sevastopol": CityInfo("sevastopol", 92, "Севастополь"),
    "donetsk": CityInfo("donetsk", 102, "Донецк"),
    "lugansk": CityInfo("lugansk", 103, "Луганск"),
    "kherson": CityInfo("kherson", 104, "Херсон"),
    "zaporizhzhia": CityInfo("zaporizhzhia", 105, "Запорожье"),
}

_CITY_ALIASES = {
    "санкт-петербург": "spb",
    "санкт петербург": "spb",
    "петербург": "spb",
    "спб": "spb",
    "москва": "moscow",
    "екатеринбург": "yekaterinburg",
    "новосибирск": "novosibirsk",
    "нижний новгород": "nizhny_novgorod",
    "ростов-на-дону": "rostov_na_donu",
    "ростов на дону": "rostov_na_donu",
    "россия": "russia",
}


class InfluenzaDbError(RuntimeError):
    """Raised when the influenza DB source cannot be queried or normalized."""


@dataclass(frozen=True)
class InfluenzaDbSettings:
    """Runtime settings for the NII influenza CSV report endpoint."""

    auth_token: str
    base_url: str = INFLUENZA_DB_BASE_URL
    report_name: str = INFLUENZA_DB_REPORT_NAME
    report_id: str = INFLUENZA_DB_REPORT_ID
    timeout_seconds: float = 60.0
    encoding: str = "utf-8"
    separator: str = "|"

    @classmethod
    def from_env(cls) -> "InfluenzaDbSettings":
        token = (
            os.getenv("INFLUENZA_DB_AUTH_TOKEN")
            or os.getenv("NIIGRIP_DB_AUTH_TOKEN")
            or os.getenv("INFLUENZA_DB_KEY")
        )
        if not token:
            raise RuntimeError(
                "Influenza DB access requires INFLUENZA_DB_AUTH_TOKEN "
                "(aliases: NIIGRIP_DB_AUTH_TOKEN, INFLUENZA_DB_KEY)."
            )
        timeout_text = os.getenv("INFLUENZA_DB_TIMEOUT_SECONDS", "60")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise RuntimeError("INFLUENZA_DB_TIMEOUT_SECONDS must be numeric.") from exc
        if timeout <= 0:
            raise RuntimeError("INFLUENZA_DB_TIMEOUT_SECONDS must be positive.")
        return cls(
            auth_token=token,
            base_url=os.getenv("INFLUENZA_DB_BASE_URL", INFLUENZA_DB_BASE_URL),
            report_name=os.getenv("INFLUENZA_DB_REPORT_NAME", INFLUENZA_DB_REPORT_NAME),
            report_id=os.getenv("INFLUENZA_DB_REPORT_ID", INFLUENZA_DB_REPORT_ID),
            timeout_seconds=timeout,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def safe_city_slug(city: str) -> str:
    slug = str(city).strip().lower().replace("ё", "е")
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^0-9a-zA-Zа-яА-Я_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError("city must not be empty.")
    return slug


def normalize_city_slug(city: str) -> str:
    raw = str(city).strip().lower().replace("ё", "е")
    raw = re.sub(r"\s+", " ", raw)
    if raw in _CITY_ALIASES:
        return _CITY_ALIASES[raw]
    slug = safe_city_slug(raw)
    if slug in CITY_REGISTRY:
        return slug
    for info in CITY_REGISTRY.values():
        if raw == info.name_ru.lower().replace("ё", "е"):
            return info.slug
    return slug


def resolve_city(city: str) -> CityInfo:
    slug = normalize_city_slug(city)
    try:
        return CITY_REGISTRY[slug]
    except KeyError as exc:
        available = ", ".join(sorted(CITY_REGISTRY))
        raise ValueError(f"Unknown city {city!r}. Available city slugs: {available}.") from exc


def list_supported_cities() -> list[dict[str, Any]]:
    return [
        {"slug": info.slug, "api_id": info.api_id, "name_ru": info.name_ru}
        for info in sorted(CITY_REGISTRY.values(), key=lambda item: item.name_ru)
    ]


def iso_week_start_date(year: int, week: int) -> date:
    try:
        return datetime.strptime(f"{year}-W{week}-1", "%G-W%V-%u").date()
    except ValueError as exc:
        raise ValueError(f"Invalid ISO week: {year} W{week:02d}.") from exc


@dataclass(frozen=True)
class InfluenzaDbRequest:
    city: str
    begin_year: int
    begin_week: int
    end_year: int
    end_week: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "city", normalize_city_slug(self.city))
        if not (1900 <= self.begin_year <= 2100 and 1900 <= self.end_year <= 2100):
            raise ValueError("begin_year and end_year must be in [1900, 2100].")
        if not (1 <= self.begin_week <= 53 and 1 <= self.end_week <= 53):
            raise ValueError("begin_week and end_week must be in [1, 53].")
        if iso_week_start_date(self.end_year, self.end_week) < iso_week_start_date(
            self.begin_year, self.begin_week
        ):
            raise ValueError("The end ISO week cannot be earlier than the begin ISO week.")

    @classmethod
    def until_latest(
        cls,
        city: str,
        *,
        begin_year: int = 2011,
        begin_week: int = 1,
        end_year: int | None = None,
        end_week: int | None = None,
    ) -> "InfluenzaDbRequest":
        if end_year is None or end_week is None:
            iso = date.today().isocalendar()
            end_year = int(iso.year) if end_year is None else end_year
            end_week = int(iso.week) if end_week is None else end_week
        return cls(city=city, begin_year=begin_year, begin_week=begin_week, end_year=end_year, end_week=end_week)


def build_influenza_db_url(request: InfluenzaDbRequest, settings: InfluenzaDbSettings) -> str:
    city_info = resolve_city(request.city)
    query = urlencode(
        {
            "reportname": settings.report_name,
            "id": settings.report_id,
            "byear": request.begin_year,
            "bweek": request.begin_week,
            "eyear": request.end_year,
            "eweek": request.end_week,
            "district": city_info.api_id,
            "auth": settings.auth_token,
        }
    )
    return f"{settings.base_url}?{query}"


def build_redacted_source_url(request: InfluenzaDbRequest, settings: InfluenzaDbSettings) -> str:
    city_info = resolve_city(request.city)
    query = urlencode(
        {
            "reportname": settings.report_name,
            "id": settings.report_id,
            "byear": request.begin_year,
            "bweek": request.begin_week,
            "eyear": request.end_year,
            "eweek": request.end_week,
            "district": city_info.api_id,
            "auth": "<redacted>",
        }
    )
    return f"{settings.base_url}?{query}"


def fetch_influenza_csv_text(
    request: InfluenzaDbRequest,
    settings: InfluenzaDbSettings,
    *,
    session: Any | None = None,
) -> str:
    if requests is None and session is None:
        raise InfluenzaDbError("Package requests is required for influenza DB access.")
    http = session or requests
    response = http.get(build_influenza_db_url(request, settings), timeout=settings.timeout_seconds)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    content = getattr(response, "content", None)
    if content is not None:
        return content.decode(settings.encoding)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    raise InfluenzaDbError("Influenza DB response contains neither content nor text.")


def _date_from_api_row(row: pd.Series) -> datetime:
    return datetime.strptime(f"{int(row['YEAR'])}-W{int(row['WEEK'])}-1", "%G-W%V-%u")


def normalize_influenza_api_frame(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in [*RAW_INFLUENZA_COLUMNS, "WEEK"] if column not in raw.columns]
    if missing:
        raise ValueError(f"Influenza DB response is missing columns: {missing}.")

    df = raw.copy()
    df["YEAR"] = df.apply(_date_from_api_row, axis=1)
    df = df.loc[:, list(RAW_INFLUENZA_COLUMNS)]
    df = df.rename(columns=RAW_INFLUENZA_COLUMNS)
    df["datetime"] = pd.to_datetime(df["datetime"])

    numeric_columns = [col for col in df.columns if col not in {"datetime", "region_name", "district_name"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sars_cases_age_group_3"] = df["sars_cases_age_group_4"] + df["sars_cases_age_group_5"]
    df["population_age_group_3"] = df["population_age_group_4"] + df["population_age_group_5"]

    tested_total = df["tested_total"].replace(0, np.nan)
    for strain_index in STRAIN_INDICES:
        tested_col = f"tested_strain_{strain_index}"
        rel_col = f"rel_strain_{strain_index}"
        real_col = f"real_cases_strain_{strain_index}"
        df[rel_col] = df[tested_col] / tested_total
        df[real_col] = (df[rel_col] * df["sars_total_cases"]).round()

    return df.drop(columns=["tested_total", *[f"tested_strain_{idx}" for idx in STRAIN_INDICES]]).sort_values(
        "datetime"
    ).reset_index(drop=True)


def parse_influenza_api_csv(csv_text: str, separator: str = "|") -> pd.DataFrame:
    if not csv_text.strip():
        raise ValueError("Influenza DB returned an empty CSV response.")
    raw = pd.read_csv(StringIO(csv_text), sep=separator)
    return normalize_influenza_api_frame(raw)


def weekly_incidence_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    df = cases.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    strain_cols = [f"real_cases_strain_{idx}" for idx in STRAIN_INDICES]
    missing = [col for col in strain_cols if col not in df.columns]
    if missing:
        raise ValueError(f"cases frame is missing strain columns: {missing}.")
    df["total_cases_formula"] = df[strain_cols].fillna(0).sum(axis=1)
    df["inc_per_10k"] = df["total_cases_formula"] / df["total_population"] * 10_000
    iso = df["datetime"].dt.isocalendar()
    out = df[["datetime", "total_population", "total_cases_formula", "inc_per_10k"]].copy()
    out.insert(1, "iso_year", iso.year.astype(int))
    out.insert(2, "iso_week", iso.week.astype(int))
    return out.dropna(subset=["datetime", "total_population", "inc_per_10k"]).sort_values("datetime").reset_index(
        drop=True
    )


def age_group_frame_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_cols = ["datetime", "region_name", "district_name"]
    for label, (cases_col, population_col) in AGE_GROUP_COLUMNS.items():
        frame = cases[base_cols].copy()
        frame["age_group"] = label
        frame["cases"] = pd.to_numeric(cases[cases_col], errors="coerce")
        frame["population"] = pd.to_numeric(cases[population_col], errors="coerce")
        frame["inc_per_10k"] = frame["cases"] / frame["population"] * 10_000
        rows.append(frame)
    age_df = pd.concat(rows, ignore_index=True)
    age_df["datetime"] = pd.to_datetime(age_df["datetime"])
    iso = age_df["datetime"].dt.isocalendar()
    age_df["iso_year"] = iso.year.astype(int)
    age_df["iso_week"] = iso.week.astype(int)
    age_df["season"] = age_df["iso_year"].where(age_df["iso_week"] < 40, age_df["iso_year"] + 1)
    age_df["season"] = (age_df["season"] - 1).astype(str) + "-" + age_df["season"].astype(str)
    return age_df.sort_values(["datetime", "age_group"]).reset_index(drop=True)


@dataclass(frozen=True)
class InfluenzaDbBundle:
    request: InfluenzaDbRequest
    city: CityInfo
    raw_cases: pd.DataFrame
    cases: pd.DataFrame
    weekly: pd.DataFrame
    age_groups: pd.DataFrame
    redacted_source_url: str


def fetch_influenza_db_bundle(
    request: InfluenzaDbRequest,
    settings: InfluenzaDbSettings | None = None,
    *,
    session: Any | None = None,
) -> InfluenzaDbBundle:
    cfg = settings or InfluenzaDbSettings.from_env()
    text = fetch_influenza_csv_text(request, cfg, session=session)
    raw = pd.read_csv(StringIO(text), sep=cfg.separator)
    cases = normalize_influenza_api_frame(raw)
    weekly = weekly_incidence_from_cases(cases)
    age_groups = age_group_frame_from_cases(cases)
    return InfluenzaDbBundle(
        request=request,
        city=resolve_city(request.city),
        raw_cases=raw,
        cases=cases,
        weekly=weekly,
        age_groups=age_groups,
        redacted_source_url=build_redacted_source_url(request, cfg),
    )


def summarize_influenza_db_bundle(bundle: InfluenzaDbBundle) -> dict[str, Any]:
    weekly = bundle.weekly
    age_groups = sorted(bundle.age_groups["age_group"].dropna().unique().tolist())
    return _jsonable(
        {
            "city": {"slug": bundle.city.slug, "api_id": bundle.city.api_id, "name_ru": bundle.city.name_ru},
            "request": {
                "begin_year": bundle.request.begin_year,
                "begin_week": bundle.request.begin_week,
                "end_year": bundle.request.end_year,
                "end_week": bundle.request.end_week,
            },
            "source_url": bundle.redacted_source_url,
            "weekly_rows": int(len(weekly)),
            "cases_rows": int(len(bundle.cases)),
            "age_group_rows": int(len(bundle.age_groups)),
            "date_range": {
                "start": weekly["datetime"].min().date().isoformat() if len(weekly) else None,
                "end": weekly["datetime"].max().date().isoformat() if len(weekly) else None,
            },
            "age_groups": age_groups,
            "latest_week": weekly.tail(1),
        }
    )


def compare_age_groups(
    age_groups: pd.DataFrame,
    *,
    season: str | None = None,
    peak_width_fraction: float = 0.5,
) -> dict[str, Any]:
    if not 0 < peak_width_fraction <= 1:
        raise ValueError("peak_width_fraction must be in (0, 1].")
    df = age_groups.copy()
    if df.empty:
        raise ValueError("age-group frame is empty.")
    if season is None:
        season = str(df["season"].dropna().iloc[-1])
    season_df = df.loc[df["season"] == season].copy()
    if season_df.empty:
        available = sorted(df["season"].dropna().unique().tolist())
        raise ValueError(f"Season {season!r} is absent. Available seasons: {available}.")

    summaries: list[dict[str, Any]] = []
    for age_group, group in season_df.groupby("age_group", sort=True):
        group = group.sort_values("datetime")
        peak_idx = group["inc_per_10k"].idxmax()
        peak_value = float(group.loc[peak_idx, "inc_per_10k"])
        threshold = peak_value * peak_width_fraction
        width_weeks = int((group["inc_per_10k"] >= threshold).sum()) if np.isfinite(peak_value) else 0
        summaries.append(
            {
                "season": season,
                "age_group": age_group,
                "weeks": int(len(group)),
                "total_cases": float(group["cases"].fillna(0).sum()),
                "mean_inc_per_10k": float(group["inc_per_10k"].mean()),
                "median_inc_per_10k": float(group["inc_per_10k"].median()),
                "seasonal_burden_inc_week": float(group["inc_per_10k"].fillna(0).sum()),
                "peak_inc_per_10k": peak_value,
                "peak_date": pd.to_datetime(group.loc[peak_idx, "datetime"]).date().isoformat(),
                "peak_iso_week": int(group.loc[peak_idx, "iso_week"]),
                "peak_width_weeks_at_fraction": width_weeks,
                "peak_width_fraction": peak_width_fraction,
            }
        )
    summary = pd.DataFrame(summaries)
    summary["rank_by_seasonal_burden"] = summary["seasonal_burden_inc_week"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary["rank_by_peak"] = summary["peak_inc_per_10k"].rank(ascending=False, method="min").astype(int)
    summary = summary.sort_values(["rank_by_seasonal_burden", "age_group"]).reset_index(drop=True)
    return _jsonable(
        {
            "season": season,
            "available_seasons": sorted(df["season"].dropna().unique().tolist()),
            "age_group_summary": summary,
        }
    )
