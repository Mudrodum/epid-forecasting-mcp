import pytest

from epid_forecasting.influenza_db import (
    InfluenzaDbRequest,
    InfluenzaDbSettings,
    compare_age_groups,
    fetch_influenza_db_bundle,
    list_supported_cities,
)


CSV_TEXT = """YEAR|WEEK|REGION_NAME|DISTRICT_NAME|ARI_TOTAL|ARI_0_2|ARI_3_6|ARI_7_14|ARI_15_64|ARI_65|POP_TOTAL|POP_0_2|POP_3_6|POP_7_14|POP_15_64|POP_65|SWB_TOTAL|A_TOTAL|PDM_TOTAL|H3_TOTAL|B_TOTAL
2025|40|СПб|СПб|1000|100|150|250|400|100|1000000|50000|70000|100000|650000|130000|100|10|20|30|40
2025|41|СПб|СПб|2000|200|300|800|500|200|1000000|50000|70000|100000|650000|130000|100|10|20|30|40
2025|42|СПб|СПб|1200|120|200|360|400|120|1000000|50000|70000|100000|650000|130000|100|10|20|30|40
"""


class FakeResponse:
    content = CSV_TEXT.encode("utf-8")

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return FakeResponse()


def test_city_registry_contains_spb():
    cities = list_supported_cities()
    assert any(city["slug"] == "spb" and city["api_id"] == 38 for city in cities)


def test_fetch_influenza_db_bundle_redacts_auth_and_builds_age_groups():
    session = FakeSession()
    settings = InfluenzaDbSettings(auth_token="secret-token", timeout_seconds=12)
    request = InfluenzaDbRequest(city="spb", begin_year=2025, begin_week=40, end_year=2025, end_week=42)

    bundle = fetch_influenza_db_bundle(request, settings, session=session)

    assert "auth=secret-token" in session.calls[0][0]
    assert "secret-token" not in bundle.redacted_source_url
    assert len(bundle.weekly) == 3
    assert set(bundle.age_groups["age_group"]) == {"0-2", "3-6", "7-14", "15-64", "65+"}
    assert bundle.weekly.loc[0, "inc_per_10k"] == pytest.approx(10.0)


def test_compare_age_groups_returns_ranked_season_summary():
    settings = InfluenzaDbSettings(auth_token="secret-token")
    request = InfluenzaDbRequest(city="spb", begin_year=2025, begin_week=40, end_year=2025, end_week=42)
    bundle = fetch_influenza_db_bundle(request, settings, session=FakeSession())

    result = compare_age_groups(bundle.age_groups, season="2025-2026")

    assert result["season"] == "2025-2026"
    assert len(result["age_group_summary"]) == 5
    first = result["age_group_summary"][0]
    assert "rank_by_seasonal_burden" in first
    assert "peak_width_weeks_at_fraction" in first
