from epid_forecasting.config import DEFAULT_DATA_PATH
from epid_forecasting.service import EpidForecastingService


def test_compare_epidemic_waves_returns_recent_wave_payload():
    service = EpidForecastingService(data_path=DEFAULT_DATA_PATH)
    payload = service.compare_epidemic_waves(n_last_seasons=3)

    assert payload["comparison_mode"] == "recent_epidemic_waves"
    assert len(payload["waves"]) == 3
    assert len(payload["latest_vs_previous"]) == 2
    assert set(payload["waves"][0]) >= {
        "season_label",
        "peak_week",
        "peak_date",
        "peak_value",
        "wave_status",
        "season_burden",
    }
