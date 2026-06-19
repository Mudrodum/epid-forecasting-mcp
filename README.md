# EpidForecasting MCP Server

A compact FastMCP server for weekly influenza forecasting in Saint Petersburg. The server is designed for CoScientist as an agent-facing analytical capability: the model requests an epidemiological forecasting run rather than coordinating internal model-training stages.

## Integration pattern

This implementation follows the existing CoScientist MCP-server conventions:

- the public MCP surface includes forecasting tools, NII influenza DB access, Open-Meteo weather-source export, age-group comparison, epidemic-wave comparison, SHAP explainability, and bulletin-context preparation tools;
- computational artifacts are written to S3-compatible storage under a `user_id/session_id/...` prefix;
- the MCP response includes the compact numerical result inline for immediate agent use;
- downloadable result files are exposed through temporary presigned URLs generated server-side, so the client never receives permanent S3 credentials.

## Public MCP tools

### `describe_influenza_dataset()`

Returns a compact dataset description: target variable, row count, date range, four-week horizon, and missing-value status.

Parameters: none.

### `list_influenza_db_cities()`

Returns the city registry supported by the NII influenza DB CSV endpoint. No database token is required for this tool.

Parameters: none.

### `export_influenza_db_dataset(session_id, user_id, city="spb", begin_year=2011, begin_week=1, end_year=None, end_week=None)`

Exports raw NII influenza DB surveillance data as normalized S3-compatible artifacts and returns compact metadata inline.

Generated S3 artifacts:

```text
user_id/session_id/epid_forecasting/influenza_db/<run_id>/weekly.csv
user_id/session_id/epid_forecasting/influenza_db/<run_id>/cases.csv
user_id/session_id/epid_forecasting/influenza_db/<run_id>/age_groups.csv
user_id/session_id/epid_forecasting/influenza_db/<run_id>/summary.json
```

The database token is read server-side from `INFLUENZA_DB_AUTH_TOKEN` or its aliases and is never returned in tool metadata.

### `export_weather_source_dataset(session_id, user_id, city="spb", start_date="2023-01-01", end_date="2026-05-31", latitude=None, longitude=None, timezone=None)`

Fetches weather covariates from the Open-Meteo Archive API, normalizes hourly `temperature_2m` and `relative_humidity_2m` to `time/temp/rh`, aggregates to Monday-anchored weekly covariates, and uploads artifacts to S3-compatible storage.

Generated S3 artifacts:

```text
user_id/session_id/epid_forecasting/weather/<run_id>/weather_hourly.csv
user_id/session_id/epid_forecasting/weather/<run_id>/weather_weekly.csv
user_id/session_id/epid_forecasting/weather/<run_id>/weather_location.json
user_id/session_id/epid_forecasting/weather/<run_id>/weather_summary.json
```

The built-in presets cover `spb`, `moscow`, `novosibirsk`, `yekaterinburg`, and `krasnodar`; other cities are resolved through Open-Meteo geocoding unless explicit latitude/longitude/timezone are supplied.

### `compare_influenza_age_groups_from_db(city="spb", season=None, begin_year=2011, begin_week=1, end_year=None, end_week=None, peak_width_fraction=0.5)`

Fetches NII influenza DB data and compares non-overlapping age groups for a selected epidemic season. If `season` is omitted, the latest available season in the fetched dataset is used.

Compared age groups:

```text
0-2, 3-6, 7-14, 15-64, 65+
```

Returned metrics include total cases, mean and median incidence per 10,000, seasonal burden, peak incidence, peak date, ISO week of peak, FWHM-like peak width, and ranks by burden and peak.


### `generate_br_model_forecast(session_id, user_id, city="spb", begin_year=2024, begin_week=40, end_year=None, end_week=None, forecast_type="total", method="mcmc", forecast_duration_weeks=4, ...)`

Runs the compact Baroyan-Rvachev-style mechanistic model as an **auxiliary** analysis. It calibrates alpha/beta parameters from normalized NII influenza DB data, produces a fitted trajectory plus calibration-sample interval, and uploads tables/figures to S3. With `method="mcmc"`, uncertainty draws target a conditional Gaussian log1p-residual pseudo-posterior using the same residual objective as the deterministic optimizer; non-MCMC methods export explicitly labelled objective-based calibration samples.

`forecast_type="total"` produces an aggregate model; `forecast_type="age"` uses two model groups, `0-14` and `15+`. The supported calibration methods are `mcmc`, `abc`, `annealing`, and `optuna` (compatibility alias to the deterministic optimization path).

### `estimate_br_model_parameters(session_id, user_id, city="spb", ..., forecast_type="total", method="mcmc")`

Runs the same BR calibration and exports alpha/beta parameter distributions. The `parameter_samples.csv` artifact contains a marked `optimizer_best_fit_reference` row plus `uncertainty_draw` rows, so the deterministic optimum and the uncertainty sample set are distinguishable. Alpha and beta are fitted calibration parameters, not direct biological measurements. The compact BR implementation does **not** estimate a separate gamma parameter: infectiousness duration is represented by its fixed kernel. If an optimizer parameter reaches a configured bound, the result explicitly flags the fit as constrained or weakly identified.

### `compute_forecast_shap_explainability(session_id, user_id, origin_date=None, max_test_samples=64, background_size=128, top_features_per_horizon=8, worst_cases_per_horizon=5, horizons=None)`

Computes SHAP forecast-driver explanations for the fixed bundled Saint Petersburg forecasting workflow and uploads the explanation artifacts to S3-compatible storage. It uses SHAP TreeExplainer when available and falls back to a permutation SHAP explainer for compatibility with the installed sklearn/SHAP versions.

Generated S3 artifacts:

```text
user_id/session_id/epid_forecasting/shap/<run_id>/shap_global_importance.csv
user_id/session_id/epid_forecasting/shap/<run_id>/shap_local_values.csv
user_id/session_id/epid_forecasting/shap/<run_id>/shap_worst_cases.csv
user_id/session_id/epid_forecasting/shap/<run_id>/shap_summary.json
```

### `prepare_influenza_bulletin_context(session_id, user_id, city="spb", ..., forecast_engine="gbdt", include_weather=True, include_forecast=True, include_shap=True, ...)`

Prepares a structured evidence packet for external bulletin writing. The tool does **not** call an LLM and does not produce final prose. The default `forecast_engine="gbdt"` fetches NII influenza DB data, loads aligned Open-Meteo weather, builds the merged weekly modeling table, computes a four-week GBDT forecast with split-conformal uncertainty, computes SHAP forecast-driver evidence, and includes wave and age-group comparisons.

When the caller explicitly sets `forecast_engine="br"`, the tool runs the compact Baroyan-Rvachev-style mechanistic calibration **instead of** GBDT. In BR mode SHAP is not calculated or included. The inline context replaces the SHAP section with `mechanistic_model_interpretation`, containing alpha/beta summaries, calibration diagnostics, parameter meanings, and an explicit note that gamma is not separately estimated in this compact implementation.

Generated S3 artifacts always include:

```text
user_id/session_id/epid_forecasting/bulletin_context/<run_id>/bulletin_context.json
user_id/session_id/epid_forecasting/bulletin_context/<run_id>/bulletin_context.md
user_id/session_id/epid_forecasting/bulletin_context/<run_id>/weekly.csv
user_id/session_id/epid_forecasting/bulletin_context/<run_id>/age_groups.csv
```

GBDT mode can additionally include:

```text
weather_hourly.csv
weather_weekly.csv
merged_weekly.csv
shap_global_importance.csv
shap_local_values.csv
shap_worst_cases.csv
```

BR mode can additionally include:

```text
br_trajectory.csv
br_parameter_samples.csv
br_summary.json
br_forecast_ru.png
br_forecast_ru.pdf
br_alpha_distribution.png
br_alpha_distribution.pdf
br_beta_distribution.png
br_beta_distribution.pdf
```

The returned metadata includes the complete compact `bulletin_context` inline, so an MCP client can immediately write a bulletin from the computed evidence. No separate prompt is returned. The full JSON/Markdown packet and tabular artifacts remain in S3 for audit and reuse. External prose must use only the inline context and referenced artifacts; the packet contains its own `writing_constraints` and `limitations`.

### `run_influenza_forecasting(session_id, user_id, origin_date=None)`

Runs the complete fixed four-week workflow:

1. build a supervised weekly feature table for horizons `h = 1..4`;
2. fit direct multi-step `HistGradientBoostingRegressor` models;
3. calculate split-conformal intervals for untouched holdout evaluation;
4. refit point-forecast models on all labelled feature-complete observations;
5. return holdout metrics and the four-week forecast inline;
6. upload output artifacts to S3-compatible storage and return temporary download links.

Parameters:

- `session_id` — session identifier used in the artifact prefix;
- `user_id` — user identifier used in the artifact prefix;
- `origin_date` — optional observed dataset week in `YYYY-MM-DD` format; omitted by default to use the latest feature-complete observation.

Model-tuning parameters, split sizes, interval settings, output paths, and storage credentials are not exposed through MCP. They are controlled by the server configuration.

## Output contract

The agent receives forecast values and validation metrics inline. Full result files are saved under:

```text
user_id/session_id/epid_forecasting/<run_id>/forecast.csv
user_id/session_id/epid_forecasting/<run_id>/metrics.json
user_id/session_id/epid_forecasting/<run_id>/run_summary.json
```

The response metadata contains an S3 URI and a temporary presigned download URL for each artifact:

```json
{
  "result_delivery": {
    "mode": "inline_summary_plus_s3_artifacts",
    "storage": "s3_compatible",
    "authentication": "server_side_s3_credentials",
    "client_download_access": "temporary_presigned_urls"
  },
  "run_id": "<uuid>",
  "storage_prefix": "<user_id>/<session_id>/epid_forecasting/<run_id>",
  "download_access": "presigned_urls",
  "presigned_url_expiration_seconds": 3600,
  "artifacts": {
    "forecast": {
      "s3_uri": "s3://<bucket>/<prefix>/forecast.csv",
      "download_url": "<temporary-url>"
    }
  }
}
```

S3 access keys are read only by the server. They are not passed to the LLM client.

## Analytical interpretation of intervals

Nominal split-conformal coverage applies to the untouched holdout evaluation. The point forecast for future weeks is produced after refitting on all labelled feature-complete observations. Therefore, its accompanying bounds are labelled `calibration-derived uncertainty bounds`; the server does not claim a new nominal coverage guarantee for the refitted production model.

## Project layout

```text
mcp-servers/epid-forecasting-mcp-server/
├── data/
│   └── influenza_weather_spb_dataset.csv
├── epid_forecasting/
│   ├── __init__.py
│   ├── bulletin_context.py
│   ├── config.py
│   ├── features.py
│   ├── influenza_db.py
│   ├── modeling.py
│   ├── weather_source.py
│   ├── explainability.py
│   ├── seasonal_analysis.py
│   ├── service.py
│   └── storage.py
├── tests/
│   ├── test_core.py
│   └── test_storage.py
├── .env.example
├── Dockerfile
├── README.md
├── epid_forecasting_server.py
└── pyproject.toml
```

## Environment variables

Create `.env` from `.env.example` and provide S3-compatible storage settings:

```bash
EPID_DATA_PATH=/app/data/influenza_weather_spb_dataset.csv
ENDPOINT_URL=http://localhost:9000
ACCESS_KEY=your-access-key
SECRET_KEY=your-secret-key
BUCKET_NAME=your-bucket-name
PRESIGNED_URL_EXPIRATION_SECONDS=3600
INFLUENZA_DB_AUTH_TOKEN=your-influenza-db-token
INFLUENZA_DB_TIMEOUT_SECONDS=60
# No Open-Meteo key is required.
```

`INFLUENZA_DB_AUTH_TOKEN` is required only for DB-backed tools. The server also accepts `NIIGRIP_DB_AUTH_TOKEN` and `INFLUENZA_DB_KEY` as aliases. Do not commit real DB tokens.

`ENDPOINT_URL`, `ACCESS_KEY`, `SECRET_KEY`, and `BUCKET_NAME` follow the names used by `dataset-collection-mcp-server`. The server also accepts `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, and `S3_BUCKET_NAME` as aliases for compatibility with the chemical MCP-server style.

## Run with uv

```bash
cd mcp-servers/epid-forecasting-mcp-server
cp .env.example .env
# Edit .env with actual object-storage credentials.
set -a && source .env && set +a
uv sync --frozen
uv run python epid_forecasting_server.py
```

The MCP endpoint is:

```text
http://localhost:7331/mcp
```

## Run with Docker Compose

When added as the next service in the repository-level MCP compose file, the external endpoint may use port `7335`, after the existing services on ports `7331` to `7334`:

```bash
docker compose -f mcp-servers/docker-compose.epid-forecasting.yml up --build
```

External endpoint:

```text
http://localhost:7335/mcp
```

## Test

```bash
cd mcp-servers/epid-forecasting-mcp-server
uv run --frozen pytest
```

Current validation status for this patch set: `18 passed`.
