# EpidForecasting MCP Server

A FastMCP server for weekly influenza forecasting in Saint Petersburg using the static dataset bundled with the server. It follows the CoScientist MCP server pattern: isolated server directory, tools declared with `@mcp.tool()`, and an HTTP `/mcp` endpoint.

## Capabilities

- Builds a supervised weekly forecasting table for horizons `h = 1..4`.
- Trains one `HistGradientBoostingRegressor` model per forecast horizon using `loss="poisson"`.
- Computes split-conformal prediction intervals with nominal 80% coverage by default.
- Generates forecasts from the latest feature-ready dataset week or an explicitly supplied historical origin date.
- Publishes exported CSV/JSON results to S3-compatible object storage for cross-service consumption.

## Project layout

```text
mcp-servers/epid-forecasting-mcp-server/
├── data/
│   └── influenza_weather_spb_dataset.csv
├── epid_forecasting/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── modeling.py
│   ├── service.py
│   └── storage.py
├── tests/
│   └── test_core.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── epid_forecasting_server.py
└── pyproject.toml
```

## Output contract

Every tool returns a JSON object with a short summary and structured metadata:

```json
{
  "answer": "Human-readable summary.",
  "metadata": {}
}
```

Small structured results are returned inline in `metadata`. Materialized export files are uploaded to S3-compatible storage and returned as `s3://...` URIs.

## MCP tools

### `describe_dataset()`

Returns dataset columns, row count, date range, target variable, missing values, and numeric summary inline in JSON.

### `get_feature_schema()`

Returns the engineered feature list, groups, dtypes, and missing counts after the valid-row filter inline in JSON.

### `train_forecast_models(test_weeks=52, calib_weeks=52, alpha=0.20, target_transform="none", persist_artifacts=True)`

Trains direct multi-step GBDT models and conformal intervals. The optional persisted model bundle is internal local server state; exported user-facing result files are handled by `export_forecast_results`.

### `forecast_next_4_weeks(origin_date=None)`

Returns a four-row inline forecast with:

- `origin_date`
- `target_date`
- `horizon_weeks`
- `inc_per_10k_pred`
- `pi80_lower`
- `pi80_upper`
- `pi80_width`

When `origin_date` is omitted, the latest dataset week with complete engineered features is used. When provided, it must exactly match an available feature-ready dataset week.

### `backtest_forecast_models()`

Returns point and interval metrics for train-only and train-plus-calibration refit models inline in JSON.

### `get_model_registry()`

Returns a machine-readable registry of the four trained direct horizon models inline in JSON.

### `export_forecast_results(user_id, session_id, origin_date=None, history_tail_n=40)`

Requires S3 configuration. It creates and uploads:

- `metrics_summary.csv`
- `test_predictions.csv`
- `forecast_next_4w.csv`
- `feature_list.csv`
- `history_plus_forecast_40.csv`
- `model_registry.json`

The object key convention is:

```text
<user_id>/<session_id>/epid_forecasting/<run_id>/<file_name>
```

Example output metadata:

```json
{
  "storage": "s3",
  "run_id": "<uuid>",
  "forecast_origin_date": "2026-04-27",
  "artifacts": {
    "forecast": "s3://<bucket>/<user_id>/<session_id>/epid_forecasting/<run_id>/forecast_next_4w.csv",
    "metrics": "s3://<bucket>/<user_id>/<session_id>/epid_forecasting/<run_id>/metrics_summary.csv"
  }
}
```

`export_forecast_results` raises a configuration error when S3 variables are not provided; it does not silently substitute container-local output paths.

## Environment variables

Copy the example file only for local execution; never commit populated credentials:

```bash
cp .env.example .env
```

```env
EPID_DATA_PATH=/app/data/influenza_weather_spb_dataset.csv
EPID_ARTIFACT_DIR=/app/artifacts

# Required only for export_forecast_results(...)
S3_ENDPOINT_URL=
S3_BUCKET_NAME=
S3_ACCESS_KEY=
S3_SECRET_KEY=
```

The inspection, training, forecast, backtest, and registry tools do not require S3 credentials. Only file export requires them.

## Local run

```bash
cd mcp-servers/epid-forecasting-mcp-server
cp .env.example .env
uv sync
uv run python epid_forecasting_server.py
```

The HTTP MCP endpoint is:

```text
http://localhost:7331/mcp
```

## Docker run

From the CoScientist repository root:

```bash
docker build \
  -f mcp-servers/epid-forecasting-mcp-server/Dockerfile \
  -t epid-forecasting-mcp-server .

docker run --rm --env-file mcp-servers/epid-forecasting-mcp-server/.env \
  -p 7334:7331 epid-forecasting-mcp-server
```

The externally exposed endpoint is:

```text
http://localhost:7334/mcp
```

## Docker Compose snippet

Add this service to `mcp-servers/docker-compose.yml`:

```yaml
epid-forecasting-mcp-server:
  build:
    context: ..
    dockerfile: mcp-servers/epid-forecasting-mcp-server/Dockerfile
  container_name: epid-forecasting-mcp-server
  env_file:
    - ./epid-forecasting-mcp-server/.env
  environment:
    PYTHONUNBUFFERED: "1"
  ports:
    - "7334:7331"
  volumes:
    - ./epid-forecasting-mcp-server/artifacts:/app/artifacts
  restart: unless-stopped
```

## Test

```bash
cd mcp-servers/epid-forecasting-mcp-server
uv run pytest
```
