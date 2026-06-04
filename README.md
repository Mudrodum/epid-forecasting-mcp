# EpidForecasting MCP Server

A compact FastMCP server for weekly influenza forecasting in Saint Petersburg. The server is designed for CoScientist as an agent-facing analytical capability: the model requests an epidemiological forecasting run rather than coordinating internal model-training stages.

## Integration pattern

This implementation follows the existing CoScientist MCP-server conventions:

- the public MCP surface is small, with one descriptive tool and one computational tool;
- computational artifacts are written to S3-compatible storage under a `user_id/session_id/...` prefix;
- the MCP response includes the compact numerical result inline for immediate agent use;
- downloadable result files are exposed through temporary presigned URLs generated server-side, so the client never receives permanent S3 credentials.

## Public MCP tools

### `describe_influenza_dataset()`

Returns a compact dataset description: target variable, row count, date range, four-week horizon, and missing-value status.

Parameters: none.

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
│   ├── config.py
│   ├── features.py
│   ├── modeling.py
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
```

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
