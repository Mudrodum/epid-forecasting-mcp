FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY mcp-servers/epid-forecasting-mcp-server/ ./
RUN uv sync --no-dev

ENV PYTHONUNBUFFERED=1
ENV EPID_DATA_PATH=/app/data/influenza_weather_spb_dataset.csv
ENV EPID_ARTIFACT_DIR=/app/artifacts

EXPOSE 7331

CMD ["uv", "run", "python", "epid_forecasting_server.py"]
