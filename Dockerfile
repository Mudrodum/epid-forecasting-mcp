FROM python:3.12-slim

WORKDIR /app

COPY mcp-servers/epid-forecasting-mcp-server /app

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

ENV EPID_DATA_PATH=/app/data/influenza_weather_spb_dataset.csv
EXPOSE 7331

CMD ["uv", "run", "python", "epid_forecasting_server.py"]
