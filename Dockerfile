FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY README.md epid_forecasting_server.py ./
COPY epid_forecasting ./epid_forecasting
COPY data ./data

RUN uv sync --frozen --no-dev

ENV EPID_DATA_PATH=/app/data/influenza_weather_spb_dataset.csv
ENV EPID_ARTIFACT_DIR=/app/artifacts

EXPOSE 7331

CMD ["uv", "run", "python", "epid_forecasting_server.py"]
