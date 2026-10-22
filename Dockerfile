FROM python:3.11-slim

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir . \
    && chown -R app:app /app

ENV MLOPS_CONFIG_DIR=/app/configs
USER app
EXPOSE 8080

CMD ["python", "-m", "mlops_cluster.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
