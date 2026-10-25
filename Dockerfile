FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Cache dependencies separately from source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore excludes keys, local figures, data and virtual environments.
COPY . .
RUN useradd --create-home appuser && \
    mkdir -p data/PaperBananaBench/diagram data/PaperBananaBench/plot results configs && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["streamlit", "run", "demo.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
