FROM python:3.12-slim

WORKDIR /app

# System deps for XGBoost/CatBoost (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements_automation.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_automation.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY predictor.py ./
COPY config.toml ./

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.automation.scheduler"]
