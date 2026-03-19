FROM python:3.10-slim as builder

WORKDIR /app

COPY requirements/ ./requirements/

RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt -r requirements/frontend.txt

# ============================================================================
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY app.py .
COPY pages/ ./pages/
COPY src/ ./src/
COPY data/ ./data/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--logger.level=error", "--server.port=8501", "--server.address=0.0.0.0"]