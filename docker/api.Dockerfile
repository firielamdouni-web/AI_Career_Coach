FROM python:3.10-slim as builder

WORKDIR /app

COPY requirements/ ./requirements/

RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt -r requirements/api.txt

# ============================================================================
# RUNTIME
# ============================================================================
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY tests/ ./tests/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# ✅ Pré-télécharger les modèles directement au build
RUN . /opt/venv/bin/activate && python -c \
    "from sentence_transformers import SentenceTransformer; \
    print('📥 Téléchargement all-mpnet-base-v2...'); \
    SentenceTransformer('all-mpnet-base-v2'); \
    print('✅ Modèle chargé')"

RUN . /opt/venv/bin/activate && python -m spacy download en_core_web_sm 2>/dev/null || true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]