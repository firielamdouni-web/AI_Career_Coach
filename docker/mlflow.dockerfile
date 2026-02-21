FROM python:3.10-slim

WORKDIR /mlflow

# Installer seulement MLflow (très léger)
RUN pip install --no-cache-dir mlflow==2.9.2

# Créer répertoires
RUN mkdir -p /mlflow/mlruns /mlflow/artifacts && \
    useradd -m -u 1000 mlflowuser && \
    chown -R mlflowuser:mlflowuser /mlflow

USER mlflowuser

EXPOSE 5000

# SQLite (plus léger que file://)
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlruns.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]