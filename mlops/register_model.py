import mlflow
from pathlib import Path

# Chemin absolu
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlops" / "mlflow_tracking"

mlflow.set_tracking_uri("file:./mlops/mlflow_tracking")

# Récupérer le meilleur run
experiment = mlflow.get_experiment_by_name("job-matcher-ml")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_accuracy DESC"])

best_run_id = runs.iloc[0]['run_id']
best_accuracy = runs.iloc[0]['metrics.test_accuracy']

print(f"🏆 Meilleur modèle : Run ID = {best_run_id}, Accuracy = {best_accuracy:.4f}")

# Enregistrer dans le Model Registry
model_uri = f"runs:/{best_run_id}/model"
model_name = "job-matcher-classifier"

mlflow.register_model(model_uri, model_name)

print(f"✅ Modèle enregistré dans le Model Registry : {model_name}")

# Promouvoir en "Production"
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = versions[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)

print(f"✅ Modèle promu en Production (version {latest_version})")