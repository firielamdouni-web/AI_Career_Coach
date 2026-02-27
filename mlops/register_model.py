import mlflow
import os
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

experiment = mlflow.get_experiment_by_name("job-matcher-ml")

if experiment is None:
    print("❌ Expérience 'job-matcher-ml' non trouvée!")
    exit(1)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"]
)

if runs.empty:
    print("❌ Aucun run trouvé!")
    exit(1)

best_run_id = runs.iloc[0]['run_id']
best_accuracy = runs.iloc[0]['metrics.test_accuracy']

print(f"🏆 Meilleur modèle : Run ID = {best_run_id}")
print(f"   Accuracy = {best_accuracy:.4f}")

model_name = "job-matcher-classifier"

# ✅ CHANGÉ : Télécharger depuis la racine des artifacts (plus de sous-dossier)
artifacts_dir = Path(__file__).parent.parent / "models" / model_name / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

local_artifacts_path = client.download_artifacts(best_run_id, ".", str(artifacts_dir))
print(f"✅ Artifacts téléchargés :")
for f in Path(local_artifacts_path).glob("*"):
    print(f"   - {f.name}")

# Créer le registered model
try:
    client.create_registered_model(model_name)
    print(f"✅ Registered Model créé : {model_name}")
except mlflow.exceptions.MlflowException as e:
    if "already exists" in str(e):
        print(f"⚠️ Registered Model '{model_name}' existe déjà")
    else:
        raise

# Créer une version
artifact_uri = client.get_run(best_run_id).info.artifact_uri
version = client.create_model_version(
    name=model_name,
    source=artifact_uri,
    run_id=best_run_id
)
print(f"✅ Version créée : {version.version}")

# Promouvoir en Production
client.transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Production"
)
print(f"✅ Modèle promu en Production (version {version.version})")