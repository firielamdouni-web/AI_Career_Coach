import mlflow
import pandas as pd
from pathlib import Path

# Chemin absolu
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlops" / "mlflow_tracking"

mlflow.set_tracking_uri("file:./mlops/mlflow_tracking")

# Charger le modèle en production
model_name = "job-matcher-classifier"
model_stage = "Production"
model_uri = f"models:/{model_name}/{model_stage}"

model = mlflow.pyfunc.load_model(model_uri)

print(f"✅ Modèle chargé : {model_name} ({model_stage})")

# Fonction de prédiction
def predict(features_dict):
    """
    Prédire le score de matching
    
    Args:
        features_dict (dict): Dictionnaire des 15 features
    
    Returns:
        dict: Prédiction + probabilités
    """
    df = pd.DataFrame([features_dict])
    prediction = model.predict(df)[0]
    
    # Décoder la prédiction
    reverse_mapping = {0: 0.0, 1: 0.5, 2: 1.0}
    score = reverse_mapping[int(prediction)]
    
    return {
        'score': score,
        'class': ['No Fit', 'Partial Fit', 'Perfect Fit'][int(prediction)]
    }

# Test
test_features = {
    'coverage': 0.75,
    'quality': 0.75,
    'nb_covered_skills': 8,
    'nb_missing_skills': 2,
    'skills_ratio': 0.8,
    'similarity_mean': 0.8,
    'similarity_max': 0.9,
    'similarity_std': 0.1,
    'top3_similarity_avg': 0.85,
    'tfidf_similarity': 0.7,
    'embedding_similarity': 0.8,
    'nb_resume_technical': 10,
    'nb_resume_soft': 5,
    'nb_job_technical': 8,
    'nb_job_soft': 4
}

result = predict(test_features)
print(f"\n📊 Prédiction : {result}")