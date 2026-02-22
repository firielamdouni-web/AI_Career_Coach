import joblib
import pandas as pd
from pathlib import Path

model_name = "job-matcher-classifier"

# ✅ CHANGÉ : Chemin sans le double dossier /artifacts/artifacts
models_dir = Path(__file__).parent.parent / "models" / model_name / "artifacts"

try:
    model = joblib.load(models_dir / "model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    with open(models_dir / "features.txt") as f:
        features = f.read().strip().split("\n")
    
    print(f"✅ Modèle chargé : {model_name}")
    print(f"   Chemin : {models_dir}")

except Exception as e:
    print(f"❌ Erreur : {e}")
    exit(1)

def predict(features_dict):
    df = pd.DataFrame([features_dict])
    scaled = scaler.transform(df[features])
    prediction = model.predict(scaled)[0]
    reverse_mapping = {0: 0.0, 1: 0.5, 2: 1.0}
    return {
        'score': reverse_mapping[int(prediction)],
        'class': ['No Fit', 'Partial Fit', 'Perfect Fit'][int(prediction)]
    }

# Test
test_features = {
    'coverage': 0.75, 'quality': 0.75, 'nb_covered_skills': 8,
    'nb_missing_skills': 2, 'skills_ratio': 0.8, 'similarity_mean': 0.8,
    'similarity_max': 0.9, 'similarity_std': 0.1, 'top3_similarity_avg': 0.85,
    'tfidf_similarity': 0.7, 'embedding_similarity': 0.8,
    'nb_resume_technical': 10, 'nb_resume_soft': 5,
    'nb_job_technical': 8, 'nb_job_soft': 4
}

result = predict(test_features)
print(f"\n📊 Prédiction : {result}")