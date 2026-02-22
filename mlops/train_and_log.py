import mlflow
import joblib
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Silencer le warning git
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("job-matcher-ml")

PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / "data" / "resume_fit_job" / "processed" / "v2_dataset_resume_job_fit_processed.xlsx"
df = pd.read_excel(data_path)

features = [
    'coverage', 'quality', 'nb_covered_skills', 'nb_missing_skills',
    'skills_ratio', 'similarity_mean', 'similarity_max', 'similarity_std',
    'top3_similarity_avg', 'tfidf_similarity', 'embedding_similarity',
    'nb_resume_technical', 'nb_resume_soft', 'nb_job_technical', 'nb_job_soft'
]

X = df[features]
y = df['score_target']

class_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
y_encoded = y.map(class_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run(run_name="XGBoost_v1"):
    
    params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    mlflow.log_params(params)
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("n_features", len(features))
    mlflow.log_param("dataset_version", "clean_v1")
    
    model = XGBClassifier(**params, n_jobs=-1, verbosity=0, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("overfitting", train_acc - test_acc)
    
    # ✅ CHANGÉ : Écrire les artifacts dans /tmp/mlflow (volume partagé)
    tmp_dir = Path("/tmp/mlflow/tmp_artifacts")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = tmp_dir / "model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path))
    
    scaler_path = tmp_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(str(scaler_path))
    
    features_path = tmp_dir / "features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(features))
    mlflow.log_artifact(str(features_path))
    
    run_id = mlflow.active_run().info.run_id
    
    print(f"✅ Modèle tracké avec MLflow")
    print(f"   - MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"   - Test Accuracy: {test_acc:.4f}")
    print(f"   - Run ID: {run_id}")