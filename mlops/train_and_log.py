import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Configuration MLflow
mlflow.set_tracking_uri("file:./mlops/mlflow_tracking")
mlflow.set_experiment("job-matcher-ml")

# Chemin absolu
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlops" / "mlflow_tracking"

# Charger les données
data_path = Path('../data/resume_fit_job/processed/v2_dataset_resume_job_fit_processed.xlsx')
df = pd.read_excel(data_path)

# Préparer les données
features = [
    'coverage', 'quality', 'nb_covered_skills', 'nb_missing_skills',
    'skills_ratio', 'similarity_mean', 'similarity_max', 'similarity_std',
    'top3_similarity_avg', 'tfidf_similarity', 'embedding_similarity',
    'nb_resume_technical', 'nb_resume_soft', 'nb_job_technical', 'nb_job_soft'
]

X = df[features]
y = df['score_target']

# Encoder les classes
class_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
y_encoded = y.map(class_mapping)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Démarrer une run MLflow
with mlflow.start_run(run_name="XGBoost_v1"):
    
    # Paramètres du modèle
    params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Logger les paramètres
    mlflow.log_params(params)
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("n_features", len(features))
    mlflow.log_param("dataset_version", "clean_v1")
    
    # Entraîner le modèle
    model = XGBClassifier(**params, n_jobs=-1, verbosity=0, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Métriques
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Logger les métriques
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("overfitting", train_acc - test_acc)
    
    # Logger le modèle
    mlflow.xgboost.log_model(model, "model")
    
    # Logger le scaler
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")
    
    # Logger les features
    with open("features.txt", "w") as f:
        f.write("\n".join(features))
    mlflow.log_artifact("features.txt")
    
    print(f"✅ Modèle tracké avec MLflow")
    print(f"   - Test Accuracy: {test_acc:.4f}")
    print(f"   - Run ID: {mlflow.active_run().info.run_id}")