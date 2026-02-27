"""
🤖 Module de prédiction ML avec XGBoost
Charge le modèle depuis MLflow et calcule les features à la volée
"""

import joblib
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "job-matcher-classifier" / "artifacts"

FEATURES = [
    'coverage', 'quality', 'nb_covered_skills', 'nb_missing_skills',
    'skills_ratio', 'similarity_mean', 'similarity_max', 'similarity_std',
    'top3_similarity_avg', 'tfidf_similarity', 'embedding_similarity',
    'nb_resume_technical', 'nb_resume_soft', 'nb_job_technical', 'nb_job_soft'
]

CLASS_LABELS = {0: 'No Fit', 1: 'Partial Fit', 2: 'Perfect Fit'}
CLASS_SCORES = {0: 0.0, 1: 0.5, 2: 1.0}


class MLPredictor:

    def __init__(self):
        self.model = None
        self.scaler = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        try:
            model_path = MODELS_DIR / "model.pkl"
            scaler_path = MODELS_DIR / "scaler.pkl"

            if not model_path.exists():
                logger.warning(f"⚠️ Modèle ML non trouvé : {model_path}")
                return

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self._loaded = True
            logger.info("✅ Modèle XGBoost chargé avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle ML : {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def compute_features(
        self,
        cv_technical_skills: List[str],
        cv_soft_skills: List[str],
        job_technical_skills: List[str],   # ✅ CHANGÉ : séparé technical/soft
        job_soft_skills: List[str],         # ✅ CHANGÉ : séparé technical/soft
        # ✅ CHANGÉ : directement skills_details de JobMatcher
        skills_details: Dict,
        cv_raw_text: str,                   # ✅ AJOUT : texte brut du CV
        job_raw_text: str,                  # ✅ AJOUT : texte brut du Job
        sentence_model                      # SentenceTransformer de JobMatcher
    ) -> Dict[str, float]:
        """
        Calculer les 15 features EXACTEMENT comme compute_features_from_huggingface.py

        Args:
            cv_technical_skills: Skills techniques du CV
            cv_soft_skills: Soft skills du CV
            job_technical_skills: Skills techniques du Job
            job_soft_skills: Soft skills du Job
            skills_details: Résultat de calculate_job_match_score()['skills_details']
            cv_raw_text: Texte brut complet du CV (pour TF-IDF et embedding)
            job_raw_text: Texte brut complet du Job (pour TF-IDF et embedding)
            sentence_model: SentenceTransformer déjà chargé dans JobMatcher
        """

        cv_all_skills = cv_technical_skills + cv_soft_skills
        job_all_skills = job_technical_skills + job_soft_skills

        # ── 1. Features Skills Matching (EXACTEMENT comme compute_features) ──
        coverage = float(skills_details.get('coverage', 0.0))
        quality = float(skills_details.get('quality', 0.0))
        nb_covered_skills = int(skills_details.get('covered_count', 0))
        total_required = int(
            skills_details.get(
                'total_required',
                len(job_all_skills)))
        nb_missing_skills = total_required - nb_covered_skills
        skills_ratio = len(cv_all_skills) / max(len(job_all_skills), 1)

        # ── 2. Features Similarité (depuis top_matches SANS diviser par 100) ──
        top_matches = skills_details.get('top_matches', [])

        if top_matches:
            # ✅ CORRECTION : similarities sont déjà en 0-100 dans JobMatcher
            similarities = [m.get('similarity', 0.0) for m in top_matches]
            similarity_mean = float(np.mean(similarities))
            similarity_max = float(np.max(similarities))
            similarity_std = float(np.std(similarities))
            top3 = sorted(similarities, reverse=True)[:3]
            top3_similarity_avg = float(np.mean(top3))
        else:
            similarity_mean = 0.0
            similarity_max = 0.0
            similarity_std = 0.0
            top3_similarity_avg = 0.0

        # ── 3. TF-IDF similarity sur TEXTE COMPLET (comme compute_features) ──
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf.fit([cv_raw_text, job_raw_text])
            tfidf_matrix = tfidf.transform([cv_raw_text, job_raw_text])
            tfidf_similarity = float(
                cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            )
        except Exception:
            tfidf_similarity = 0.0

        # ── 4. Embedding similarity sur TEXTE COMPLET (comme compute_features) ──
        try:
            embeddings = sentence_model.encode(
                [cv_raw_text, job_raw_text],
                show_progress_bar=False
            )
            embedding_similarity = float(
                cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            )
        except Exception:
            embedding_similarity = 0.0

        # ── 5. Features Contexte ─────────────────────────────────────────────
        nb_resume_technical = len(cv_technical_skills)
        nb_resume_soft = len(cv_soft_skills)
        nb_job_technical = len(job_technical_skills)   # ✅ CORRECTION
        nb_job_soft = len(job_soft_skills)              # ✅ CORRECTION

        return {
            'coverage': coverage,
            'quality': quality,
            'nb_covered_skills': float(nb_covered_skills),
            'nb_missing_skills': float(nb_missing_skills),
            'skills_ratio': skills_ratio,
            'similarity_mean': similarity_mean,
            'similarity_max': similarity_max,
            'similarity_std': similarity_std,
            'top3_similarity_avg': top3_similarity_avg,
            'tfidf_similarity': tfidf_similarity,
            'embedding_similarity': embedding_similarity,
            'nb_resume_technical': float(nb_resume_technical),
            'nb_resume_soft': float(nb_resume_soft),
            'nb_job_technical': float(nb_job_technical),
            'nb_job_soft': float(nb_job_soft),
        }

    def predict(self, features: Dict[str, float]) -> Dict:
        if not self._loaded:
            return {
                'ml_label': 'N/A',
                'ml_score': None,
                'ml_probabilities': None,
                'ml_available': False
            }

        try:
            feature_vector = [[features[f] for f in FEATURES]]
            scaled = self.scaler.transform(feature_vector)
            prediction = int(self.model.predict(scaled)[0])
            probabilities = self.model.predict_proba(scaled)[0].tolist()

            return {
                'ml_label': CLASS_LABELS[prediction],
                'ml_score': CLASS_SCORES[prediction],
                'ml_probabilities': {
                    'no_fit': round(probabilities[0], 3),
                    'partial_fit': round(probabilities[1], 3),
                    'perfect_fit': round(probabilities[2], 3),
                },
                'ml_available': True,
                'ml_features': features
            }

        except Exception as e:
            logger.error(f"❌ Erreur prédiction ML : {e}")
            return {
                'ml_label': 'N/A',
                'ml_score': None,
                'ml_probabilities': None,
                'ml_available': False
            }


_ml_predictor: Optional[MLPredictor] = None


def get_ml_predictor() -> MLPredictor:
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor()
    return _ml_predictor
