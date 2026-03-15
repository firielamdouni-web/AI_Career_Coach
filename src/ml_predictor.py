"""
🤖 Module de prédiction ML avec XGBoost
Charge le modèle depuis MLflow et calcule les features à la volée
"""

import joblib
import numpy as np
import re 
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
    'nb_resume_technical', 'nb_resume_soft', 'nb_job_technical', 'nb_job_soft',
    'resume_text_length', 'resume_text_word_count', 'resume_text_unique_words',
    'resume_text_avg_word_length', 'resume_text_sentence_count', 'resume_text_capital_ratio',
    'job_description_text_length', 'job_description_text_word_count', 'job_description_text_unique_words',
    'job_description_text_avg_word_length', 'job_description_text_sentence_count', 'job_description_text_capital_ratio',
]

CLASS_LABELS = {0: 'No Fit', 1: 'Partial Fit', 2: 'Perfect Fit'}
CLASS_SCORES = {0: 0.0, 1: 0.5, 2: 1.0}

def _compute_text_features(text: str, prefix: str) -> Dict:
    if not text:
        return {
            f'{prefix}_text_length': 0,
            f'{prefix}_text_word_count': 0,
            f'{prefix}_text_unique_words': 0,
            f'{prefix}_text_avg_word_length': 0.0,
            f'{prefix}_text_sentence_count': 0,
            f'{prefix}_text_capital_ratio': 0.0,
        }
    words = text.split()
    return {
        f'{prefix}_text_length':          len(text),
        f'{prefix}_text_word_count':       len(words),
        f'{prefix}_text_unique_words':     len(set(w.lower() for w in words)),
        f'{prefix}_text_avg_word_length':  float(np.mean([len(w) for w in words])) if words else 0.0,
        f'{prefix}_text_sentence_count':   len(re.findall(r'[.!?]+', text)),
        f'{prefix}_text_capital_ratio':    sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
    }


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
        job_technical_skills: List[str],
        job_soft_skills: List[str],
        skills_details: Dict,
        cv_raw_text: str,
        job_raw_text: str,
        sentence_model
    ) -> Dict[str, float]:

        cv_all_skills  = cv_technical_skills + cv_soft_skills
        job_all_skills = job_technical_skills + job_soft_skills

        # ── 1. Skills Matching ────────────────────────────────────────────────
        # ✅ IDENTIQUE au dataset : seuils 65 / 40
        THRESHOLD_STRICT   = 65
        THRESHOLD_MODERATE = 40

        # ── 2. ✅ FIX PRINCIPAL : recalculer les similarités SKILL PAR SKILL
        #        exactement comme dans compute_features_from_huggingface.py
        if len(cv_all_skills) > 0 and len(job_all_skills) > 0:
            cv_embs  = sentence_model.encode([s.lower() for s in cv_all_skills],  show_progress_bar=False)
            job_embs = sentence_model.encode([s.lower() for s in job_all_skills], show_progress_bar=False)

            similarities      = []
            nb_covered_skills = 0
            nb_missing_count  = 0

            for i, job_skill in enumerate(job_all_skills):
                best_sim = 0.0

                # Exact match en premier
                for cv_skill in cv_all_skills:
                    if cv_skill.lower() == job_skill.lower():
                        best_sim = 100.0
                        break

                # Sinon similarité cosine * 100 (→ 0-100 comme dataset)
                if best_sim < 100.0:
                    sims     = cosine_similarity([job_embs[i]], cv_embs)[0] * 100
                    best_sim = float(np.max(sims))

                similarities.append(best_sim)

                if best_sim >= THRESHOLD_STRICT:   nb_covered_skills += 1
                if best_sim <  THRESHOLD_MODERATE: nb_missing_count  += 1

            sim_array = np.array(similarities)
            n_job     = len(job_all_skills)

            # ✅ IDENTIQUE au dataset
            coverage          = (nb_covered_skills / n_job) * 100
            covered_sims      = [s for s in similarities if s >= THRESHOLD_STRICT]
            quality           = float(np.mean(covered_sims)) if covered_sims else 0.0
            nb_missing_skills = nb_missing_count
            skills_ratio      = len(cv_all_skills) / max(n_job, 1)

            similarity_mean     = float(sim_array.mean())
            similarity_max      = float(sim_array.max())
            similarity_std      = float(sim_array.std())
            top3                = sorted(similarities, reverse=True)[:3]
            top3_similarity_avg = float(np.mean(top3))

        else:
            # Cas vide
            coverage = quality = skills_ratio = 0.0
            nb_covered_skills = nb_missing_skills = 0
            similarity_mean = similarity_max = similarity_std = top3_similarity_avg = 0.0

        # ── 3. TF-IDF ─────────────────────────────────────────────────────────
        try:
            tfidf        = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf.fit([cv_raw_text, job_raw_text])
            tfidf_matrix = tfidf.transform([cv_raw_text, job_raw_text])
            tfidf_similarity = float(
                cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            )
        except Exception:
            tfidf_similarity = 0.0

        # ── 4. Embedding ──────────────────────────────────────────────────────
        try:
            embeddings = sentence_model.encode(
                [cv_raw_text, job_raw_text], show_progress_bar=False
            )
            embedding_similarity = float(
                cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            )
        except Exception:
            embedding_similarity = 0.0

        # ── 5. Contexte ───────────────────────────────────────────────────────
        nb_resume_technical = len(cv_technical_skills)
        nb_resume_soft      = len(cv_soft_skills)
        nb_job_technical    = len(job_technical_skills)
        nb_job_soft         = len(job_soft_skills)

        # ── 6. Features textuelles ────────────────────────────────────────────
        resume_feats = _compute_text_features(cv_raw_text,  'resume')
        job_feats    = _compute_text_features(job_raw_text, 'job_description')

        # ── 7. Assembler les 27 features ──────────────────────────────────────
        result = {
            'coverage':             coverage,
            'quality':              quality,
            'nb_covered_skills':    float(nb_covered_skills),
            'nb_missing_skills':    float(nb_missing_skills),
            'skills_ratio':         skills_ratio,
            'similarity_mean':      similarity_mean,
            'similarity_max':       similarity_max,
            'similarity_std':       similarity_std,
            'top3_similarity_avg':  top3_similarity_avg,
            'tfidf_similarity':     tfidf_similarity,
            'embedding_similarity': embedding_similarity,
            'nb_resume_technical':  float(nb_resume_technical),
            'nb_resume_soft':       float(nb_resume_soft),
            'nb_job_technical':     float(nb_job_technical),
            'nb_job_soft':          float(nb_job_soft),
        }
        result.update(resume_feats)
        result.update(job_feats)

        # Debug temporaire
        logger.warning(f"🔍 DEBUG : coverage={coverage:.1f} | quality={quality:.1f} | sim_mean={similarity_mean:.1f} | sim_max={similarity_max:.1f}")

        return result

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