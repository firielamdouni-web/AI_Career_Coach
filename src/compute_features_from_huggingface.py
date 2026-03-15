"""
Script pour calculer les features ML à partir du dataset Hugging Face
Utilise SkillsExtractor et JobMatcher pour extraire les features PERTINENTES
Version optimisée : 27 features (15 originales + 12 textuelles) + SUPPORT GPU
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PATH
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch  # ✅ AJOUT

from src.skills_extractor import SkillsExtractor
from src.job_matcher import JobMatcher

# Configuration du logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== INITIALISATION =====
print("=" * 70)
print("🚀 CALCUL DES FEATURES ML - DATASET HUGGING FACE (VERSION 27 FEATURES + GPU)")
print("=" * 70)

# ✅ DÉTECTION DU GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device détecté : {device.upper()}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   • GPU : {gpu_name}")
    print(f"   • VRAM : {gpu_memory:.1f} GB")
    print(f"   ⚡ Accélération GPU activée !\n")
else:
    print(f"   ⚠️  Aucun GPU détecté, utilisation du CPU")
    print(f"   💡 Installe CUDA pour accélérer : https://pytorch.org/get-started/locally/\n")

print("🔧 Initialisation des modules...")
skills_extractor = SkillsExtractor()
job_matcher = JobMatcher()

# Modèle pour embedding similarity (avec GPU)
print(f"📦 Chargement du modèle Sentence-Transformer sur {device.upper()}...")
embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Vectorizer pour TF-IDF (CPU seulement)
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

print("✅ Modules initialisés avec succès\n")

# ===== CHARGEMENT DU DATASET =====
print("📥 Chargement du dataset...")

csv_path = Path('data/processed/huggingface_resume_job_fit.xlsx')

if not csv_path.exists():
    print(f"❌ Fichier non trouvé : {csv_path}")
    print("💡 Exécute d'abord le notebook 08_exploration_dataset.ipynb (Cellule 12)")
    sys.exit(1)

df = pd.read_excel(csv_path, engine='openpyxl')

print(f"✅ Dataset chargé : {len(df):,} samples")

# ===== LIMITATION À SAMPLE_LIMIT SAMPLES =====
SAMPLE_LIMIT = None

if SAMPLE_LIMIT is not None and len(df) > SAMPLE_LIMIT:
    print(f"\n⚠️  Limitation du dataset à {SAMPLE_LIMIT} samples (pour tests)")
    df = df.head(SAMPLE_LIMIT).copy()
    print(f"✅ Dataset réduit : {len(df)} samples\n")
else:
    print(f"✅ Traitement du dataset complet : {len(df):,} samples\n")

# Réinitialiser les index pour éviter les bugs
df = df.reset_index(drop=True)
print(f"✅ Index réinitialisés (0 à {len(df)-1})\n")

# ===== PRÉ-CALCUL DES EMBEDDINGS (OPTIMISATION + GPU) =====
print(f"🔄 Pré-calcul des embeddings sur {device.upper()} (pour optimisation)...")

# Extraire les textes
resume_texts = df['resume'].astype(str).tolist()
job_texts = df['job_description'].astype(str).tolist()

# ✅ OPTIMISATION GPU : Batch size augmenté
batch_size = 64 if device == "cuda" else 32

# Générer les embeddings par batch (plus rapide)
print(f"   • Embeddings des CVs (batch={batch_size})...")
resume_embeddings = embedding_model.encode(
    resume_texts, 
    show_progress_bar=True, 
    batch_size=batch_size,
    device=device
)

print(f"   • Embeddings des Jobs (batch={batch_size})...")
job_embeddings = embedding_model.encode(
    job_texts, 
    show_progress_bar=True, 
    batch_size=batch_size,
    device=device
)

# ✅ Transférer sur CPU pour la suite (économie VRAM)
if device == "cuda":
    resume_embeddings = resume_embeddings
    job_embeddings = job_embeddings
    torch.cuda.empty_cache()  # Libérer la mémoire GPU

print("✅ Embeddings pré-calculés\n")

# ===== PRÉ-CALCUL DES TF-IDF (OPTIMISATION) =====
print("🔄 Pré-calcul des TF-IDF...")

# Combiner tous les textes pour fit le vectorizer
all_texts = resume_texts + job_texts
tfidf_vectorizer.fit(all_texts)

# Transformer les textes
resume_tfidf = tfidf_vectorizer.transform(resume_texts)
job_tfidf = tfidf_vectorizer.transform(job_texts)

print("✅ TF-IDF pré-calculés\n")

# ===== FONCTION D'EXTRACTION DE FEATURES =====
def compute_text_features(text):
    """
    Calculer les 6 features textuelles pour un texte donné
    """
    if not text or pd.isna(text):
        return {
            'text_length': 0,
            'text_word_count': 0,
            'text_unique_words': 0,
            'text_avg_word_length': 0.0,
            'text_sentence_count': 0,
            'text_capital_ratio': 0.0
        }
    
    text = str(text)
    words = text.split()
    
    return {
        'text_length':        len(text),
        'text_word_count':    len(words),
        'text_unique_words':  len(set(w.lower() for w in words)),
        'text_avg_word_length': float(np.mean([len(w) for w in words])) if words else 0.0,
        'text_sentence_count': len(re.findall(r'[.!?]+', text)),
        'text_capital_ratio':  sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0.0
    }

def compute_features_for_row(row, row_idx):
    """
    Calculer les 27 features ML pour une paire (CV, Job)
    
    Features calculées :
    - Skills Matching (5) : coverage, quality, nb_covered_skills, nb_missing_skills, skills_ratio
    - Similarité (4)      : similarity_mean, similarity_max, similarity_std, top3_similarity_avg
    - Sémantique (2)      : tfidf_similarity, embedding_similarity
    - Contexte (4)        : nb_resume_technical, nb_resume_soft, nb_job_technical, nb_job_soft
    - Texte CV (6)        : resume_text_length, resume_text_word_count, ...
    - Texte Job (6)       : job_description_text_length, job_description_text_word_count, ...
    """
    THRESHOLD_STRICT   = 65
    THRESHOLD_MODERATE = 40
    
    try:
        # ===== 1. EXTRACTION DES SKILLS =====
        cv_result = skills_extractor.extract_from_cv(str(row['resume']))
        cv_technical = cv_result.get('technical_skills', [])
        cv_soft = cv_result.get('soft_skills', [])
        cv_skills_list = cv_technical + cv_soft
        
        job_result = skills_extractor.extract_from_cv(str(row['job_description']))
        job_technical = job_result.get('technical_skills', [])
        job_soft = job_result.get('soft_skills', [])
        job_skills_list = job_technical + job_soft
        
        # ===== 2. FEATURES DE CONTEXTE (4 features) =====
        nb_resume_technical = len(cv_technical)
        nb_resume_soft = len(cv_soft)
        nb_job_technical = len(job_technical)
        nb_job_soft = len(job_soft)
        
        # ===== 3. FEATURES SÉMANTIQUES (pré-calculées) =====
        tfidf_sim = cosine_similarity(resume_tfidf[row_idx], job_tfidf[row_idx])[0][0]
        embedding_sim = cosine_similarity(
            resume_embeddings[row_idx].reshape(1, -1),
            job_embeddings[row_idx].reshape(1, -1)
        )[0][0]

        # ===== 4. FEATURES TEXTUELLES (12 features) =====
        resume_text_feats = compute_text_features(row['resume'])
        job_text_feats    = compute_text_features(row['job_description'])
        
        # Préfixer les noms pour distinguer CV et Job
        resume_feats = {f'resume_{k}': v for k, v in resume_text_feats.items()}
        job_feats = {f'job_description_{k}': v for k, v in job_text_feats.items()}

        # ===== 5. GESTION DES CAS VIDES =====
        if len(cv_skills_list) == 0 or len(job_skills_list) == 0:
            return {
                # Skills Matching (5)
                'coverage': 0.0, 'quality': 0.0,
                'nb_covered_skills': 0,
                'nb_missing_skills': len(job_skills_list),
                'skills_ratio': 0.0,
                # Similarité (4)
                'similarity_mean': 0.0, 'similarity_max': 0.0,
                'similarity_std': 0.0, 'top3_similarity_avg': 0.0,
                # Sémantique (2)
                'tfidf_similarity': float(tfidf_sim),
                'embedding_similarity': float(embedding_sim),
                # Contexte (4)
                'nb_resume_technical': nb_resume_technical,
                'nb_resume_soft': nb_resume_soft,
                'nb_job_technical': nb_job_technical,
                'nb_job_soft': nb_job_soft,
                # Texte CV (6)
                **resume_feats,
                # Texte Job (6)
                **job_feats,
            }
        
        # ===== 6. MATCHING SKILL PAR SKILL (seuils 65% / 40%) =====
        cv_embs  = embedding_model.encode([s.lower() for s in cv_skills_list], show_progress_bar=False)
        job_embs = embedding_model.encode([s.lower() for s in job_skills_list], show_progress_bar=False)

        similarities = []
        nb_covered_skills = 0
        nb_missing_count  = 0

        for i, job_skill in enumerate(job_skills_list):
            best_sim = 0.0

            # Exact match en premier
            for cv_skill in cv_skills_list:
                if cv_skill.lower() == job_skill.lower():
                    best_sim = 100.0
                    break

            # Sinon similarité sémantique
            if best_sim < 100.0:
                sims = cosine_similarity([job_embs[i]], cv_embs)[0] * 100
                best_sim = float(np.max(sims))

            similarities.append(best_sim)

            if best_sim >= THRESHOLD_STRICT:   nb_covered_skills += 1
            if best_sim <  THRESHOLD_MODERATE: nb_missing_count  += 1

        sim_array = np.array(similarities)
        n_job     = len(job_skills_list)

        # ===== 7. FEATURES SKILLS MATCHING (5 features) =====
        coverage          = (nb_covered_skills / n_job) * 100
        covered_sims      = [s for s in similarities if s >= THRESHOLD_STRICT]
        quality           = float(np.mean(covered_sims)) if covered_sims else 0.0
        nb_missing_skills = nb_missing_count
        skills_ratio      = len(cv_skills_list) / max(n_job, 1)

        # ===== 8. FEATURES SIMILARITÉ (4 features) =====
        similarity_mean     = float(sim_array.mean())
        similarity_max      = float(sim_array.max())
        similarity_std      = float(sim_array.std())
        top3                = sorted(similarities, reverse=True)[:3]
        top3_similarity_avg = float(np.mean(top3))

        # ===== 9. RETOUR DES 27 FEATURES =====
        return {
            # Skills Matching (5)
            'coverage':            float(coverage),
            'quality':             float(quality),
            'nb_covered_skills':   int(nb_covered_skills),
            'nb_missing_skills':   int(nb_missing_skills),
            'skills_ratio':        float(skills_ratio),
            # Similarité (4)
            'similarity_mean':     float(similarity_mean),
            'similarity_max':      float(similarity_max),
            'similarity_std':      float(similarity_std),
            'top3_similarity_avg': float(top3_similarity_avg),
            # Sémantique (2)
            'tfidf_similarity':    float(tfidf_sim),
            'embedding_similarity': float(embedding_sim),
            # Contexte (4)
            'nb_resume_technical': int(nb_resume_technical),
            'nb_resume_soft':      int(nb_resume_soft),
            'nb_job_technical':    int(nb_job_technical),
            'nb_job_soft':         int(nb_job_soft),
            # Texte CV (6)
            **resume_feats,
            # Texte Job (6)
            **job_feats,
        }
    
    except Exception as e:
        logger.error(f"⚠️ Erreur ligne {row_idx} : {e}")
        # Retourner features par défaut
        return {
            'coverage': 0.0, 'quality': 0.0,
            'nb_covered_skills': 0, 'nb_missing_skills': 0,
            'skills_ratio': 0.0, 'similarity_mean': 0.0,
            'similarity_max': 0.0, 'similarity_std': 0.0,
            'top3_similarity_avg': 0.0, 'tfidf_similarity': 0.0,
            'embedding_similarity': 0.0, 'nb_resume_technical': 0,
            'nb_resume_soft': 0, 'nb_job_technical': 0, 'nb_job_soft': 0,
            'resume_text_length': 0, 'resume_text_word_count': 0,
            'resume_text_unique_words': 0, 'resume_text_avg_word_length': 0.0,
            'resume_text_sentence_count': 0, 'resume_text_capital_ratio': 0.0,
            'job_description_text_length': 0, 'job_description_text_word_count': 0,
            'job_description_text_unique_words': 0, 'job_description_text_avg_word_length': 0.0,
            'job_description_text_sentence_count': 0, 'job_description_text_capital_ratio': 0.0,
        }


# ===== CALCUL DES FEATURES POUR TOUT LE DATASET =====
estimated_time = len(df) * 0.5 / 60 if device == "cuda" else len(df) * 1.0 / 60
print(f"🔄 Calcul des 27 features ML pour {len(df):,} samples...")
print(f"   ⏱️ Temps estimé : ~{estimated_time:.1f} minutes ({device.upper()})\n")

features_list = []
errors_count = 0

for seq_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="📊 Traitement", unit="sample")):
    features = compute_features_for_row(row, seq_idx)
    
    # Compter les erreurs
    if features['coverage'] == 0.0 and features['quality'] == 0.0:
        errors_count += 1
    
    features_list.append(features)

# ===== CRÉATION DU DATAFRAME FINAL =====
print(f"\n✅ Traitement terminé")
print(f"   • Samples traités : {len(features_list):,}")
print(f"   • Erreurs/vides   : {errors_count:,} ({errors_count/len(df)*100:.1f}%)\n")

features_df = pd.DataFrame(features_list)
features_df['score_target'] = df['score_target'].values

print(f"📊 Features calculées :")
print(f"   • Nombre de features : {len(features_df.columns) - 1}")
feature_names = [col for col in features_df.columns if col != 'score_target']
print(f"   • Features : {feature_names}\n")

# ===== VÉRIFICATION DE LA QUALITÉ =====
print("🔍 Vérification de la qualité des données...\n")

# Valeurs manquantes
missing = features_df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante")
else:
    print("⚠️ Valeurs manquantes détectées :")
    print(missing[missing > 0])

# Statistiques descriptives
print(f"\n📊 Statistiques descriptives :")
print(features_df.describe())

# Corrélations avec target
print(f"\n📊 Corrélations avec score_target :")
correlations = features_df.corr()['score_target'].drop('score_target').sort_values(ascending=False)
print(correlations)

# Top 3 features corrélées
print(f"\n🎯 Top 3 features les plus corrélées :")
for i, (feat, corr) in enumerate(correlations.head(3).items(), 1):
    print(f"   {i}. {feat:25} → {corr:.3f}")

# ===== SAUVEGARDE =====
output_path = Path('data/processed/dataset_resume_job_fit_processed.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

features_df.to_csv(output_path, index=False)

print(f"\n✅ Dataset avec features sauvegardé : {output_path}")
print(f"   • Taille         : {len(features_df):,} samples")
print(f"   • Features       : {len(features_df.columns) - 1}")
print(f"   • Taille fichier : {output_path.stat().st_size / 1024:.2f} KB")

# ===== RÉSUMÉ FINAL =====
print("\n" + "=" * 70)
print("🎯 RÉSUMÉ DU TRAITEMENT")
print("=" * 70)
print(f"✅ Device utilisé    : {device.upper()}")
print(f"✅ Dataset source    : {len(df):,} samples")
print(f"✅ Features extraites : {len(features_df.columns) - 1} (15 originales + 12 textuelles)")
print(f"✅ Samples finaux     : {len(features_df):,}")
print(f"✅ Taux de réussite   : {(len(df) - errors_count) / len(df) * 100:.1f}%")
print(f"\n📊 Composition des features :")
print(f"   • Skills Matching (5) : coverage, quality, nb_covered_skills, nb_missing_skills, skills_ratio")
print(f"   • Similarité (4)      : similarity_mean, similarity_max, similarity_std, top3_similarity_avg")
print(f"   • Sémantique (2)      : tfidf_similarity, embedding_similarity")
print(f"   • Contexte (4)        : nb_resume_technical, nb_resume_soft, nb_job_technical, nb_job_soft")
print(f"   • Texte CV (6)        : resume_text_length, resume_text_word_count, resume_text_unique_words, resume_text_avg_word_length, resume_text_sentence_count, resume_text_capital_ratio")
print(f"   • Texte Job (6)       : job_description_text_length, job_description_text_word_count, job_description_text_unique_words, job_description_text_avg_word_length, job_description_text_sentence_count, job_description_text_capital_ratio")
print(f"\n📂 Fichier sauvegardé : {output_path}")
print(f"\n🎯 Prochaine étape    : Entraîner le modèle ML avec 27 features")
print("=" * 70)