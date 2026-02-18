"""
🎯 AI Career Coach - Dashboard Streamlit (VERSION API)
Application de recommandation d'offres d'emploi basée sur l'analyse de CV
Utilise l'API FastAPI pour tous les traitements
"""

import streamlit as st
import requests
import json
import os 
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_URL", "http://ai-career-coach-api:8000")

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé (identique à avant)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f0f8ff;
    }
    .job-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .job-card-excellent {
        border-color: #4CAF50;
        background-color: #f1f8f4;
    }
    .job-card-good {
        border-color: #FFC107;
        background-color: #fffbf0;
    }
    .job-card-medium {
        border-color: #FF9800;
        background-color: #fff8f0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .score-excellent {
        background-color: #4CAF50;
        color: white;
    }
    .score-good {
        background-color: #FFC107;
        color: white;
    }
    .score-medium {
        background-color: #FF9800;
        color: white;
    }
    .score-low {
        background-color: #9E9E9E;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .api-status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .api-status-disconnected {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS D'APPEL API
# ============================================================================

def check_api_health():
    """
    Vérifier que l'API est accessible
    
    Returns:
        dict ou None: Réponse de l'API ou None si erreur
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_api_stats():
    """
    Obtenir les statistiques de l'API
    
    Returns:
        dict ou None: Statistiques ou None si erreur
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def extract_skills_via_api(cv_file):
    """
    Extraire les compétences via l'API
    
    Args:
        cv_file: Fichier PDF uploadé (UploadedFile de Streamlit)
        
    Returns:
        dict ou None: {technical_skills, soft_skills, total_skills, cv_text_length}
    """
    try:
        # Préparer le fichier pour l'upload
        # Remettre le curseur au début
        cv_file.seek(0)
        
        files = {
            "file": (cv_file.name, cv_file, "application/pdf")
        }
        
        # Appeler l'API
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract-skills",
            files=files,
            timeout=480  # 8 minutes max
        )
        
        # Vérifier le statut
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur API (Code {response.status_code})")
            error_detail = response.json().get('detail', 'Erreur inconnue')
            st.error(f"Détail : {error_detail}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API")
        st.info("💡 Vérifiez que l'API tourne : `uvicorn src.api:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout : L'API met trop de temps à répondre")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {str(e)}")
        return None


def recommend_jobs_via_api(cv_file, top_n=10, min_score=40.0):
    """
    Obtenir des recommandations via l'API
    
    Args:
        cv_file: Fichier PDF uploadé
        top_n: Nombre de recommandations
        min_score: Score minimum
        
    Returns:
        dict ou None: {recommendations, total_jobs_analyzed, cv_skills_count}
    """
    try:
        # Préparer le fichier
        cv_file.seek(0)
        
        files = {
            "file": (cv_file.name, cv_file, "application/pdf")
        }
        
        params = {
            "top_n": top_n,
            "min_score": min_score
        }
        
        # Appeler l'API
        response = requests.post(
            f"{API_BASE_URL}/api/v1/recommend-jobs",
            files=files,
            params=params,
            timeout=480  # 8 minutes max
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur API (Code {response.status_code})")
            error_detail = response.json().get('detail', 'Erreur inconnue')
            st.error(f"Détail : {error_detail}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout : L'API met trop de temps à répondre")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {str(e)}")
        return None


def get_jobs_list(category=None, remote=None, limit=25):
    """
    Obtenir la liste des offres via l'API
    
    Args:
        category: Filtrer par catégorie (optionnel)
        remote: Filtrer par télétravail (optionnel)
        limit: Nombre maximum de résultats
        
    Returns:
        list ou None: Liste d'offres ou None si erreur
    """
    try:
        params = {"limit": limit}
        if category:
            params["category"] = category
        if remote is not None:
            params["remote"] = remote
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/jobs",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except requests.exceptions.RequestException:
        return None

def get_api_url():
    """
    Détecter automatiquement l'URL de l'API
    
    Returns:
        str: URL de l'API (localhost ou nom Docker)
    """
    # 1. Vérifier si on est dans Docker (variable d'env)
    if os.getenv("RUNNING_IN_DOCKER"):
        return "http://ai-career-coach-api:8000"
    
    # 2. Sinon, vérifier si l'API est accessible sur localhost
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            return "http://localhost:8000"
    except:
        pass
    
    # 3. Fallback : URL depuis variable d'environnement ou défaut Docker
    return os.getenv("API_BASE_URL", "http://ai-career-coach-api:8000")

# Utiliser la fonction pour obtenir l'URL
API_BASE_URL = get_api_url()

print(f"🔌 API detectée : {API_BASE_URL}")


# ============================================================================
# FONCTIONS D'AFFICHAGE
# ============================================================================

def get_score_class(score):
    """Retourner la classe CSS selon le score"""
    if score >= 70:
        return "excellent", "🟢"
    elif score >= 50:
        return "good", "🟡"
    elif score >= 40:
        return "medium", "🟠"
    else:
        return "low", "🔴"


def display_job_card(job, rank):
    """Afficher une carte d'offre d'emploi (VERSION SIMPLIFIÉE)"""
    score = job['score']
    score_class, emoji = get_score_class(score)
    
    card_class = f"job-card job-card-{score_class}" if score_class != "low" else "job-card"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    # En-tête
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {emoji} #{rank} - {job['title']}")
        st.markdown(f"**🏢 {job['company']}** | 📍 {job['location']}")
    
    with col2:
        st.markdown(
            f'<div class="score-badge score-{score_class}">{score:.1f}%</div>', 
            unsafe_allow_html=True
        )
    
    # Détails simplifiés
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**💼 Expérience** : {job['experience_required']}")
        st.markdown(f"**🏠 Remote** : {'Oui ✅' if job['remote'] else 'Non ❌'}")
    
    with col2:
        st.markdown(f"**🎯 Score de matching** : {score:.1f}%")
    
    # Compétences matchées
    with st.expander("🔧 Compétences matchées"):
        matching_skills = job.get('matching_skills', [])
        
        if matching_skills:
            st.markdown(f"**{len(matching_skills)} compétences correspondent à cette offre :**")
            st.markdown("---")
            
            # Afficher en colonnes pour meilleure lisibilité
            num_cols = 3
            cols = st.columns(num_cols)
            
            for idx, skill in enumerate(matching_skills):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.markdown(f"✓ {skill}")
        else:
            st.info("Aucune compétence matchée disponible")

    # Compétences manquantes
    missing_skills = job.get('missing_skills', [])
    if missing_skills:
        with st.expander(f"⚠️ Compétences manquantes ({len(missing_skills)})"):
            st.markdown(f"**{len(missing_skills)} compétences requises que vous ne possédez pas (ou non détectées) :**")
            st.markdown("---")
            
            # Afficher en colonnes (même format)
            num_cols = 3
            cols = st.columns(num_cols)
            
            for idx, skill in enumerate(missing_skills):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.markdown(f"❌ {skill}")
            
            # Message d'encouragement
            st.markdown("---")
            st.info("💡 **Conseil** : Ajoutez ces compétences à votre CV ou suivez une formation pour améliorer votre score !")

    # NOUVEAU : Bouton simulation d'entretien
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🎤 Simuler un entretien", key=f"interview_{job['job_id']}", use_container_width=True):
            # Sauvegarder le job_id dans session_state
            st.session_state.selected_job_for_interview = job['job_id']
            # Rediriger vers la page interview
            st.switch_page("pages/1_Interview_Simulator.py")
    
    with col2:
        if st.button(f"📄 Voir l'offre complète", key=f"details_{job['job_id']}", use_container_width=True):
            st.info("Fonctionnalité en développement")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    """Application principale"""
    
    # Header
    st.markdown('<div class="main-header">🎯 AI Career Coach</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Trouvez les offres d\'emploi parfaites pour votre profil</div>', 
        unsafe_allow_html=True
    )
    
    # Vérifier la connexion à l'API
    st.sidebar.header("🔌 État de l'API")
    
    with st.spinner("🔍 Vérification de l'API..."):
        health = check_api_health()
    
    if health:
        st.sidebar.markdown(
            f'<p class="api-status-connected">✅ API connectée</p>',
            unsafe_allow_html=True
        )
        st.sidebar.markdown(f"**Version** : {health.get('version', 'N/A')}")
        st.sidebar.markdown(f"**Offres disponibles** : {health.get('jobs_available', 0)}")
        
        # Obtenir les stats
        stats = get_api_stats()
        if stats:
            with st.sidebar.expander("📊 Statistiques système"):
                st.markdown(f"**Total offres** : {stats['total_jobs']}")
                st.markdown(f"**Remote** : {stats['remote_jobs']}")
                st.markdown(f"**On-site** : {stats['on_site_jobs']}")
                st.markdown(f"**Compétences techniques** : {stats['total_technical_skills']}")
                st.markdown(f"**Soft skills** : {stats['total_soft_skills']}")
                st.markdown(f"**Modèle** : {stats['model_used']}")
    else:
        st.sidebar.markdown(
            '<p class="api-status-disconnected">❌ API non accessible</p>',
            unsafe_allow_html=True
        )
        st.error("❌ Impossible de se connecter à l'API")
        st.info("💡 Lancez l'API avec : `uvicorn src.api:app --reload --port 8000`")
        st.code("uvicorn src.api:app --reload --port 8000", language="bash")
        st.stop()
    
    # Initialiser session state
    if 'cv_processed' not in st.session_state:
        st.session_state.cv_processed = False
        st.session_state.cv_skills = []
        st.session_state.recommendations = []
        st.session_state.cv_skills_count = 0
    
    # Zone d'upload
    st.markdown("---")
    st.header("📤 Upload de CV")
    
    uploaded_file = st.file_uploader(
        "Choisissez votre CV (PDF)",
        type=['pdf'],
        help="Uploadez votre CV au format PDF pour obtenir des recommandations personnalisées"
    )
    
    # Bouton d'analyse
    if uploaded_file is not None:
        st.markdown(f"**Fichier uploadé** : {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Analyser mon CV", type="primary", use_container_width=True):
                
                # Étape 1 : Extraire les compétences
                with st.spinner("🔍 Extraction des compétences via API... (30-60 secondes)"):
                    skills_result = extract_skills_via_api(uploaded_file)
                
                if not skills_result:
                    st.error("❌ Échec de l'extraction des compétences")
                    st.stop()
                
                st.success(f"✅ {skills_result['total_skills']} compétences détectées")
                
                # Étape 2 : Obtenir les recommandations
                with st.spinner("🎯 Calcul des recommandations via API... (30-60 secondes)"):
                    recommendations_result = recommend_jobs_via_api(uploaded_file, top_n=25, min_score=0)
                
                if not recommendations_result:
                    st.error("❌ Échec de la génération des recommandations")
                    st.stop()
                
                st.success(f"✅ {len(recommendations_result['recommendations'])} offres analysées")
                
                # Sauvegarder dans session state
                st.session_state.cv_processed = True
                st.session_state.cv_skills = skills_result['technical_skills']
                st.session_state.recommendations = recommendations_result['recommendations']
                st.session_state.cv_skills_count = skills_result['total_skills']
                
                st.rerun()
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.cv_processed = False
                st.session_state.cv_skills = []
                st.session_state.recommendations = []
                st.session_state.cv_skills_count = 0
                st.rerun()
    
    # Si pas de CV traité, afficher les instructions
    if not st.session_state.cv_processed:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### 📄 Comment ça marche ?")
        st.markdown("""
        1. **Uploadez votre CV** au format PDF
        2. **Cliquez sur "Analyser mon CV"**
        3. **Obtenez des recommandations personnalisées** basées sur vos compétences
        
        Notre système utilise l'IA pour :
        - ✅ Extraire automatiquement vos compétences (via API)
        - ✅ Comparer votre profil avec 25+ offres d'emploi
        - ✅ Calculer un score de matching sémantique
        - ✅ Recommander les meilleures opportunités
        
        ⏱️ **Temps de traitement estimé** : 30-60 secondes (appels API)
        
        🔌 **Architecture** : Streamlit → API FastAPI → Modèles IA
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.stop()
    
    # Si CV traité, afficher les résultats
    cv_skills = st.session_state.cv_skills
    recommendations = st.session_state.recommendations
    cv_skills_count = st.session_state.cv_skills_count
    
    # Sidebar - Filtres
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par score minimum
    min_score = st.sidebar.slider(
        "Score minimum (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Afficher uniquement les offres avec un score ≥ ce seuil"
    )
    
    # Filtre Remote
    remote_filter = st.sidebar.radio(
        "Type de travail",
        options=["Tous", "Remote uniquement", "On-site uniquement"],
        index=0,
        help="Filtrer par mode de travail"
    )

    # Filtre par niveau d'expérience
    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 Niveau d'expérience")
    
    # Extraire les niveaux uniques
    all_experience_levels = sorted(list(set(job['experience_required'] for job in recommendations)))
    
    # Multiselect
    selected_experiences = st.sidebar.multiselect(
        "Sélectionnez un ou plusieurs niveaux",
        options=all_experience_levels,
        default=all_experience_levels,  # Tous sélectionnés par défaut
        help="Maintenez Ctrl (Windows) ou Cmd (Mac) pour sélectionner plusieurs niveaux"
    )
    
    # Si rien n'est sélectionné, afficher tous
    if not selected_experiences:
        selected_experiences = all_experience_levels
    
    # Appliquer les filtres
    filtered_recs = recommendations.copy()
    
    # Filtre score
    filtered_recs = [job for job in filtered_recs if job['score'] >= min_score]
    
    # Filtre remote
    if remote_filter == "Remote uniquement":
        filtered_recs = [job for job in filtered_recs if job['remote']]
    elif remote_filter == "On-site uniquement":
        filtered_recs = [job for job in filtered_recs if not job['remote']]

    # Filtre expérience
    filtered_recs = [job for job in filtered_recs if job['experience_required'] in selected_experiences]
    
    # Statistiques globales
    st.markdown("---")
    st.header("📊 Vue d'ensemble")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Compétences CV", cv_skills_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Offres analysées", len(recommendations))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Offres filtrées", len(filtered_recs))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if filtered_recs:
            st.metric("Meilleur score", f"{filtered_recs[0]['score']:.1f}%")
        else:
            st.metric("Meilleur score", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution des scores
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Distribution des matches")
        excellent = len([j for j in filtered_recs if j['score'] >= 70])
        good = len([j for j in filtered_recs if 50 <= j['score'] < 70])
        medium = len([j for j in filtered_recs if 40 <= j['score'] < 50])
        low = len([j for j in filtered_recs if j['score'] < 40])
        
        st.markdown(f"🟢 **Excellent match (≥70%)** : {excellent} offres")
        st.markdown(f"🟡 **Bon match (50-70%)** : {good} offres")
        st.markdown(f"🟠 **Match moyen (40-50%)** : {medium} offres")
        st.markdown(f"🔴 **Match faible (<40%)** : {low} offres")
        
        st.info("💡 Le score est basé **uniquement** sur le matching sémantique des compétences")

    with col2:
        st.subheader("🔧 Vos compétences")
        for i, skill in enumerate(cv_skills[:10], 1):
            st.markdown(f"{i}. {skill}")
        
        if len(cv_skills) > 10:
            with st.expander(f"Voir les {len(cv_skills) - 10} autres compétences"):
                for i, skill in enumerate(cv_skills[10:], 11):
                    st.markdown(f"{i}. {skill}")
    
    # Liste des offres
    st.markdown("---")
    st.header(f"🏆 Top {min(10, len(filtered_recs))} Offres Recommandées")
    
    if not filtered_recs:
        st.warning("Aucune offre ne correspond aux critères sélectionnés")
        st.info("💡 Essayez de réduire le score minimum ou d'élargir les filtres")
    else:
        # Nombre d'offres à afficher
        num_to_show = st.selectbox(
            "Nombre d'offres à afficher",
            options=[5, 10, 15, 20, len(filtered_recs)],
            index=1 if len(filtered_recs) >= 10 else 0
        )
        
        # Afficher les offres
        for i, job in enumerate(filtered_recs[:num_to_show], 1):
            display_job_card(job, i)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🎯 AI Career Coach | Powered by FastAPI + Sentence-Transformers + Streamlit<br>"
        f"🔌 Connected to API: {API_BASE_URL}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()