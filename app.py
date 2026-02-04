"""
🎯 AI Career Coach - Dashboard Streamlit (VERSION API)
Application de recommandation d'offres d'emploi basée sur l'analyse de CV
Utilise l'API FastAPI pour tous les traitements
"""

import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"  # URL de l'API FastAPI

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS MODERNE ET DESIGN SYSTEM
# ============================================================================

st.markdown("""
<style>
    /* ========== POLICE & THÈME GLOBAL ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ========== HEADER ANIMÉ ========== */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #ffffff;
        margin-bottom: 3rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* ========== ZONE D'UPLOAD MODERNE ========== */
    .upload-box {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .upload-box:hover {
        border-color: #764ba2;
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
    }
    
    /* ========== CARTES DE JOBS REDESIGNÉES ========== */
    .job-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #e0e0e0;
        position: relative;
        overflow: hidden;
    }
    
    .job-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .job-card:hover {
        transform: translateX(8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .job-card-excellent {
        border-left-color: #10b981;
    }
    
    .job-card-excellent::before {
        background: linear-gradient(180deg, #10b981 0%, #059669 100%);
    }
    
    .job-card-excellent:hover::before {
        width: 100%;
        opacity: 0.05;
    }
    
    .job-card-good {
        border-left-color: #f59e0b;
    }
    
    .job-card-good::before {
        background: linear-gradient(180deg, #f59e0b 0%, #d97706 100%);
    }
    
    .job-card-good:hover::before {
        width: 100%;
        opacity: 0.05;
    }
    
    .job-card-medium {
        border-left-color: #ef4444;
    }
    
    .job-card-medium::before {
        background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%);
    }
    
    .job-card-medium:hover::before {
        width: 100%;
        opacity: 0.05;
    }
    
    /* ========== BADGES DE SCORE MODERNES ========== */
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .score-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .score-good {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .score-low {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
    }
    
    /* ========== CARTES MÉTRIQUES GLASSMORPHISM ========== */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.15);
    }
    
    /* ========== STATUT API AVEC ICÔNES ========== */
    .api-status-connected {
        color: #10b981;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .api-status-disconnected {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* ========== BOUTONS REDESIGNÉS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* ========== SIDEBAR MODERNE ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* ========== EXPANDER STYLÉ ========== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-weight: 600 !important;
    }
    
    /* ========== ANIMATIONS D'ENTRÉE ========== */
    .element-container {
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ========== SCROLLBAR PERSONNALISÉE ========== */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed #667eea;
    }
    
    /* ========== SELECTBOX ========== */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
    }
    
    /* ========== SLIDER ========== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
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
        cv_file.seek(0)
        
        files = {
            "file": (cv_file.name, cv_file, "application/pdf")
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract-skills",
            files=files,
            timeout=120
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
    Obtenir des recommandations via l'API FAISS (rapide)
    """
    try:
        cv_file.seek(0)
        
        files = {
            "file": ("cv.pdf", cv_file, "application/pdf")
        }
        
        faiss_min_score = min_score / 100.0 if min_score > 1 else min_score
        
        params = {
            "top_k": top_n,
            "min_score": faiss_min_score
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/jobs/recommend-fast",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            adapted_recommendations = []
            for job in data['recommendations']:
                adapted_job = {
                    "job_id": job.get("job_id", "N/A"),
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", "N/A"),
                    "location": job.get("location", "N/A"),
                    "remote": job.get("remote", False),
                    "experience_required": job.get("experience_required", "N/A"),
                    "category": job.get("category", "Non spécifié"),
                    "score": job.get("faiss_score_percent", 0),
                    "skills_match": job.get("faiss_score_percent", 0),
                    "experience_match": 0,
                    "location_match": 0,
                    "competition_factor": 0,
                    "matching_skills": []
                }
                adapted_recommendations.append(adapted_job)
            
            return {
                "recommendations": adapted_recommendations,
                "total_jobs_analyzed": data.get("total_jobs_indexed", 0),
                "cv_skills_count": 0
            }
        else:
            st.error(f"❌ Erreur API (Code {response.status_code})")
            try:
                error_detail = response.json().get('detail', 'Erreur inconnue')
                st.error(f"Détail : {error_detail}")
            except:
                st.error(f"Réponse : {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout : L'API met trop de temps à répondre")
        return None
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
    """Afficher une carte d'offre d'emploi moderne"""
    
    score = job.get('score', job.get('faiss_score_percent', 0))
    score_class, emoji = get_score_class(score)
    
    card_class = f"job-card job-card-{score_class}" if score_class != "low" else "job-card"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    # En-tête
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {emoji} #{rank} - {job.get('title', 'N/A')}")
        st.markdown(f"**🏢 {job.get('company', 'N/A')}** | 📍 {job.get('location', 'N/A')}")
    
    with col2:
        st.markdown(
            f'<div class="score-badge score-{score_class}">{score:.1f}%</div>', 
            unsafe_allow_html=True
        )
    
    # Détails
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**💼 Expérience** : {job.get('experience_required', 'N/A')}")
        st.markdown(f"**🏠 Remote** : {'Oui ✅' if job.get('remote', False) else 'Non ❌'}")
    
    with col2:
        skills_match = job.get('skills_match', score)
        st.markdown(f"**🎯 Match compétences** : {skills_match:.1f}%")
        competition = job.get('competition_factor', 0)
        st.markdown(f"**📊 Facteur compétition** : {competition}%")
    
    # Scores détaillés
    if job.get('experience_match') or job.get('location_match'):
        with st.expander("📊 Voir les scores détaillés"):
            cols = st.columns(4)
            cols[0].metric("Compétences", f"{job.get('skills_match', score):.1f}%")
            cols[1].metric("Expérience", f"{job.get('experience_match', 0)}%")
            cols[2].metric("Localisation", f"{job.get('location_match', 0)}%")
            cols[3].metric("Compétition", f"{job.get('competition_factor', 0)}%")
    
    # Compétences matchées
    matching_skills = job.get('matching_skills', [])
    if matching_skills:
        with st.expander("🔧 Compétences matchées"):
            for skill in matching_skills:
                st.markdown(f"- {skill}")
    else:
        if job.get('category'):
            st.info(f"📂 Catégorie : {job['category']}")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    """Application principale"""
    
    # Header moderne
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
                
                with st.spinner("🔍 Extraction des compétences via API..."):
                    skills_result = extract_skills_via_api(uploaded_file)
                
                if not skills_result:
                    st.error("❌ Échec de l'extraction des compétences")
                    st.stop()
                
                st.success(f"✅ {skills_result['total_skills']} compétences détectées")
                
                with st.spinner("🎯 Calcul des recommandations via API..."):
                    recommendations_result = recommend_jobs_via_api(uploaded_file, top_n=25, min_score=0)
                
                if not recommendations_result:
                    st.error("❌ Échec de la génération des recommandations")
                    st.stop()
                
                st.success(f"✅ {len(recommendations_result['recommendations'])} offres analysées")
                
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
    
    # Si pas de CV traité
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
    
    min_score = st.sidebar.slider(
        "Score minimum (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5
    )
    
    remote_filter = st.sidebar.radio(
        "Type de travail",
        options=["Tous", "Remote uniquement", "On-site uniquement"],
        index=0
    )
    
    # Appliquer les filtres
    filtered_recs = recommendations.copy()
    
    filtered_recs = [
        job for job in filtered_recs 
        if job.get('score', job.get('faiss_score_percent', 0)) >= min_score
    ]
    
    if remote_filter == "Remote uniquement":
        filtered_recs = [job for job in filtered_recs if job['remote']]
    elif remote_filter == "On-site uniquement":
        filtered_recs = [job for job in filtered_recs if not job['remote']]
    
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
            best_score = filtered_recs[0].get('score', filtered_recs[0].get('faiss_score_percent', 0))
            st.metric("Meilleur score", f"{best_score:.1f}%")
        else:
            st.metric("Meilleur score", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution des scores
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Distribution des matches")
        
        def get_job_score(job):
            return job.get('score', job.get('faiss_score_percent', 0))

        excellent = len([j for j in filtered_recs if get_job_score(j) >= 70])
        good = len([j for j in filtered_recs if 50 <= get_job_score(j) < 70])
        medium = len([j for j in filtered_recs if 40 <= get_job_score(j) < 50])
        low = len([j for j in filtered_recs if get_job_score(j) < 40])
        
        st.markdown(f"🟢 **Excellent match (≥70%)** : {excellent} offres")
        st.markdown(f"🟡 **Bon match (50-70%)** : {good} offres")
        st.markdown(f"🟠 **Match moyen (40-50%)** : {medium} offres")
        st.markdown(f"🔴 **Match faible (<40%)** : {low} offres")
    
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
        num_to_show = st.selectbox(
            "Nombre d'offres à afficher",
            options=[5, 10, 15, 20, len(filtered_recs)],
            index=1 if len(filtered_recs) >= 10 else 0
        )
        
        for i, job in enumerate(filtered_recs[:num_to_show], 1):
            display_job_card(job, i)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: white; font-size: 0.9rem;'>"
        "🎯 AI Career Coach | Powered by FastAPI + Sentence-Transformers + Streamlit<br>"
        f"🔌 Connected to API: {API_BASE_URL}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()