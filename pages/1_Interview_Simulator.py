"""
🎤 Simulation d'Entretien Interactive
Page Streamlit pour pratiquer les entretiens d'embauche avec IA
"""

import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(
    page_title="🎤 Simulateur d'Entretien",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .interview-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .question-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #1f77b4;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .answer-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }
    .evaluation-excellent {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    .evaluation-good {
        background-color: #FFC107;
        color: black;
        padding: 1rem;
        border-radius: 8px;
    }
    .evaluation-medium {
        background-color: #FF9800;
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    .progress-bar {
        width: 100%;
        height: 30px;
        background-color: #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.3s ease;
    }
    .tip-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS API
# ============================================================================

def generate_interview_questions(cv_skills, job_id, num_questions=8):
    """Générer des questions d'entretien via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/simulate-interview",
            json={
                "cv_skills": cv_skills,
                "job_id": job_id,
                "num_questions": num_questions
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur API {response.status_code}: {response.json().get('detail', 'Erreur inconnue')}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None


def evaluate_answer_api(question, answer, question_type, target_skill=None):
    """Évaluer une réponse via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/evaluate-answer",
            json={
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "target_skill": target_skill
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur évaluation: {response.json().get('detail')}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None


def get_job_details(job_id):
    """Récupérer les détails d'un job via API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/jobs/{job_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# ============================================================================
# INITIALISATION SESSION STATE
# ============================================================================

if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
    st.session_state.questions = []
    st.session_state.current_question_index = 0
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.interview_completed = False
    st.session_state.job_title = ""
    st.session_state.job_id = ""

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.markdown('<div class="interview-header">🎤 Simulateur d\'Entretien IA</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    Entraînez-vous aux entretiens d'embauche avec notre IA basée sur <strong>Groq (Llama 3.3 70B)</strong>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# ÉTAPE 1 : SÉLECTION DU JOB
# ============================================================================

if not st.session_state.interview_started:
    st.markdown("---")
    st.header("📋 Étape 1 : Sélectionnez un poste")
    
    # Vérifier si l'utilisateur a uploadé un CV
    if 'cv_skills' not in st.session_state or not st.session_state.cv_skills:
        st.warning("⚠️ Vous devez d'abord uploader votre CV sur la page principale")
        st.markdown("[👉 Retour à l'accueil](../)")
        st.stop()
    
    cv_skills = st.session_state.cv_skills
    
    st.success(f"✅ Compétences détectées : {len(cv_skills)}")
    with st.expander("🔍 Voir vos compétences"):
        st.markdown(", ".join(cv_skills[:20]))
    
    # Sélection du job
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Pour quel poste voulez-vous vous entraîner ?")
        
        # Récupérer la liste des jobs
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/jobs?limit=25")
            if response.status_code == 200:
                jobs = response.json()
                
                # Créer un dict {nom_affiché: job_id}
                job_options = {
                    f"{job['title']} - {job['company']} ({job['location']})": job['job_id']
                    for job in jobs
                }
                
                selected_job_display = st.selectbox(
                    "Choisissez un poste",
                    options=list(job_options.keys()),
                    help="Sélectionnez le poste pour lequel vous souhaitez simuler un entretien"
                )
                
                selected_job_id = job_options[selected_job_display]
                
                # Afficher les détails
                job_details = get_job_details(selected_job_id)
                if job_details:
                    st.markdown(f"**📍 Localisation**: {job_details['location']}")
                    st.markdown(f"**💼 Expérience**: {job_details['experience_required']}")
                    st.markdown(f"**🏠 Remote**: {'Oui' if job_details['remote'] else 'Non'}")
                
            else:
                st.error("❌ Impossible de charger les offres")
                st.stop()
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.stop()
    
    with col2:
        st.markdown("### ⚙️ Paramètres")
        num_questions = st.slider(
            "Nombre de questions",
            min_value=4,
            max_value=12,
            value=8,
            step=2,
            help="Nombre total de questions (50% RH + 50% techniques)"
        )
        
        st.markdown("### 💡 Conseils")
        st.markdown("""
        - Prenez votre temps
        - Donnez des exemples concrets
        - Structurez vos réponses (STAR)
        - Soyez authentique
        """)
    
    # Bouton de démarrage
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Démarrer la simulation", type="primary", use_container_width=True):
            with st.spinner("⏳ Génération des questions avec Groq... (30-45 secondes)"):
                result = generate_interview_questions(
                    cv_skills=cv_skills,
                    job_id=selected_job_id,
                    num_questions=num_questions
                )
                
                if result:
                    # Fusionner les questions
                    all_questions = result['rh_questions'] + result['technical_questions']
                    
                    # Sauvegarder dans session state
                    st.session_state.interview_started = True
                    st.session_state.questions = all_questions
                    st.session_state.job_title = result['job_title']
                    st.session_state.job_id = selected_job_id
                    st.session_state.current_question_index = 0
                    st.session_state.answers = []
                    st.session_state.evaluations = []
                    
                    st.success(f"✅ {len(all_questions)} questions générées !")
                    st.rerun()
                else:
                    st.error("❌ Échec de la génération des questions")

# ============================================================================
# ÉTAPE 2 : QUESTIONS & RÉPONSES
# ============================================================================

elif st.session_state.interview_started and not st.session_state.interview_completed:
    
    questions = st.session_state.questions
    current_index = st.session_state.current_question_index
    current_question = questions[current_index]
    
    # Barre de progression
    progress_percentage = (current_index / len(questions)) * 100
    
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percentage}%;"></div>
    </div>
    <p style="text-align: center; margin-top: 0.5rem;">
        Question {current_index + 1} / {len(questions)} ({progress_percentage:.0f}%)
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Afficher la question
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    
    question_type_emoji = {
        'présentation': '👤',
        'motivation': '🎯',
        'soft_skills': '🤝',
        'projet': '📂',
        'compétence_technique': '💻',
        'architecture': '🏗️',
        'outil': '🔧',
        'résolution_problème': '🔍'
    }
    
    emoji = question_type_emoji.get(current_question['type'], '❓')
    
    st.markdown(f"## {emoji} Question {current_index + 1}")
    st.markdown(f"### {current_question['question']}")
    
    if 'skill' in current_question:
        st.markdown(f"**🎯 Compétence évaluée**: {current_question['skill']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone de réponse
    st.markdown("### 💬 Votre réponse")
    
    answer = st.text_area(
        "Répondez ici",
        height=200,
        placeholder="Prenez votre temps pour structurer votre réponse...",
        key=f"answer_{current_index}"
    )
    
    # Conseils contextuels
    st.markdown('<div class="tip-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Conseils pour cette question")
    
    if current_question['type'] in ['présentation', 'projet']:
        st.markdown("""
        - Utilisez la structure **STAR** : Situation, Tâche, Action, Résultat
        - Donnez des **exemples concrets** avec des chiffres si possible
        - Mentionnez les **technologies utilisées**
        """)
    elif current_question['type'] == 'motivation':
        st.markdown("""
        - Montrez que vous avez **recherché l'entreprise**
        - Liez vos **objectifs** avec la mission de l'entreprise
        - Soyez **authentique et passionné**
        """)
    elif current_question['type'] in ['compétence_technique', 'architecture', 'outil']:
        st.markdown("""
        - Commencez par une **définition claire**
        - Donnez des **exemples d'utilisation**
        - Mentionnez les **bonnes pratiques**
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Boutons d'action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_index > 0:
            if st.button("⬅️ Question précédente"):
                st.session_state.current_question_index -= 1
                st.rerun()
    
    with col2:
        if st.button("✅ Valider et continuer", type="primary", use_container_width=True):
            if not answer or len(answer.strip()) < 20:
                st.error("⚠️ Votre réponse est trop courte (minimum 20 caractères)")
            else:
                with st.spinner("⏳ Évaluation de votre réponse avec Groq... (10-20 secondes)"):
                    evaluation = evaluate_answer_api(
                        question=current_question['question'],
                        answer=answer,
                        question_type=current_question['type'],
                        target_skill=current_question.get('skill')
                    )
                    
                    if evaluation:
                        # Sauvegarder
                        st.session_state.answers.append({
                            'question_id': current_question['id'],
                            'question': current_question['question'],
                            'answer': answer,
                            'type': current_question['type']
                        })
                        st.session_state.evaluations.append(evaluation)
                        
                        # Afficher le score immédiatement
                        score = evaluation['score']
                        if score >= 70:
                            st.success(f"🎉 Excellent ! Score: {score:.0f}/100")
                        elif score >= 50:
                            st.info(f"👍 Bien ! Score: {score:.0f}/100")
                        else:
                            st.warning(f"📈 À améliorer. Score: {score:.0f}/100")
                        
                        # Passer à la question suivante
                        if current_index < len(questions) - 1:
                            st.session_state.current_question_index += 1
                            st.rerun()
                        else:
                            # Fin de l'entretien
                            st.session_state.interview_completed = True
                            st.rerun()
                    else:
                        st.error("❌ Erreur lors de l'évaluation")
    
    with col3:
        if st.button("⏭️ Passer", help="Passer cette question (non évaluée)"):
            st.session_state.current_question_index += 1
            if st.session_state.current_question_index >= len(questions):
                st.session_state.interview_completed = True
            st.rerun()

# ============================================================================
# ÉTAPE 3 : RÉSULTATS
# ============================================================================

elif st.session_state.interview_completed:
    
    st.markdown("---")
    st.header("🎉 Simulation terminée !")
    
    evaluations = st.session_state.evaluations
    
    if not evaluations:
        st.warning("Aucune question n'a été évaluée")
        if st.button("🔄 Recommencer"):
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.rerun()
        st.stop()
    
    # Statistiques globales
    scores = [ev['score'] for ev in evaluations]
    avg_score = sum(scores) / len(scores)
    
    st.markdown("---")
    st.subheader("📊 Résultats globaux")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score moyen", f"{avg_score:.1f}/100")
    with col2:
        st.metric("Questions répondues", f"{len(evaluations)}/{len(st.session_state.questions)}")
    with col3:
        st.metric("Score max", f"{max(scores):.0f}/100")
    with col4:
        st.metric("Score min", f"{min(scores):.0f}/100")
    
    # Distribution des scores
    st.markdown("---")
    st.subheader("📈 Distribution des performances")
    
    excellent = sum(1 for s in scores if s >= 70)
    bon = sum(1 for s in scores if 50 <= s < 70)
    moyen = sum(1 for s in scores if 30 <= s < 50)
    faible = sum(1 for s in scores if s < 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"🟢 **Excellent (≥70)** : {excellent} ({excellent/len(scores)*100:.0f}%)")
        st.markdown(f"🟡 **Bon (50-70)** : {bon} ({bon/len(scores)*100:.0f}%)")
        st.markdown(f"🟠 **Moyen (30-50)** : {moyen} ({moyen/len(scores)*100:.0f}%)")
        st.markdown(f"🔴 **Faible (<30)** : {faible} ({faible/len(scores)*100:.0f}%)")
    
    with col2:
        # Décision finale
        if avg_score >= 75:
            st.success("🎉 **Décision** : Excellent profil ! Vous êtes prêt pour l'entretien.")
        elif avg_score >= 60:
            st.info("👍 **Décision** : Bon profil. Quelques axes d'amélioration à travailler.")
        elif avg_score >= 45:
            st.warning("📈 **Décision** : Profil prometteur. Entraînez-vous davantage.")
        else:
            st.error("🔄 **Décision** : À retravailler. Pratiquez régulièrement.")
    
    # Détail par question
    st.markdown("---")
    st.subheader("📋 Détail des réponses")
    
    for idx, (answer_data, evaluation) in enumerate(zip(st.session_state.answers, evaluations), 1):
        score = evaluation['score']
        
        # Classe CSS selon le score
        if score >= 70:
            card_class = "evaluation-excellent"
            emoji = "🟢"
        elif score >= 50:
            card_class = "evaluation-good"
            emoji = "🟡"
        else:
            card_class = "evaluation-medium"
            emoji = "🟠"
        
        with st.expander(f"{emoji} Question {idx} - Score: {score:.0f}/100"):
            st.markdown(f"**❓ Question** : {answer_data['question']}")
            st.markdown(f"**💬 Votre réponse** : {answer_data['answer'][:200]}...")
            
            st.markdown("---")
            st.markdown(f"**📊 Score** : {score:.0f}/100")
            st.markdown(f"**📝 Évaluation** : {evaluation['evaluation']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Points forts** :")
                for point in evaluation['points_forts']:
                    st.markdown(f"- {point}")
            
            with col2:
                st.markdown("**⚠️ À améliorer** :")
                for point in evaluation['points_amelioration']:
                    st.markdown(f"- {point}")
            
            if 'recommandations' in evaluation:
                st.markdown("**💡 Recommandations** :")
                for reco in evaluation['recommandations']:
                    st.markdown(f"- {reco}")
    
    # Actions finales
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Nouvelle simulation", use_container_width=True):
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            st.session_state.answers = []
            st.session_state.evaluations = []
            st.rerun()
    
    with col2:
        if st.button("📊 Voir les statistiques", use_container_width=True):
            st.info("Fonctionnalité en développement")
    
    with col3:
        if st.button("🏠 Retour à l'accueil", use_container_width=True):
            st.switch_page("app.py")

# ============================================================================
# SIDEBAR - AIDE
# ============================================================================

st.sidebar.header("❓ Aide")
st.sidebar.markdown("""
### Comment ça marche ?

1. **Sélectionnez un poste** pour lequel vous voulez vous entraîner
2. **Répondez aux questions** une par une
3. **Recevez un feedback immédiat** après chaque réponse
4. **Analysez vos résultats** à la fin

### 💡 Conseils

- **Structurez vos réponses** (STAR: Situation, Tâche, Action, Résultat)
- **Donnez des exemples concrets** de projets
- **Mentionnez des chiffres** (amélioration de X%, réduction de Y%)
- **Soyez authentique**

### 🎯 Objectif

S'entraîner aux entretiens pour :
- Identifier vos points forts
- Travailler vos axes d'amélioration
- Gagner en confiance
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **Propulsé par Groq (Llama 3.3 70B)**")