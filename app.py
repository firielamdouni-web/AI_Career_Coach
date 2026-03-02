"""
🎯 AI Career Coach - Dashboard Streamlit
Interface unique : offres locales (JSON) + offres réelles (JSearch)
"""

import streamlit as st
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header { font-size:2.5rem; font-weight:bold; text-align:center; color:#1f77b4; margin-bottom:.5rem; }
.sub-header  { font-size:1.1rem; text-align:center; color:#666; margin-bottom:2rem; }
.source-local   { background:#e3f2fd; color:#1565c0; padding:2px 8px; border-radius:12px; font-size:.8rem; font-weight:bold; }
.source-scraped { background:#e8f5e9; color:#2e7d32; padding:2px 8px; border-radius:12px; font-size:.8rem; font-weight:bold; }
.score-badge { display:inline-block; padding:.3rem .8rem; border-radius:20px; font-weight:bold; font-size:1.1rem; }
.score-excellent { background:#4CAF50; color:white; }
.score-good      { background:#FFC107; color:#333; }
.score-medium    { background:#FF9800; color:white; }
.score-low       { background:#9E9E9E; color:white; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS API
# ============================================================================

@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_api_stats():
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def extract_skills_via_api(cv_file):
    try:
        cv_file.seek(0)
        r = requests.post(
            f"{API_BASE_URL}/api/v1/extract-skills",
            files={"file": (cv_file.name, cv_file, "application/pdf")},
            timeout=480
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"❌ Extraction échouée ({r.status_code}) : {r.json().get('detail','')}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout extraction")
        return None
    except Exception as e:
        st.error(f"❌ {e}")
        return None


def recommend_jobs_via_api(cv_file, top_n=50, min_score=0.0, live_scrape=True):
    """
    Appelle /api/v1/recommend-jobs avec live_scrape=True pour inclure
    les offres JSearch en temps réel.
    """
    try:
        cv_file.seek(0)
        r = requests.post(
            f"{API_BASE_URL}/api/v1/recommend-jobs",
            files={"file": (cv_file.name, cv_file, "application/pdf")},
            params={
                "top_n":        top_n,
                "min_score":    min_score,
                "use_faiss":    "false",
                "live_scrape":  str(live_scrape).lower()
            },
            timeout=600  # 10 min car JSearch peut être lent
        )
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get('detail', r.text)
        except Exception:
            detail = r.text
        st.error(f"❌ Recommandations échouées ({r.status_code}) : {detail}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout — le scraping JSearch a pris trop de temps")
        return None
    except Exception as e:
        st.error(f"❌ {e}")
        return None


# ============================================================================
# AFFICHAGE D'UNE CARTE D'OFFRE
# ============================================================================

def get_score_class(score):
    if score >= 70:   return "excellent", "🟢"
    if score >= 50:   return "good",      "🟡"
    if score >= 40:   return "medium",    "🟠"
    return "low", "🔴"


def display_job_card(job: dict, rank: int, cv_skills: list):
    score = job['score']
    score_class, emoji = get_score_class(score)
    # ← CHANGEMENT : détecter aussi via is_scraped (pas seulement l'URL)
    is_real = bool(job.get('is_scraped') or job.get('url', '').startswith('http'))

    col_title, col_score = st.columns([4, 1])
    with col_title:
        source_html = (
            '<span class="source-scraped">🌐 Offre réelle (JSearch)</span>'
            if is_real else
            '<span class="source-local">📁 Offre locale</span>'
        )
        st.markdown(
            f"### {emoji} #{rank} — {job['title']}  "
            f"<br>{source_html}",
            unsafe_allow_html=True
        )
        st.markdown(f"**🏢 {job['company']}** &nbsp;|&nbsp; 📍 {job['location']}")
        # ← NOUVEAU : afficher la source (hellowork, indeed, etc.)
        if job.get('source') and job['source'] not in ('local',):
            st.caption(f"🔗 Source : {job['source']}")

    with col_score:
        st.markdown(
            f'<div class="score-badge score-{score_class}" '
            f'style="margin-top:1.5rem;">{score:.1f}%</div>',
            unsafe_allow_html=True
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**💼 Expérience** : {job.get('experience_required','N/A')}")
        st.markdown(f"**🏠 Remote** : {'✅ Oui' if job.get('remote') else '❌ Non'}")
        if job.get('employment_type'):
            st.markdown(f"**📋 Contrat** : {job['employment_type']}")
    with c2:
        st.markdown(f"**🎯 Score matching** : {score:.1f}%")
        if job.get('salary_min') and job.get('salary_max'):
            st.markdown(f"**💰 Salaire** : {job['salary_min']:,.0f} – {job['salary_max']:,.0f} €")
        elif job.get('salary_min'):
            st.markdown(f"**💰 Salaire** : à partir de {job['salary_min']:,.0f} €")
    with c3:
        if job.get('ml_available'):
            ml_label = job.get('ml_label', 'N/A')
            colors = {'Perfect Fit':('🟢','#4CAF50'), 'Partial Fit':('🟡','#FFC107'), 'No Fit':('🔴','#f44336')}
            ml_e, ml_c = colors.get(ml_label, ('⚪','#9E9E9E'))
            st.markdown(
                f"**🤖 Prédiction ML** : "
                f"<span style='color:{ml_c};font-weight:bold;'>{ml_e} {ml_label}</span>",
                unsafe_allow_html=True
            )
            proba = job.get('ml_probabilities')
            if proba:
                with st.expander("📊 Probabilités ML"):
                    pa, pb, pc = st.columns(3)
                    pa.metric("🔴 No Fit",      f"{proba.get('no_fit',0)*100:.1f}%")
                    pb.metric("🟡 Partial Fit", f"{proba.get('partial_fit',0)*100:.1f}%")
                    pc.metric("🟢 Perfect Fit", f"{proba.get('perfect_fit',0)*100:.1f}%")
        else:
            st.markdown("**🤖 Prédiction ML** : ⚪ N/A")

    # ← NOUVEAU : description pour les offres réelles
    if is_real and job.get('description'):
        with st.expander("📄 Description du poste"):
            desc = job['description']
            st.markdown(desc[:1000] + ("..." if len(desc) > 1000 else ""))

    matching = job.get('matching_skills', [])
    missing  = job.get('missing_skills',  [])

    col_m, col_miss = st.columns(2)
    with col_m:
        with st.expander(f"✅ Compétences matchées ({len(matching)})"):
            if matching:
                cols = st.columns(3)
                for i, s in enumerate(matching):
                    cols[i % 3].markdown(f"✓ {s}")
            else:
                st.info("Aucune compétence matchée")
    with col_miss:
        with st.expander(f"⚠️ Compétences manquantes ({len(missing)})"):
            if missing:
                cols = st.columns(3)
                for i, s in enumerate(missing):
                    cols[i % 3].markdown(f"❌ {s}")
                st.info("💡 Formez-vous sur ces compétences pour améliorer votre score !")
            else:
                st.success("Vous avez toutes les compétences requises 🎉")

    st.markdown("---")
    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button("🎤 Simuler un entretien", key=f"iv_{job['job_id']}", use_container_width=True):
            st.session_state.selected_job_for_interview = job['job_id']
            st.session_state.cv_skills_for_interview    = cv_skills
            st.switch_page("pages/1_Interview_Simulator.py")

    with btn2:
        # ← CHANGEMENT : condition améliorée (is_real inclut is_scraped sans URL)
        if job.get('url') and job['url'].startswith('http'):
            st.markdown(
                f'<a href="{job["url"]}" target="_blank">'
                f'<button style="width:100%;padding:.5rem;background:#4CAF50;'
                f'color:white;border:none;border-radius:4px;cursor:pointer;font-size:1rem;">'
                f'🌐 Voir l\'offre en ligne</button></a>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("&nbsp;")

    st.markdown("---")


# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    st.markdown('<div class="main-header">🎯 AI Career Coach</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Matching intelligent CV ↔ Offres réelles (LinkedIn · Indeed · Glassdoor)</div>',
        unsafe_allow_html=True
    )

    # ── Sidebar : état API ────────────────────────────────────────────────
    st.sidebar.header("🔌 État de l'API")
    health = check_api_health()

    if not health:
        st.sidebar.error("❌ API non accessible")
        st.error("❌ API non accessible. Lancez : `uvicorn src.api:app --reload --port 8000`")
        st.stop()

    st.sidebar.success("✅ API connectée")
    st.sidebar.markdown(f"**Version** : {health.get('version','N/A')}")

    stats = get_api_stats()
    if stats:
        with st.sidebar.expander("📊 Statistiques"):
            st.markdown(f"- Offres locales : **{stats['total_jobs']}**")
            st.markdown(f"- Remote : **{stats['remote_jobs']}**")
            st.markdown(f"- Compétences tech : **{stats['total_technical_skills']}**")
            st.markdown(f"- Soft skills : **{stats['total_soft_skills']}**")

    # ── Session state ─────────────────────────────────────────────────────
    defaults = {
        'cv_processed':       False,
        'cv_skills':          [],
        'recommendations':    [],
        'cv_skills_count':    0,
        'total_analyzed':     0,
        'local_count':        0,
        'scraped_count':      0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # ── Upload CV ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📤 Uploadez votre CV")
    uploaded_file = st.file_uploader(
        "Choisissez votre CV (PDF)", type=['pdf'],
        help="Votre CV sera analysé pour extraire vos compétences"
    )

    if uploaded_file:
        st.markdown(f"📎 **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB)")

        col_analyze, col_reset = st.columns([1, 1])

        with col_analyze:
            if st.button("🚀 Analyser mon CV", type="primary", use_container_width=True):
                # Étape 1 : extraction des compétences
                with st.spinner("🔍 Extraction des compétences..."):
                    skills_result = extract_skills_via_api(uploaded_file)

                if not skills_result:
                    st.stop()

                st.success(f"✅ {skills_result['total_skills']} compétences détectées")

                # Étape 2 : recommandations (avec scraping temps réel)
                with st.spinner(
                    "🌐 Scraping JSearch en temps réel + calcul des scores... "
                    "(peut prendre 1-2 minutes)"
                ):
                    reco_result = recommend_jobs_via_api(
                        uploaded_file,
                        top_n=200,         # ← CHANGEMENT : était 25, inclut toutes les offres DB
                        min_score=0.0,
                        live_scrape=False  # ← le scheduler alimente la DB 2x/jour
                    )

                if not reco_result:
                    st.stop()

                total = reco_result.get('total_jobs_analyzed', 0)
                local = reco_result.get('local_jobs_count', 0)
                scraped = reco_result.get('scraped_jobs_count', 0)
                st.success(
                    f"✅ {total} offres analysées "
                    f"({local} locales + {scraped} réelles JSearch)"
                )

                st.session_state.cv_processed    = True
                st.session_state.cv_skills       = skills_result['technical_skills']
                st.session_state.recommendations = reco_result['recommendations']
                st.session_state.cv_skills_count = skills_result['total_skills']
                st.session_state.total_analyzed  = total
                st.session_state.local_count     = local
                st.session_state.scraped_count   = scraped
                st.rerun()

        with col_reset:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                for k in defaults:
                    st.session_state[k] = defaults[k]
                st.rerun()

    # ── Instructions si aucun CV traité ──────────────────────────────────
    if not st.session_state.cv_processed:
        st.markdown("---")
        with st.container():
            st.markdown("""
            ### 📄 Comment ça marche ?

            1. **Uploadez votre CV** au format PDF
            2. **Cliquez sur "Analyser mon CV"**
            3. Le système :
               - 🔍 Extrait automatiquement vos compétences
               - 🌐 Scrape les offres en **temps réel** sur LinkedIn, Indeed, Glassdoor
               - 🤖 Calcule un score de matching sémantique pour chaque offre
               - 📊 Prédit avec XGBoost si vous êtes **No Fit / Partial Fit / Perfect Fit**
               - 🎤 Simule un entretien personnalisé pour chaque poste

            ⏱️ **Temps estimé** : 1–2 minutes (scraping inclus)
            """)
        st.stop()

    # ── Résultats ─────────────────────────────────────────────────────────
    recommendations = st.session_state.recommendations
    cv_skills       = st.session_state.cv_skills

    # ── Sidebar : filtres ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtres")

    min_score = st.sidebar.slider("Score minimum (%)", 0, 100, 0, 5)

    source_filter = st.sidebar.radio(
        "Source des offres",
        ["Toutes", "📁 Locales uniquement", "🌐 JSearch uniquement"],
        index=0
    )

    remote_filter = st.sidebar.radio(
        "Mode de travail",
        ["Tous", "Remote uniquement", "On-site uniquement"],
        index=0
    )

    all_exp = sorted(set(j.get('experience_required', 'N/A') for j in recommendations))
    sel_exp = st.sidebar.multiselect("Expérience", all_exp, default=all_exp)
    if not sel_exp:
        sel_exp = all_exp

    # ── Appliquer filtres ─────────────────────────────────────────────────
    filtered = recommendations.copy()
    filtered = [j for j in filtered if j['score'] >= min_score]

    if source_filter == "📁 Locales uniquement":
        filtered = [j for j in filtered if not j.get('is_scraped')]
    elif source_filter == "🌐 JSearch uniquement":
        filtered = [j for j in filtered if j.get('is_scraped')]

    if remote_filter == "Remote uniquement":
        filtered = [j for j in filtered if j.get('remote')]
    elif remote_filter == "On-site uniquement":
        filtered = [j for j in filtered if not j.get('remote')]

    filtered = [j for j in filtered if j.get('experience_required', 'N/A') in sel_exp]

    # ── Métriques ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Vue d'ensemble")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🧠 Compétences CV",     st.session_state.cv_skills_count)
    m2.metric("📋 Offres analysées",   st.session_state.total_analyzed)
    m3.metric("📁 Locales",            st.session_state.local_count)
    m4.metric("🌐 JSearch (réelles)",  st.session_state.scraped_count)
    m5.metric("🔍 Offres filtrées",    len(filtered))

    st.markdown("---")
    col_dist, col_skills = st.columns(2)

    with col_dist:
        st.subheader("🎯 Distribution des scores")
        exc  = sum(1 for j in filtered if j['score'] >= 70)
        good = sum(1 for j in filtered if 50 <= j['score'] < 70)
        med  = sum(1 for j in filtered if 40 <= j['score'] < 50)
        low  = sum(1 for j in filtered if j['score'] < 40)
        real = sum(1 for j in filtered if j.get('is_scraped'))
        loc  = len(filtered) - real

        st.markdown(f"🟢 **Excellent (≥70%)** : {exc} offres")
        st.markdown(f"🟡 **Bon (50–70%)** : {good} offres")
        st.markdown(f"🟠 **Moyen (40–50%)** : {med} offres")
        st.markdown(f"🔴 **Faible (<40%)** : {low} offres")
        st.markdown("---")
        st.markdown(f"🌐 **Offres réelles JSearch** : {real}")
        st.markdown(f"📁 **Offres locales** : {loc}")

    with col_skills:
        st.subheader("🔧 Vos compétences détectées")
        for i, s in enumerate(cv_skills[:12], 1):
            st.markdown(f"{i}. {s}")
        if len(cv_skills) > 12:
            with st.expander(f"Voir {len(cv_skills)-12} autres compétences"):
                for i, s in enumerate(cv_skills[12:], 13):
                    st.markdown(f"{i}. {s}")

    # ── Liste des offres ──────────────────────────────────────────────────
    st.markdown("---")
    st.header(f"🏆 Offres Recommandées ({len(filtered)} résultats)")

    if not filtered:
        st.warning("Aucune offre ne correspond aux filtres sélectionnés.")
        st.info("💡 Réduisez le score minimum ou changez les filtres.")
        st.stop()

    num_show = st.selectbox(
        "Nombre d'offres à afficher",
        options=[10, 20, 30, 50, len(filtered)],
        index=0
    )

    for i, job in enumerate(filtered[:num_show], 1):
        with st.container():
            display_job_card(job, i, cv_skills)


if __name__ == "__main__":
    main()