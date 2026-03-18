import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
import plotly.graph_objects as go

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(page_title="Monitoring MLOps", page_icon="📊", layout="wide")

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.markdown("## 📊 Dashboard MLOps & Système (Temps Réel)")
with col_btn:
    if st.button("🔄 Rafraîchir"):
        st.rerun()

st.markdown("Visualisation en temps réel des performances de l'application et de la base de données réelle.")
st.divider()

try:
    stats_data = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5).json()
    monit_data = requests.get(f"{API_BASE_URL}/api/v1/monitoring-data", timeout=5).json()
    
    total_jobs = stats_data.get('total_jobs', 0)
    local_jobs = stats_data.get('jobs_by_category', {}).get('Local', 0)
    scraped_jobs = total_jobs - local_jobs
    
    df_logs = pd.DataFrame(monit_data.get('logs', []))
    scores = monit_data.get('scores', [])
    skills = monit_data.get('top_skills', [])
    
except Exception as e:
    st.error(f"Erreur de connexion à l'API : {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

total_req = len(df_logs) if not df_logs.empty else 0
avg_time = f"{df_logs['response_time_ms'].mean() / 1000:.2f}s" if not df_logs.empty else "0s"
success_rate = f"{(len(df_logs[df_logs['status_code'] < 400]) / total_req * 100):.1f}%" if total_req > 0 else "100%"

with col1:
    st.metric(label="Total Requêtes (Actives)", value=total_req, delta=f"+{total_req} depuis lancement")
with col2:
    st.metric(label="Temps de réponse API (moy)", value=avg_time, delta="-0.1s optimisé", delta_color="inverse")
with col3:
    st.metric(label="Taux de succès HTTP", value=success_rate, delta="Opérationnel")
with col4:
    st.metric(label="Offres indexées", value=total_jobs, delta=f"{scraped_jobs} récupérées JSearch")

st.markdown("### 📈 Surveillances des flux API")
col_api1, col_api2 = st.columns(2)

if not df_logs.empty:
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    with col_api1:
        df_time = df_logs.groupby(['timestamp', 'endpoint'])['response_time_ms'].mean().reset_index()
        fig_time = px.line(df_time, x='timestamp', y='response_time_ms', color='endpoint', title="Temps de réponse par Endpoint (ms)")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_api2:
        df_status = df_logs['status_code'].astype(str).value_counts().reset_index()
        df_status.columns = ['status_code', 'count']
        
        color_map = {'200': '#22c55e', '404': '#f59e0b', '422': '#f59e0b', '500': '#ef4444'}
        
        fig_pie = px.pie(df_status, names='status_code', values='count', title="Répartition des Codes HTTP",
                         color='status_code', color_discrete_map=color_map)
        fig_pie.update_traces(hole=.4)
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("Aucune requête API enregistrée pour le moment. Naviguez sur l'application !")

st.markdown("### 🧠 Modèles MLOps & Entraînements")
col_ia1, col_ia2, col_ia3, col_ia4 = st.columns([1.5, 1.5, 1, 1]) 

with col_ia1:
    if len(scores) > 0:
        df_scores = pd.DataFrame({'score': scores})
        fig_hist = px.histogram(df_scores, x='score', nbins=15, title="Distribution des Scores de Matching", color_discrete_sequence=['#8b5cf6'])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Aucun CV n'a encore été scoré.")

with col_ia2:
    if len(skills) > 0:
        df_skills = pd.DataFrame(skills).sort_values(by="count", ascending=True)
        fig_bar = px.bar(df_skills, x='count', y='skill', orientation='h', title="Top Compétences Réelles Extraites", color_discrete_sequence=['#0ea5e9'])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Aucune compétence extraite.")

with col_ia3:
    st.markdown("##### 🧱 Sources des Offres")
    df_sources = pd.DataFrame({
        'Source': ['Base locale', 'Scraping JSearch'],
        'Quantité': [local_jobs, scraped_jobs]
    })
    fig_donut = px.pie(df_sources, names='Source', values='Quantité', hole=0.5, color='Source',
                       color_discrete_map={'Base locale': '#f43f5e', 'Scraping JSearch': '#14b8a6'})
    fig_donut.update_layout(margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_donut, use_container_width=True)

with col_ia4:
    st.markdown("##### ⚠️ Quota API JSearch")
    QUOTA_MAX = 200
    estimated_req = min(int(scraped_jobs / 15), QUOTA_MAX)
    quota_left = QUOTA_MAX - estimated_req
    
    color = "green" if quota_left > 50 else "orange" if quota_left > 20 else "red"
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = estimated_req,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Restant estimé: {quota_left}", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [0, QUOTA_MAX]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 150], 'color': "rgba(34, 197, 94, 0.2)"},   
                {'range': [150, 180], 'color': "rgba(245, 158, 11, 0.2)"}, 
                {'range': [180, 200], 'color': "rgba(239, 68, 68, 0.2)"}   
            ],
        }
    ))
    fig_gauge.update_layout(margin=dict(t=40, b=10, l=20, r=20), height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)