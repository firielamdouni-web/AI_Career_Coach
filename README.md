[![CI/CD Pipeline](https://github.com/firielamdouni-web/AI_Career_Coach/actions/workflows/ci.yml/badge.svg)](https://github.com/firielamdouni-web/AI_Career_Coach/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Tests](https://img.shields.io/badge/tests-168%20passed-brightgreen.svg)]()

# AI Career Coach - Système Intelligent de Matching CV ↔ Offres d'Emploi

## Description du Projet

AI Career Coach est un système intelligent d'aide à l'emploi destiné aux profils juniors (Data Science et ML Engineering). Le projet combine NLP, embeddings sémantiques, machine learning et recherche vectorielle afin de proposer des recommandations d'offres personnalisées basées sur l'analyse automatique d'un CV.

### Objectifs Principaux

1. Extraction automatique des compétences techniques et soft skills depuis un CV PDF
2. Matching sémantique entre profil candidat et offres d'emploi (dataset local + offres scrapées)
3. Scoring intelligent basé sur la couverture et la qualité sémantique
4. Recommandations personnalisées avec explication des forces et faiblesses
5. Simulation d'entretiens avec génération de questions contextuelles et évaluation de réponses
6. Pipeline MLOps avec tracking MLflow et enregistrement de modèle

## Structure du projet

```
AI_Career_Coach/
|-- app.py                              # Frontend Streamlit
|-- pages/                              # Pages Streamlit
|   |-- 1_Interview_Simulator.py
|   |-- 2_Monitoring.py
|   |-- styles.css
|
|-- src/                                # Logique métier + API
|   |-- api.py                          # FastAPI (endpoints REST)
|   |-- compute_features_from_huggingface.py  # Features ML (dataset HF)
|   |-- cv_parser.py                    # Parsing PDF
|   |-- skills_extractor.py             # Extraction skills (spaCy + regex)
|   |-- job_matcher.py                  # Matching sémantique + scoring
|   |-- vector_store.py                 # FAISS (optionnel)
|   |-- ml_predictor.py                 # Features + prédiction XGBoost
|   |-- interview_simulator.py          # Interview simulator (Groq)
|   |-- job_scraper.py                  # Scraping (JSearch)
|   |-- database.py                     # PostgreSQL
|   |-- scheduler.py                    # Scrape planifié + sync
|
|-- data/
|   |-- skills_reference.json           # Référentiel skills + variations
|   |-- jobs/
|   |   |-- jobs_dataset.json           # Dataset jobs local
|   |-- resume_fit_job/
|       |-- raw/
|       |   |-- huggingface_resume_job_fit_RAW.xlsx
|       |-- processed/
|           |-- dataset_resume_job_fit_processed_v2.xlsx
|           |-- v2_dataset_resume_job_fit_processed.xlsx
|
|-- db/
|   |-- init/
|       |-- init_db.sql                 # Schéma Postgres
|
|-- docker/
|   |-- api.Dockerfile
|   |-- streamlit.Dockerfile
|-- docker-compose.yml
|
|-- mlops/
|   |-- train_and_log.py                # Entraînement + MLflow
|   |-- register_model.py               # Model Registry
|   |-- load_model.py
|
|-- models/
|   |-- classifier_clean_metadata.json
|   |-- features.txt
|   |-- job-matcher-classifier/
|       |-- artifacts/
|           |-- features.txt
|           |-- metadata.json
|
|-- notebooks/                          # Notebooks d'exploration / dev
|-- requirements/
|   |-- base.txt
|   |-- api.txt
|   |-- frontend.txt
|
|-- tests/
|
|-- .github/workflows/ci.yml            # CI/CD
|-- .streamlit/config.toml              # Config Streamlit
|-- .env.example
|-- .dockerignore
|-- .gitignore
|-- .flake8
|-- pytest.ini
```

## Quick Start

### Option 1 : Démarrage avec Docker (Recommandé)

Prérequis : Docker + Docker Compose.

```bash
# 1. Cloner le repo
git clone https://github.com/firielamdouni-web/AI_Career_Coach.git
cd AI_Career_Coach

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et renseigner au minimum : GROQ_API_KEY
# (optionnel) JSEARCH_API_KEY si vous activez le scraping

# 3. Lancer tous les services
docker compose up -d --build

# 4. Vérifier que tout est UP
docker compose ps
```

Accès aux interfaces :

- API Swagger : http://localhost:8000/docs
- Streamlit UI : http://localhost:8501
- MLflow UI : http://localhost:5000
- Grafana : http://localhost:3000

Vérification rapide :

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/stats
```

Arrêter les services :

```bash
docker compose down           # arrêter sans supprimer les données
docker compose down -v        # arrêter et supprimer les volumes (reset complet)
```

### Option 2 : Démarrage en local (Développement)

```bash
# 1. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements/base.txt -r requirements/api.txt -r requirements/frontend.txt

# 3. Télécharger un modèle spaCy
python -m spacy download en_core_web_sm
# optionnel (FR) : python -m spacy download fr_core_news_lg

# 4. Configurer les variables d'environnement
cp .env.example .env

# 5. Lancer l'API
uvicorn src.api:app --reload --port 8000

# 6. Lancer Streamlit (dans un autre terminal)
streamlit run app.py
```

## Architecture du Système

### Architecture Docker (6 services)

Le déploiement Docker orchestre 6 services : PostgreSQL, MLflow, API, Streamlit, Scheduler et Grafana.

```
┌─────────────────────────────────────────────────────────────────┐
│                     UTILISATEUR / NAVIGATEUR                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────┐
        │     STREAMLIT FRONTEND (Port 8501)      │
        │     • Upload CV                         │
        │     • Recommandations                   │
        │     • Simulation d'entretiens           │
        │     • Monitoring                        │
        └─────────────────────────────────────────┘
                              │
                              ↓ HTTP
        ┌─────────────────────────────────────────┐
        │     FASTAPI BACKEND (Port 8000)         │
        │     • Extraction skills                 │
        │     • Matching + scoring                │
        │     • ML predict                        │
        │     • Scrape + sync jobs                │
        └─────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┬─────────────┐
                ↓             ↓             ↓             ↓
    ┌───────────────────┐  ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐
    │ POSTGRESQL (5432)  │  │ MLFLOW (5000)│ │ GRAFANA (3000)│ │ SCHEDULER         │
    │ • logs + jobs      │  │ • runs       │ │ • dashboards  │ │ • scrape quotidien │
    └───────────────────┘  └──────────────┘ └──────────────┘ └───────────────────┘
```

### Endpoints API Disponibles

Ces endpoints correspondent à l'implémentation actuelle dans `src/api.py`.

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut de l'API |
| GET | `/api/v1/stats` | Statistiques globales (jobs, skills) |
| POST | `/api/v1/extract-skills` | Extraire compétences d'un CV PDF |
| POST | `/api/v1/recommend-jobs` | Recommander des jobs (top_n / min_score) |
| GET | `/api/v1/jobs` | Lister tous les jobs (locaux + scrapés) |
| GET | `/api/v1/jobs/{job_id}` | Détails d'un job |
| POST | `/api/v1/simulate-interview` | Générer questions d'entretien |
| POST | `/api/v1/evaluate-answer` | Évaluer une réponse |
| GET | `/api/v1/scrape-jobs` | Scraping manuel via JSearch |
| GET | `/api/v1/scraped-jobs` | Lire les jobs scrapés depuis Postgres |
| POST | `/api/v1/sync-jobs` | Recharger DB → exporter JSON + rebuild FAISS |
| POST | `/api/v1/ml-predict` | Prédiction ML (XGBoost) |
| GET | `/api/v1/faiss-stats` | Statistiques FAISS |
| GET | `/api/v1/monitoring-data` | Données DB pour le monitoring |

## Pipeline de Matching CV ↔ Jobs

```
┌─────────────────────────────────────────────────────────────────┐
│  1. UPLOAD CV (Streamlit)                                       │
│     • Utilisateur upload CV PDF via interface                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. PARSING (src/cv_parser.py)                                  │
│     • Extraction texte via pdfplumber                           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. EXTRACTION SKILLS (src/skills_extractor.py)                 │
│     • spaCy (en_core_web_sm / fr_core_news_lg)                  │
│     • Keyword matching sur skills_reference.json                │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. PRÉ-FILTRAGE FAISS (src/vector_store.py) [OPTIONNEL]        │
│     • Pré-sélection d'offres proches sémantiquement             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. SCORING (src/job_matcher.py)                                │
│     • overall_score = (coverage × 0.8) + (quality × 0.2)        │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. TRI & FILTRAGE (src/api.py)                                 │
│     • Tri par score décroissant, min_score, top_n               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. AFFICHAGE (app.py)                                          │
│     • Cards : score + compétences matchées/manquantes           │
└─────────────────────────────────────────────────────────────────┘
```

## Tests et Validation

### Tester l'API avec cURL

```bash
# 1. Extraction de compétences
curl -X POST "http://localhost:8000/api/v1/extract-skills" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/My_CV.pdf"

# 2. Recommandation de jobs (TOP 5)
curl -X POST "http://localhost:8000/api/v1/recommend-jobs?top_n=5" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/My_CV.pdf"

# 3. Simulation d'entretien
curl -X POST "http://localhost:8000/api/v1/simulate-interview" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_1",
    "cv_skills": ["Python", "Machine Learning"],
    "num_questions": 6
  }'
```

### Script de test complet

Le script ci-dessous est optionnel et nécessite `jq`.

```bash
cat > test_api.sh << 'EOF'
#!/bin/bash
set -e

CV_PATH="data/My_CV.pdf"

echo "API health..."
curl -s http://localhost:8000/health | jq .

echo "Stats..."
curl -s http://localhost:8000/api/v1/stats | jq .

echo "Recommendations (top 3)..."
curl -s -X POST "http://localhost:8000/api/v1/recommend-jobs?top_n=3" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$CV_PATH" | jq .

echo "Done"
EOF

chmod +x test_api.sh
./test_api.sh
```

## Modèle ML Entraîné

### Caractéristiques du Modèle

- Type : XGBoost Classifier
- Classes : 3 (No Fit, Partial Fit, Perfect Fit)
- Dataset :
  - Brut (HuggingFace) : `data/resume_fit_job/raw/huggingface_resume_job_fit_RAW.xlsx` (6,241 échantillons)
  - Processed v1 : `data/resume_fit_job/processed/v2_dataset_resume_job_fit_processed.xlsx` (4,524 échantillons)
  - Processed v2 (27 features) : `data/resume_fit_job/processed/dataset_resume_job_fit_processed_v2.xlsx` (6,241 échantillons)
- Tracking : MLflow (runs + artifacts, via `MLFLOW_TRACKING_URI`)
- Features : 27 (versionnées dans `models/job-matcher-classifier/artifacts/features.txt`)

Métriques disponibles dans les artefacts versionnés (`models/job-matcher-classifier/artifacts/metadata.json`) :

- Test accuracy : 0.6781
- Test precision (weighted) : 0.6750
- Test recall (weighted) : 0.6781
- Test F1 (weighted) : 0.6727
- Split : 4,992 train / 1,249 test (6,241 échantillons)

Note : le fichier `models/classifier_clean_metadata.json` référence aussi un export/registre MLflow (stage Production) avec une accuracy reportée à 0.7050 et un compteur de features à 15. Les artefacts présents dans `models/job-matcher-classifier/artifacts/` correspondent à la version v2 (27 features) et exposent leurs propres métriques dans `metadata.json`.

### Features utilisées (27)

```text
coverage
quality
nb_covered_skills
nb_missing_skills
skills_ratio
similarity_mean
similarity_max
similarity_std
top3_similarity_avg
tfidf_similarity
embedding_similarity
nb_resume_technical
nb_resume_soft
nb_job_technical
nb_job_soft
resume_text_length
resume_text_word_count
resume_text_unique_words
resume_text_avg_word_length
resume_text_sentence_count
resume_text_capital_ratio
job_description_text_length
job_description_text_word_count
job_description_text_unique_words
job_description_text_avg_word_length
job_description_text_sentence_count
job_description_text_capital_ratio
```

### Entraîner et tracker le modèle

```bash
# Entraîner le modèle et logger dans MLflow
python mlops/train_and_log.py

# Enregistrer dans le Model Registry
python mlops/register_model.py
```

## Technologies Utilisées

### Backend

- FastAPI
- PostgreSQL

### NLP & ML

- spaCy (extraction skills)
- SentenceTransformers (embeddings)
- FAISS (vector search, optionnel)
- XGBoost (classification)
- Groq (LLM pour interview simulator)

### MLOps

- MLflow
- Docker / Docker Compose

### Frontend

- Streamlit

## Performances

Ces métriques sont indicatives et dépendent des ressources machine et de l'utilisation de FAISS.

| Métrique | Valeur |
|---|---|
| Jobs disponibles | 25 offres (dataset local `data/jobs/jobs_dataset.json`) |
| Jobs scrapés (JSearch) | Variable (dépend des requêtes, de `num_pages` et des résultats du moment) ; stockés en DB Postgres (table `scraped_jobs`) |
| Skills trackés | 653 compétences techniques + 190 soft skills (`data/skills_reference.json`) |
| Variations skills | 472 variations (mapping vers canonicals) |
| Temps parsing CV | ~2–3 secondes |
| Temps scraping (live_scrape) | Variable (réseau + API externe) ; peut dominer le temps total quand activé |
| Temps matching | ~0.1 s / job (≈ 2.5 s pour 25 jobs, hors scraping) |
| Temps total pipeline | ~7–10 secondes (parsing + extraction + matching + rendu UI) |
| Temps total pipeline (avec scraping) | Variable ; dépend surtout du scraping, puis du matching sur (jobs locaux + jobs scrapés) |
| Accuracy modèle ML | ~0.68–0.70 (selon version du modèle) |
| Index FAISS / Embeddings | 768 dimensions (SentenceTransformer) |
| Scheduler scraping | Exécution quotidienne (08:00) + synchronisation DB → FAISS via endpoint `/api/v1/sync-jobs` |


## URLs Clés (Mode Docker)

| Service | URL | Description |
|---|---|---|
| API Swagger | http://localhost:8000/docs | Documentation interactive API |
| API Health | http://localhost:8000/health | Statut de l'API |
| Streamlit | http://localhost:8501 | Interface utilisateur |
| MLflow UI | http://localhost:5000 | Tracking des modèles |
| Grafana | http://localhost:3000 | Dashboards |
| PostgreSQL | localhost:5432 | Base de données (psql / client DB) |

## CI/CD

Le pipeline GitHub Actions est défini dans `.github/workflows/ci.yml` et suit l'ordre : lint (flake8), tests (pytest), build Docker, puis déploiement conditionnel.

Commandes locales (équivalent CI) :

```bash
flake8 src/ tests/ --config=.flake8
pytest -q
```

## Variables d'environnement

Le template est dans `.env.example`.

- `GROQ_API_KEY` : requis pour la génération et l'évaluation des réponses d'entretien
- `JSEARCH_API_KEY` : requis si vous activez le scraping via JSearch (RapidAPI)
- `DATABASE_URL` : URL PostgreSQL
- `MLFLOW_TRACKING_URI` : URL MLflow
- `API_BASE_URL` : URL de l'API consommée par Streamlit (en Docker : `http://api:8000`)
- `HF_TOKEN` : optionnel (Hugging Face)

