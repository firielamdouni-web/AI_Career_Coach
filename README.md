# 🎯 AI Career Coach - Système Intelligent de Matching CV ↔ Offres d'Emploi

## 📖 Description du Projet

**AI Career Coach** est un système intelligent d'aide à l'emploi destiné aux **profils juniors en Data Science et ML Engineering**. Le projet combine **NLP**, **embeddings sémantiques**, **machine learning** et **recherche vectorielle** pour proposer des recommandations d'emploi personnalisées basées sur l'analyse automatique de CV.

###  Objectifs Principaux

1. **Extraction automatique** des compétences techniques et soft skills depuis un CV PDF
2. **Matching sémantique** entre profil candidat et offres d'emploi
3. **Scoring intelligent** basé sur la couverture et la qualité des compétences
4. **Recommandations personnalisées** avec explication des forces et faiblesses
5. **Simulation d'entretiens** avec génération de questions contextuelles
6. **MLOps pipeline** avec tracking des expériences et déploiement de modèles

## 📁 Structure du projet

```
AI_Career_Coach/
│
├── 📁 data/                               # Données et artifacts
│   ├── 📁 jobs/                           # Offres d'emploi et embeddings
│   │   └── jobs_dataset.json              # 25 offres d'emploi (Data Science/ML)
│   │
│   ├── 📁 resume_fit_job/                   # Dataset CV-Job
│   │   ├── 📁 processed/                    # Données nettoyées
│   │   │   └── v2_dataset_resume_job_fit_processed.xlsx  # Dataset nettoyé (4,524 samples)
│   │   └── 📁 raw/                          # Données brutes
│   │       └── huggingface_resume_job_fit_RAW.xlsx  # Dataset brut (6,241 samples)
│   │
│   ├── skills_reference.json                # Compétences techniques + soft skills
│   └── RESUME_*.pdf                         # CVs de test
│
├── 📁 db/ 
│   ├── 📁 init/                         
│       └── init_db.sql                     # Schéma PostgreSQL
│
├── 📁 docker/                             # Dockerfiles
│   ├── api.Dockerfile                      # Image Docker API FastAPI
│   └── streamlit.Dockerfile                # Image Docker Streamlit
│
├── 📁 mlops/                                # Pipeline MLOps
│   ├── train_and_log.py                     # Entraînement + tracking MLflow
│   ├── register_model.py                    # Enregistrement Model Registry
│   └── serve_model.py                       # Test de prédiction
│
├── 📁 models/                               # Modèles entraînés (metadata uniquement)
│   └── classifier_clean_metadata.json       # Métadonnées du modèle XGBoost
│
├── 📁 notebooks/                            # Notebooks de développement
│   ├── 01_cv_parser.ipynb                   # Parsing de CV PDF
│   ├── 02_skills_extraction_simple.ipynb    # Extraction de compétences CV
│   ├── 03_extraction_skills_job_offers.ipynb # Extraction de compétences jobs
│   ├── 03_semantic_matching.ipynb            # Tests de matching sémantique
│   ├── 04_job_generation.ipynb              # Génération du dataset d'offres
│   ├── 05_job_recommendation.ipynb          # Système de recommandation
│   ├── 06_faiss_indexing.ipynb              # Base vectorielle
│   ├── 07_interview_simulation.ipynb        # Simulation d'entretiens
│   ├── 08_exploration_dataset_RAW.ipynb     # Exploration dataset brute
│   └── 09_ml_model_training.ipynb           # Entraînement modèle ML (XGBoost, 70% accuracy)
│
├── 📁 src/                                   # Code source principal
│   ├── api.py                               # API FastAPI (endpoints REST)
│   ├── cv_parser.py                         # Parser CV (PyPDF2 + pdfplumber)
│   ├── skills_extractor.py                  # Extraction compétences (spaCy + regex)
│   ├── job_matcher.py                       # Matching sémantique (SentenceTransformer)
│   ├── vector_store.py                      # Recherche vectorielle (FAISS)
│   ├── database.py                          # Gestion PostgreSQL (SQLAlchemy)
│   ├── interview_simulator.py               # Génération questions d'entretien
│   └── compute_features_from_huggingface.py # Calcul features ML
│
├── 📁 pages/                               # Pages Streamlit
│   └── 1_Interview_Simulation.py           # Page simulation entretien
│
├── 📁 tests/                                 # Tests unitaires
│   └── ...
│
├── 📁 requirements/ 
│   ├── api.txt                                  # Dépendances API (FastAPI, Groq...)
│   ├── frontend.txt                             # Dépendances Streamlit
│   └── base.txt                                 # Dépendances communs
│
├── app.py                                    # Dashboard Streamlit (frontend)
├── requirements.txt                          # Dépendances Python
├── docker-compose.yml                        # Orchestration 4 services Docker
├── .env.example                              # Template variables d'environnement
├── .dockerignore                             # Fichiers exclus du build
├── .gitignore                                
└── README.md                                
```

## 🚀 Quick Start

### **Option 1 : Démarrage avec Docker (Recommandé)**

```bash
# 1. Cloner le repo
git clone https://github.com/firielamdouni-web/AI_Career_Coach/tree/Firiel
cd AI_Career_Coach

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos valeurs et ajouter votre GROQ_API_KEY

# 3. Lancer tous les services (PostgreSQL + API + Streamlit + MLflow)
docker-compose up -d

# 4. Vérifier que tout est UP
docker-compose ps

# 5. Entraîner et enregistrer le modèle
docker-compose exec api python mlops/train_and_log.py
docker-compose exec api python mlops/register_model.py

# 6. Accéder aux interfaces
# - API Swagger : http://localhost:8000/docs
# - Streamlit UI : http://localhost:8501
# - MLflow UI : http://localhost:5000
```

**Vérification rapide :**

```bash
# Health check API
curl http://localhost:8000/health

# Stats du système
curl http://localhost:8000/api/v1/stats

# Tester une recommandation
curl -X POST "http://localhost:8000/api/v1/recommend-jobs" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/CV_exemple.pdf" \
  -F "top_k=5"
```

**Arrêter les services :**

```bash
cd deployment/docker
docker-compose down           # Arrêter sans supprimer les données
docker-compose down -v        # Arrêter et supprimer les volumes (reset complet)
```

---

### **Option 2 : Démarrage en local (Développement)**

```bash
# 1. Cloner le repo
git clone https://github.com/firielamdouni-web/AI_Career_Coach/tree/Firiel
cd AI_Career_Coach

# 2. Créer l'environnement virtuel
python -m venv env
source env/bin/activate  # (ou env\Scripts\activate sur Windows)

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle spaCy
python -m spacy download en_core_news_lg

# 5. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et ajouter votre GROQ_API_KEY

# 6. (Optionnel) Entraîner le modèle ML et tracker avec MLflow
python mlops/train_and_log.py
python mlops/register_model.py

# 7. Lancer MLflow UI (dans un terminal séparé)
mlflow ui --backend-store-uri file:./mlops/mlflow_tracking --port 5000
# Accéder à MLflow UI : http://localhost:5000

# 8. Lancer l'API FastAPI (dans un autre terminal)
uvicorn src.api:app --reload --port 8000
# Documentation interactive : http://localhost:8000/docs

# 9. Lancer le dashboard Streamlit (dans un troisième terminal)
streamlit run app.py
# Interface utilisateur : http://localhost:8501
```

---

## 🎯 **Architecture du Système**

### **🐳 Architecture Docker (4-tiers)**

```
┌─────────────────────────────────────────────────────────────────┐
│                     UTILISATEUR / NAVIGATEUR                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌─────────────────────────────────────────┐
        │     STREAMLIT FRONTEND (Port 8501)      │
        │     • Upload CV                         │
        │     • Affichage recommandations         │
        │     • Simulation d'entretiens           │
        └─────────────────────────────────────────┘
                              │
                              ↓ HTTP POST
        ┌─────────────────────────────────────────┐
        │     FASTAPI BACKEND (Port 8000)         │
        │     • 8 endpoints REST                  │
        │     • Extraction skills                 │
        │     • Matching sémantique               │
        │     • Scoring intelligent               │
        └─────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ↓                           ↓
    ┌───────────────────────┐   ┌───────────────────────┐
    │  POSTGRESQL (5432)    │   │  MLFLOW SERVER (5000) │
    │  • Stockage CVs       │   │  • Model Registry     │
    │  • Historique matchs  │   │  • Tracking runs      │
    │  • Logs candidats     │   │  • Artifacts ML       │
    └───────────────────────┘   └───────────────────────┘
```

### **📡 Endpoints API Disponibles**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Statut de l'API |
| `GET` | `/api/v1/stats` | Statistiques globales (jobs, skills) |
| `POST` | `/api/v1/extract-skills` | Extraire compétences d'un CV PDF |
| `POST` | `/api/v1/recommend-jobs` | Recommander des jobs (TOP-K) |
| `GET` | `/api/v1/jobs` | Lister tous les jobs disponibles |
| `GET` | `/api/v1/jobs/{job_id}` | Détails d'un job spécifique |
| `POST` | `/api/v1/simulate-interview` | Générer questions d'entretien |
| `POST` | `/api/v1/evaluate-answer` | Évaluer une réponse candidat |
| `POST` | `/api/v1/search` | Recherche sémantique de jobs |
| `POST` | `/api/v1/match` | Matching CV ↔ Job spécifique |

---

## 🎯 **Pipeline de Matching CV ↔ Jobs**

```
┌─────────────────────────────────────────────────────────────────┐
│  1. UPLOAD CV (Streamlit)                                       │
│     • Utilisateur upload CV PDF via interface                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. PARSING (cv_parser.py)                                      │
│     • pdfplumber                                                │
│     • Extraction texte brut (~2000 caractères)                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. EXTRACTION SKILLS (skills_extractor.py)                     │
│     • spaCy (fr_core_news_lg)                                   │
│     • Pattern matching sur 1250 skills                           │
│     • Résultat : ["python", "pandas", "numpy", ...]             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. PRÉ-FILTRAGE FAISS (vector_store.py) [OPTIONNEL]            │
│     • Embedding CV avec SentenceTransformer                     │
│     • Recherche Top-50 dans index FAISS                         │
│     • Temps : ~0.5s vs 2.5s (brute force)                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. SCORING DÉTAILLÉ (job_matcher.py)                           │
│     • Calcul similarité CV ↔ Job (cosinus)                      │
│     • Score = (Coverage × 0.5) + (Quality × 0.5)                │
│     • Coverage : Skills couverts / Skills requis                │
│     • Quality : Moyenne similarités sémantiques                 │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. TRI & FILTRAGE (api.py)                                     │
│     • Tri par score décroissant                                 │
│     • Filtrage score minimum (défaut: 40%)                      │
│     • Limitation Top-K (défaut: 25)                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. AFFICHAGE (app.py)                                          │
│     • Cards avec score + compétences matchées/manquantes        │
│     • Filtres interactifs (remote, expérience, score)           │
│     • Graphiques de répartition                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 **Tests et Validation**

### **Tester l'API avec cURL**

```bash
# 1. Extraction de compétences
curl -X POST "http://localhost:8000/api/v1/extract-skills" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/CV_exemple.pdf"

# 2. Recommandation de jobs (TOP 5)
curl -X POST "http://localhost:8000/api/v1/recommend-jobs" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/CV_exemple.pdf" \
  -F "top_k=5"

# 3. Recherche sémantique
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Machine learning engineer with Python", "top_k": 5}'

# 4. Simulation d'entretien
curl -X POST "http://localhost:8000/api/v1/simulate-interview" \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Data Scientist",
    "cv_skills": ["Python", "Machine Learning", "TensorFlow"],
    "num_questions": 3
  }'
```

### **Script de test complet**

```bash
# Créer le script
cat > test_api.sh << 'EOF'
#!/bin/bash
echo "🧪 TEST COMPLET DE L'API"
echo "========================"

CV_PATH="data/CV_exemple.pdf"

echo "1️⃣ Health Check..."
curl -s http://localhost:8000/health | jq .

echo "2️⃣ Statistiques..."
curl -s http://localhost:8000/api/v1/stats | jq .

echo "3️⃣ Recommandations TOP 3..."
curl -s -X POST "http://localhost:8000/api/v1/recommend-jobs" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$CV_PATH" \
  -F "top_k=3" | jq .

echo "✅ TESTS TERMINÉS"
EOF

# Rendre exécutable
chmod +x test_api.sh

# Lancer
./test_api.sh
```

---

## 🎯 **Modèle ML Entraîné**

### **Caractéristiques du Modèle**

- **Type** : XGBoost Classifier
- **Classes** : 3 (No Fit, Partial Fit, Perfect Fit)
- **Features** : 15 (coverage, quality, similarities, etc.)
- **Performance** : ~70% accuracy (Test Set)
- **Dataset** : 4,524 samples (nettoyé depuis 6,241 bruts)
- **Tracking** : MLflow (expériences + Model Registry)

### **Features Utilisées (15)**

```python
[
    'job_title_similarity',
    'description_similarity', 
    'requirements_similarity',
    'responsibilities_similarity',
    'matching_skills_count',
    'missing_skills_count',
    'skills_coverage',
    'avg_skill_similarity',
    'max_skill_similarity',
    'min_skill_similarity',
    'cv_job_cosine_similarity',
    'quality_score',
    'has_remote',
    'experience_level',
    'company_type'
]
```

### **Entraîner et Tracker le Modèle**

```bash
# Entraîner le modèle et logger dans MLflow
python mlops/train_and_log.py

# Enregistrer dans le Model Registry
python mlops/register_model.py

# Tester une prédiction
python mlops/serve_model.py

# Consulter les runs dans MLflow UI
mlflow ui --backend-store-uri file:./mlops/mlflow_tracking --port 5000
# Ouvrir http://localhost:5000
```

---

## 🛠️ **Technologies Utilisées**

### **Backend**
- **FastAPI** : API REST moderne et performante
- **PostgreSQL** : Base de données relationnelle
- **SQLAlchemy** : ORM Python

### **NLP & ML**
- **spaCy** : Extraction de compétences (en_core_news_lg)
- **SentenceTransformers** : Embeddings sémantiques (all-mpnet-base-v2)
- **FAISS** : Recherche vectorielle ultra-rapide
- **XGBoost** : Classification des matchs CV-Job
- **Groq** : LLM pour simulation d'entretiens

### **MLOps**
- **MLflow** : Tracking expériences + Model Registry
- **Docker** : Containerisation 4-tiers
- **Docker Compose** : Orchestration multi-conteneurs

### **Frontend**
- **Streamlit** : Dashboard interactif
- **Plotly** : Visualisations graphiques

### **Parsing PDF**
- **pdfplumber** : Extraction texte 

---

## 📊 **Performances**

| Métrique | Valeur |
|----------|--------|
| **Jobs disponibles** | 25 offres (Data Science/ML) |
| **Skills trackés** | 171 compétences techniques |
| **Temps parsing CV** | ~2-3 secondes |
| **Temps matching** | ~0.1s/job (2.5s pour 25 jobs) |
| **Temps total pipeline** | ~7-10 secondes |
| **Accuracy modèle ML** | 70% (test set) |
| **Index FAISS** | 768 dimensions (SentenceTransformer) |

---

## 🌐 **URLs Clés (Mode Docker)**

| Service | URL | Description |
|---------|-----|-------------|
| **API Swagger** | http://localhost:8000/docs | Documentation interactive API |
| **API Health** | http://localhost:8000/health | Statut de l'API |
| **Streamlit** | http://localhost:8501 | Interface utilisateur |
| **MLflow UI** | http://localhost:5000 | Tracking des modèles |
| **PostgreSQL** | localhost:5432 | Base de données (psql uniquement) |


---


[![CI/CD Pipeline](https://github.com/firielamdouni-web/AI_Career_Coach/actions/workflows/ci.yml/badge.svg)](https://github.com/firielamdouni-web/AI_Career_Coach/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Tests](https://img.shields.io/badge/tests-149%20passed-brightgreen.svg)]() 