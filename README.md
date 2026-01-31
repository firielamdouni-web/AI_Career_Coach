# 🎯 ROADMAP PFE - Système d'Aide à l'Emploi pour Juniors

## 📅 SEMAINE 1-2 : CORE FONCTIONNEL
- [x] Parser CV (01_cv_parser.ipynb)
- [x] Extraction compétences (02_skills_extraction_simple.ipynb)
- [x] Matching sémantique (03_semantic_matching.ipynb)
- [X] Scraping offres (04_job_scraping.ipynb) 
- [X] Matching CV ↔ Offres (05_job_recommendation.ipynb)
- [X] Dashboard Streamlit v1 (app.py)

**Livrable Semaine 2** : Système fonctionnel de bout en bout

## 📅 SEMAINE 3-4 : ENRICHISSEMENT
- [ ] API FastAPI (src/api.py) ← MAINTENANT
- [ ] Base vectorielle FAISS (src/vector_store.py)
- [ ] Simulation entretien LLM (06_interview_simulation.ipynb)
- [ ] Clustering profils KMeans (07_profile_clustering.ipynb)

**Livrable Semaine 4** : API + Features ML avancées

## 📅 SEMAINE 5-6 : INDUSTRIALISATION
- [ ] Tests unitaires (tests/)
- [ ] Dashboard Streamlit v2 (graphiques, stats)
- [ ] Scraping offres réelles via API (optionnel)
- [ ] Monitoring performances (logs, métriques)

**Livrable Semaine 6** : Code robuste et testé

## 📅 SEMAINE 7-8 : FINALISATION
- [ ] Documentation complète (README, docstrings)
- [ ] Rapport PFE (40-60 pages)
- [ ] Préparation soutenance (slides)
- [ ] Déploiement cloud (optionnel)

**Livrable Semaine 8** : PFE complet prêt à soutenir

Pipeline :

┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 : 01_cv_parser.ipynb                               │
│   CV PDF → cv_text_pdfplumber.txt                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 : 02_skills_extraction_simple.ipynb                │
│   cv_text.txt → extracted_skills_simple.json ✅ NÉCESSAIRE │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 3 : 03_semantic_matching.ipynb (OPTIONNEL)           │
│   Test de matching sémantique                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 4 : 04_job_generation.ipynb                          │
│   Génère jobs_dataset.json ✅ NÉCESSAIRE                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 5 : 05_job_recommendation.ipynb                      │
│   extracted_skills_simple.json + jobs_dataset.json         │
│   → job_recommendations.json                                │
└─────────────────────────────────────────────────────────────┘