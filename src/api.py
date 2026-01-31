"""
🎯 API FastAPI pour le système de matching CV ↔ Jobs
Endpoints pour extraction de compétences et recommandations d'emploi
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import os
import json

# Import des modules locaux
from .cv_parser import CVParser
from .skills_extractor import SkillsExtractor
from .job_matcher import JobMatcher

# ============================================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================================

app = FastAPI(
    title="🎯 AI Career Coach API",
    description="API de matching CV ↔ Offres d'emploi avec IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (permettre les requêtes depuis le frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CHARGEMENT DES DONNÉES ET MODÈLES
# ============================================================================

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
JOBS_DATASET_PATH = PROJECT_ROOT / "data" / "jobs" / "jobs_dataset.json"
SKILLS_DB_PATH = PROJECT_ROOT / "data" / "skills_reference.json"

# Cache pour éviter de recharger à chaque requête
_cv_parser = None
_skills_extractor = None
_job_matcher = None
_jobs_dataset = None


def get_cv_parser() -> CVParser:
    """Obtenir le parser de CV (singleton)"""
    global _cv_parser
    if _cv_parser is None:
        _cv_parser = CVParser(method='pdfplumber')
    return _cv_parser


def get_skills_extractor() -> SkillsExtractor:
    """Obtenir l'extracteur de compétences (singleton)"""
    global _skills_extractor
    if _skills_extractor is None:
        _skills_extractor = SkillsExtractor(skills_db_path=str(SKILLS_DB_PATH))
    return _skills_extractor


def get_job_matcher() -> JobMatcher:
    """Obtenir le matcher (singleton)"""
    global _job_matcher
    if _job_matcher is None:
        _job_matcher = JobMatcher(model_name='all-mpnet-base-v2')
    return _job_matcher


def get_jobs_dataset() -> Dict:
    """Charger le dataset d'offres d'emploi (singleton)"""
    global _jobs_dataset
    if _jobs_dataset is None:
        if not JOBS_DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset d'offres non trouvé : {JOBS_DATASET_PATH}\n"
                "Exécutez le notebook 04_job_generation.ipynb"
            )
        with open(JOBS_DATASET_PATH, 'r', encoding='utf-8') as f:
            _jobs_dataset = json.load(f)
    return _jobs_dataset


# ============================================================================
# MODÈLES PYDANTIC (VALIDATION DES RÉPONSES)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    models_loaded: bool
    jobs_available: int


class SkillsResponse(BaseModel):
    technical_skills: List[str]
    soft_skills: List[str]
    total_skills: int
    cv_text_length: int


class JobRecommendation(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    remote: bool
    experience_required: str
    score: float
    skills_match: float
    experience_match: int
    location_match: int
    competition_factor: int
    matching_skills: List[str]


class RecommendationsResponse(BaseModel):
    recommendations: List[JobRecommendation]
    total_jobs_analyzed: int
    cv_skills_count: int


class JobDetail(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    remote: bool
    experience_required: str
    category: str
    description: str
    skills_required: List[str]


class StatsResponse(BaseModel):
    total_jobs: int
    jobs_by_category: Dict[str, int]
    remote_jobs: int
    on_site_jobs: int
    total_technical_skills: int
    total_soft_skills: int
    model_used: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint racine - Informations sur l'API
    """
    return {
        "message": "🎯 AI Career Coach API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "stats": "/api/v1/stats",
            "extract_skills": "/api/v1/extract-skills",
            "recommend_jobs": "/api/v1/recommend-jobs",
            "list_jobs": "/api/v1/jobs",
            "get_job": "/api/v1/jobs/{job_id}"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifier l'état de santé de l'API
    
    Returns:
        État de l'API et disponibilité des ressources
    """
    try:
        # Vérifier que le dataset existe
        dataset = get_jobs_dataset()
        jobs_count = len(dataset.get('jobs', []))
        
        # Vérifier si les modèles sont chargés
        models_loaded = _job_matcher is not None and _skills_extractor is not None
        
        return {
            "status": "healthy",
            "message": "API opérationnelle",
            "version": "1.0.0",
            "models_loaded": models_loaded,
            "jobs_available": jobs_count
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": str(e),
                "version": "1.0.0",
                "models_loaded": False,
                "jobs_available": 0
            }
        )


@app.post("/api/v1/extract-skills", response_model=SkillsResponse, tags=["Skills"])
async def extract_skills(file: UploadFile = File(...)):
    """
    Extraire les compétences d'un CV PDF
    
    Args:
        file: Fichier PDF du CV
        
    Returns:
        Liste des compétences techniques et soft skills détectées
    """
    # Vérifier le type de fichier
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être un PDF"
        )
    
    tmp_file_path = None
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # 1. Parser le CV
        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Le CV est vide ou illisible. Vérifiez que le PDF contient du texte."
            )
        
        # 2. Extraire les compétences
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)  
        
        technical_skills = skills_result['technical_skills']
        soft_skills = skills_result['soft_skills']
        
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "total_skills": len(technical_skills) + len(soft_skills),
            "cv_text_length": len(cv_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du CV: {str(e)}"
        )
    finally:
        # Nettoyer le fichier temporaire
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass


@app.post("/api/v1/recommend-jobs", response_model=RecommendationsResponse, tags=["Jobs"])
async def recommend_jobs(
    file: UploadFile = File(...),
    top_n: int = Query(10, ge=1, le=25, description="Nombre de recommandations"),
    min_score: float = Query(40.0, ge=0.0, le=100.0, description="Score minimum")
):
    """
    Obtenir des recommandations d'emploi basées sur un CV
    
    Args:
        file: Fichier PDF du CV
        top_n: Nombre de recommandations à retourner (défaut: 10)
        min_score: Score minimum pour filtrer (défaut: 40.0)
        
    Returns:
        Liste des jobs recommandés avec scores de matching
    """
    # Vérifier le type de fichier
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être un PDF"
        )
    
    tmp_file_path = None
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # 1. Parser le CV
        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Le CV est vide ou illisible"
            )
        
        # 2. Extraire les compétences
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)  
        cv_skills = skills_result['technical_skills']
        
        if not cv_skills:
            raise HTTPException(
                status_code=400,
                detail="Aucune compétence technique détectée dans le CV"
            )
        
        # 3. Charger les jobs
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']  
        
        # 4. Calculer les recommandations
        matcher = get_job_matcher()
        ranked_jobs = matcher.rank_jobs(cv_skills, jobs)  
        
        # 5. Filtrer par score minimum et top_n
        filtered_jobs = [
            job for job in ranked_jobs 
            if job['global_score'] >= min_score
        ][:top_n]
        
        # 6. Formater la réponse
        recommendations = []
        for job in filtered_jobs:
            recommendations.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job['remote_ok'],  
                "experience_required": job['experience'],  
                "score": job['global_score'],  
                "skills_match": job['skills_score'],  
                "experience_match": job['experience_score'],
                "location_match": job['location_score'],
                "competition_factor": job['competition_score'],
                "matching_skills": job['skills_details']['top_skills'][:10]  
            })
        
        return {
            "recommendations": recommendations,
            "total_jobs_analyzed": len(jobs),
            "cv_skills_count": len(cv_skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )
    finally:
        # Nettoyer le fichier temporaire
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass


@app.get("/api/v1/jobs", response_model=List[JobDetail], tags=["Jobs"])
async def list_jobs(
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    remote: Optional[bool] = Query(None, description="Filtrer par télétravail"),
    limit: int = Query(25, ge=1, le=100, description="Nombre maximum de résultats")
):
    """
    Lister toutes les offres d'emploi disponibles
    
    Args:
        category: Filtrer par catégorie (optionnel)
        remote: Filtrer par télétravail (optionnel)
        limit: Nombre maximum de résultats (défaut: 25)
        
    Returns:
        Liste des offres d'emploi
    """
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']  # ✅ ACCÈS CORRECT
        
        # Appliquer les filtres
        filtered_jobs = jobs
        
        if category:
            filtered_jobs = [
                job for job in filtered_jobs 
                if job.get('category', '').lower() == category.lower()
            ]
        
        if remote is not None:
            filtered_jobs = [
                job for job in filtered_jobs 
                if job.get('remote_ok', False) == remote
            ]
        
        # Limiter le nombre de résultats
        filtered_jobs = filtered_jobs[:limit]
        
        # Formater la réponse
        result = []
        for job in filtered_jobs:
            result.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job.get('remote_ok', False),
                "experience_required": job['experience'],
                "category": job.get('category', 'Non spécifié'),
                "description": job['description'],
                "skills_required": job.get('requirements', [])
            })
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des jobs: {str(e)}"
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobDetail, tags=["Jobs"])
async def get_job(job_id: str):
    """
    Obtenir les détails d'une offre d'emploi spécifique
    
    Args:
        job_id: Identifiant du job (ex: job_001)
        
    Returns:
        Détails complets du job
    """
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']
        
        # Rechercher le job
        job = next((j for j in jobs if j['job_id'] == job_id), None)
        
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job avec l'ID '{job_id}' non trouvé"
            )
        
        return {
            "job_id": job['job_id'],
            "title": job['title'],
            "company": job['company'],
            "location": job['location'],
            "remote": job.get('remote_ok', False),
            "experience_required": job['experience'],
            "category": job.get('category', 'Non spécifié'),
            "description": job['description'],
            "skills_required": job.get('requirements', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération du job: {str(e)}"
        )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """
    Obtenir des statistiques sur le système
    
    Returns:
        Statistiques générales (jobs, compétences, modèles)
    """
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']
        extractor = get_skills_extractor()
        
        # Calculer les statistiques
        categories = {}
        remote_count = 0
        
        for job in jobs:
            category = job.get('category', 'Non spécifié')
            categories[category] = categories.get(category, 0) + 1
            if job.get('remote_ok', False):
                remote_count += 1
        
        return {
            "total_jobs": len(jobs),
            "jobs_by_category": categories,
            "remote_jobs": remote_count,
            "on_site_jobs": len(jobs) - remote_count,
            "total_technical_skills": len(extractor.skills_database['technical_skills']),
            "total_soft_skills": len(extractor.skills_database['soft_skills']),
            "model_used": "all-mpnet-base-v2 (768 dimensions)"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )


# ============================================================================
# GESTION DES ERREURS GLOBALES
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint non trouvé. Consultez /docs pour la liste complète."}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur. Consultez les logs."}
    )


# ============================================================================
# ÉVÉNEMENTS DE DÉMARRAGE/ARRÊT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Actions au démarrage de l'API"""
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DE L'API")
    print("="*60)
    
    # Vérifier que les fichiers nécessaires existent
    if not JOBS_DATASET_PATH.exists():
        print("⚠️  ATTENTION : Dataset d'offres manquant")
        print(f"   Chemin attendu : {JOBS_DATASET_PATH}")
        print("   Exécutez : notebooks/04_job_generation.ipynb")
    
    if not SKILLS_DB_PATH.exists():
        print("⚠️  ATTENTION : Base de compétences manquante")
        print(f"   Chemin attendu : {SKILLS_DB_PATH}")
    
    print("\n✅ API prête à recevoir des requêtes")
    print("📖 Documentation : http://127.0.0.1:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt de l'API"""
    print("\n🛑 Arrêt de l'API")