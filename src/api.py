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
import logging
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn

# ============================================================================
# ✅ CHARGEMENT DES VARIABLES D'ENVIRONNEMENT (AVANT TOUS LES IMPORTS)
# ============================================================================
from dotenv import load_dotenv
load_dotenv()

# Vérifier que DATABASE_URL est chargé
if not os.getenv("DATABASE_URL"):
    raise RuntimeError(
        "❌ DATABASE_URL non trouvé. Vérifiez que le fichier .env existe à la racine du projet."
    )

# ============================================================================
# IMPORTS DES MODULES LOCAUX
# ============================================================================
from .cv_parser import CVParser
from .skills_extractor import SkillsExtractor
from .job_matcher import JobMatcher
from .vector_store import JobVectorStore
from .interview_simulator import InterviewSimulator, get_interview_simulator
from .database import get_db_manager, close_db_connection

# ============================================================================
# CONFIGURATION DU LOGGER
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MLFLOW
# ============================================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = "ai-career-coach-production"

# Initialiser MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

logger.info(f"✅ MLflow configuré : {MLFLOW_TRACKING_URI}")

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
ML_MODEL_PATH = PROJECT_ROOT / "models" / "ml_classifier_clean_v1.pkl" 

# Cache pour éviter de recharger à chaque requête
_cv_parser = None
_skills_extractor = None
_job_matcher = None
_jobs_dataset = None
_vector_store = None
_ml_classifier = None 

def get_cv_parser() -> CVParser:
    """Obtenir le parser de CV (singleton)"""
    global _cv_parser
    if _cv_parser is None:
        _cv_parser = CVParser(method='pdfplumber')
        logger.info("✅ CVParser initialisé")
    return _cv_parser


def get_skills_extractor() -> SkillsExtractor:
    """Obtenir l'extracteur de compétences (singleton)"""
    global _skills_extractor
    if _skills_extractor is None:
        _skills_extractor = SkillsExtractor(skills_db_path=str(SKILLS_DB_PATH))
        logger.info("✅ SkillsExtractor initialisé")
    return _skills_extractor


def get_job_matcher() -> JobMatcher:
    """Obtenir le matcher (singleton)"""
    global _job_matcher
    if _job_matcher is None:
        _job_matcher = JobMatcher()
        logger.info("✅ JobMatcher initialisé")
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
        logger.info(f"✅ Dataset chargé : {len(_jobs_dataset.get('jobs', []))} offres")
    return _jobs_dataset


def get_vector_store() -> JobVectorStore:
    """Obtenir le vector store FAISS (singleton)"""
    global _vector_store
    if _vector_store is None:
        _vector_store = JobVectorStore(model_name='all-mpnet-base-v2')
        
        # Chemins de l'index FAISS
        index_path = PROJECT_ROOT / "data" / "faiss_index" / "jobs.index"
        metadata_path = PROJECT_ROOT / "data" / "faiss_index" / "jobs_metadata.pkl"
        
        # Charger l'index si disponible
        if index_path.exists() and metadata_path.exists():
            _vector_store.load(str(index_path), str(metadata_path))
            logger.info(f"✅ Index FAISS chargé : {_vector_store.index.ntotal} offres")
        else:
            # Construire l'index si absent
            logger.warning("⚠️  Index FAISS non trouvé, construction en cours...")
            dataset = get_jobs_dataset()
            _vector_store.build_index(dataset['jobs'], index_type='flat')
            
            # Sauvegarder pour la prochaine fois
            index_path.parent.mkdir(parents=True, exist_ok=True)
            _vector_store.save(str(index_path), str(metadata_path))
            logger.info("✅ Index FAISS construit et sauvegardé")
    
    return _vector_store

def get_ml_classifier():
    """Obtenir le modèle ML XGBoost (singleton)"""
    global _ml_classifier
    if _ml_classifier is None:
        if ML_MODEL_PATH.exists():
            try:
                _ml_classifier = joblib.load(ML_MODEL_PATH)
                logger.info(f"✅ Modèle ML chargé : {ML_MODEL_PATH}")
                logger.info(f"   • Type : {type(_ml_classifier)}")
                logger.info(f"   • Features : {_ml_classifier.n_features_in_}")
            except Exception as e:
                logger.error(f"❌ Erreur chargement modèle ML : {e}")
                _ml_classifier = None
        else:
            logger.warning(f"⚠️ Modèle ML non trouvé : {ML_MODEL_PATH}")
            _ml_classifier = None
    return _ml_classifier


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
    missing_skills: List[str] = []


class RecommendationsResponse(BaseModel):
    recommendations: List[JobRecommendation]
    total_jobs_analyzed: int
    cv_skills_count: int
    faiss_used: bool = False


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


class InterviewRequest(BaseModel):
    """Requête de simulation d'entretien"""
    cv_skills: List[str]
    job_id: str
    num_questions: int = 8


class InterviewResponse(BaseModel):
    """Réponse avec questions d'entretien"""
    job_title: str
    rh_questions: List[Dict]
    technical_questions: List[Dict]
    total_questions: int


class AnswerEvaluationRequest(BaseModel):
    """Requête d'évaluation de réponse"""
    question: str
    answer: str
    question_type: str
    target_skill: Optional[str] = None


class AnswerEvaluationResponse(BaseModel):
    """Réponse d'évaluation"""
    score: float
    evaluation: str
    points_forts: List[str]
    points_amelioration: List[str]
    recommandations: List[str]


class SearchRequest(BaseModel):
    query: str = Field(..., description="Requête de recherche")
    top_k: int = Field(5, ge=1, le=25, description="Nombre de résultats")


class MatchRequest(BaseModel):
    cv_text: str = Field(..., description="Texte du CV")
    top_k: int = Field(5, ge=1, le=25, description="Nombre de résultats")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine - Informations sur l'API"""
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
            "get_job": "/api/v1/jobs/{job_id}",
            "simulate_interview": "/api/v1/simulate-interview",
            "evaluate_answer": "/api/v1/evaluate-answer",
            "search": "/api/v1/search",
            "match": "/api/v1/match"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifier l'état de santé de l'API"""
    try:
        dataset = get_jobs_dataset()
        jobs_count = len(dataset.get('jobs', []))
        models_loaded = (
            _job_matcher is not None and 
            _skills_extractor is not None and
            _ml_classifier is not None  # ← AJOUTER
        )
        
        return {
            "status": "healthy",
            "message": "API opérationnelle",
            "version": "1.0.0",
            "models_loaded": models_loaded,
            "jobs_available": jobs_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
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
    """Extraire les compétences d'un CV PDF"""
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
        
        # Parser le CV
        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Le CV est vide ou illisible. Vérifiez que le PDF contient du texte."
            )
        
        # Extraire les compétences
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)
        
        technical_skills = skills_result['technical_skills']
        soft_skills = skills_result['soft_skills']
        
        logger.info(f"✅ Extraction réussie : {len(technical_skills)} tech skills, {len(soft_skills)} soft skills")
        
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "total_skills": len(technical_skills) + len(soft_skills),
            "cv_text_length": len(cv_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur extraction skills: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du CV: {str(e)}"
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass


@app.post("/api/v1/recommend-jobs", response_model=RecommendationsResponse, tags=["Jobs"])
async def recommend_jobs(
    file: UploadFile = File(...),
    top_n: int = Query(10, ge=1, le=25, description="Nombre de recommandations"),
    min_score: float = Query(40.0, ge=0.0, le=100.0, description="Score minimum"),
    use_faiss: bool = Query(False, description="Utiliser FAISS pour pré-filtrage")
):
    """
    Obtenir des recommandations d'emploi basées sur un CV
    SCORING BASÉ SUR APPROCHE 4 : Coverage + Quality
    ✅ INTÉGRATION MLFLOW : Tracking automatique des runs
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
    tmp_file_path = None
    
    try:
        # ====================================================================
        # ÉTAPE 1 : SAUVEGARDE TEMPORAIRE DU FICHIER
        # ====================================================================
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # ====================================================================
        # ÉTAPE 2 : PARSING DU CV
        # ====================================================================
        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Le CV est vide ou illisible")
        
        # ====================================================================
        # ÉTAPE 3 : EXTRACTION DES COMPÉTENCES
        # ====================================================================
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)
        technical_skills = skills_result['technical_skills']
        soft_skills = skills_result['soft_skills']
        cv_skills = technical_skills + soft_skills
        
        if not cv_skills:
            raise HTTPException(
                status_code=400,
                detail="Aucune compétence détectée dans le CV"
            )
        
        logger.info(f"✅ CV analysé : {len(cv_skills)} compétences détectées")
        
        # ====================================================================
        # 🎯 MLFLOW : DÉMARRAGE DU RUN
        # ====================================================================
        with mlflow.start_run(run_name=f"recommend_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log des paramètres de la requête
            mlflow.log_param("cv_filename", file.filename)
            mlflow.log_param("top_n", top_n)
            mlflow.log_param("min_score", min_score)
            mlflow.log_param("use_faiss", use_faiss)
            mlflow.log_param("cv_text_length", len(cv_text))
            mlflow.log_param("cv_skills_count", len(cv_skills))
            mlflow.log_param("technical_skills_count", len(technical_skills))
            mlflow.log_param("soft_skills_count", len(soft_skills))
            
            # ====================================================================
            # ÉTAPE 4 : OBTENIR LES CANDIDATS (FAISS OU COMPLET)
            # ====================================================================
            if use_faiss:
                vector_store = get_vector_store()
                faiss_candidates = vector_store.search(
                    cv_skills=cv_skills,
                    cv_text=cv_text[:500],
                    top_k=min(50, vector_store.index.ntotal)
                )
                candidate_jobs = [job for job, _ in faiss_candidates]
                logger.info(f"✅ FAISS: {len(candidate_jobs)} candidats pré-filtrés")
                mlflow.log_metric("faiss_candidates_count", len(candidate_jobs))
            else:
                dataset = get_jobs_dataset()
                candidate_jobs = dataset['jobs']
                mlflow.log_metric("total_jobs_in_dataset", len(candidate_jobs))
            
            # ====================================================================
            # ÉTAPE 5 : SCORING AVEC JOBMATCHER (APPROCHE 4)
            # ====================================================================
            matcher = get_job_matcher()
            detailed_results = []
            
            for job in candidate_jobs:
                detailed_score = matcher.calculate_job_match_score(cv_skills, job)
                
                # Extraire les skills matchées
                matching_skills = []
                matched_job_skills = set()
                
                for match in detailed_score['skills_details']['top_matches']:
                    skill_name = match['cv_skill']
                    if skill_name not in matching_skills:
                        matching_skills.append(skill_name)
                    matched_job_skills.add(match['job_skill'])
                
                # Calculer les skills manquantes
                all_job_skills = matcher.extract_job_skills(job)
                missing_skills = [
                    skill for skill in all_job_skills 
                    if skill not in matched_job_skills
                ]
                
                detailed_results.append({
                    'job_id': job['job_id'],
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'remote_ok': job.get('remote_ok', False),
                    'experience': job['experience'],
                    'score': detailed_score['score'],
                    'skills_details': detailed_score['skills_details'],
                    'matching_skills': matching_skills,
                    'missing_skills': missing_skills
                })
            
            # ====================================================================
            # ÉTAPE 6 : TRI ET FILTRAGE
            # ====================================================================
            detailed_results.sort(key=lambda x: x['score'], reverse=True)
            
            filtered_jobs = [
                job for job in detailed_results 
                if job['score'] >= min_score
            ][:top_n]
            
            logger.info(f"✅ {len(filtered_jobs)} recommandations générées (score ≥ {min_score})")
            
            # ====================================================================
            # 🎯 MLFLOW : LOG DES MÉTRIQUES DE SCORING
            # ====================================================================
            mlflow.log_metric("total_jobs_analyzed", len(candidate_jobs))
            mlflow.log_metric("jobs_after_filtering", len(filtered_jobs))
            
            if filtered_jobs:
                scores = [job['score'] for job in filtered_jobs]
                mlflow.log_metric("top1_score", scores[0])
                mlflow.log_metric("avg_score", sum(scores) / len(scores))
                mlflow.log_metric("min_score_returned", scores[-1])
                mlflow.log_metric("max_score", max(scores))
                mlflow.log_metric("median_score", sorted(scores)[len(scores)//2])
                
                # Log distribution des scores
                mlflow.log_metric("scores_above_80", sum(1 for s in scores if s >= 80))
                mlflow.log_metric("scores_60_to_80", sum(1 for s in scores if 60 <= s < 80))
                mlflow.log_metric("scores_below_60", sum(1 for s in scores if s < 60))
            else:
                mlflow.log_metric("top1_score", 0.0)
                mlflow.log_metric("avg_score", 0.0)
                logger.warning(f"⚠️ Aucune recommandation avec score ≥ {min_score}")
            
            # ====================================================================
            # 🎯 MLFLOW : SAUVEGARDE DES ARTEFACTS
            # ====================================================================
            try:
                # Créer un fichier JSON avec les résultats complets
                artifacts_data = {
                    "cv_filename": file.filename,
                    "timestamp": datetime.now().isoformat(),
                    "cv_skills": {
                        "technical": technical_skills,
                        "soft": soft_skills,
                        "total": len(cv_skills)
                    },
                    "recommendations": [
                        {
                            "rank": idx + 1,
                            "job_id": job['job_id'],
                            "title": job['title'],
                            "company": job['company'],
                            "score": round(job['score'], 2),
                            "matching_skills": job['matching_skills'],
                            "missing_skills": job['missing_skills']
                        }
                        for idx, job in enumerate(filtered_jobs)
                    ],
                    "parameters": {
                        "top_n": top_n,
                        "min_score": min_score,
                        "use_faiss": use_faiss
                    }
                }
                
                # Sauvegarder dans un fichier temporaire
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(artifacts_data, f, indent=2, ensure_ascii=False)
                    artifact_path = f.name
                
                # Logger l'artifact dans MLflow
                mlflow.log_artifact(artifact_path, "recommendations")
                
                # Nettoyer le fichier temporaire
                os.unlink(artifact_path)
                
                logger.info("✅ Artefacts MLflow sauvegardés")
                
            except Exception as e:
                logger.error(f"⚠️ Erreur sauvegarde artefacts MLflow : {e}")
            
            # ====================================================================
            # ÉTAPE 7 : SAUVEGARDE DANS POSTGRESQL
            # ====================================================================
            try:
                db = get_db_manager()
                
                # Sauvegarder l'analyse du CV
                cv_analysis_id = db.save_cv_analysis(
                    cv_filename=file.filename,
                    cv_text=cv_text,
                    technical_skills=technical_skills,
                    soft_skills=soft_skills,
                    user_id=None
                )
                
                logger.info(f"✅ CV analysis sauvegardé en BDD (ID: {cv_analysis_id})")
                
                # Log de l'ID dans MLflow
                mlflow.log_param("cv_analysis_id", cv_analysis_id)
                
                # Sauvegarder les recommandations
                for job in filtered_jobs:
                    db.save_job_recommendation(
                        cv_analysis_id=cv_analysis_id,
                        job_id=job['job_id'],
                        job_title=job['title'],
                        company=job['company'],
                        score=job['score'],
                        skills_match=job['score'],
                        experience_match=0,
                        location_match=0,
                        competition_factor=0,
                        matching_skills=job['matching_skills']
                    )
                
                logger.info(f"✅ {len(filtered_jobs)} recommandations sauvegardées en BDD")
                
            except Exception as e:
                logger.error(f"⚠️ Erreur sauvegarde PostgreSQL (non bloquant): {e}")
            
            # ====================================================================
            # ÉTAPE 8 : FORMATER LA RÉPONSE
            # ====================================================================
            recommendations = []
            for job in filtered_jobs:
                recommendations.append({
                    "job_id": job['job_id'],
                    "title": job['title'],
                    "company": job['company'],
                    "location": job['location'],
                    "remote": job['remote_ok'],
                    "experience_required": job['experience'],
                    "score": job['score'],
                    "skills_match": job['score'],
                    "experience_match": 0,
                    "location_match": 0,
                    "competition_factor": 0,
                    "matching_skills": job['matching_skills'],
                    "missing_skills": job['missing_skills']
                })
            
            # ====================================================================
            # 🎯 MLFLOW : TAG DU RUN
            # ====================================================================
            mlflow.set_tag("cv_filename", file.filename)
            mlflow.set_tag("success", "true")
            mlflow.set_tag("environment", "production")
            
            logger.info(f"✅ MLflow run complété pour {file.filename}")
            
            # ====================================================================
            # RETOUR DE LA RÉPONSE
            # ====================================================================
            return {
                "recommendations": recommendations,
                "total_jobs_analyzed": len(candidate_jobs),
                "cv_skills_count": len(cv_skills),
                "faiss_used": use_faiss
            }
        
    except HTTPException:
        # ====================================================================
        # 🎯 MLFLOW : LOG DES ERREURS
        # ====================================================================
        try:
            with mlflow.start_run(run_name=f"error_{file.filename}_{datetime.now().strftime('%H%M%S')}"):
                mlflow.set_tag("success", "false")
                mlflow.set_tag("error_type", "HTTPException")
                mlflow.log_param("cv_filename", file.filename)
        except:
            pass
        raise
        
    except Exception as e:
        # ====================================================================
        # 🎯 MLFLOW : LOG DES ERREURS SYSTÈME
        # ====================================================================
        logger.error(f"Erreur génération recommandations: {e}")
        try:
            with mlflow.start_run(run_name=f"error_{file.filename}_{datetime.now().strftime('%H%M%S')}"):
                mlflow.set_tag("success", "false")
                mlflow.set_tag("error_type", type(e).__name__)
                mlflow.log_param("cv_filename", file.filename)
                mlflow.log_param("error_message", str(e))
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )
        
    finally:
        # ====================================================================
        # NETTOYAGE DU FICHIER TEMPORAIRE
        # ====================================================================
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass


@app.get("/api/v1/jobs", response_model=List[JobDetail], tags=["Jobs"])
async def list_jobs(
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    remote: Optional[bool] = Query(None, description="Filtrer par télétravail"),
    experience: Optional[str] = Query(None, description="Filtrer par niveau d'expérience"),
    limit: int = Query(25, ge=1, le=100, description="Nombre maximum de résultats")
):
    """Lister toutes les offres d'emploi disponibles"""
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']
        
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
        
        if experience:
            filtered_jobs = [
                job for job in filtered_jobs 
                if job.get('experience', '').lower() == experience.lower()
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
        logger.error(f"Erreur liste jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des jobs: {str(e)}"
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobDetail, tags=["Jobs"])
async def get_job(job_id: str):
    """Obtenir les détails d'une offre d'emploi spécifique"""
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']
        
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
        logger.error(f"Erreur get job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération du job: {str(e)}"
        )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Obtenir des statistiques sur le système"""
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
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )


@app.get("/api/v1/faiss-stats", tags=["Stats"])
async def get_faiss_stats():
    """Obtenir des statistiques sur l'index FAISS"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return {
            "faiss_enabled": stats['indexed'],
            "total_jobs_indexed": stats['total_jobs'],
            "model_used": stats['model_name'],
            "embedding_dimension": stats['dimension'],
            "index_type": "Flat L2 (exact search)"
        }
        
    except Exception as e:
        logger.error(f"Erreur faiss stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de la récupération des stats FAISS: {str(e)}"}
        )


@app.get("/api/v1/history/cv-analyses", tags=["History"])
async def get_cv_analyses_history(limit: int = Query(10, ge=1, le=100)):
    """Récupérer l'historique des analyses de CV"""
    try:
        db = get_db_manager()
        analyses = db.get_recent_cv_analyses(limit=limit)
        
        return {
            "total": len(analyses),
            "analyses": analyses
        }
        
    except Exception as e:
        logger.error(f"Erreur history cv-analyses: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur BDD: {str(e)}")


@app.get("/api/v1/history/recommendations/{cv_analysis_id}", tags=["History"])
async def get_recommendations_history(cv_analysis_id: int):
    """Récupérer les recommandations d'une analyse CV"""
    try:
        db = get_db_manager()
        recommendations = db.get_recommendations_for_cv(cv_analysis_id)
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune recommandation trouvée pour l'analyse {cv_analysis_id}"
            )
        
        return {
            "cv_analysis_id": cv_analysis_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur history recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur BDD: {str(e)}")


@app.get("/api/v1/stats/database", tags=["Statistics"])
async def get_database_statistics():
    """Statistiques globales de la base de données"""
    try:
        db = get_db_manager()
        stats = db.get_statistics()
        
        return {
            "database_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur BDD: {str(e)}")


@app.post("/api/v1/simulate-interview", response_model=InterviewResponse, tags=["Interview"])
async def simulate_interview(request: InterviewRequest):
    """Générer des questions d'entretien personnalisées avec Groq"""
    try:
        # Récupérer l'offre
        dataset = get_jobs_dataset()
        job = next((j for j in dataset['jobs'] if j['job_id'] == request.job_id), None)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Offre {request.job_id} introuvable"
            )
        
        # Générer les questions avec Groq
        simulator = get_interview_simulator()
        
        questions = simulator.generate_questions(
            cv_skills=request.cv_skills,
            job_title=job['title'],
            job_description=job['description'],
            job_requirements=job['requirements'],
            num_questions=request.num_questions
        )
        
        # ====================================================================
        # SAUVEGARDE DANS POSTGRESQL
        # ====================================================================
        try:
            db = get_db_manager()
            
            cv_analysis_id = db.save_cv_analysis(
                cv_filename="interview_simulation.pdf",
                cv_text="",
                technical_skills=request.cv_skills,
                soft_skills=[],
                user_id=None
            )
            
            simulation_id = db.save_interview_simulation(
                cv_analysis_id=cv_analysis_id,
                job_id=request.job_id,
                questions=questions,
                answers=[],
                scores=[],
                average_score=0.0
            )
            
            logger.info(f"✅ Simulation d'entretien sauvegardée (ID: {simulation_id})")
            
        except Exception as e:
            logger.error(f"⚠️ Erreur sauvegarde simulation: {e}")
        
        return {
            "job_title": job['title'],
            "rh_questions": questions['rh_questions'],
            "technical_questions": questions['technical_questions'],
            "total_questions": len(questions['rh_questions']) + len(questions['technical_questions'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur génération questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur génération questions : {str(e)}"
        )


@app.post("/api/v1/evaluate-answer", response_model=AnswerEvaluationResponse, tags=["Interview"])
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Évaluer la réponse d'un candidat avec Groq"""
    try:
        if not request.answer or len(request.answer.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="La réponse est trop courte (minimum 10 caractères)"
            )
        
        simulator = get_interview_simulator()
        
        evaluation = simulator.evaluate_answer(
            question=request.question,
            answer=request.answer,
            question_type=request.question_type,
            target_skill=request.target_skill
        )
        
        return evaluation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur évaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur évaluation : {str(e)}"
        )


@app.post("/api/v1/search", tags=["Jobs"])
async def search_jobs(request: SearchRequest):
    """Rechercher des offres d'emploi par similarité sémantique"""
    try:
        vector_store = get_vector_store()
        
        results = vector_store.search(
            cv_skills=[],
            cv_text=request.query,
            top_k=request.top_k
        )
        
        formatted_results = []
        for job, score in results:
            formatted_results.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job.get('remote_ok', False),
                "experience_required": job['experience'],
                "score": float(score) * 100,
                "description": job['description'][:200] + "..."
            })
        
        return {
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Erreur search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche: {str(e)}"
        )


@app.post("/api/v1/match", tags=["Jobs"])
async def match_cv(request: MatchRequest):
    """Matcher un texte de CV avec les meilleures offres"""
    try:
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(request.cv_text)
        cv_skills = skills_result['technical_skills'] + skills_result['soft_skills']
        
        vector_store = get_vector_store()
        results = vector_store.search(
            cv_skills=cv_skills,
            cv_text=request.cv_text,
            top_k=request.top_k
        )
        
        matches = []
        for job, score in results:
            matches.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job.get('remote_ok', False),
                "experience_required": job['experience'],
                "score": float(score) * 100,
                "description": job['description'][:200] + "..."
            })
        
        return {
            "cv_summary": request.cv_text[:200] + "...",
            "cv_skills_found": len(cv_skills),
            "matches": matches,
            "count": len(matches)
        }
        
    except Exception as e:
        logger.error(f"Erreur match: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du matching: {str(e)}"
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
    logger.info("="*60)
    logger.info("🚀 DÉMARRAGE DE L'API")
    logger.info("="*60)
    
    # Vérifier que les fichiers nécessaires existent
    if not JOBS_DATASET_PATH.exists():
        logger.warning(f"⚠️  Dataset d'offres manquant: {JOBS_DATASET_PATH}")
    
    if not SKILLS_DB_PATH.exists():
        logger.warning(f"⚠️  Base de compétences manquante: {SKILLS_DB_PATH}")

    # ✅ AJOUTER : Pré-charger le modèle ML
    logger.info("⏳ Chargement des modèles...")
    
    try:
        # Forcer le chargement du JobMatcher
        _ = get_job_matcher()
        logger.info("✅ JobMatcher chargé")
    except Exception as e:
        logger.error(f"❌ Erreur JobMatcher : {e}")
    
    try:
        # Forcer le chargement du SkillsExtractor
        _ = get_skills_extractor()
        logger.info("✅ SkillsExtractor chargé")
    except Exception as e:
        logger.error(f"❌ Erreur SkillsExtractor : {e}")
    
    try:
        # Forcer le chargement du modèle ML
        ml_model = get_ml_classifier()
        if ml_model is not None:
            logger.info("✅ Modèle ML chargé avec succès")
        else:
            logger.warning("⚠️ Modèle ML non disponible (fonctionnement en mode dégradé)")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle ML : {e}")
    
    # Pré-charger FAISS au démarrage
    logger.info("⏳ Pré-chargement du vector store FAISS...")
    try:
        _ = get_vector_store()
        logger.info("✅ FAISS chargé avec succès")
    except Exception as e:
        logger.error(f"⚠️  Erreur FAISS : {e}")
    
    logger.info("✅ API prête à recevoir des requêtes")
    logger.info("📖 Documentation : http://127.0.0.1:8000/docs")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt de l'API"""
    logger.info("🔄 Arrêt de l'API - Fermeture de la connexion PostgreSQL...")
    close_db_connection()
    logger.info("✅ API arrêtée proprement")


# ============================================================================
# POINT D'ENTRÉE POUR UVICORN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)