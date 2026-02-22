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
from .vector_store import JobVectorStore
from .interview_simulator import InterviewSimulator, get_interview_simulator
from src.database import get_db_manager

# Import des modules locaux
from .cv_parser import CVParser
from .skills_extractor import SkillsExtractor
from .job_matcher import JobMatcher
from .ml_predictor import get_ml_predictor

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
        _job_matcher = JobMatcher()
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

_vector_store = None

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
            print(f"✅ Index FAISS chargé : {_vector_store.index.ntotal} offres")
        else:
            # Construire l'index si absent
            print("⚠️  Index FAISS non trouvé, construction en cours...")
            dataset = get_jobs_dataset()
            _vector_store.build_index(dataset['jobs'], index_type='flat')
            
            # Sauvegarder pour la prochaine fois
            index_path.parent.mkdir(parents=True, exist_ok=True)
            _vector_store.save(str(index_path), str(metadata_path))
            print(f"✅ Index FAISS construit et sauvegardé")
    
    return _vector_store

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
    ml_label: Optional[str] = None     
    ml_score: Optional[float] = None     
    ml_probabilities: Optional[Dict] = None  
    ml_available: bool = False           

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
    min_score: float = Query(40.0, ge=0.0, le=100.0, description="Score minimum"),
    use_faiss: bool = Query(False, description="Utiliser FAISS pour pré-filtrage")
):
    """
    Obtenir des recommandations d'emploi basées sur un CV
    SCORING BASÉ SUR APPROCHE 4 : Coverage + Quality
    
    **Workflow :**
    1. Extraction des compétences du CV
    2. Si use_faiss=True : Pré-filtrage rapide avec FAISS (top 50)
    3. Scoring détaillé avec JobMatcher (Approche 4)
    4. Tri et filtrage final
    
    Args:
        file: Fichier PDF du CV
        top_n: Nombre de recommandations (défaut: 10)
        min_score: Score minimum (défaut: 40.0)
        use_faiss: Utiliser FAISS (défaut: False)
        
    Returns:
        Liste des jobs recommandés avec scores détaillés
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
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
            raise HTTPException(status_code=400, detail="Le CV est vide ou illisible")
        
        # 2. Extraire les compétences
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)
        technical_skills = skills_result['technical_skills']
        soft_skills = skills_result['soft_skills']
        cv_skills = technical_skills + soft_skills 
        
        if not cv_skills:
            raise HTTPException(
                status_code=400,
                detail="Aucune compétence technique détectée dans le CV"
            )
        
        # 3. Obtenir les candidats
        if use_faiss:
            vector_store = get_vector_store()
            faiss_candidates = vector_store.search(
                cv_skills=cv_skills,
                cv_text=cv_text[:500],
                top_k=min(50, vector_store.index.ntotal)
            )
            candidate_jobs = [job for job, _ in faiss_candidates]
        else:
            dataset = get_jobs_dataset()
            candidate_jobs = dataset['jobs']
        
        # 4. Scoring avec JobMatcher (Approche 4)
        matcher = get_job_matcher()
        ml_predictor = get_ml_predictor()
        detailed_results = []
        
        for job in candidate_jobs:
            # Calcul du score avec Approche 4
            detailed_score = matcher.calculate_job_match_score(cv_skills, job)
            
            # Extraire TOUTES les skills matchées
            matching_skills = []
            matched_job_skills = set()  # Tracker les job skills matchées
            
            for match in detailed_score['skills_details']['top_matches']:
                # Ajouter cv_skill à matching_skills
                skill_name = match['cv_skill']
                if skill_name not in matching_skills:
                    matching_skills.append(skill_name)
                
                # Tracker la job_skill correspondante
                matched_job_skills.add(match['job_skill'])
            
            # Calculer les skills manquantes
            # = Toutes les skills du job - celles qui ont matché
            all_job_skills = matcher.extract_job_skills(job)
            missing_skills = [
                skill for skill in all_job_skills 
                if skill not in matched_job_skills
            ]

            # ✅ FIX : Prédiction ML avec la bonne signature
            ml_result = {'ml_available': False, 'ml_label': 'N/A', 'ml_score': None}
            if ml_predictor.is_loaded:
                try:
                    # ✅ Séparer skills techniques et soft du job
                    known_technical = set(
                        s.lower() for s in matcher.skills_db.get('technical_skills', [])
                    )
                    known_soft = set(
                        s.lower() for s in matcher.skills_db.get('soft_skills', [])
                    )
                    job_technical_skills = [
                        s for s in all_job_skills if s.lower() in known_technical
                    ]
                    job_soft_skills = [
                        s for s in all_job_skills if s.lower() in known_soft
                    ]
                    # Skills non catégorisées → technical par défaut
                    categorized = set(job_technical_skills + job_soft_skills)
                    job_technical_skills += [
                        s for s in all_job_skills if s not in categorized
                    ]

                    # ✅ Construire le texte brut du job pour TF-IDF et embedding
                    job_raw_text = " ".join([
                        job.get('title', ''),
                        job.get('description', ''),
                        " ".join(job.get('requirements', [])),
                        " ".join(job.get('nice_to_have', []))
                    ]).strip()

                    ml_features = ml_predictor.compute_features(
                        cv_technical_skills=technical_skills,
                        cv_soft_skills=soft_skills,
                        job_technical_skills=job_technical_skills,   
                        job_soft_skills=job_soft_skills,             
                        skills_details=detailed_score['skills_details'],  
                        cv_raw_text=cv_text,                          
                        job_raw_text=job_raw_text,                    
                        sentence_model=matcher.model
                    )
                    ml_result = ml_predictor.predict(ml_features)
                except Exception as e:
                    print(f"⚠️ ML prediction error: {e}")
            
            detailed_results.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'remote_ok': job.get('remote_ok', False),
                'experience': job['experience'],
                'score': detailed_score['score'],
                'skills_details': detailed_score['skills_details'],
                'matching_skills': matching_skills,  # Skills CV qui matchent
                'missing_skills': missing_skills,  # Skills job non matchées
                'ml_label': ml_result.get('ml_label', 'N/A'),          
                'ml_score': ml_result.get('ml_score'),                   
                'ml_probabilities': ml_result.get('ml_probabilities'),  
                'ml_available': ml_result.get('ml_available', False),    
            })
        
        # 5. Tri par score
        detailed_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 6. Filtrer par score minimum et top_n
        filtered_jobs = [
            job for job in detailed_results 
            if job['score'] >= min_score
        ][:top_n]
        
        # 7. Formater la réponse
        recommendations = []
        for job in filtered_jobs:
            recommendations.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job['remote_ok'],
                "experience_required": job['experience'],
                "score": float(job['score']),  # ✅ Convertir en float Python
                "skills_match": float(job['score']),  # ✅ Convertir en float Python
                "experience_match": 0,  # Deprecated (compatibilité)
                "location_match": 0,  # Deprecated (compatibilité)
                "competition_factor": 0,  # Deprecated (compatibilité)
                "matching_skills": job['matching_skills'],  
                "missing_skills": job['missing_skills'],
                "ml_label": job['ml_label'],                
                "ml_score": job['ml_score'],               
                "ml_probabilities": job['ml_probabilities'], 
                "ml_available": job['ml_available'],         
            })

        # 1️⃣ Sauvegarder l'analyse CV
        db = get_db_manager()
        cv_id = db.save_cv_analysis(
            cv_filename=file.filename,
            cv_text=cv_text,  # Texte extrait du CV
            technical_skills=technical_skills,  # Liste des compétences
            soft_skills=soft_skills,
            user_id=1  # anonymous
        )
        
        # 2️⃣ Sauvegarder chaque recommandation
        for job in filtered_jobs:
            db.save_job_recommendation(
                cv_analysis_id=cv_id,
                job_id=job['job_id'],
                job_title=job['title'],
                company=job['company'],
                score=float(job['score']),  # ✅ AJOUTER float()
                coverage=float(job.get('skills_details', {}).get('coverage', 0)),  # ✅ AJOUTER float()
                quality=float(job.get('skills_details', {}).get('quality', 0)),  # ✅ AJOUTER float()
                matching_skills=job.get('matching_skills', []),
                missing_skills=job.get('missing_skills', [])
            )
        
        return {
            "recommendations": recommendations,
            "total_jobs_analyzed": len(candidate_jobs),
            "cv_skills_count": len(cv_skills),
            "database_id": cv_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )
    finally:
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

        # Filtre expérience
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

@app.get("/api/v1/faiss-stats", tags=["Stats"])
async def get_faiss_stats():
    """
    Obtenir des statistiques sur l'index FAISS
    
    Returns:
        Statistiques de l'index vectoriel
    """
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
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de la récupération des stats FAISS: {str(e)}"}
        )
    
@app.post("/api/v1/simulate-interview", response_model=InterviewResponse, tags=["Interview"])
async def simulate_interview(request: InterviewRequest):
    """
    Générer des questions d'entretien personnalisées avec Groq (Llama 3.1 70B)
    
    **Workflow:**
    1. Récupération de l'offre d'emploi
    2. Génération de questions RH et techniques par LLM
    3. Questions adaptées au profil candidat
    
    Args:
        cv_skills: Compétences du candidat
        job_id: ID de l'offre ciblée
        num_questions: Nombre de questions (défaut: 8)
        
    Returns:
        Questions RH et techniques générées par IA
    """
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
        
        return {
            "job_title": job['title'],
            "rh_questions": questions['rh_questions'],
            "technical_questions": questions['technical_questions'],
            "total_questions": len(questions['rh_questions']) + len(questions['technical_questions'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur génération questions : {str(e)}"
        )


@app.post("/api/v1/evaluate-answer", response_model=AnswerEvaluationResponse, tags=["Interview"])
async def evaluate_answer(request: AnswerEvaluationRequest):
    """
    Évaluer la réponse d'un candidat avec Groq (Llama 3.1 70B)
    
    **Workflow:**
    1. Analyse de la réponse par LLM
    2. Scoring automatique (0-100)
    3. Génération de feedback personnalisé
    
    Args:
        question: Question posée
        answer: Réponse du candidat
        question_type: Type de question (présentation, technique, etc.)
        target_skill: Compétence évaluée (optionnel)
        
    Returns:
        Score, feedback et recommandations d'amélioration
    """
    try:
        # Validation
        if not request.answer or len(request.answer.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="La réponse est trop courte (minimum 10 caractères)"
            )
        
        # Évaluer la réponse avec Groq
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
        raise HTTPException(
            status_code=500,
            detail=f"Erreur évaluation : {str(e)}"
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
    
    # ✅ PRÉ-CHARGER FAISS AU DÉMARRAGE
    print("\n⏳ Pré-chargement du vector store FAISS...")
    try:
        _ = get_vector_store()  # Force le chargement
        print("✅ FAISS chargé avec succès")
    except Exception as e:
        print(f"⚠️  Erreur FAISS : {e}")

    # ✅ Pré-charger le modèle ML
    print("\n⏳ Pré-chargement du modèle XGBoost...")
    try:
        ml = get_ml_predictor()
        if ml.is_loaded:
            print("✅ Modèle XGBoost chargé avec succès")
        else:
            print("⚠️  Modèle XGBoost non disponible (lance train_and_log.py)")
    except Exception as e:
        print(f"⚠️  Erreur ML : {e}")

    print("\n✅ API prête à recevoir des requêtes")
    print("📖 Documentation : http://127.0.0.1:8000/docs")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt de l'API"""
    print("\n🛑 Arrêt de l'API")