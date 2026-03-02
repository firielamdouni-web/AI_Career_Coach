"""
🎯 API FastAPI pour le système de matching CV ↔ Jobs
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
from .interview_simulator import get_interview_simulator
from src.database import get_db_manager
import logging

from .cv_parser import CVParser
from .skills_extractor import SkillsExtractor
from .job_matcher import JobMatcher
from .ml_predictor import get_ml_predictor
from src.job_scraper import get_job_scraper

logger = logging.getLogger(__name__)

app = FastAPI(
    title="🎯 AI Career Coach API",
    description="API de matching CV ↔ Offres d'emploi avec IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent
JOBS_DATASET_PATH = PROJECT_ROOT / "data" / "jobs" / "jobs_dataset.json"
SKILLS_DB_PATH = PROJECT_ROOT / "data" / "skills_reference.json"

_cv_parser = None
_skills_extractor = None
_job_matcher = None
_jobs_dataset = None
_vector_store = None


def get_cv_parser() -> CVParser:
    global _cv_parser
    if _cv_parser is None:
        _cv_parser = CVParser(method='pdfplumber')
    return _cv_parser


def get_skills_extractor() -> SkillsExtractor:
    global _skills_extractor
    if _skills_extractor is None:
        _skills_extractor = SkillsExtractor(skills_db_path=str(SKILLS_DB_PATH))
    return _skills_extractor


def get_job_matcher() -> JobMatcher:
    global _job_matcher
    if _job_matcher is None:
        _job_matcher = JobMatcher()
    return _job_matcher


def get_jobs_dataset() -> Dict:
    global _jobs_dataset
    if _jobs_dataset is None:
        if not JOBS_DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset non trouvé : {JOBS_DATASET_PATH}")
        with open(JOBS_DATASET_PATH, 'r', encoding='utf-8') as f:
            _jobs_dataset = json.load(f)
    return _jobs_dataset


def get_vector_store() -> JobVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = JobVectorStore(model_name='all-mpnet-base-v2')
        index_path = PROJECT_ROOT / "data" / "faiss_index" / "jobs.index"
        metadata_path = PROJECT_ROOT / "data" / "faiss_index" / "jobs_metadata.pkl"
        if index_path.exists() and metadata_path.exists():
            _vector_store.load(str(index_path), str(metadata_path))
            print(f"✅ Index FAISS chargé : {_vector_store.index.ntotal} offres")
        else:
            print("⚠️  Index FAISS non trouvé, construction en cours...")
            dataset = get_jobs_dataset()
            _vector_store.build_index(dataset['jobs'], index_type='flat')
            index_path.parent.mkdir(parents=True, exist_ok=True)
            _vector_store.save(str(index_path), str(metadata_path))
    return _vector_store


# ============================================================================
# NORMALISATION DES JOBS SCRAPÉS
# ============================================================================

def _parse_skills_field(raw) -> List[str]:
    """Parse un champ skills qui peut être str JSON, list, ou None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        # S'assurer que chaque élément est une string
        return [str(s).strip() for s in raw if s]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        # Tenter JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if s]
        except Exception:
            pass
        # Fallback : split virgule
        return [s.strip() for s in raw.split(',') if s.strip()]
    return []


def _normalize_scraped_job(row: Dict) -> Dict:
    """
    Convertit une ligne scraped_jobs en format uniforme pour le JobMatcher.
    Garantit que job_id est unique (préfixe 'sc_') pour éviter collision avec jobs locaux.
    """
    job = dict(row)

    # ── job_id unique ──────────────────────────────────────────────────────
    raw_id = str(job.get('job_id') or job.get('id') or '')
    if not raw_id.startswith('sc_'):
        job['job_id'] = f"sc_{raw_id}" if raw_id else f"sc_{id(job)}"

    # ── skills ────────────────────────────────────────────────────────────
    job['requirements'] = _parse_skills_field(
        job.get('required_skills') or job.get('requirements')
    )
    job['nice_to_have'] = []

    # ── champs obligatoires pour JobMatcher ───────────────────────────────
    job['title'] = job.get('title') or 'Sans titre'
    job['company'] = job.get('company') or 'Inconnue'
    job['location'] = job.get('location') or 'Non spécifié'
    job['description'] = job.get('description') or ''
    job['experience'] = job.get('experience') or 'Non spécifié'
    job['remote_ok'] = bool(job.get('is_remote') or job.get('remote_ok'))
    job['url'] = job.get('url') or ''
    job['source'] = job.get('source') or 'scraped'
    job['is_scraped'] = True

    return job


# ============================================================================
# HELPER : RÉCUPÉRER JOBS SCRAPÉS DEPUIS LA DB
# ============================================================================

def _get_scraped_jobs_from_db() -> List[Dict]:
    """Charge et normalise tous les jobs scrapés depuis PostgreSQL."""
    try:
        db = get_db_manager()
        db.cursor.execute("SELECT * FROM scraped_jobs ORDER BY scraped_at DESC")
        rows = db.cursor.fetchall()
        jobs = [_normalize_scraped_job(dict(r)) for r in rows]
        logger.info(f"✅ {len(jobs)} jobs scrapés chargés depuis la DB")
        return jobs
    except Exception as e:
        logger.warning(f"⚠️ Impossible de charger les jobs scrapés : {e}")
        return []


# ============================================================================
# HELPER : SCRAPER EN TEMPS RÉEL puis sauvegarder
# ============================================================================

def _scrape_and_save(query: str, location: str = "France",
                     num_pages: int = 2) -> List[Dict]:
    try:
        scraper = get_job_scraper()
        raw_jobs = scraper.search_jobs(
            query=query, location=location,
            num_pages=num_pages, remote_only=False, date_posted="month"
        )
        logger.info(f"🌐 JSearch : {len(raw_jobs)} offres trouvées pour '{query}'")

        try:
            db = get_db_manager()
            saved = 0
            for j in raw_jobs:
                try:
                    db.cursor.execute(
                        "SELECT 1 FROM scraped_jobs WHERE job_id = %s LIMIT 1",
                        (j.get('job_id') or j.get('id'),)
                    )
                    if not db.cursor.fetchone():
                        db.cursor.execute("""
                            INSERT INTO scraped_jobs
                                (job_id, title, company, location, description,
                                 url, source, employment_type, is_remote,
                                 salary_min, salary_max, required_skills, scraped_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                        """, (
                            j.get('job_id') or j.get('id'),
                            j.get('title', ''), j.get('company', ''),
                            j.get('location', ''), j.get('description', ''),
                            j.get('url', ''), j.get('source', 'jsearch'),
                            j.get('employment_type', ''),
                            bool(j.get('remote_ok') or j.get('is_remote')),
                            j.get('salary_min'), j.get('salary_max'),
                            json.dumps(_parse_skills_field(
                                j.get('required_skills') or j.get('requirements', [])
                            )),
                        ))
                        saved += 1
                except Exception as ins_err:
                    logger.warning(f"Insert job error: {ins_err}")
                    db.conn.rollback()          # ← était db.connection.rollback()

            db.conn.commit()                   # ← était db.connection.commit()
            logger.info(f"💾 {saved} nouveaux jobs sauvegardés en DB")
        except Exception as db_err:
            logger.warning(f"DB save error: {db_err}")

        return [_normalize_scraped_job(j) for j in raw_jobs]

    except Exception as e:
        logger.warning(f"⚠️ Scraping échoué : {e}")
        return []


# ============================================================================
# MODÈLES PYDANTIC
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
    url: Optional[str] = None
    source: Optional[str] = None          # "local" | "scraped"
    is_scraped: Optional[bool] = False
    employment_type: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    ml_label: Optional[str] = None
    ml_score: Optional[float] = None
    ml_probabilities: Optional[Dict] = None
    ml_available: bool = False


class RecommendationsResponse(BaseModel):
    recommendations: List[JobRecommendation]
    total_jobs_analyzed: int
    cv_skills_count: int
    local_jobs_count: int
    scraped_jobs_count: int


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
    cv_skills: List[str]
    job_id: str
    num_questions: int = 8


class InterviewResponse(BaseModel):
    job_title: str
    rh_questions: List[Dict]
    technical_questions: List[Dict]
    total_questions: int


class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str
    question_type: str
    target_skill: Optional[str] = None


class AnswerEvaluationResponse(BaseModel):
    score: float
    evaluation: str
    points_forts: List[str]
    points_amelioration: List[str]
    recommandations: List[str]


class MLPredictRequest(BaseModel):
    cv_technical_skills: List[str] = Field(..., description="Skills techniques du CV")
    cv_soft_skills: List[str] = Field(default=[], description="Soft skills du CV")
    job_id: str = Field(..., description="ID de l'offre d'emploi")


class MLPredictResponse(BaseModel):
    job_id: str
    job_title: str
    ml_label: str
    ml_score: float
    ml_probabilities: Dict
    features_used: Dict
    model_info: Dict


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🎯 AI Career Coach API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        dataset = get_jobs_dataset()
        jobs_count = len(dataset.get('jobs', []))
        models_loaded = _job_matcher is not None and _skills_extractor is not None
        return {
            "status": "healthy",
            "message": "API opérationnelle",
            "version": "1.0.0",
            "models_loaded": models_loaded,
            "jobs_available": jobs_count
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "status": "unhealthy", "message": str(e),
            "version": "1.0.0", "models_loaded": False, "jobs_available": 0
        })


@app.post("/api/v1/extract-skills", response_model=SkillsResponse, tags=["Skills"])
async def extract_skills(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="CV vide ou illisible")
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)
        return {
            "technical_skills": skills_result['technical_skills'],
            "soft_skills": skills_result['soft_skills'],
            "total_skills": len(skills_result['technical_skills']) + len(skills_result['soft_skills']),
            "cv_text_length": len(cv_text)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur CV: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


# ============================================================================
# ENDPOINT PRINCIPAL : RECOMMEND JOBS
# ============================================================================

@app.post("/api/v1/recommend-jobs",
          response_model=RecommendationsResponse,
          tags=["Jobs"])
async def recommend_jobs(
    file: UploadFile = File(...),
    top_n: int = Query(50, ge=1, le=200, description="Nombre de recommandations"),  # ← le=200
    min_score: float = Query(0.0, ge=0.0, le=100.0, description="Score minimum"),
    use_faiss: bool = Query(False, description="Utiliser FAISS"),
    live_scrape: bool = Query(True, description="Scraper en temps réel via JSearch")
):
    """
    🎯 Analyse un CV PDF et retourne les meilleures offres (locales + réelles JSearch).

    Workflow :
    1. Parse le CV → extrait les compétences
    2. Si live_scrape=True → lance JSearch avec les skills détectées (temps réel)
    3. Fusionne jobs locaux (JSON) + jobs scrapés (DB + temps réel)
    4. Score chaque offre avec JobMatcher (semantic matching)
    5. Prédiction ML XGBoost (si dispo)
    6. Retourne toutes les offres triées par score
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")

    tmp_file_path = None
    try:
        # ── 1. Parse CV ───────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        parser = get_cv_parser()
        cv_text = parser.parse(tmp_file_path)
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="CV vide ou illisible")

        # ── 2. Extraire compétences ───────────────────────────────────────
        extractor = get_skills_extractor()
        skills_result = extractor.extract_from_cv(cv_text)
        technical_skills = skills_result['technical_skills']
        soft_skills = skills_result['soft_skills']
        cv_skills = technical_skills + soft_skills

        if not cv_skills:
            raise HTTPException(status_code=400, detail="Aucune compétence détectée")

        # ── 3. Matcher + ML predictor ────────────────────────────────────
        matcher = get_job_matcher()
        ml_predictor = get_ml_predictor()

        # ── 4. Construire le pool de jobs ────────────────────────────────
        # 4a. Jobs locaux JSON
        dataset = get_jobs_dataset()
        local_jobs = dataset['jobs']
        for j in local_jobs:
            j.setdefault('source', 'local')
            j.setdefault('is_scraped', False)

        # 4b. Jobs scrapés existants en DB
        db_scraped = _get_scraped_jobs_from_db()

        # 4c. Scraping temps réel JSearch (si activé)
        live_scraped = []
        if live_scrape:
            # Utiliser les top 3 compétences techniques comme query
            top_skills = technical_skills[:3] if technical_skills else ['Data Science']
            query = " ".join(top_skills)
            logger.info(f"🌐 Scraping JSearch pour : '{query}'")
            live_scraped = _scrape_and_save(query=query, location="France", num_pages=2)
            logger.info(f"🌐 {len(live_scraped)} offres récupérées en temps réel")

        # Fusionner en évitant les doublons (par job_id)
        seen_ids = set()
        candidate_jobs = []

        for j in local_jobs:
            jid = j.get('job_id', '')
            if jid not in seen_ids:
                seen_ids.add(jid)
                candidate_jobs.append(j)

        # Fusionner DB + live (live_scraped en priorité car plus récents)
        all_scraped = {j['job_id']: j for j in db_scraped}
        for j in live_scraped:
            all_scraped[j['job_id']] = j  # écrase avec version fraîche

        for j in all_scraped.values():
            jid = j.get('job_id', '')
            if jid not in seen_ids:
                seen_ids.add(jid)
                candidate_jobs.append(j)

        local_count = len(local_jobs)
        scraped_count = len(candidate_jobs) - local_count
        logger.info(
            f"📊 Pool total : {len(candidate_jobs)} jobs "
            f"({local_count} locaux + {scraped_count} scrapés)"
        )

        # ── 5. Scorer chaque job ──────────────────────────────────────────
        detailed_results = []

        for job in candidate_jobs:
            try:
                detailed_score = matcher.calculate_job_match_score(cv_skills, job)
            except Exception as e:
                logger.warning(f"Score error on {job.get('job_id')}: {e}")
                continue

            # Skills matchées
            matching_skills = []
            matched_job_skills = set()
            for match in detailed_score['skills_details']['top_matches']:
                s = match['cv_skill']
                if s not in matching_skills:
                    matching_skills.append(s)
                matched_job_skills.add(match['job_skill'])

            # Skills manquantes
            all_job_skills = matcher.extract_job_skills(job)
            missing_skills = [s for s in all_job_skills if s not in matched_job_skills]

            # ML
            ml_result = {'ml_available': False, 'ml_label': 'N/A',
                         'ml_score': None, 'ml_probabilities': None}
            if ml_predictor.is_loaded:
                try:
                    known_tech = set(s.lower() for s in matcher.skills_db.get('technical_skills', []))
                    known_soft = set(s.lower() for s in matcher.skills_db.get('soft_skills', []))
                    job_tech = [s for s in all_job_skills if s.lower() in known_tech]
                    job_soft = [s for s in all_job_skills if s.lower() in known_soft]
                    categorized = set(job_tech + job_soft)
                    job_tech += [s for s in all_job_skills if s not in categorized]

                    job_raw = " ".join([
                        job.get('title', ''), job.get('description', ''),
                        " ".join(job.get('requirements', [])),
                        " ".join(job.get('nice_to_have', []))
                    ]).strip()

                    ml_feat = ml_predictor.compute_features(
                        cv_technical_skills=technical_skills,
                        cv_soft_skills=soft_skills,
                        job_technical_skills=job_tech,
                        job_soft_skills=job_soft,
                        skills_details=detailed_score['skills_details'],
                        cv_raw_text=cv_text,
                        job_raw_text=job_raw,
                        sentence_model=matcher.model
                    )
                    ml_result = ml_predictor.predict(ml_feat)
                except Exception as e:
                    logger.warning(f"ML error: {e}")

            detailed_results.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'location': job.get('location', 'Non spécifié'),
                'remote_ok': job.get('remote_ok', False),
                'experience': job.get('experience', 'Non spécifié'),
                'description': job.get('description', ''),
                'requirements': job.get('requirements', []),
                'url': job.get('url', ''),
                'source': job.get('source', 'local'),
                'is_scraped': job.get('is_scraped', False),
                'employment_type': job.get('employment_type'),
                'salary_min': job.get('salary_min'),
                'salary_max': job.get('salary_max'),
                'score': detailed_score['score'],
                'skills_details': detailed_score['skills_details'],
                'matching_skills': matching_skills,
                'missing_skills': missing_skills,
                'ml_label': ml_result.get('ml_label', 'N/A'),
                'ml_score': ml_result.get('ml_score'),
                'ml_probabilities': ml_result.get('ml_probabilities'),
                'ml_available': ml_result.get('ml_available', False),
            })

        # ── 6. Trier + filtrer ───────────────────────────────────────────
        detailed_results.sort(key=lambda x: x['score'], reverse=True)
        filtered = [j for j in detailed_results if j['score'] >= min_score][:top_n]

        # ── 7. Formater réponse ──────────────────────────────────────────
        recommendations = []
        for job in filtered:
            recommendations.append({
                "job_id": job['job_id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "remote": job['remote_ok'],
                "experience_required": job['experience'],
                "score": float(job['score']),
                "skills_match": float(job['score']),
                "experience_match": 0,
                "location_match": 0,
                "competition_factor": 0,
                "matching_skills": job['matching_skills'],
                "missing_skills": job['missing_skills'],
                "url": job.get('url', ''),
                "source": job.get('source', 'local'),
                "is_scraped": job.get('is_scraped', False),
                "employment_type": job.get('employment_type'),
                "salary_min": float(job['salary_min']) if job.get('salary_min') else None,
                "salary_max": float(job['salary_max']) if job.get('salary_max') else None,
                "ml_label": job['ml_label'],
                "ml_score": job['ml_score'],
                "ml_probabilities": job['ml_probabilities'],
                "ml_available": job['ml_available'],
            })

        # ── 8. Sauvegarder en DB ─────────────────────────────────────────
        try:
            db = get_db_manager()
            cv_id = db.save_cv_analysis(
                cv_filename=file.filename, cv_text=cv_text,
                technical_skills=technical_skills, soft_skills=soft_skills,
                user_id=1
            )
            for job in filtered:
                db.save_job_recommendation(
                    cv_analysis_id=cv_id, job_id=job['job_id'],
                    job_title=job['title'], company=job['company'],
                    score=float(job['score']),
                    coverage=float(job.get('skills_details', {}).get('coverage', 0)),
                    quality=float(job.get('skills_details', {}).get('quality', 0)),
                    matching_skills=job.get('matching_skills', []),
                    missing_skills=job.get('missing_skills', [])
                )
        except Exception as e:
            logger.warning(f"DB save error: {e}")
            cv_id = None

        return {
            "recommendations": recommendations,
            "total_jobs_analyzed": len(candidate_jobs),
            "cv_skills_count": len(cv_skills),
            "local_jobs_count": local_count,
            "scraped_jobs_count": scraped_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur recommandations: {str(e)}"
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


# ============================================================================
# SIMULATE INTERVIEW (local + scrapés)
# ============================================================================

@app.post("/api/v1/simulate-interview",
          response_model=InterviewResponse,
          tags=["Interview"])
async def simulate_interview(request: InterviewRequest):
    """Génère des questions d'entretien — supporte jobs locaux ET scrapés."""
    try:
        job = None

        # 1) Dataset local
        dataset = get_jobs_dataset()
        job = next((j for j in dataset['jobs'] if j['job_id'] == request.job_id), None)

        # 2) DB scrapés (avec ou sans préfixe sc_)
        if job is None:
            try:
                db = get_db_manager()
                raw_id = request.job_id.replace('sc_', '')
                db.cursor.execute(
                    "SELECT * FROM scraped_jobs WHERE job_id = %s OR job_id = %s LIMIT 1",
                    (request.job_id, raw_id)
                )
                row = db.cursor.fetchone()
                if row:
                    row = dict(row)
                    requirements = _parse_skills_field(row.get('required_skills'))
                    job = {
                        "job_id": row.get('job_id'),
                        "title": row.get('title', ''),
                        "description": row.get('description', ''),
                        "requirements": requirements,
                    }
            except Exception as e:
                logger.warning(f"DB lookup error: {e}")
                job = None  # ← IMPORTANT : s'assurer que job reste None si DB échoue

        # ← CORRECTION : vérification APRÈS les deux lookups
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Offre '{request.job_id}' introuvable"
            )

        simulator = get_interview_simulator()
        questions = simulator.generate_questions(
            cv_skills=request.cv_skills,
            job_title=job.get('title', ''),
            job_description=job.get('description', ''),
            job_requirements=job.get('requirements', []),
            num_questions=request.num_questions
        )
        return {
            "job_title": job.get('title', ''),
            "rh_questions": questions['rh_questions'],
            "technical_questions": questions['technical_questions'],
            "total_questions": len(questions['rh_questions']) + len(questions['technical_questions'])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur simulation : {str(e)}")


# ============================================================================
# AUTRES ENDPOINTS (inchangés)
# ============================================================================

@app.post("/api/v1/evaluate-answer",
          response_model=AnswerEvaluationResponse,
          tags=["Interview"])
async def evaluate_answer(request: AnswerEvaluationRequest):
    try:
        if not request.answer or len(request.answer.strip()) < 10:
            raise HTTPException(status_code=400, detail="Réponse trop courte")
        simulator = get_interview_simulator()
        evaluation = simulator.evaluate_answer(
            question=request.question, answer=request.answer,
            question_type=request.question_type, target_skill=request.target_skill
        )
        return evaluation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur évaluation : {str(e)}")


@app.get("/api/v1/jobs", response_model=List[JobDetail], tags=["Jobs"])
async def list_jobs(
    category: Optional[str] = None,
    remote: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200)
):
    try:
        dataset = get_jobs_dataset()
        local_jobs = dataset['jobs']
        db_scraped = _get_scraped_jobs_from_db()

        formatted = []
        for j in local_jobs:
            formatted.append({
                "job_id": j['job_id'], "title": j['title'],
                "company": j['company'], "location": j['location'],
                "remote": j.get('remote_ok', False),
                "experience_required": j['experience'],
                "category": j.get('category', 'Local'),
                "description": j['description'],
                "skills_required": j.get('requirements', [])
            })
        for j in db_scraped:
            formatted.append({
                "job_id": j['job_id'], "title": j['title'],
                "company": j['company'], "location": j['location'],
                "remote": j.get('remote_ok', False),
                "experience_required": j.get('experience', 'Non spécifié'),
                "category": "Réel (JSearch)",
                "description": j.get('description', ''),
                "skills_required": j.get('requirements', [])
            })

        if category:
            formatted = [j for j in formatted if j['category'] == category]
        if remote is not None:
            formatted = [j for j in formatted if j['remote'] == remote]

        return formatted[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs/{job_id}", response_model=JobDetail, tags=["Jobs"])
async def get_job(job_id: str):
    try:
        dataset = get_jobs_dataset()
        job = next((j for j in dataset['jobs'] if j['job_id'] == job_id), None)
        if job:
            return {
                "job_id": job['job_id'], "title": job['title'],
                "company": job['company'], "location": job['location'],
                "remote": job.get('remote_ok', False),
                "experience_required": job['experience'],
                "category": job.get('category', 'Local'),
                "description": job['description'],
                "skills_required": job.get('requirements', [])
            }
        # Chercher dans scrapés
        raw_id = job_id.replace('sc_', '')
        try:                                   # ← AJOUT : try/except autour de la DB
            db = get_db_manager()
            db.cursor.execute(
                "SELECT * FROM scraped_jobs WHERE job_id = %s OR job_id = %s LIMIT 1",
                (job_id, raw_id)
            )
            row = db.cursor.fetchone()
            if row:
                row = dict(row)
                return {
                    "job_id": row['job_id'], "title": row['title'],
                    "company": row['company'], "location": row['location'],
                    "remote": bool(row.get('is_remote')),
                    "experience_required": "Non spécifié",
                    "category": "Réel (JSearch)",
                    "description": row.get('description', ''),
                    "skills_required": _parse_skills_field(row.get('required_skills'))
                }
        except Exception:
            pass                               # ← Si DB down, on tombe sur le 404

        raise HTTPException(status_code=404, detail=f"Job '{job_id}' introuvable")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    try:
        dataset = get_jobs_dataset()
        jobs = dataset['jobs']
        extractor = get_skills_extractor()
        categories = {}
        remote_count = 0
        for job in jobs:
            cat = job.get('category', 'Non spécifié')
            categories[cat] = categories.get(cat, 0) + 1
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scrape-jobs", tags=["Jobs"])
async def scrape_jobs(
    query: str = "Data Scientist",
    location: str = "France",
    num_pages: int = 1,
    remote_only: bool = False,
    date_posted: str = "month"
):
    """Scraper manuel via JSearch."""
    try:
        scraper = get_job_scraper()
        jobs = scraper.search_jobs(
            query=query, location=location,
            num_pages=num_pages, remote_only=remote_only,
            date_posted=date_posted
        )
        return {"status": "success", "query": query,
                "location": location, "total_found": len(jobs), "jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scraped-jobs", tags=["Jobs"])
async def get_scraped_jobs(
    limit: int = Query(50, ge=1, le=200),
    source: Optional[str] = None
):
    try:
        db = get_db_manager()
        q = "SELECT * FROM scraped_jobs WHERE 1=1"
        params = []
        if source:
            q += " AND source = %s"
            params.append(source)
        q += " ORDER BY scraped_at DESC LIMIT %s"
        params.append(limit)
        db.cursor.execute(q, params)
        rows = db.cursor.fetchall()
        result = []
        for row in rows:
            row = dict(row)
            result.append({
                "id": row.get('id'),
                "job_id": row.get('job_id'),
                "title": row.get('title'),
                "company": row.get('company'),
                "location": row.get('location'),
                "description": row.get('description'),
                "url": row.get('url'),
                "source": row.get('source'),
                "employment_type": row.get('employment_type'),
                "is_remote": row.get('is_remote'),
                "salary_min": float(row['salary_min']) if row.get('salary_min') else None,
                "salary_max": float(row['salary_max']) if row.get('salary_max') else None,
                "required_skills": _parse_skills_field(row.get('required_skills')),
                "scraped_at": row['scraped_at'].isoformat() if row.get('scraped_at') else None,
            })
        return {"total": len(result), "jobs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml-predict", response_model=MLPredictResponse, tags=["ML"])
async def ml_predict(request: MLPredictRequest):
    try:
        ml_predictor = get_ml_predictor()
        if not ml_predictor.is_loaded:
            raise HTTPException(status_code=503, detail="Modèle ML non disponible")
        dataset = get_jobs_dataset()
        job = next((j for j in dataset['jobs'] if j['job_id'] == request.job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail=f"Offre '{request.job_id}' introuvable")
        cv_skills = request.cv_technical_skills + request.cv_soft_skills
        if not cv_skills:
            raise HTTPException(status_code=400, detail="Au moins une compétence requise")
        matcher = get_job_matcher()
        detailed_score = matcher.calculate_job_match_score(cv_skills, job)
        all_job_skills = matcher.extract_job_skills(job)
        known_tech = set(s.lower() for s in matcher.skills_db.get('technical_skills', []))
        known_soft = set(s.lower() for s in matcher.skills_db.get('soft_skills', []))
        job_tech = [s for s in all_job_skills if s.lower() in known_tech]
        job_soft = [s for s in all_job_skills if s.lower() in known_soft]
        categorized = set(job_tech + job_soft)
        job_tech += [s for s in all_job_skills if s not in categorized]
        job_raw = " ".join([job.get('title', ''), job.get('description', ''),
                            " ".join(job.get('requirements', [])),
                            " ".join(job.get('nice_to_have', []))]).strip()
        ml_feat = ml_predictor.compute_features(
            cv_technical_skills=request.cv_technical_skills,
            cv_soft_skills=request.cv_soft_skills,
            job_technical_skills=job_tech, job_soft_skills=job_soft,
            skills_details=detailed_score['skills_details'],
            cv_raw_text=" ".join(cv_skills),
            job_raw_text=job_raw, sentence_model=matcher.model
        )
        ml_result = ml_predictor.predict(ml_feat)
        return {
            "job_id": request.job_id, "job_title": job['title'],
            "ml_label": ml_result['ml_label'], "ml_score": ml_result['ml_score'],
            "ml_probabilities": ml_result['ml_probabilities'],
            "features_used": ml_feat,
            "model_info": {"model_type": "XGBoost", "accuracy": "70%",
                           "classes": ["No Fit", "Partial Fit", "Perfect Fit"], "nb_features": 15}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur ML : {str(e)}")


@app.get("/api/v1/faiss-stats", tags=["Stats"])
async def get_faiss_stats():
    try:
        vs = get_vector_store()
        stats = vs.get_stats()
        return {"faiss_enabled": stats['indexed'], "total_jobs_indexed": stats['total_jobs'],
                "model_used": stats['model_name'], "embedding_dimension": stats['dimension']}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404,
                        content={"detail": "Endpoint non trouvé. Consultez /docs"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500,
                        content={"detail": "Erreur interne. Consultez les logs."})


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("🚀 DÉMARRAGE DE L'API AI Career Coach")
    print("=" * 60)
    if not JOBS_DATASET_PATH.exists():
        print(f"⚠️  Dataset manquant : {JOBS_DATASET_PATH}")
    print("\n⏳ Pré-chargement FAISS...")
    try:
        _ = get_vector_store()
        print("✅ FAISS OK")
    except Exception as e:
        print(f"⚠️  FAISS : {e}")
    print("\n⏳ Pré-chargement XGBoost...")
    try:
        ml = get_ml_predictor()
        print("✅ XGBoost OK" if ml.is_loaded else "⚠️  XGBoost non disponible")
    except Exception as e:
        print(f"⚠️  XGBoost : {e}")
    # Vérifier combien de jobs scrapés sont en DB
    try:
        scraped = _get_scraped_jobs_from_db()
        print(f"\n📊 Jobs scrapés en DB : {len(scraped)}")
    except Exception:
        pass
    print("\n✅ API prête — http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n🛑 Arrêt de l'API")
