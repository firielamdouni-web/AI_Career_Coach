"""
Job Scraper - JSearch API (RapidAPI)
Agrège les offres LinkedIn, Indeed, Glassdoor légalement
"""
import os
import logging
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com"


class JobScraper:
    """
    Scraper d'offres d'emploi via JSearch API (RapidAPI)
    Agrège : LinkedIn, Indeed, Glassdoor
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JSEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("JSEARCH_API_KEY manquant — définir dans .env")

        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        logger.info("✅ JobScraper initialisé")

    def search_jobs(
        self,
        query: str,
        location: str = "France",
        num_pages: int = 1,
        employment_types: Optional[str] = None,
        remote_only: bool = False,
        date_posted: str = "month"
    ) -> List[Dict]:
        """
        Chercher des offres d'emploi via JSearch

        Args:
            query           : ex. "Data Scientist", "Python Developer"
            location        : ex. "Paris", "France", "Remote"
            num_pages       : nombre de pages (10 résultats/page)
            employment_types: "FULLTIME", "PARTTIME", "INTERN", "CONTRACTOR"
            remote_only     : filtrer uniquement remote
            date_posted     : "all", "today", "3days", "week", "month"

        Returns:
            Liste de jobs normalisés
        """
        params = {
            "query": f"{query} {location}",
            "page": "1",
            "num_pages": str(num_pages),
            "date_posted": date_posted,
            "language": "fr"
        }

        if employment_types:
            params["employment_types"] = employment_types
        if remote_only:
            params["remote_jobs_only"] = "true"

        try:
            logger.info(f"🔍 Recherche JSearch : '{query}' à '{location}'")
            response = requests.get(
                f"{JSEARCH_BASE_URL}/search",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            raw_jobs = data.get("data", [])
            logger.info(f"📥 {len(raw_jobs)} offres reçues de JSearch")

            return [self._normalize_job(job) for job in raw_jobs]

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error("❌ Quota JSearch dépassé (200 req/mois sur tier gratuit)")
            else:
                logger.error(f"❌ Erreur HTTP JSearch : {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur réseau JSearch : {e}")
            raise

    def _normalize_job(self, raw: Dict) -> Dict:
        """Normaliser un job JSearch vers notre format interne"""
        return {
            "job_id":           raw.get("job_id", ""),
            "title":            raw.get("job_title", ""),
            "company":          raw.get("employer_name", ""),
            "location":         self._extract_location(raw),
            "description":      raw.get("job_description", ""),
            "url":              raw.get("job_apply_link", ""),
            "source":           raw.get("job_publisher", "").lower(),
            "employment_type":  raw.get("job_employment_type", ""),
            "is_remote":        raw.get("job_is_remote", False),
            "salary_min":       raw.get("job_min_salary"),
            "salary_max":       raw.get("job_max_salary"),
            "required_skills":  self._extract_skills_from_description(
                                    raw.get("job_description", "")
                                ),
            "scraped_at":       datetime.now().isoformat()
        }

    def _extract_location(self, raw: Dict) -> str:
        """Construire la localisation depuis les champs JSearch"""
        city    = raw.get("job_city", "")
        country = raw.get("job_country", "")
        state   = raw.get("job_state", "")

        parts = [p for p in [city, state, country] if p]
        return ", ".join(parts) if parts else "Non précisé"

    def _extract_skills_from_description(self, description: str) -> List[str]:
        """Extraction basique de skills depuis la description"""
        if not description:
            return []

        COMMON_SKILLS = [
            "python", "java", "javascript", "typescript", "react", "node.js",
            "sql", "postgresql", "mysql", "mongodb", "redis",
            "docker", "kubernetes", "aws", "azure", "gcp",
            "machine learning", "deep learning", "tensorflow", "pytorch",
            "fastapi", "django", "flask", "spring",
            "git", "ci/cd", "agile", "scrum"
        ]

        description_lower = description.lower()
        return [skill for skill in COMMON_SKILLS if skill in description_lower]

    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """Récupérer les détails d'un job spécifique"""
        try:
            response = requests.get(
                f"{JSEARCH_BASE_URL}/job-details",
                headers=self.headers,
                params={"job_id": job_id, "extended_publisher_details": "false"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            jobs = data.get("data", [])
            return self._normalize_job(jobs[0]) if jobs else None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur get_job_details : {e}")
            return None
        
# les méthodes de sauvegarde dans PostgreSQL et FAISS

    def search_and_save(
        self,
        query: str,
        location: str = "France",
        num_pages: int = 1,
        save_to_db: bool = True,
        save_to_faiss: bool = True
    ) -> Dict:
        """
        Scraper + sauvegarder en PostgreSQL + indexer dans FAISS
        Retourne un résumé des opérations
        """
        # ── 1. Scraper ──────────────────────────────────────────────────────
        jobs = self.search_jobs(query=query, location=location, num_pages=num_pages)

        saved_db    = 0
        saved_faiss = 0
        errors      = []

        # ── 2. Sauvegarder dans PostgreSQL ──────────────────────────────────
        if save_to_db and jobs:
            try:
                from src.database import get_db_manager
                db = get_db_manager()
                saved_db = self._save_jobs_to_db(db, jobs)
                logger.info(f"💾 {saved_db} jobs sauvegardés en DB")
            except Exception as e:
                logger.error(f"❌ Erreur sauvegarde DB : {e}")
                errors.append(f"DB: {str(e)}")

        # ── 3. Indexer dans FAISS ────────────────────────────────────────────
        if save_to_faiss and jobs:
            try:
                from src.vector_store import get_vector_store
                vs = get_vector_store()
                saved_faiss = self._save_jobs_to_faiss(vs, jobs)
                logger.info(f"🔍 {saved_faiss} jobs indexés dans FAISS")
            except Exception as e:
                logger.error(f"❌ Erreur indexation FAISS : {e}")
                errors.append(f"FAISS: {str(e)}")

        return {
            "total_scraped": len(jobs),
            "saved_to_db":   saved_db,
            "saved_to_faiss": saved_faiss,
            "errors":        errors,
            "jobs":          jobs
        }
        
    def _save_jobs_to_db(self, db, jobs: List[Dict]) -> int:
        """Insérer les jobs dans PostgreSQL (ignore les doublons via job_id)"""
        saved = 0
        query = """
            INSERT INTO scraped_jobs
                (job_id, title, company, location, description, url,
                 source, employment_type, is_remote,
                 salary_min, salary_max, required_skills, scraped_at)
            VALUES
                (%(job_id)s, %(title)s, %(company)s, %(location)s,
                 %(description)s, %(url)s, %(source)s, %(employment_type)s,
                 %(is_remote)s, %(salary_min)s, %(salary_max)s,
                 %(required_skills)s, %(scraped_at)s)
            ON CONFLICT (job_id) DO NOTHING
        """
        import json
        for job in jobs:
            try:
                job_data = job.copy()
                job_data["required_skills"] = json.dumps(job.get("required_skills", []))
                db.cursor.execute(query, job_data)
                db.conn.commit()
                saved += 1
            except Exception as e:
                logger.warning(f"⚠️ Job ignoré ({job.get('job_id')}) : {e}")
                db.conn.rollback()
        return saved
    
    def _save_jobs_to_faiss(self, vs, jobs: List[Dict]) -> int:
        """Indexer les descriptions de jobs dans FAISS"""
        saved = 0
        for job in jobs:
            try:
                if job.get("description"):
                    # Si ton vector_store a une méthode add_job
                    vs.add_job(job)  # ou vs.add_jobs([job])
                    saved += 1
            except Exception as e:
                logger.warning(f"⚠️ Job non indexé ({job.get('job_id')}) : {e}")
        return saved
    

# ─────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────
_scraper_instance: Optional[JobScraper] = None


def get_job_scraper() -> JobScraper:
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = JobScraper()
    return _scraper_instance