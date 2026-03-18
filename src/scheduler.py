"""
Scheduler automatique - Scrape JSearch 2x/jour
Lance ce fichier en parallèle de l'API :
    python -m src.scheduler
"""

from src.database import get_db_manager
from src.job_scraper import get_job_scraper
import schedule
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import requests 

sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SCHEDULER] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_QUERIES = [
    {"query": "Data Scientist", "location": "France", "num_pages": 1},
    {"query": "Machine Learning Engineer", "location": "France", "num_pages": 1},
    {"query": "Data Engineer", "location": "France", "num_pages": 1},
    {"query": "MLOps Engineer", "location": "France", "num_pages": 1},
    {"query": "AI Engineer", "location": "France", "num_pages": 1},
]


def _parse_skills(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if s]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [s.strip() for s in raw.split(',') if s.strip()]
    return []


def _get_fresh_db():
    """
    Retourne une connexion DB valide.
    Vérifie que conn est ouvert, sinon reconnecte via _connect().
    """
    db = get_db_manager()
    try:
        if db.conn is None or db.conn.closed:
            logger.info(" Reconnexion DB...")
            db._connect()
        else:
            db.cursor.execute("SELECT 1")
    except Exception:
        logger.info(" Reconnexion DB après erreur...")
        try:
            db._connect()
        except Exception as e:
            raise RuntimeError(f"Impossible de se connecter à la DB : {e}")
    return db


def run_daily_scrape():
    """Scrape toutes les requêtes et sauvegarde les nouveaux jobs en DB."""
    logger.info("=" * 50)
    logger.info(f" Démarrage scraping — {datetime.now():%Y-%m-%d %H:%M}")
    logger.info("=" * 50)

    scraper = get_job_scraper()
    total_new = 0
    total_found = 0

    for q in SCRAPE_QUERIES:
        logger.info(f" Scraping : '{q['query']}' @ {q['location']}")
        try:
            jobs = scraper.search_jobs(
                query=q['query'],
                location=q['location'],
                num_pages=q['num_pages'],
                remote_only=False,
                date_posted="week"
            )
            total_found += len(jobs)
            logger.info(f"   → {len(jobs)} offres trouvées")

            db = _get_fresh_db()
            saved = 0

            for job in jobs:
                try:
                    job_id = str(job.get('job_id') or job.get('id') or '')
                    if not job_id:
                        continue

                    db.cursor.execute(
                        "SELECT 1 FROM scraped_jobs WHERE job_id = %s LIMIT 1",
                        (job_id,)
                    )
                    if db.cursor.fetchone():
                        continue 

                    skills = _parse_skills(
                        job.get('required_skills') or job.get('requirements', [])
                    )

                    db.cursor.execute("""
                        INSERT INTO scraped_jobs (
                            job_id, title, company, location, description,
                            url, source, employment_type, is_remote,
                            salary_min, salary_max, required_skills, scraped_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                    """, (
                        job_id,
                        job.get('title', ''),
                        job.get('company', ''),
                        job.get('location', ''),
                        job.get('description', ''),
                        job.get('url', ''),
                        job.get('source', 'jsearch'),
                        job.get('employment_type', ''),
                        bool(job.get('remote_ok') or job.get('is_remote')),
                        job.get('salary_min'),
                        job.get('salary_max'),
                        json.dumps(skills),
                    ))
                    saved += 1

                except Exception as e:
                    logger.warning(f" Insert error ({job_id}): {e}")
                    try:
                        db.conn.rollback()
                    except Exception:
                        pass

            db.conn.commit()
            total_new += saved
            logger.info(f" {saved} nouveaux jobs sauvegardés")

        except Exception as e:
            logger.error(f" Scraping échoué pour '{q['query']}': {e}")

    try:
        cleanup_db = _get_fresh_db()
        cleanup_db.clean_old_scraped_jobs(days_to_keep=30)
    except Exception as e:
        logger.error(f"Impossible de nettoyer la DB : {e}")

    logger.info("-" * 50)
    logger.info(f"Terminé : {total_new} nouveaux / {total_found} trouvés")
    logger.info("=" * 50)

    logger.info("Déclenchement de la synchronisation DB -> FAISS sur l'API...")
    try:
        response = requests.post("http://api:8000/api/v1/sync-jobs")
        if response.status_code == 200:
            logger.info(f"API synchronisée avec succès: {response.json().get('jobs_loaded')} offres chargées.")
        else:
            logger.error(f"Erreur de synchronisation API: {response.text}")
    except Exception as e:
        logger.error(f"Impossible de contacter l'API pour la synchronisation: {e}")


def main():
    logger.info("Scheduler démarré — scraping 1x/jour (plan Free JSearch)")
    logger.info("Horaire : 08:00")

    run_daily_scrape()

    schedule.every().day.at("08:00").do(run_daily_scrape)

    while True:
        next_run = schedule.next_run()
        logger.info(f"Prochain scraping : {next_run:%Y-%m-%d %H:%M}")
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
