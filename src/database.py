"""
🗄️ Database Manager - Gestion PostgreSQL pour AI Career Coach
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire de connexion PostgreSQL (Singleton Pattern)"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Établir la connexion"""
        try:
            self.conn = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor
            )
            self.cursor = self.conn.cursor()
            logger.info("✅ PostgreSQL connecté")
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur connexion PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}")
    
    def _ensure_connection(self):
        """Vérifier et rétablir la connexion si nécessaire"""
        try:
            if self.conn is None or self.conn.closed:
                self._connect()
            else:
                self.cursor.execute("SELECT 1")
        except psycopg2.Error:
            self._connect()
    
    # ========================================================================
    # MÉTHODES POUR CV ANALYSES
    # ========================================================================
    
    def save_cv_analysis(
        self,
        cv_filename: str,
        cv_text: str,
        technical_skills: List[str],
        soft_skills: List[str],
        user_id: int = 1  # ✅ Par défaut anonymous (id=1)
    ) -> int:
        """Sauvegarder une analyse de CV"""
        self._ensure_connection()
        
        try:
            total_skills = len(technical_skills) + len(soft_skills)
            
            self.cursor.execute("""
                INSERT INTO cv_analyses 
                (user_id, cv_filename, cv_text, technical_skills, soft_skills, total_skills)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                user_id,
                cv_filename,
                cv_text,  # ✅ Pas de limitation
                technical_skills,
                soft_skills,
                total_skills
            ))
            
            cv_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(f"✅ CV sauvegardé (ID: {cv_id})")
            return cv_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde CV: {e}")
            raise
    
    def get_recent_cv_analyses(self, limit: int = 10) -> List[Dict]:
        """Récupérer les N analyses récentes"""
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT id, cv_filename, total_skills, analyzed_at
                FROM cv_analyses
                ORDER BY analyzed_at DESC
                LIMIT %s;
            """, (limit,))
            
            return [dict(row) for row in self.cursor.fetchall()]
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture CSV analyses: {e}")
            return []
    
    # ========================================================================
    # MÉTHODES POUR JOB RECOMMENDATIONS
    # ========================================================================
    
    def save_job_recommendation(
        self,
        cv_analysis_id: int,
        job_id: str,
        job_title: str,
        company: str,
        score: float,
        coverage: float,
        quality: float,
        matching_skills: List[str],
        missing_skills: List[str]
    ) -> int:
        """Sauvegarder une recommandation"""
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                INSERT INTO job_recommendations 
                (cv_analysis_id, job_id, job_title, company, score, 
                 coverage, quality, matching_skills, missing_skills)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                cv_analysis_id, job_id, job_title, company,
                round(score, 2), round(coverage, 2), round(quality, 2),
                matching_skills, missing_skills
            ))
            
            rec_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(f"✅ Recommandation sauvegardée (ID: {rec_id})")
            return rec_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde recommandation: {e}")
            raise
    
    def get_recommendations_for_cv(self, cv_analysis_id: int) -> List[Dict]:
        """Récupérer toutes les recommandations d'une analyse"""
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT * FROM job_recommendations
                WHERE cv_analysis_id = %s
                ORDER BY score DESC;
            """, (cv_analysis_id,))
            
            return [dict(row) for row in self.cursor.fetchall()]
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture recommandations: {e}")
            return []
    
    # ========================================================================
    # MÉTHODES POUR INTERVIEW SIMULATIONS
    # ========================================================================
    
    def save_interview_simulation(
        self,
        cv_analysis_id: int,
        job_id: str,
        rh_questions: List[Dict],
        technical_questions: List[Dict],
        answers: List[Dict] = None,
        scores: List[Dict] = None,
        average_score: float = 0.0
    ) -> int:
        """Sauvegarder une simulation d'entretien"""
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                INSERT INTO interview_simulations 
                (cv_analysis_id, job_id, rh_questions, technical_questions, 
                 answers, scores, average_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                cv_analysis_id, job_id,
                json.dumps(rh_questions),
                json.dumps(technical_questions),
                json.dumps(answers or []),
                json.dumps(scores or []),
                round(average_score, 2)
            ))
            
            sim_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(f"✅ Simulation sauvegardée (ID: {sim_id})")
            return sim_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde simulation: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Récupérer les statistiques globales"""
        self._ensure_connection()
        
        try:
            stats = {}
            
            self.cursor.execute("SELECT COUNT(*) as count FROM cv_analyses;")
            stats['total_cv_analyses'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM job_recommendations;")
            stats['total_recommendations'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM interview_simulations;")
            stats['total_simulations'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT AVG(score) as avg_score FROM job_recommendations;")
            result = self.cursor.fetchone()['avg_score']
            stats['average_score'] = round(float(result), 2) if result else 0.0
            
            logger.info(f"📊 Stats: {stats}")
            return stats
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur stats: {e}")
            return {}
    
    def disconnect(self):
        """Fermer la connexion"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✅ PostgreSQL déconnecté")


# ============================================================================
# SINGLETON GLOBAL
# ============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Obtenir l'instance singleton du DatabaseManager"""
    global _db_manager
    
    if _db_manager is None:
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            raise ValueError(
                "❌ DATABASE_URL not found. Check .env file."
            )
        
        _db_manager = DatabaseManager(database_url)
    
    return _db_manager


def close_db_connection():
    """Fermer proprement la connexion"""
    global _db_manager
    if _db_manager:
        _db_manager.disconnect()
        _db_manager = None