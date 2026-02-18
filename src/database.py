"""
🗄️ Database Manager - Gestion PostgreSQL pour AI Career Coach

Ce module gère toutes les interactions avec PostgreSQL :
- Connexion via singleton pattern (évite les connexions multiples)
- Sauvegarde des analyses CV
- Sauvegarde des recommandations
- Sauvegarde des simulations d'entretien
- Lecture de l'historique
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Gestionnaire de connexion PostgreSQL (Singleton Pattern)
    
    Utilise le pattern Singleton pour éviter de créer plusieurs connexions.
    La connexion est persistante pendant toute la durée de vie de l'API.
    """
    
    def __init__(self, database_url: str):
        """
        Initialiser le gestionnaire de BDD
        
        Args:
            database_url: URL de connexion PostgreSQL (from .env)
        """
        self.database_url = database_url
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Établir la connexion à PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor  # Retourne des dicts au lieu de tuples
            )
            self.cursor = self.conn.cursor()
            logger.info("✅ Connexion PostgreSQL établie avec succès")
            
            # Test de connexion
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()
            logger.info(f"📊 PostgreSQL version: {version['version'][:50]}...")
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur connexion PostgreSQL: {e}")
            raise ConnectionError(
                f"Impossible de se connecter à PostgreSQL. "
                f"Vérifiez que DATABASE_URL est correct dans .env et que PostgreSQL tourne. "
                f"Détail: {e}"
            )
    
    def disconnect(self):
        """Fermer proprement la connexion"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✅ Connexion PostgreSQL fermée")
    
    def _ensure_connection(self):
        """Vérifier que la connexion est active (reconnexion si nécessaire)"""
        try:
            if self.conn is None or self.conn.closed:
                logger.warning("⚠️ Connexion PostgreSQL fermée, reconnexion...")
                self._connect()
            else:
                # Test ping
                self.cursor.execute("SELECT 1")
        except psycopg2.Error:
            logger.warning("⚠️ Connexion PostgreSQL perdue, reconnexion...")
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
        user_id: Optional[int] = None
    ) -> int:
        """
        Sauvegarder une analyse de CV dans la table cv_analyses
        
        Args:
            cv_filename: Nom du fichier PDF
            cv_text: Texte complet extrait du CV
            technical_skills: Liste des compétences techniques
            soft_skills: Liste des soft skills
            user_id: ID utilisateur (optionnel, pour futur multi-users)
        
        Returns:
            ID de l'analyse créée
        """
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
                cv_text[:5000],  # Limiter à 5000 chars pour éviter les CVs trop longs
                technical_skills,
                soft_skills,
                total_skills
            ))
            
            cv_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(
                f"✅ CV analysis sauvegardé (ID: {cv_id}) - "
                f"{total_skills} compétences ({len(technical_skills)} tech, {len(soft_skills)} soft)"
            )
            
            return cv_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde CV analysis: {e}")
            raise
    
    def get_cv_analysis(self, cv_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupérer une analyse de CV par son ID
        
        Args:
            cv_id: ID de l'analyse
        
        Returns:
            Dict contenant les données de l'analyse ou None
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT * FROM cv_analyses WHERE id = %s;
            """, (cv_id,))
            
            result = self.cursor.fetchone()
            return dict(result) if result else None
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture CV analysis: {e}")
            return None
    
    def get_recent_cv_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupérer les N analyses les plus récentes
        
        Args:
            limit: Nombre d'analyses à retourner
        
        Returns:
            Liste des analyses
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT id, cv_filename, total_skills, analyzed_at
                FROM cv_analyses
                ORDER BY analyzed_at DESC
                LIMIT %s;
            """, (limit,))
            
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture CV analyses: {e}")
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
        skills_match: float,
        experience_match: float,
        location_match: float,
        competition_factor: float,
        matching_skills: List[str]
    ) -> int:
        """
        Sauvegarder une recommandation de job
        
        Args:
            cv_analysis_id: ID de l'analyse CV associée
            job_id: ID du job (ex: "job_001")
            job_title: Titre du poste
            company: Nom de l'entreprise
            score: Score global de matching (0-100)
            skills_match: Score compétences (0-100)
            experience_match: Score expérience (0-100)
            location_match: Score localisation (0-100)
            competition_factor: Facteur de compétition (0-100)
            matching_skills: Liste des compétences matchées
        
        Returns:
            ID de la recommandation créée
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                INSERT INTO job_recommendations 
                (cv_analysis_id, job_id, job_title, company, score, 
                 skills_match, experience_match, location_match, competition_factor, matching_skills)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                cv_analysis_id,
                job_id,
                job_title,
                company,
                round(score, 2),
                round(skills_match, 2),
                round(experience_match, 2),
                round(location_match, 2),
                round(competition_factor, 2),
                matching_skills
            ))
            
            recommendation_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(
                f"✅ Recommandation sauvegardée (ID: {recommendation_id}) - "
                f"Job: {job_title} @ {company} (Score: {score:.1f}%)"
            )
            
            return recommendation_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde recommandation: {e}")
            raise
    
    def get_recommendations_for_cv(self, cv_analysis_id: int) -> List[Dict[str, Any]]:
        """
        Récupérer toutes les recommandations d'une analyse CV
        
        Args:
            cv_analysis_id: ID de l'analyse CV
        
        Returns:
            Liste des recommandations triées par score
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT * FROM job_recommendations
                WHERE cv_analysis_id = %s
                ORDER BY score DESC;
            """, (cv_analysis_id,))
            
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
            
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
        questions: List[Dict[str, Any]],
        answers: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
        average_score: float
    ) -> int:
        """
        Sauvegarder une simulation d'entretien
        
        Args:
            cv_analysis_id: ID de l'analyse CV
            job_id: ID du job concerné
            questions: Liste des questions générées (JSON)
            answers: Liste des réponses du candidat (JSON)
            scores: Liste des scores détaillés (JSON)
            average_score: Score moyen global
        
        Returns:
            ID de la simulation créée
        """
        self._ensure_connection()
        
        try:
            # Convertir les listes en JSONB
            import json
            
            self.cursor.execute("""
                INSERT INTO interview_simulations 
                (cv_analysis_id, job_id, questions, answers, scores, average_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                cv_analysis_id,
                job_id,
                json.dumps(questions),
                json.dumps(answers),
                json.dumps(scores),
                round(average_score, 2)
            ))
            
            simulation_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(
                f"✅ Simulation d'entretien sauvegardée (ID: {simulation_id}) - "
                f"Job: {job_id} (Score moyen: {average_score:.1f}%)"
            )
            
            return simulation_id
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"❌ Erreur sauvegarde simulation: {e}")
            raise
    
    def get_interview_simulation(self, simulation_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupérer une simulation d'entretien par son ID
        
        Args:
            simulation_id: ID de la simulation
        
        Returns:
            Dict contenant les données de la simulation ou None
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("""
                SELECT * FROM interview_simulations WHERE id = %s;
            """, (simulation_id,))
            
            result = self.cursor.fetchone()
            return dict(result) if result else None
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture simulation: {e}")
            return None
    
    # ========================================================================
    # MÉTHODES UTILITAIRES
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Récupérer les statistiques globales de la BDD
        
        Returns:
            Dict avec les compteurs (total_cv_analyses, total_recommendations, etc.)
        """
        self._ensure_connection()
        
        try:
            stats = {}
            
            # Compter les analyses CV
            self.cursor.execute("SELECT COUNT(*) as count FROM cv_analyses;")
            stats['total_cv_analyses'] = self.cursor.fetchone()['count']
            
            # Compter les recommandations
            self.cursor.execute("SELECT COUNT(*) as count FROM job_recommendations;")
            stats['total_recommendations'] = self.cursor.fetchone()['count']
            
            # Compter les simulations d'entretien
            self.cursor.execute("SELECT COUNT(*) as count FROM interview_simulations;")
            stats['total_interview_simulations'] = self.cursor.fetchone()['count']
            
            # Score moyen des recommandations
            self.cursor.execute("SELECT AVG(score) as avg_score FROM job_recommendations;")
            avg_score_result = self.cursor.fetchone()['avg_score']
            stats['average_recommendation_score'] = round(float(avg_score_result), 2) if avg_score_result else 0.0
            
            logger.info(f"📊 Statistiques BDD: {stats}")
            
            return stats
            
        except psycopg2.Error as e:
            logger.error(f"❌ Erreur lecture statistiques: {e}")
            return {}


# ============================================================================
# SINGLETON GLOBAL (Pattern Factory)
# ============================================================================

_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Récupérer l'instance singleton du DatabaseManager
    
    Cette fonction garantit qu'une seule connexion PostgreSQL est créée
    pendant toute la durée de vie de l'API (évite les connexions multiples).
    
    Returns:
        Instance du DatabaseManager
    
    Raises:
        ConnectionError: Si impossible de se connecter à PostgreSQL
    """
    global _db_manager_instance
    
    if _db_manager_instance is None:
        # Charger l'URL depuis les variables d'environnement
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            raise ValueError(
                "❌ DATABASE_URL non trouvé dans les variables d'environnement. "
                "Vérifiez que le fichier .env existe et contient DATABASE_URL."
            )
        
        logger.info("🔄 Initialisation du DatabaseManager (première connexion)...")
        _db_manager_instance = DatabaseManager(database_url)
    
    return _db_manager_instance


def close_db_connection():
    """
    Fermer proprement la connexion PostgreSQL
    
    À appeler lors de l'arrêt de l'API (shutdown event).
    """
    global _db_manager_instance
    
    if _db_manager_instance is not None:
        _db_manager_instance.disconnect()
        _db_manager_instance = None
        logger.info("✅ DatabaseManager fermé")