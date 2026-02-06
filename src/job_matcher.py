"""
Module de matching sémantique CV ↔ Offres d'emploi
Utilise Sentence-Transformers pour la similarité sémantique
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# ✅ Initialisation du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobMatcher:
    """
    Classe pour matcher un CV avec des offres d'emploi
    """
    
    def __init__(self):
        """Initialize JobMatcher with sentence transformer model"""
        logger.info("Initialisation du JobMatcher...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("✅ JobMatcher initialisé avec all-mpnet-base-v2")
        
    def calculate_skills_similarity(
        self, 
        cv_skills: List[str], 
        job_description: str
    ) -> Dict:
        """
        Calculer la similarité entre compétences CV et description d'offre
        SCORES BRUTS (sans round) pour précision maximale du tri
        """
        if not cv_skills or not job_description:
            return {
                'overall_score': 0,
                'high_matches': 0,
                'total_skills': 0,
                'matches': []
            }
        
        job_sentences = [
            s.strip() 
            for s in job_description.split('\n') 
            if s.strip() and len(s.strip()) > 10
        ]
        
        if not job_sentences:
            job_sentences = [job_description]
        
        try:
            job_embeddings = self.model.encode(job_sentences, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Erreur encodage phrases : {e}")
            return {
                'overall_score': 0,
                'high_matches': 0,
                'total_skills': len(cv_skills),
                'matches': []
            }
        
        matches = []
        
        for skill in cv_skills:
            try:
                skill_embedding = self.model.encode([skill.lower()], show_progress_bar=False)
                similarities = cosine_similarity(skill_embedding, job_embeddings)[0]
                
                # ✅ SCORE BRUT (pas de round)
                max_similarity = max(similarities) * 100
                
                best_match_idx = similarities.argmax()
                best_sentence = job_sentences[best_match_idx]
                
                matches.append({
                    'skill': skill,
                    'similarity': max_similarity,  # ← PAS de round()
                    'match': 'high' if max_similarity >= 40 else 'medium' if max_similarity >= 30 else 'low',
                    'matched_sentence': best_sentence[:60] + '...' if len(best_sentence) > 60 else best_sentence
                })
            except Exception as e:
                logger.warning(f"Erreur skill '{skill}' : {e}")
                continue
        
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        
        # ✅ SCORE GLOBAL BRUT (pas de round)
        if matches:
            avg_similarity = sum(m['similarity'] for m in matches) / len(matches)
            high_matches = len([m for m in matches if m['match'] == 'high'])
        else:
            avg_similarity = 0
            high_matches = 0
        
        return {
            'overall_score': avg_similarity,  # ← PAS de round()
            'high_matches': high_matches,
            'total_skills': len(cv_skills),
            'matches': matches
        }
    
    def calculate_job_match_score(
        self, 
        cv_skills: List[str], 
        job: Dict
    ) -> Dict:
        """
        Calculer un score de matching entre CV et offre
        SCORE BRUT stocké, arrondi uniquement à l'affichage
        """
        job_requirements = job.get('requirements', [])
        job_description = job.get('description', '')
        job_text = ' '.join(job_requirements) + '\n' + job_description
        
        skills_result = self.calculate_skills_similarity(cv_skills, job_text)
        skills_score = skills_result['overall_score']  # ← Score brut
        
        return {
            'job_id': job.get('job_id', 'unknown'),
            'title': job.get('title', 'N/A'),
            'company': job.get('company', 'N/A'),
            'location': job.get('location', 'N/A'),
            'type': job.get('type', 'CDI'),
            'experience': job.get('experience', 'N/A'),
            'salary': job.get('salary', 'N/A'),
            'remote_ok': job.get('remote_ok', False),
            'applicants': int(job.get('applicants', 50)),
            'url': job.get('url', ''),
            
            # ✅ SCORE BRUT (pas de round ici, sera fait à l'affichage)
            'score': float(skills_score),  # ← PAS de round()
            
            'skills_details': {
                'high_matches': skills_result['high_matches'],
                'top_skills': [m['skill'] for m in skills_result['matches'][:5]]
            },
            
            'requirements': job.get('requirements', []),
            'nice_to_have': job.get('nice_to_have', []),
            'description_preview': job.get('description', '')[:200] + '...'
        }
    
    
    def rank_jobs(
        self, 
        cv_skills: List[str], 
        jobs: List[Dict]
    ) -> List[Dict]:
        """
        Classer toutes les offres par score de compétences décroissant
        
        Args:
            cv_skills: Liste de compétences du CV
            jobs: Liste d'offres d'emploi
            
        Returns:
            Liste d'offres triées par score
        """
        if not cv_skills or not jobs:
            logger.warning("⚠️  CV skills ou jobs vides")
            return []
        
        import time
        start_time = time.time()
        
        logger.info(f"🎯 Ranking {len(jobs)} offres avec {len(cv_skills)} compétences")
        logger.info("📊 Scoring basé uniquement sur les compétences")
        
        recommendations = []
        
        for i, job in enumerate(jobs):
            try:
                score_data = self.calculate_job_match_score(cv_skills, job)
                recommendations.append(score_data)
            except Exception as e:
                logger.error(f"Erreur job {job.get('job_id', 'unknown')} : {e}")
                continue
            
            if (i + 1) % 10 == 0:
                logger.info(f"  ✅ Traité {i+1}/{len(jobs)} offres...")
        
        recommendations = sorted(
            recommendations, 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Ranking terminé en {elapsed:.2f}s")
        
        if recommendations:
            logger.info(f"📊 Meilleur score: {recommendations[0]['score']:.1f}% | "
                       f"Pire score: {recommendations[-1]['score']:.1f}%")
        
        return recommendations


# Fonction utilitaire pour charger le matcher
def load_matcher(model_name: str = 'all-mpnet-base-v2') -> JobMatcher:
    """
    Charger un JobMatcher pré-configuré
    
    Args:
        model_name: Nom du modèle Sentence-Transformer
        
    Returns:
        Instance de JobMatcher
    """
    return JobMatcher(model_name)