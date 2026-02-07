"""
Module de matching sémantique CV ↔ Offres d'emploi
Utilise Sentence-Transformers pour la similarité sémantique
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re
from pathlib import Path

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

        # Charger skills_reference.json
        skills_db_path = Path(__file__).parent.parent / "data" / "skills_reference.json"

        if skills_db_path.exists():
            with open(skills_db_path, 'r', encoding='utf-8') as f:
                self.skills_db = json.load(f)
            logger.info(f"✅ skills_reference.json chargé ({len(self.skills_db.get('technical_skills', []))} skills)")
        else:
            logger.warning(f"⚠️ skills_reference.json non trouvé : {skills_db_path}")
            self.skills_db = {'technical_skills': [], 'variations': {}, 'soft_skills': []}
        
        # Construire le mapping de variations
        self.variations_map = self._build_variations_map()
        
        logger.info("✅ JobMatcher initialisé avec all-mpnet-base-v2")

    def _build_variations_map(self) -> dict:
        """
        Construire le mapping bidirectionnel canonical ↔ variations
            
        Returns:
            Dict {forme_canonique: [toutes_variations]}
        """
        variations_map = {}
            
        # Charger depuis la section "variations"
        if 'variations' in self.skills_db:
            for canonical, variations_list in self.skills_db['variations'].items():
                variations_map[canonical] = variations_list
            
        # Ajouter les skills techniques (eux-mêmes = forme canonique)
        if 'technical_skills' in self.skills_db:
            for skill in self.skills_db['technical_skills']:
                skill_lower = skill.lower().strip()
                if skill_lower not in variations_map:
                    variations_map[skill_lower] = [skill_lower]
            
        return variations_map
    
    def _normalize_skill(self, skill: str) -> str:
        """
        Normaliser un skill en utilisant le mapping de variations
        
        Args:
            skill: Compétence à normaliser
            
        Returns:
            Forme canonique du skill
        """
        skill_clean = skill.lower().strip()
        skill_clean = skill_clean.replace('-', ' ').replace('_', ' ')
        skill_clean = ' '.join(skill_clean.split())
        
        # Chercher dans le mapping de variations
        for canonical, variations in self.variations_map.items():
            if skill_clean in variations or skill_clean == canonical:
                return canonical
        
        return skill_clean

    def extract_job_skills(self, job: Dict) -> List[str]:
        """
        Extraire et normaliser les compétences de l'offre avec skills_reference.json
        
        Args:
            job: Dictionnaire de l'offre d'emploi
            
        Returns:
            Liste de compétences extraites et normalisées
        """
        # Créer une liste de tous les skills reconnus
        all_known_skills = set()
        for canonical, variations in self.variations_map.items():
            all_known_skills.add(canonical)
            all_known_skills.update(variations)
        
        skills = []
        
        # 1. Requirements (priorité haute)
        if 'requirements' in job and job['requirements']:
            for req in job['requirements']:
                # Extraire mots entre parenthèses
                match = re.search(r'\((.*?)\)', req)
                if match:
                    keywords = [k.strip() for k in match.group(1).split(',')]
                    skills.extend(keywords)
                
                # Extraire skills connus du texte
                req_lower = req.lower()
                for known_skill in all_known_skills:
                    pattern = r'\b' + re.escape(known_skill) + r'\b'
                    if re.search(pattern, req_lower):
                        skills.append(known_skill)
        
        # 2. Nice-to-have (priorité moyenne)
        if 'nice_to_have' in job and job['nice_to_have']:
            for nice in job['nice_to_have']:
                match = re.search(r'\((.*?)\)', nice)
                if match:
                    keywords = [k.strip() for k in match.group(1).split(',')]
                    skills.extend(keywords)
                
                nice_lower = nice.lower()
                for known_skill in all_known_skills:
                    pattern = r'\b' + re.escape(known_skill) + r'\b'
                    if re.search(pattern, nice_lower):
                        skills.append(known_skill)
        
        # 3. Normaliser avec le mapping
        normalized = []
        seen = set()
        
        for skill in skills:
            if not skill or not skill.strip():
                continue
            
            skill_norm = self._normalize_skill(skill)
            
            if skill_norm not in seen:
                normalized.append(skill_norm)
                seen.add(skill_norm)
        
        logger.info(f"💼 Skills extraits de l'offre : {len(normalized)}")
        if normalized:
            logger.info(f"   Exemples : {', '.join(normalized[:3])}")
        
        return normalized
        
    def calculate_skills_similarity(
        self, 
        cv_skills: List[str], 
        job: Dict
    ) -> Dict:
        """
        APPROCHE 4 : Matching Skills Offre → CV avec cache d'embeddings
        Pour chaque skill requis par l'offre, trouver le meilleur match dans le CV
        
        Args:
            cv_skills: Liste de compétences du CV
            job: Dictionnaire de l'offre d'emploi
            
        Returns:
            Dict avec score, coverage, quality et détails par compétence
        """
        if not cv_skills:
            return {
                'overall_score': 0,
                'coverage': 0,
                'quality': 0,
                'covered_count': 0,
                'total_required': 0,
                'matches': []
            }
        
        # Extraire les skills de l'offre
        job_skills = self.extract_job_skills(job)
        
        if not job_skills:
            logger.warning("⚠️ Aucune compétence trouvée dans l'offre")
            return {
                'overall_score': 0,
                'coverage': 0,
                'quality': 0,
                'covered_count': 0,
                'total_required': 0,
                'matches': []
            }
        
        logger.info(f"🔍 Matching {len(job_skills)} skills offre ↔ {len(cv_skills)} skills CV")
        
        try:
            # ✅ OPTIMISATION : Encoder tous les skills UNE SEULE FOIS (cache)
            cv_embeddings = {
                skill: self.model.encode([skill.lower()], show_progress_bar=False)[0]
                for skill in cv_skills
            }
            
            job_embeddings = {
                skill: self.model.encode([skill.lower()], show_progress_bar=False)[0]
                for skill in job_skills
            }
            
            matches = []
            
            # Pour chaque skill de l'OFFRE
            for job_skill in job_skills:
                job_emb = job_embeddings[job_skill]
                
                best_similarity = 0
                best_cv_skill = None
                
                # Comparer avec CHAQUE skill du CV (réutilise le cache)
                for cv_skill in cv_skills:
                    cv_emb = cv_embeddings[cv_skill]
                    similarity = cosine_similarity([job_emb], [cv_emb])[0][0] * 100
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cv_skill = cv_skill
                
                # Déterminer le statut du match
                if best_similarity >= 40:
                    status = 'covered'
                    match_level = 'high'
                elif best_similarity >= 30:
                    status = 'partial'
                    match_level = 'medium'
                else:
                    status = 'missing'
                    match_level = 'low'
                
                matches.append({
                    'job_skill': job_skill,
                    'cv_skill': best_cv_skill,
                    'similarity': best_similarity,
                    'status': status,
                    'match': match_level
                })
            
            # Trier par similarité décroissante
            matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
            
            # Calculer les métriques
            covered = [m for m in matches if m['status'] == 'covered']
            
            # Coverage : % de skills de l'offre couverts
            coverage = (len(covered) / len(job_skills)) * 100 if job_skills else 0
            
            # Quality : Qualité moyenne des matchs couverts
            quality = sum(m['similarity'] for m in covered) / len(covered) if covered else 0
            
            # Score final : Moyenne pondérée Coverage (50%) + Quality (50%)
            overall_score = (coverage * 0.5) + (quality * 0.5)
            
            logger.info(f"✅ Coverage: {coverage:.1f}% | Quality: {quality:.1f}% | Score: {overall_score:.1f}%")
            
            return {
                'overall_score': overall_score,
                'coverage': coverage,
                'quality': quality,
                'covered_count': len(covered),
                'total_required': len(job_skills),
                'matches': matches
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du matching : {e}")
            return {
                'overall_score': 0,
                'coverage': 0,
                'quality': 0,
                'covered_count': 0,
                'total_required': len(job_skills),
                'matches': []
            }
    
    def calculate_job_match_score(
        self, 
        cv_skills: List[str], 
        job: Dict
    ) -> Dict:
        """
        Calculer un score de matching entre CV et offre (APPROCHE 4)
        
        Args:
            cv_skills: Liste de compétences du CV
            job: Dictionnaire de l'offre d'emploi
            
        Returns:
            Dict avec les informations de l'offre + score détaillé
        """
        # Normaliser les skills du CV
        cv_skills_normalized = []
        seen = set()
        
        for skill in cv_skills:
            skill_norm = self._normalize_skill(skill)
            if skill_norm not in seen:
                cv_skills_normalized.append(skill_norm)
                seen.add(skill_norm)
        
        # Calculer le matching avec l'Approche 4
        skills_result = self.calculate_skills_similarity(cv_skills_normalized, job)
        
        # Extraire les top skills matchés
        top_matched_skills = []
        for match in skills_result.get('matches', []):
            if match['status'] == 'covered':
                top_matched_skills.append({
                    'job_skill': match['job_skill'],
                    'cv_skill': match['cv_skill'],
                    'similarity': match['similarity']
                })
        
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
            
            # Score global (brut, sera arrondi à l'affichage)
            'score': float(skills_result['overall_score']),
            
            # Détails du matching
            'skills_details': {
                'coverage': skills_result['coverage'],          # % skills offre couverts
                'quality': skills_result['quality'],            # Qualité moyenne
                'covered_count': skills_result['covered_count'], # Nombre skills couverts
                'total_required': skills_result['total_required'], # Total skills requis
                'top_matches': top_matched_skills               # Top 5 matchs
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
def load_matcher() -> JobMatcher:
    """
    Charger un JobMatcher pré-configuré
    
    Args:
        model_name: Nom du modèle Sentence-Transformer
        
    Returns:
        Instance de JobMatcher
    """
    return JobMatcher()