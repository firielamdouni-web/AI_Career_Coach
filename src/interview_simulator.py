"""
🎤 Simulateur d'Entretien IA
Génération de questions personnalisées et évaluation des réponses avec Groq (Llama 3.1 70B)
"""

import os
from typing import List, Dict, Optional
from groq import Groq
import json
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()


class InterviewSimulator:
    """
    Simulateur d'entretien basé sur Groq (Llama 3.1 70B Versatile)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialiser le simulateur

        Args:
            api_key: Clé API Groq (ou None pour utiliser variable d'environnement)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "❌ Clé API Groq manquante.\n"
                "Définissez GROQ_API_KEY dans vos variables d'environnement ou passez api_key en paramètre.\n"
                "Obtenez une clé gratuite sur : https://console.groq.com/")

        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

        print(f"✅ InterviewSimulator initialisé avec {self.model}")

    def generate_questions(
        self,
        cv_skills: List[str],
        job_title: str,
        job_description: str,
        job_requirements: List[str],
        num_questions: int = 8
    ) -> Dict:
        """
        Générer des questions d'entretien personnalisées

        Args:
            cv_skills: Compétences extraites du CV
            job_title: Titre du poste
            job_description: Description du poste
            job_requirements: Compétences requises
            num_questions: Nombre de questions (défaut: 8)

        Returns:
            Dict avec questions RH et techniques
        """
        # Limiter les compétences pour le prompt (éviter overflow)
        cv_skills_str = ', '.join(cv_skills[:20])
        requirements_str = ', '.join(job_requirements[:15])

        # Construire le prompt
        prompt = f"""Tu es un recruteur technique expérimenté. Tu dois préparer un entretien pour un candidat junior.

**PROFIL CANDIDAT:**
Compétences: {cv_skills_str}

**POSTE CIBLÉ:**
Titre: {job_title}
Description: {job_description[:300]}...
Compétences requises: {requirements_str}

**CONSIGNES:**
Génère exactement {num_questions} questions d'entretien (50% RH + 50% techniques) au format JSON strict.

Format JSON attendu:
{{
  "rh_questions": [
    {{"id": 1, "question": "Parlez-moi de vous et de votre parcours.", "type": "présentation"}},
    {{"id": 2, "question": "Pourquoi ce poste vous intéresse ?", "type": "motivation"}},
    {{"id": 3, "question": "Décrivez une situation où vous avez travaillé en équipe.", "type": "soft_skills"}},
    {{"id": 4, "question": "Parlez-moi d'un projet dont vous êtes fier.", "type": "projet"}}
  ],
  "technical_questions": [
    {{"id": 5, "question": "Comment utilisez-vous Python dans vos projets ?", "type": "compétence_technique", "skill": "Python"}},
    {{"id": 6, "question": "Expliquez comment vous structureriez une API REST.", "type": "architecture", "skill": "API"}},
    {{"id": 7, "question": "Décrivez votre expérience avec Docker.", "type": "outil", "skill": "Docker"}},
    {{"id": 8, "question": "Comment débugguez-vous un bug en production ?", "type": "résolution_problème", "skill": "Debugging"}}
  ]
}}

**RÈGLES IMPORTANTES:**
- Questions RH adaptées à un profil junior (0-2 ans d'expérience)
- Questions techniques basées sur les compétences du CV ET du poste
- Pas de questions system design complexe pour un junior
- Chaque question technique doit mentionner une compétence spécifique dans "skill"
- Retourne UNIQUEMENT le JSON, aucun texte avant ou après
- Le JSON doit être valide (pas de virgule finale, guillemets corrects)
"""

        try:
            # Appel API Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant RH expert. Tu réponds UNIQUEMENT en JSON valide, sans markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9
            )

            # Extraire le JSON
            content = response.choices[0].message.content.strip()

            # Nettoyer le JSON (enlever markdown si présent)
            if content.startswith("```json"):
                content = content.replace(
                    "```json", "").replace(
                    "```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            # Parser le JSON
            questions = json.loads(content)

            # Validation
            if "rh_questions" not in questions or "technical_questions" not in questions:
                raise ValueError(
                    "Format JSON invalide : clés 'rh_questions' ou 'technical_questions' manquantes")

            if not isinstance(
                    questions['rh_questions'],
                    list) or not isinstance(
                    questions['technical_questions'],
                    list):
                raise ValueError(
                    "Format JSON invalide : 'rh_questions' et 'technical_questions' doivent être des listes")

            print(f"✅ {len(questions['rh_questions'])} questions RH générées")
            print(
                f"✅ {len(questions['technical_questions'])} questions techniques générées")

            return questions

        except json.JSONDecodeError as e:
            print(f"❌ Erreur parsing JSON : {e}")
            print(f"Réponse brute (300 premiers caractères) : {content[:300]}")
            raise ValueError(f"Le LLM n'a pas retourné un JSON valide : {e}")
        except Exception as e:
            print(f"❌ Erreur génération questions : {e}")
            raise

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        question_type: str,
        target_skill: Optional[str] = None
    ) -> Dict:
        """
        Évaluer la réponse d'un candidat

        Args:
            question: Question posée
            answer: Réponse du candidat
            question_type: Type de question (présentation, motivation, technique, etc.)
            target_skill: Compétence ciblée (pour questions techniques)

        Returns:
            Dict avec score, feedback et points forts/faibles
        """
        # Construire le contexte
        skill_context = f"\nCompétence évaluée: {target_skill}" if target_skill else ""

        # Prompt d'évaluation
        prompt = f"""Tu es un évaluateur d'entretien technique bienveillant mais rigoureux. Évalue cette réponse.

**QUESTION:**
{question}

**TYPE DE QUESTION:** {question_type}{skill_context}

**RÉPONSE DU CANDIDAT:**
{answer}

**CONSIGNES:**
Évalue la réponse sur une échelle de 0 à 100 et donne un feedback constructif.

Format JSON attendu:
{{
  "score": 75,
  "evaluation": "Réponse claire et structurée. Le candidat démontre une bonne compréhension...",
  "points_forts": [
    "Exemples concrets mentionnés",
    "Bonne structure de réponse",
    "Démontre l'autonomie"
  ],
  "points_amelioration": [
    "Pourrait ajouter des métriques chiffrées",
    "Manque de détails techniques"
  ],
  "recommandations": [
    "Préparer 2-3 exemples avec résultats mesurables",
    "Approfondir les aspects techniques"
  ]
}}

**CRITÈRES D'ÉVALUATION:**
- Clarté et structure (0-20 points)
- Pertinence et exemples concrets (0-30 points)
- Profondeur technique si applicable (0-30 points)
- Communication et soft skills (0-20 points)

**RÈGLES:**
- Sois bienveillant mais honnête
- Pour un profil junior, valorise l'apprentissage et la motivation
- Un score < 50 = réponse insuffisante
- Un score 50-70 = correct mais à améliorer
- Un score 70-85 = bonne réponse
- Un score > 85 = excellente réponse
- Retourne UNIQUEMENT le JSON, sans markdown
"""

        try:
            # Appel API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un évaluateur RH expert. Tu réponds UNIQUEMENT en JSON valide."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Plus déterministe pour l'évaluation
                max_tokens=1200
            )

            content = response.choices[0].message.content.strip()

            # Nettoyer JSON
            if content.startswith("```json"):
                content = content.replace(
                    "```json", "").replace(
                    "```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            evaluation = json.loads(content)

            # Validation
            required_keys = [
                "score",
                "evaluation",
                "points_forts",
                "points_amelioration"]
            missing_keys = [
                key for key in required_keys if key not in evaluation]
            if missing_keys:
                raise ValueError(
                    f"Clés manquantes dans le JSON : {missing_keys}")

            # Normaliser le score (0-100)
            evaluation["score"] = max(0, min(100, float(evaluation["score"])))

            # S'assurer que les listes sont bien des listes
            if not isinstance(evaluation["points_forts"], list):
                evaluation["points_forts"] = [str(evaluation["points_forts"])]
            if not isinstance(evaluation["points_amelioration"], list):
                evaluation["points_amelioration"] = [
                    str(evaluation["points_amelioration"])]

            # Ajouter recommandations si manquantes
            if "recommandations" not in evaluation:
                evaluation["recommandations"] = [
                    "Continuer à pratiquer", "Préparer des exemples concrets"]

            print(f"✅ Réponse évaluée : {evaluation['score']:.0f}/100")

            return evaluation

        except json.JSONDecodeError as e:
            print(f"❌ Erreur parsing JSON : {e}")
            print(f"Réponse brute : {content[:300]}")
            raise ValueError(f"Le LLM n'a pas retourné un JSON valide : {e}")
        except Exception as e:
            print(f"❌ Erreur évaluation : {e}")
            raise

    def generate_final_feedback(
        self,
        evaluations: List[Dict],
        job_title: str
    ) -> Dict:
        """
        Générer un feedback global sur la simulation

        Args:
            evaluations: Liste des évaluations individuelles
            job_title: Titre du poste

        Returns:
            Dict avec feedback global et recommandations
        """
        if not evaluations:
            return {
                "global_score": 0,
                "decision": "Aucune évaluation",
                "synthese": "Aucune réponse n'a été évaluée.",
                "competences_validees": [],
                "axes_progression": [],
                "prochaines_etapes": ["Commencer la simulation d'entretien"]
            }

        # Calculer statistiques
        scores = [eval_data["score"] for eval_data in evaluations]
        avg_score = sum(scores) / len(scores)

        # Agréger points forts/faibles
        all_strengths = []
        all_improvements = []

        for eval_data in evaluations:
            all_strengths.extend(eval_data.get("points_forts", []))
            all_improvements.extend(eval_data.get("points_amelioration", []))

        # Prompt pour feedback global
        prompt = f"""Tu es un recruteur senior. Génère un feedback global sur cette simulation d'entretien.

**POSTE CIBLÉ:** {job_title}

**RÉSULTATS DE LA SIMULATION:**
- Score moyen: {avg_score:.1f}/100
- Nombre de questions: {len(evaluations)}
- Scores individuels: {', '.join(f'{s:.0f}' for s in scores)}

**POINTS FORTS IDENTIFIÉS:**
{chr(10).join(f"- {s}" for s in list(set(all_strengths))[:8])}

**POINTS D'AMÉLIORATION:**
{chr(10).join(f"- {i}" for i in list(set(all_improvements))[:8])}

**CONSIGNES:**
Génère un feedback global bienveillant mais constructif.

Format JSON attendu:
{{
  "global_score": {avg_score:.1f},
  "decision": "À retravailler",
  "synthese": "En 2-3 phrases, résume la performance globale du candidat.",
  "competences_validees": ["Compétence 1", "Compétence 2", "..."],
  "axes_progression": ["Axe 1", "Axe 2", "..."],
  "prochaines_etapes": ["Étape 1", "Étape 2", "..."]
}}

**RÈGLES:**
- decision: "À retravailler" si score < 50, "Prometteur" si 50-75, "Excellent" si > 75
- synthese: 2-3 phrases sur la performance globale
- competences_validees: 3-5 points forts principaux
- axes_progression: 3-5 axes d'amélioration concrets
- prochaines_etapes: 3-4 actions concrètes à faire
- Ton bienveillant et encourageant
- Retourne UNIQUEMENT le JSON
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un coach carrière expert. Réponds en JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content.replace(
                    "```json", "").replace(
                    "```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            feedback = json.loads(content)

            print(f"✅ Feedback global généré")

            return feedback

        except Exception as e:
            print(f"⚠️  Erreur génération feedback global : {e}")
            # Fallback avec feedback basique
            if avg_score < 50:
                decision = "À retravailler"
            elif avg_score < 75:
                decision = "Prometteur"
            else:
                decision = "Excellent"

            return {
                "global_score": round(avg_score, 1),
                "decision": decision,
                "synthese": f"Vous avez obtenu un score moyen de {avg_score:.1f}/100 sur {len(evaluations)} questions. {'Continuez à vous entraîner.' if avg_score < 70 else 'Bonne performance globale !'}",
                "competences_validees": list(set(all_strengths[:5])),
                "axes_progression": list(set(all_improvements[:5])),
                "prochaines_etapes": [
                    "Pratiquer les questions d'entretien régulièrement",
                    "Préparer 3-5 exemples concrets de projets",
                    "Travailler la structure STAR (Situation, Tâche, Action, Résultat)"
                ]
            }


# ============================================================================
# SINGLETON POUR API
# ============================================================================

_interview_simulator = None


def get_interview_simulator() -> InterviewSimulator:
    """Obtenir le simulateur (singleton)"""
    global _interview_simulator
    if _interview_simulator is None:
        _interview_simulator = InterviewSimulator()
    return _interview_simulator
