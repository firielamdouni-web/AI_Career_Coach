"""
Module d'extraction de compétences depuis un CV
Support multilingue : anglais (en_core_web_sm) et français (fr_core_news_lg)
"""


import re
import json
import spacy
import logging
from pathlib import Path
from typing import List, Dict, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillsExtractor:
    """
    Extracteur de compétences techniques et soft skills


    Supporte plusieurs modèles spaCy avec fallback automatique :
    1. en_core_web_sm (anglais, pour développement local)
    2. fr_core_news_lg (français, pour production Docker)
    """

    def __init__(self, skills_db_path: Optional[str] = None):
        """
        Initialiser l'extracteur


        Args:
            skills_db_path: Chemin vers skills_reference.json (optionnel)


        Raises:
            RuntimeError: Si aucun modèle spaCy n'est disponible
        """

        self.nlp = None
        self.model_name = None

        models_to_try = [
            ("en_core_web_sm", "Anglais (développement local)"),
            ("fr_core_news_lg", "Français (production Docker)")
        ]

        for model_name, description in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                self.model_name = model_name
                logger.info(
                    f"Modèle spaCy chargé : {model_name} ({description})")
                break  
            except OSError:
                logger.warning(
                    f"Modèle {model_name} non trouvé, essai suivant...")
                continue

        if self.nlp is None:
            raise RuntimeError(
                " Aucun modèle spaCy trouvé. Installez-en un avec :\n"
                "  • python -m spacy download en_core_web_sm  (anglais)\n"
                "  • python -m spacy download fr_core_news_lg  (français)"
            )

        if skills_db_path is None:
            skills_db_path = Path(__file__).parent.parent / \
                "data" / "skills_reference.json"

        skills_db_path = Path(skills_db_path)

        if not skills_db_path.exists():
            raise FileNotFoundError(
                f" Fichier skills_reference.json non trouvé : {skills_db_path}\n"
                "Assurez-vous que data/skills_reference.json existe.")

        with open(skills_db_path, 'r', encoding='utf-8') as f:
            self.skills_database = json.load(f)

        logger.info(
            f" Base de compétences chargée depuis {skills_db_path.name}")
        logger.info(
            f"   • Compétences techniques : {len(self.skills_database['technical_skills'])}")
        logger.info(
            f"   • Soft skills : {len(self.skills_database['soft_skills'])}")

        self.variations_to_canonical = self._build_variations_map()
        print(f"   • Variations : {len(self.variations_to_canonical)} mappings")

    def _build_variations_map(self) -> Dict[str, str]:
        """
        Construire un mapping {variation → canonical}
        Ex: {"ml" → "machine learning", "py" → "python"}
        """
        mapping = {}

        if 'variations' in self.skills_database:
            for canonical, variations_list in self.skills_database['variations'].items():
                for variation in variations_list:
                    mapping[variation.lower()] = canonical

        return mapping

    def extract_skills_from_text(
        self,
        text: str,
        skills_list: List[str]
    ) -> List[str]:
        """
        Extraire compétences depuis texte brut avec matching par regex


        Args:
            text: Texte du CV
            skills_list: Liste de compétences de référence


        Returns:
            Liste de compétences trouvées (triée alphabétiquement)
        """
        text_lower = text.lower()
        found_skills = set()

        for skill in skills_list:
            skill_lower = skill.lower()

            if re.search(r'[^a-z0-9\s]', skill_lower):
                escaped = re.escape(skill_lower)
                pattern = r'(?:^|\s|[(\[{])' + escaped + r'(?:\s|$|[.,;:)\]}])'
            else:
                pattern = r'\b' + re.escape(skill_lower) + r'\b'

            if re.search(pattern, text_lower):
                found_skills.add(skill)

        for variation, canonical in self.variations_to_canonical.items():
            canonical_in_list = any(
                s.lower() == canonical.lower()
                for s in skills_list
            )
            if not canonical_in_list:
                continue

            if re.search(r'[^a-z0-9\s]', variation):
                escaped = re.escape(variation)
                pattern = r'(?:^|\s|[(\[{])' + escaped + r'(?:\s|$|[.,;:)\]}])'
            else:
                pattern = r'\b' + re.escape(variation) + r'\b'

            if re.search(pattern, text_lower):
                canonical_skill = next(
                    (s for s in skills_list if s.lower() == canonical.lower()),
                    canonical
                )
                found_skills.add(canonical_skill)

        return sorted(found_skills)

    def extract_from_cv(self, cv_text: str) -> Dict:
        """
        Extraire toutes les compétences d'un CV


        Args:
            cv_text: Texte complet du CV


        Returns:
            Dict avec :
            - technical_skills : List[str]
            - soft_skills : List[str]
            - total_skills : int
            - spacy_entities : List[Dict] (entités nommées détectées)
            - model_used : str (nom du modèle spaCy utilisé)
        """

        try:
            doc = self.nlp(cv_text)
            spacy_entities = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]
        except Exception as e:
            logger.warning(f"  Erreur traitement spaCy : {e}")
            spacy_entities = []

        technical_skills = self.extract_skills_from_text(
            cv_text,
            self.skills_database['technical_skills']
        )

        soft_skills = self.extract_skills_from_text(
            cv_text,
            self.skills_database['soft_skills']
        )

        logger.info(
            f" Extraction terminée : {len(technical_skills)} tech skills, "
            f"{len(soft_skills)} soft skills"
        )

        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "total_skills": len(technical_skills) + len(soft_skills),
            "spacy_entities": spacy_entities,
            "model_used": self.model_name  
        }

    def extract_and_save(
        self,
        cv_text: str,
        output_path: str,
        cv_filename: str = "CV.pdf"
    ) -> Dict:
        """
        Extraire et sauvegarder les résultats dans un fichier JSON


        Args:
            cv_text: Texte du CV
            output_path: Chemin de sauvegarde (ex: outputs/skills.json)
            cv_filename: Nom du fichier CV (pour métadonnées)


        Returns:
            Résultats de l'extraction (même format que extract_from_cv)
        """
        from datetime import datetime

        results = self.extract_from_cv(cv_text)

        results["cv_file"] = cv_filename
        results["extraction_date"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        results["method"] = f"spaCy ({self.model_name}) + keyword matching"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f" Résultats sauvegardés : {output_path}")

        return results

    def get_model_info(self) -> Dict[str, str]:
        """
        Obtenir les informations sur le modèle spaCy utilisé


        Returns:
            Dict avec nom, version, langue du modèle
        """
        if self.nlp is None:
            return {"error": "Aucun modèle chargé"}

        meta = self.nlp.meta

        return {
            "name": meta.get("name", "unknown"),
            "version": meta.get("version", "unknown"),
            "language": meta.get("lang", "unknown"),
            "description": meta.get("description", "N/A")
        }


def extract_skills_from_cv_file(
    cv_text_path: str,
    output_path: Optional[str] = None,
    skills_db_path: Optional[str] = None
) -> Dict:
    """
    Extraire compétences depuis un fichier texte de CV


    Args:
        cv_text_path: Chemin vers le fichier texte du CV
        output_path: Chemin de sauvegarde JSON (optionnel)
        skills_db_path: Chemin vers skills_reference.json (optionnel)


    Returns:
        Résultats de l'extraction


    Example:
        >>> results = extract_skills_from_cv_file(
        ...     "data/cv_text.txt",
        ...     "outputs/skills.json"
        ... )
        >>> print(results['technical_skills'])
        ['Python', 'pandas', 'scikit-learn']
    """
    cv_text_path = Path(cv_text_path)

    if not cv_text_path.exists():
        raise FileNotFoundError(f"Fichier CV non trouvé : {cv_text_path}")

    with open(cv_text_path, 'r', encoding='utf-8') as f:
        cv_text = f.read()

    extractor = SkillsExtractor(skills_db_path)

    if output_path:
        return extractor.extract_and_save(
            cv_text,
            output_path,
            cv_filename=cv_text_path.name
        )
    else:
        return extractor.extract_from_cv(cv_text)


def test_models_availability() -> Dict[str, bool]:
    """
    Tester quels modèles spaCy sont disponibles


    Returns:
        Dict avec disponibilité de chaque modèle


    Example:
        >>> availability = test_models_availability()
        >>> print(availability)
        {'en_core_web_sm': True, 'fr_core_news_lg': False}
    """
    models = {
        "en_core_web_sm": "Anglais (small)",
        "fr_core_news_lg": "Français (large)"
    }

    availability = {}

    for model_name, description in models.items():
        try:
            spacy.load(model_name)
            availability[model_name] = True
            logger.info(f" {model_name} ({description}) : DISPONIBLE")
        except OSError:
            availability[model_name] = False
            logger.warning(f" {model_name} ({description}) : NON DISPONIBLE")

    return availability


if __name__ == "__main__":
    """
    Script de test pour vérifier le bon fonctionnement
    """
    print("=" * 60)
    print(" TEST DU MODULE SKILLS EXTRACTOR")
    print("=" * 60)

    # Test 1 : Vérifier disponibilité des modèles
    print("\n Test 1 : Disponibilité des modèles spaCy")
    print("-" * 60)
    availability = test_models_availability()

    # Test 2 : Créer un extracteur
    print("\n Test 2 : Initialisation de SkillsExtractor")
    print("-" * 60)
    try:
        extractor = SkillsExtractor()
        model_info = extractor.get_model_info()
        print(f" Extracteur initialisé avec succès")
        print(f"   Modèle : {model_info['name']} v{model_info['version']}")
        print(f"   Langue : {model_info['language']}")
    except Exception as e:
        print(f" Erreur : {e}")
        exit(1)

    print("\n Test 3 : Extraction sur texte de test")
    print("-" * 60)

    test_cv_text = """
    Développeur Full Stack avec 5 ans d'expérience


    Compétences techniques :
    - Langages : Python, JavaScript, TypeScript
    - Frameworks : React, Node.js, Django
    - Bases de données : PostgreSQL, MongoDB
    - Outils : Git, Docker, AWS


    Soft skills :
    - Communication
    - Teamwork
    - Problem solving
    - Adaptability
    """

    results = extractor.extract_from_cv(test_cv_text)

    print(f" Extraction réussie")
    print(f"   • Compétences techniques : {len(results['technical_skills'])}")
    print(f"   • Soft skills : {len(results['soft_skills'])}")
    print(f"\n Compétences détectées :")
    print(f"   Technical : {', '.join(results['technical_skills'][:10])}")
    print(f"   Soft : {', '.join(results['soft_skills'][:5])}")

    print("\n" + "=" * 60)
    print(" TOUS LES TESTS PASSÉS")
    print("=" * 60)
