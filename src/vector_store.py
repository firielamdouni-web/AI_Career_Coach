"""
Base vectorielle FAISS pour recherche rapide d'offres d'emploi
Indexation et recherche sémantique avec Sentence-Transformers
"""

import json
import logging
from typing import Optional
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
from pathlib import Path


class JobVectorStore:
    """
    Gère l'indexation et la recherche d'offres dans FAISS
    Utilise Sentence-Transformers pour générer les embeddings
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialiser le vector store

        Args:
            model_name: Nom du modèle Sentence-Transformers (défaut: all-mpnet-base-v2)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.jobs_metadata = []

        print(
            f" JobVectorStore initialisé avec {model_name} ({self.dimension} dimensions)")

    def build_index(self, jobs: List[Dict], index_type: str = 'flat') -> None:
        """
        Construire l'index FAISS à partir d'une liste d'offres

        Args:
            jobs: Liste d'offres (chaque offre = dict avec title, description, requirements, etc.)
            index_type: Type d'index FAISS ('flat' pour exact search, 'ivf' pour approximate)
        """
        if not jobs:
            raise ValueError("La liste d'offres est vide")

        print(f"\n Construction de l'index FAISS...")
        print(f"   Nombre d'offres : {len(jobs)}")
        print(f"   Type d'index : {index_type}")

        job_texts = []
        for job in jobs:
            text = f"{job['title']} {job['description']}"

            if 'requirements' in job and job['requirements']:
                text += " " + " ".join(job['requirements'])

            if 'nice_to_have' in job and job['nice_to_have']:
                text += " " + " ".join(job['nice_to_have'])

            job_texts.append(text)

        print(f"   Génération des embeddings...")
        embeddings = self.model.encode(
            job_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True 
        )

        if index_type == 'flat':

            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == 'ivf':
            nlist = min(100, len(jobs) // 10) 
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings.astype('float32'))
        else:
            raise ValueError(f"Type d'index non supporté : {index_type}")

        self.index.add(embeddings.astype('float32'))

        self.jobs_metadata = jobs

        print(f" Index construit avec succès !")
        print(f"   Total d'offres indexées : {self.index.ntotal}")

    def search(
        self,
        cv_skills: List[str],
        top_k: int = 10,
        cv_text: str = None
    ) -> List[Tuple[Dict, float]]:
        """
        Rechercher les offres les plus similaires à un profil

        Args:
            cv_skills: Liste de compétences extraites du CV
            top_k: Nombre de résultats à retourner
            cv_text: Texte complet du CV (optionnel, pour enrichir la recherche)

        Returns:
            Liste de tuples (job, similarity_score) triée par score décroissant
        """
        if self.index is None:
            raise ValueError(
                "L'index FAISS n'est pas construit. Appelez build_index() d'abord.")

        if not cv_skills:
            raise ValueError("La liste de compétences est vide")

        query_text = " ".join(cv_skills)
        if cv_text:
            query_text += " " + cv_text[:500] 
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(
            query_embedding.astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            job = self.jobs_metadata[idx]
            similarity_score = float(scores[0][i])

            similarity_percentage = similarity_score * 100

            results.append((job, similarity_percentage))

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Sauvegarder l'index FAISS et les metadata sur disque

        Args:
            index_path: Chemin pour sauvegarder l'index FAISS (.index)
            metadata_path: Chemin pour sauvegarder les metadata (.pkl)
        """
        if self.index is None:
            raise ValueError(
                "Aucun index à sauvegarder. Appelez build_index() d'abord.")

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'jobs_metadata': self.jobs_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }, f)

        print(f" Index sauvegardé : {index_path}")
        print(f" Metadata sauvegardées : {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Charger l'index FAISS et les metadata depuis le disque

        Args:
            index_path: Chemin de l'index FAISS (.index)
            metadata_path: Chemin des metadata (.pkl)
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")

        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata introuvables : {metadata_path}")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.jobs_metadata = data['jobs_metadata']

            if data['model_name'] != self.model_name:
                print(f"  Attention : modèle différent")
                print(f"   Index créé avec : {data['model_name']}")
                print(f"   Modèle actuel : {self.model_name}")

        print(f" Index chargé : {self.index.ntotal} offres")
        print(f" Metadata chargées : {len(self.jobs_metadata)} offres")

    def get_stats(self) -> Dict:
        """
        Obtenir des statistiques sur l'index

        Returns:
            Dictionnaire avec les statistiques
        """
        if self.index is None:
            return {
                'indexed': False,
                'total_jobs': 0,
                'model_name': self.model_name,
                'dimension': self.dimension
            }

        return {
            'indexed': True,
            'total_jobs': self.index.ntotal,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'jobs_with_metadata': len(self.jobs_metadata)
        }

    def add_job(self, job: Dict) -> bool:
        """
        Ajouter un job à l'index FAISS

        Args:
            job: Dictionnaire contenant les informations du job

        Returns:
            True si ajouté, False sinon
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            if not job.get("description"):
                return False

            embedding = self.model.encode([job["description"]])[0]

            self.index.add(embedding.reshape(1, -1))

            self.jobs_metadata.append(job)

            logger.info(f" Job ajouté à FAISS : {job.get('title', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f" Erreur ajout job à FAISS : {e}")
            return False


# ─────────────────────────────────────────
# Singleton pour VectorStore
# ─────────────────────────────────────────

_vector_store_instance: Optional[JobVectorStore] = None


def get_vector_store() -> JobVectorStore:
    """Retourne l'instance singleton du JobVectorStore"""
    global _vector_store_instance
    if _vector_store_instance is None:
        from src.job_matcher import JOB_DATA_PATH

        logger = logging.getLogger(__name__)
        _vector_store_instance = JobVectorStore()

        try:
            with open(JOB_DATA_PATH, 'r') as f:
                jobs_data = json.load(f)
                jobs = jobs_data.get("jobs", [])
                if jobs:
                    _vector_store_instance.build_index(jobs)
                    logger.info(f"{len(jobs)} jobs chargés dans FAISS")
        except Exception as e:
            logger.warning(f" Impossible de charger les jobs dans FAISS: {e}")

    return _vector_store_instance
