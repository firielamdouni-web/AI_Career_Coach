"""
Tests unitaires - JobVectorStore (FAISS)
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import pickle
import tempfile
import os


MOCK_JOBS = [
    {
        "job_id": "job_001",
        "title": "Data Scientist",
        "company": "TechCorp",
        "description": "Python and ML required.",
        "requirements": ["Python", "Machine Learning", "SQL"],
        "nice_to_have": ["Docker"]
    },
    {
        "job_id": "job_002",
        "title": "ML Engineer",
        "company": "StartupAI",
        "description": "Deploy ML models.",
        "requirements": ["Python", "Docker", "TensorFlow"],
        "nice_to_have": ["Kubernetes"]
    },
    {
        "job_id": "job_003",
        "title": "Data Engineer",
        "company": "DataCo",
        "description": "Build data pipelines.",
        "requirements": ["Python", "Spark", "SQL"],
        "nice_to_have": ["Airflow"]
    }
]

CV_SKILLS = ["Python", "Machine Learning", "SQL"]


@pytest.fixture
def vector_store():
    """Fixture JobVectorStore avec SentenceTransformer mocké"""
    with patch('src.vector_store.SentenceTransformer') as mock_st:  
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 64

        def mock_encode(texts, **kwargs):
            np.random.seed(42)
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.random.rand(n, 64).astype('float32')
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock_model.encode.side_effect = mock_encode
        mock_st.return_value = mock_model

        from src.vector_store import JobVectorStore
        vs = JobVectorStore(model_name='all-mpnet-base-v2')
        return vs


class TestJobVectorStoreInit:

    def test_init_sets_model_name(self, vector_store):
        assert vector_store.model_name == 'all-mpnet-base-v2'

    def test_init_index_is_none(self, vector_store):
        assert vector_store.index is None

    def test_init_metadata_is_empty(self, vector_store):
        assert vector_store.jobs_metadata == []

    def test_init_dimension_set(self, vector_store):
        assert vector_store.dimension == 64


class TestBuildIndex:

    def test_build_index_creates_index(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        assert vector_store.index is not None

    def test_build_index_correct_count(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        assert vector_store.index.ntotal == len(MOCK_JOBS)

    def test_build_index_saves_metadata(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        assert len(vector_store.jobs_metadata) == len(MOCK_JOBS)

    def test_build_index_empty_jobs_raises(self, vector_store):
        with pytest.raises(ValueError, match="vide"):
            vector_store.build_index([])

    def test_build_index_invalid_type_raises(self, vector_store):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            vector_store.build_index("not a list")


class TestSearch:

    def test_search_returns_list(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(CV_SKILLS, top_k=2)
        assert isinstance(results, list)

    def test_search_returns_tuples(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(CV_SKILLS, top_k=2)
        for job, score in results:
            assert isinstance(job, dict)
            assert isinstance(score, float)

    def test_search_top_k_respected(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(CV_SKILLS, top_k=2)
        assert len(results) <= 2

    def test_search_top_k_exceeds_total(self, vector_store):
        """top_k > nombre d'offres → retourne tout"""
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(CV_SKILLS, top_k=100)
        assert len(results) <= len(MOCK_JOBS)

    def test_search_without_index_raises(self, vector_store):
        with pytest.raises(ValueError, match="index"):
            vector_store.search(CV_SKILLS, top_k=2)

    def test_search_empty_skills_raises(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        with pytest.raises(ValueError, match="vide"):
            vector_store.search([], top_k=2)

    def test_search_results_contain_job_id(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(CV_SKILLS, top_k=3)
        for job, _ in results:
            assert "job_id" in job

    def test_search_with_cv_text(self, vector_store):
        """Recherche avec cv_text optionnel doit fonctionner"""
        vector_store.build_index(MOCK_JOBS)
        results = vector_store.search(
            CV_SKILLS, top_k=2,
            cv_text="Experienced Python developer with ML skills"
        )
        assert len(results) > 0


class TestGetStats:

    def test_stats_not_indexed(self, vector_store):
        stats = vector_store.get_stats()
        assert stats['indexed'] is False
        assert stats['total_jobs'] == 0

    def test_stats_indexed(self, vector_store):
        vector_store.build_index(MOCK_JOBS)
        stats = vector_store.get_stats()
        assert stats['indexed'] is True
        assert stats['total_jobs'] == len(MOCK_JOBS)

    def test_stats_contains_model_name(self, vector_store):
        stats = vector_store.get_stats()
        assert stats['model_name'] == 'all-mpnet-base-v2'

    def test_stats_contains_dimension(self, vector_store):
        stats = vector_store.get_stats()
        assert stats['dimension'] == 64


class TestSaveLoad:

    def test_save_without_index_raises(self, vector_store):
        with pytest.raises(ValueError, match="index"):
            vector_store.save("/tmp/test.index", "/tmp/test.pkl")

    def test_save_and_load(self, vector_store):
        """Sauvegarder puis recharger l'index"""
        vector_store.build_index(MOCK_JOBS)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            meta_path = os.path.join(tmpdir, "test.pkl")

            vector_store.save(index_path, meta_path)

            assert os.path.exists(index_path)
            assert os.path.exists(meta_path)

            # Recréer et charger
            with patch('sentence_transformers.SentenceTransformer') as mock_st2:
                mock_model2 = MagicMock()
                mock_model2.get_sentence_embedding_dimension.return_value = 64
                mock_st2.return_value = mock_model2

                from src.vector_store import JobVectorStore
                vs2 = JobVectorStore()
                vs2.load(index_path, meta_path)

                assert vs2.index.ntotal == len(MOCK_JOBS)
                assert len(vs2.jobs_metadata) == len(MOCK_JOBS)

    def test_load_missing_index_raises(self, vector_store):
        with pytest.raises(FileNotFoundError):
            vector_store.load("/nonexistent/path.index", "/nonexistent/meta.pkl")

    def test_load_missing_metadata_raises(self, vector_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            # Créer uniquement l'index, pas les metadata
            vector_store.build_index(MOCK_JOBS)
            import faiss
            faiss.write_index(vector_store.index, index_path)

            with pytest.raises(FileNotFoundError):
                vector_store.load(index_path, "/nonexistent/meta.pkl")