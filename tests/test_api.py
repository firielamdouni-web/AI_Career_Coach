"""
Tests unitaires - API FastAPI
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import io


MOCK_JOBS_DATASET = {
    "jobs": [
        {
            "job_id": "job_001",
            "title": "Data Scientist",
            "company": "TechCorp",
            "location": "Paris",
            "experience": "1-2 ans",
            "remote_ok": True,
            "applicants": 30,
            "category": "Data Science",
            "description": "Analyse de données avec Python et ML.",
            "requirements": ["Python", "Machine Learning", "SQL"],
            "nice_to_have": ["Docker"]
        },
        {
            "job_id": "job_002",
            "title": "ML Engineer",
            "company": "StartupAI",
            "location": "Lyon",
            "experience": "0-1 an",
            "remote_ok": False,
            "applicants": 10,
            "category": "ML Engineering",
            "description": "Déploiement de modèles ML.",
            "requirements": ["Python", "Docker", "TensorFlow"],
            "nice_to_have": ["Kubernetes"]
        }
    ]
}

MOCK_SKILLS_DB = {
    "technical_skills": ["Python", "SQL", "Docker", "TensorFlow", "Machine Learning"],
    "soft_skills": ["Communication", "Teamwork"],
    "variations": {}
}

MOCK_CV_TEXT = "Python developer with experience in Machine Learning and SQL databases."

MOCK_SKILLS_RESULT = {
    "technical_skills": ["Python", "Machine Learning", "SQL"],
    "soft_skills": ["Communication"]
}

MOCK_MATCH_SCORE = {
    "score": 75.0,
    "skills_details": {
        "coverage": 80.0,
        "quality": 70.0,
        "covered_count": 3,
        "total_required": 4,
        "top_matches": [
            {"cv_skill": "Python", "job_skill": "Python", "similarity": 1.0}
        ]
    }
}


@pytest.fixture
def client():
    """Client de test FastAPI avec tous les modules mockés"""
    with patch('src.api.get_jobs_dataset', return_value=MOCK_JOBS_DATASET), \
         patch('src.api.get_cv_parser') as mock_parser, \
         patch('src.api.get_skills_extractor') as mock_extractor, \
         patch('src.api.get_job_matcher') as mock_matcher, \
         patch('src.api.get_ml_predictor') as mock_ml, \
         patch('src.api.get_vector_store') as mock_vs, \
         patch('src.api.get_db_manager') as mock_db, \
         patch('src.api.get_interview_simulator') as mock_sim:

        # Parser CV
        mock_parser_inst = MagicMock()
        mock_parser_inst.parse.return_value = MOCK_CV_TEXT
        mock_parser.return_value = mock_parser_inst

        # Extractor
        mock_extractor_inst = MagicMock()
        mock_extractor_inst.extract_from_cv.return_value = MOCK_SKILLS_RESULT
        mock_extractor_inst.skills_database = MOCK_SKILLS_DB
        mock_extractor.return_value = mock_extractor_inst

        # Matcher
        mock_matcher_inst = MagicMock()
        mock_matcher_inst.calculate_job_match_score.return_value = MOCK_MATCH_SCORE
        mock_matcher_inst.extract_job_skills.return_value = ["Python", "SQL"]
        mock_matcher_inst.skills_db = MOCK_SKILLS_DB
        mock_matcher_inst.model = MagicMock()
        mock_matcher.return_value = mock_matcher_inst

        # ML Predictor
        mock_ml_inst = MagicMock()
        mock_ml_inst.is_loaded = False
        mock_ml.return_value = mock_ml_inst

        # Vector Store
        mock_vs_inst = MagicMock()
        mock_vs_inst.index = MagicMock()
        mock_vs_inst.index.ntotal = 2
        mock_vs.return_value = mock_vs_inst

        # Database
        mock_db_inst = MagicMock()
        mock_db_inst.save_cv_analysis.return_value = 1
        mock_db_inst.save_job_recommendation.return_value = 1
        mock_db.return_value = mock_db_inst

        # Interview Simulator
        mock_sim_inst = MagicMock()
        mock_sim_inst.generate_questions.return_value = {
            "rh_questions": [{"id": 1, "question": "Parlez-moi de vous.", "type": "présentation"}],
            "technical_questions": [{"id": 2, "question": "Expliquez Python.", "type": "technique", "skill": "Python"}]
        }
        mock_sim_inst.evaluate_answer.return_value = {
            "score": 75.0,
            "evaluation": "Bonne réponse",
            "points_forts": ["Clarté"],
            "points_amelioration": ["Plus de détails"],
            "recommandations": ["Continuer"]
        }
        mock_sim.return_value = mock_sim_inst

        from src.api import app
        with TestClient(app) as test_client:
            yield test_client


# ============================================================================
# TESTS ROOT
# ============================================================================

class TestRoot:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_version(self, client):
        response = client.get("/")
        assert response.json()["version"] == "1.0.0"


# ============================================================================
# TESTS HEALTH
# ============================================================================

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_jobs_available(self, client):
        response = client.get("/health")
        assert response.json()["jobs_available"] == 2


# ============================================================================
# TESTS STATS
# ============================================================================

class TestStats:

    def test_stats_returns_200(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200

    def test_stats_contains_total_jobs(self, client):
        response = client.get("/api/v1/stats")
        assert response.json()["total_jobs"] == 2

    def test_stats_contains_remote_jobs(self, client):
        response = client.get("/api/v1/stats")
        assert response.json()["remote_jobs"] == 1


# ============================================================================
# TESTS LIST JOBS
# ============================================================================

class TestListJobs:

    def test_list_jobs_returns_200(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200

    def test_list_jobs_returns_all_jobs(self, client):
        response = client.get("/api/v1/jobs")
        assert len(response.json()) == 2

    def test_list_jobs_filter_remote(self, client):
        response = client.get("/api/v1/jobs?remote=true")
        jobs = response.json()
        assert all(job["remote"] for job in jobs)

    def test_list_jobs_filter_category(self, client):
        response = client.get("/api/v1/jobs?category=Data+Science")
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["title"] == "Data Scientist"

    def test_list_jobs_limit(self, client):
        response = client.get("/api/v1/jobs?limit=1")
        assert len(response.json()) == 1


# ============================================================================
# TESTS GET JOB BY ID
# ============================================================================

class TestGetJob:

    def test_get_existing_job(self, client):
        response = client.get("/api/v1/jobs/job_001")
        assert response.status_code == 200
        assert response.json()["job_id"] == "job_001"

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/v1/jobs/job_999")
        assert response.status_code == 404

    def test_get_job_contains_required_fields(self, client):
        response = client.get("/api/v1/jobs/job_001")
        data = response.json()
        for field in ["job_id", "title", "company", "location", "description"]:
            assert field in data


# ============================================================================
# TESTS EXTRACT SKILLS
# ============================================================================

class TestExtractSkills:

    def _make_pdf(self):
        """Créer un faux PDF en bytes"""
        return io.BytesIO(b"%PDF fake content")

    def test_extract_skills_non_pdf_returns_400(self, client):
        fake_file = io.BytesIO(b"not a pdf")
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.txt", fake_file, "text/plain")}
        )
        assert response.status_code == 400

    def test_extract_skills_returns_technical_and_soft(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "technical_skills" in data
        assert "soft_skills" in data

    def test_extract_skills_returns_total_count(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        expected_total = len(data["technical_skills"]) + len(data["soft_skills"])
        assert data["total_skills"] == expected_total


# ============================================================================
# TESTS RECOMMEND JOBS
# ============================================================================

class TestRecommendJobs:

    def _make_pdf(self):
        return io.BytesIO(b"%PDF fake content")

    def test_recommend_jobs_non_pdf_returns_400(self, client):
        fake_file = io.BytesIO(b"not a pdf")
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.txt", fake_file, "text/plain")}
        )
        assert response.status_code == 400

    def test_recommend_jobs_returns_200(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.status_code == 200

    def test_recommend_jobs_contains_recommendations(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        assert "recommendations" in data
        assert "cv_skills_count" in data

    def test_recommend_jobs_top_n_limit(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs?top_n=1",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        assert len(data["recommendations"]) <= 1


# ============================================================================
# TESTS SIMULATE INTERVIEW
# ============================================================================

class TestSimulateInterview:

    def test_simulate_interview_valid_job(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python", "SQL"], "job_id": "job_001", "num_questions": 4}
        )
        assert response.status_code == 200

    def test_simulate_interview_invalid_job(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_999", "num_questions": 4}
        )
        assert response.status_code == 404

    def test_simulate_interview_returns_questions(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_001", "num_questions": 4}
        )
        data = response.json()
        assert "rh_questions" in data
        assert "technical_questions" in data
        assert "total_questions" in data


# ============================================================================
# TESTS EVALUATE ANSWER
# ============================================================================

class TestEvaluateAnswer:

    def test_evaluate_answer_returns_200(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "J'utilise Python depuis 2 ans pour faire du data science.",
                "question_type": "technique"
            }
        )
        assert response.status_code == 200

    def test_evaluate_answer_too_short_returns_400(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "Oui.",
                "question_type": "technique"
            }
        )
        assert response.status_code == 400

    def test_evaluate_answer_contains_score(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "J'utilise Python depuis 2 ans pour faire du data science.",
                "question_type": "technique"
            }
        )
        data = response.json()
        assert "score" in data
        assert 0 <= data["score"] <= 100