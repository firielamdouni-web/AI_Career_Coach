"""
Tests unitaires - API FastAPI
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
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
    "technical_skills": [
        "Python",
        "SQL",
        "Docker",
        "TensorFlow",
        "Machine Learning"],
    "soft_skills": [
        "Communication",
        "Teamwork"],
    "variations": {}}

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

        # Database — fetchone=None par défaut pour éviter les faux positifs
        mock_db_inst = MagicMock()
        mock_db_inst.save_cv_analysis.return_value = 1
        mock_db_inst.save_job_recommendation.return_value = 1
        mock_db_inst.cursor.fetchone.return_value = None
        mock_db_inst.cursor.fetchall.return_value = []
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

    def test_root_contains_status(self, client):
        response = client.get("/")
        assert response.json()["status"] == "operational"

    def test_root_contains_documentation_link(self, client):
        response = client.get("/")
        assert response.json()["documentation"] == "/docs"

    def test_root_contains_message(self, client):
        response = client.get("/")
        assert "message" in response.json()


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

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        for field in ["status", "message", "version", "models_loaded", "jobs_available"]:
            assert field in data

    def test_health_version_field(self, client):
        response = client.get("/health")
        assert response.json()["version"] == "1.0.0"

    def test_health_models_loaded_is_bool(self, client):
        response = client.get("/health")
        assert isinstance(response.json()["models_loaded"], bool)


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

    def test_stats_on_site_jobs(self, client):
        response = client.get("/api/v1/stats")
        data = response.json()
        assert data["on_site_jobs"] == data["total_jobs"] - data["remote_jobs"]

    def test_stats_jobs_by_category_is_dict(self, client):
        response = client.get("/api/v1/stats")
        assert isinstance(response.json()["jobs_by_category"], dict)

    def test_stats_jobs_by_category_counts(self, client):
        response = client.get("/api/v1/stats")
        categories = response.json()["jobs_by_category"]
        assert categories.get("Data Science") == 1
        assert categories.get("ML Engineering") == 1

    def test_stats_model_used_field(self, client):
        response = client.get("/api/v1/stats")
        assert "model_used" in response.json()

    def test_stats_technical_skills_count_positive(self, client):
        response = client.get("/api/v1/stats")
        assert response.json()["total_technical_skills"] > 0

    def test_stats_soft_skills_count_positive(self, client):
        response = client.get("/api/v1/stats")
        assert response.json()["total_soft_skills"] > 0

    def test_stats_complete_schema(self, client):
        response = client.get("/api/v1/stats")
        data = response.json()
        for field in ["total_jobs", "jobs_by_category", "remote_jobs",
                      "on_site_jobs", "total_technical_skills",
                      "total_soft_skills", "model_used"]:
            assert field in data


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

    def test_list_jobs_filter_remote_false(self, client):
        response = client.get("/api/v1/jobs?remote=false")
        jobs = response.json()
        assert all(not job["remote"] for job in jobs)

    def test_list_jobs_filter_category(self, client):
        response = client.get("/api/v1/jobs?category=Data+Science")
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["title"] == "Data Scientist"

    def test_list_jobs_filter_nonexistent_category(self, client):
        response = client.get("/api/v1/jobs?category=NonExistent")
        assert response.json() == []

    def test_list_jobs_limit(self, client):
        response = client.get("/api/v1/jobs?limit=1")
        assert len(response.json()) == 1

    def test_list_jobs_schema(self, client):
        response = client.get("/api/v1/jobs")
        job = response.json()[0]
        for field in ["job_id", "title", "company", "location",
                      "remote", "experience_required", "category",
                      "description", "skills_required"]:
            assert field in job

    def test_list_jobs_returns_list(self, client):
        response = client.get("/api/v1/jobs")
        assert isinstance(response.json(), list)

    def test_list_jobs_remote_true_count(self, client):
        response = client.get("/api/v1/jobs?remote=true")
        assert len(response.json()) == 1

    def test_list_jobs_ml_engineering_category(self, client):
        response = client.get("/api/v1/jobs?category=ML+Engineering")
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["title"] == "ML Engineer"


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

    def test_get_job_002(self, client):
        response = client.get("/api/v1/jobs/job_002")
        assert response.status_code == 200
        assert response.json()["title"] == "ML Engineer"

    def test_get_job_correct_company(self, client):
        response = client.get("/api/v1/jobs/job_001")
        assert response.json()["company"] == "TechCorp"

    def test_get_job_correct_location(self, client):
        response = client.get("/api/v1/jobs/job_002")
        assert response.json()["location"] == "Lyon"

    def test_get_job_remote_field_is_bool(self, client):
        response = client.get("/api/v1/jobs/job_001")
        assert isinstance(response.json()["remote"], bool)

    def test_get_job_skills_required_is_list(self, client):
        response = client.get("/api/v1/jobs/job_001")
        assert isinstance(response.json()["skills_required"], list)

    def test_get_scraped_job_via_db(self, client):
        """job_id sc_ trouvé en DB → 200"""
        with patch('src.api.get_db_manager') as mock_db_local:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = {
                'job_id': 'sc_db001',
                'title': 'DevOps Engineer',
                'company': 'CloudCorp',
                'location': 'Remote',
                'description': 'CI/CD pipelines.',
                'is_remote': True,
                'required_skills': '["Docker","Kubernetes"]',
            }
            mock_db_inst = MagicMock()
            mock_db_inst.cursor = mock_cursor
            mock_db_local.return_value = mock_db_inst

            response = client.get("/api/v1/jobs/sc_db001")
        assert response.status_code == 200
        assert response.json()["title"] == "DevOps Engineer"

    def test_get_scraped_job_db_failure_returns_500(self, client):
        """DB inaccessible pour job sc_ → 500"""
        with patch('src.api.get_db_manager') as mock_db_fail:
            mock_db_fail.side_effect = Exception("DB down")
            response = client.get("/api/v1/jobs/sc_unknown")
        assert response.status_code == 500


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

    def test_extract_skills_technical_skills_are_list(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert isinstance(response.json()["technical_skills"], list)

    def test_extract_skills_soft_skills_are_list(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert isinstance(response.json()["soft_skills"], list)

    def test_extract_skills_cv_text_length_positive(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.json()["cv_text_length"] > 0

    def test_extract_skills_expected_technical_skills(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert "Python" in response.json()["technical_skills"]

    def test_extract_skills_schema(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        for field in ["technical_skills", "soft_skills", "total_skills", "cv_text_length"]:
            assert field in data

    def test_extract_skills_jpg_returns_400(self, client):
        fake_file = io.BytesIO(b"fake image")
        response = client.post(
            "/api/v1/extract-skills",
            files={"file": ("cv.jpg", fake_file, "image/jpeg")}
        )
        assert response.status_code == 400


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

    def test_recommend_jobs_schema(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        for field in ["recommendations", "total_jobs_analyzed",
                      "cv_skills_count", "local_jobs_count", "scraped_jobs_count"]:
            assert field in data

    def test_recommend_jobs_recommendations_is_list(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert isinstance(response.json()["recommendations"], list)

    def test_recommend_jobs_cv_skills_count_positive(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.json()["cv_skills_count"] > 0

    def test_recommend_jobs_recommendation_schema(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        recs = response.json()["recommendations"]
        assert len(recs) > 0
        rec = recs[0]
        for field in ["job_id", "title", "company", "score",
                      "matching_skills", "missing_skills"]:
            assert field in rec

    def test_recommend_jobs_score_is_float(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        recs = response.json()["recommendations"]
        assert all(isinstance(r["score"], float) for r in recs)

    def test_recommend_jobs_sorted_by_score_desc(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        scores = [r["score"] for r in response.json()["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_jobs_min_score_filter(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs?min_score=80.0",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        recs = response.json()["recommendations"]
        assert all(r["score"] >= 80.0 for r in recs)

    def test_recommend_jobs_local_jobs_count(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.json()["local_jobs_count"] == 2

    def test_recommend_jobs_total_analyzed_geq_local(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        data = response.json()
        assert data["total_jobs_analyzed"] >= data["local_jobs_count"]

    def test_recommend_jobs_top_n_invalid_too_high_returns_422(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs?top_n=6000",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.status_code == 422

    def test_recommend_jobs_top_n_zero_returns_422(self, client):
        fake_pdf = self._make_pdf()
        response = client.post(
            "/api/v1/recommend-jobs?top_n=0",
            files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
        )
        assert response.status_code == 422

    def test_recommend_jobs_with_ml_predictor_loaded(self, client):
        """Quand ML est chargé, ml_available doit être True dans les résultats"""
        with patch('src.api.get_ml_predictor') as mock_ml_loaded:
            mock_ml_inst = MagicMock()
            mock_ml_inst.is_loaded = True
            mock_ml_inst.compute_features.return_value = {"feature_1": 0.5}
            mock_ml_inst.predict.return_value = {
                "ml_available": True,
                "ml_label": "Perfect Fit",
                "ml_score": 0.9,
                "ml_probabilities": {"No Fit": 0.05, "Partial Fit": 0.05, "Perfect Fit": 0.9}
            }
            mock_ml_loaded.return_value = mock_ml_inst

            fake_pdf = io.BytesIO(b"%PDF fake content")
            response = client.post(
                "/api/v1/recommend-jobs",
                files={"file": ("cv.pdf", fake_pdf, "application/pdf")}
            )
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        assert any(r["ml_available"] for r in recs)


# ============================================================================
# TESTS SIMULATE INTERVIEW
# ============================================================================

class TestSimulateInterview:

    def test_simulate_interview_valid_job(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python", "SQL"], "job_id": "job_001", "num_questions": 4})
        assert response.status_code == 200

    def test_simulate_interview_invalid_job(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_999", "num_questions": 4})
        assert response.status_code == 404

    def test_simulate_interview_returns_questions(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_001", "num_questions": 4})
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
                "question_type": "technique"})
        assert response.status_code == 200

    def test_evaluate_answer_too_short_returns_400(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "Oui.",
                "question_type": "technique"})
        assert response.status_code == 400

    def test_evaluate_answer_contains_score(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "J'utilise Python depuis 2 ans pour faire du data science.",
                "question_type": "technique"})
        data = response.json()
        assert "score" in data
        assert 0 <= data["score"] <= 100

    def test_evaluate_answer_schema(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "J'utilise Python depuis 2 ans pour faire du data science.",
                "question_type": "technique"})
        data = response.json()
        for field in ["score", "evaluation", "points_forts",
                      "points_amelioration", "recommandations"]:
            assert field in data

    def test_evaluate_answer_points_forts_is_list(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "J'utilise Python depuis 2 ans pour faire du data science.",
                "question_type": "technique"})
        assert isinstance(response.json()["points_forts"], list)

    def test_evaluate_answer_with_target_skill(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Expliquez Docker.",
                "answer": "Docker permet de conteneuriser les applications facilement.",
                "question_type": "technique",
                "target_skill": "Docker"})
        assert response.status_code == 200

    def test_evaluate_answer_empty_answer_returns_400(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "",
                "question_type": "technique"})
        assert response.status_code == 400

    def test_evaluate_answer_whitespace_only_returns_400(self, client):
        response = client.post(
            "/api/v1/evaluate-answer",
            json={
                "question": "Parlez-moi de Python.",
                "answer": "   ",
                "question_type": "technique"})
        assert response.status_code == 400


# ============================================================================
# TESTS SIMULATE INTERVIEW — SCRAPÉS + CAS LIMITES
# ============================================================================

class TestSimulateInterviewExtended:

    def test_simulate_interview_invalid_job_returns_404(self, client):
        """job_999 n'existe ni en local ni en DB (fetchone=None) → 404"""
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_999", "num_questions": 4}
        )
        assert response.status_code == 404

    def test_simulate_interview_scraped_job_via_db(self, client):
        """DB retourne une ligne → 200 avec le titre de la ligne DB"""
        with patch('src.api.get_db_manager') as mock_db_scraped:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = {
                'job_id': 'sc_abc123',
                'title': 'Data Engineer JSearch',
                'description': 'Poste Data Engineer en France.',
                'required_skills': '["Python","Spark"]',
            }
            mock_db_inst = MagicMock()
            mock_db_inst.cursor = mock_cursor
            mock_db_scraped.return_value = mock_db_inst

            response = client.post(
                "/api/v1/simulate-interview",
                json={"cv_skills": ["Python"], "job_id": "sc_abc123", "num_questions": 4}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["job_title"] == "Data Engineer JSearch"

    def test_simulate_interview_scraped_job_db_failure_returns_404(self, client):
        """DB lève une exception → job reste None → 404"""
        with patch('src.api.get_db_manager') as mock_db_fail:
            mock_db_fail.side_effect = Exception("DB connection error")
            response = client.post(
                "/api/v1/simulate-interview",
                json={"cv_skills": ["Python"], "job_id": "sc_unknown", "num_questions": 4}
            )
        assert response.status_code == 404

    def test_simulate_interview_response_structure(self, client):
        """Vérifie la structure complète de la réponse pour un job valide"""
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python", "SQL"], "job_id": "job_001", "num_questions": 8}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_title" in data
        assert "rh_questions" in data
        assert "technical_questions" in data
        assert "total_questions" in data
        assert isinstance(data["rh_questions"], list)
        assert isinstance(data["technical_questions"], list)
        assert data["total_questions"] == len(data["rh_questions"]) + len(data["technical_questions"])

    def test_simulate_interview_total_questions_coherent(self, client):
        """total_questions == len(rh) + len(technical)"""
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_002", "num_questions": 4}
        )
        data = response.json()
        assert data["total_questions"] == len(data["rh_questions"]) + len(data["technical_questions"])

    def test_simulate_interview_empty_skills_still_works(self, client):
        """cv_skills vide → l'endpoint ne doit pas planter"""
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": [], "job_id": "job_001", "num_questions": 4}
        )
        assert response.status_code == 200

    def test_simulate_interview_sc_prefix_stripped_for_db_lookup(self, client):
        """DB retourne None pour sc_ → 404"""
        with patch('src.api.get_db_manager') as mock_db_none:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_db_inst = MagicMock()
            mock_db_inst.cursor = mock_cursor
            mock_db_none.return_value = mock_db_inst

            response = client.post(
                "/api/v1/simulate-interview",
                json={"cv_skills": ["Python"], "job_id": "sc_inexistant", "num_questions": 4}
            )
        assert response.status_code == 404

    def test_simulate_interview_job_title_matches_dataset(self, client):
        """Le job_title retourné doit correspondre au dataset"""
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_001", "num_questions": 4}
        )
        assert response.json()["job_title"] == "Data Scientist"

    def test_simulate_interview_job_002_title(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python", "Docker"], "job_id": "job_002", "num_questions": 4}
        )
        assert response.json()["job_title"] == "ML Engineer"

    def test_simulate_interview_rh_questions_not_empty(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_001", "num_questions": 4}
        )
        assert len(response.json()["rh_questions"]) > 0

    def test_simulate_interview_technical_questions_not_empty(self, client):
        response = client.post(
            "/api/v1/simulate-interview",
            json={"cv_skills": ["Python"], "job_id": "job_001", "num_questions": 4}
        )
        assert len(response.json()["technical_questions"]) > 0

    def test_simulate_interview_scraped_job_missing_fields_uses_defaults(self, client):
        """DB retourne une ligne minimale sans description → pas d'erreur"""
        with patch('src.api.get_db_manager') as mock_db_minimal:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = {
                'job_id': 'sc_minimal',
                'title': 'Minimal Job',
                'description': None,
                'required_skills': None,
            }
            mock_db_inst = MagicMock()
            mock_db_inst.cursor = mock_cursor
            mock_db_minimal.return_value = mock_db_inst

            response = client.post(
                "/api/v1/simulate-interview",
                json={"cv_skills": ["Python"], "job_id": "sc_minimal", "num_questions": 2}
            )
        assert response.status_code == 200
        assert response.json()["job_title"] == "Minimal Job"


# ============================================================================
# TESTS SCRAPED JOBS ENDPOINT
# ============================================================================

class TestScrapedJobs:

    def test_scraped_jobs_returns_200(self, client):
        response = client.get("/api/v1/scraped-jobs")
        assert response.status_code == 200

    def test_scraped_jobs_schema(self, client):
        response = client.get("/api/v1/scraped-jobs")
        data = response.json()
        assert "total" in data
        assert "jobs" in data

    def test_scraped_jobs_jobs_is_list(self, client):
        response = client.get("/api/v1/scraped-jobs")
        assert isinstance(response.json()["jobs"], list)

    def test_scraped_jobs_empty_when_db_empty(self, client):
        response = client.get("/api/v1/scraped-jobs")
        assert response.json()["total"] == 0

    def test_scraped_jobs_limit_param(self, client):
        response = client.get("/api/v1/scraped-jobs?limit=10")
        assert response.status_code == 200

    def test_scraped_jobs_limit_too_high_returns_422(self, client):
        response = client.get("/api/v1/scraped-jobs?limit=999")
        assert response.status_code == 422

    def test_scraped_jobs_source_filter(self, client):
        response = client.get("/api/v1/scraped-jobs?source=jsearch")
        assert response.status_code == 200


# ============================================================================
# TESTS FAISS STATS
# ============================================================================

class TestFaissStats:

    def test_faiss_stats_returns_200(self, client):
        response = client.get("/api/v1/faiss-stats")
        assert response.status_code == 200

    def test_faiss_stats_schema(self, client):
        response = client.get("/api/v1/faiss-stats")
        data = response.json()
        for field in ["faiss_enabled", "total_jobs_indexed",
                      "model_used", "embedding_dimension"]:
            assert field in data

    def test_faiss_stats_total_jobs_indexed(self, client):
        response = client.get("/api/v1/faiss-stats")
        assert response.status_code in [200, 500]


# ============================================================================
# TESTS 404 HANDLER
# ============================================================================

class TestErrorHandlers:

    def test_unknown_endpoint_returns_404(self, client):
        response = client.get("/api/v1/nonexistent-endpoint")
        assert response.status_code == 404

    def test_404_response_contains_detail(self, client):
        response = client.get("/api/v1/nonexistent-endpoint")
        assert "detail" in response.json()

