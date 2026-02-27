"""
Tests unitaires - JobMatcher
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.job_matcher import JobMatcher


MOCK_SKILLS_DB = {
    "technical_skills": ["python", "machine learning", "sql", "docker"],
    "soft_skills": ["communication", "teamwork"],
    "variations": {
        "python": ["python", "py"],
        "machine learning": ["machine learning", "ml"]
    }
}

MOCK_JOB = {
    "job_id": "job_001",
    "title": "Junior ML Engineer",
    "company": "TechCorp",
    "location": "Paris",
    "type": "CDI",
    "experience": "1-2 ans",
    "salary": "40-50k",
    "remote_ok": True,
    "applicants": 50,
    "url": "http://example.com",
    "description": "We need a Python developer",
    "requirements": ["Python (python, ml)", "SQL experience"],
    "nice_to_have": ["Docker (docker)"]
}


@pytest.fixture
def matcher():
    """Fixture pour créer un JobMatcher mocké"""
    with patch('src.job_matcher.SentenceTransformer') as mock_st, \
            patch('builtins.open'), \
            patch('json.load', return_value=MOCK_SKILLS_DB), \
            patch('pathlib.Path.exists', return_value=True):

        mock_model = MagicMock()
        # Retourner des embeddings factices (768-dim)
        mock_model.encode.return_value = np.random.rand(1, 768)[0]
        mock_st.return_value = mock_model

        m = JobMatcher.__new__(JobMatcher)
        m.model = mock_model
        m.skills_db = MOCK_SKILLS_DB
        m.variations_map = m._build_variations_map()
        return m


class TestJobMatcher:

    def test_normalize_skill_basic(self, matcher):
        """Test normalisation skill basique"""
        result = matcher._normalize_skill("Python")
        assert result == "python"

    def test_normalize_skill_with_dash(self, matcher):
        """Test normalisation skill avec tiret"""
        result = matcher._normalize_skill("machine-learning")
        assert result == "machine learning"

    def test_normalize_skill_with_spaces(self, matcher):
        """Test normalisation skill avec espaces multiples"""
        result = matcher._normalize_skill("  python  ")
        assert result == "python"

    def test_normalize_skill_variation(self, matcher):
        """Test normalisation via variations"""
        result = matcher._normalize_skill("py")
        assert result == "python"

    def test_extract_job_skills(self, matcher):
        """Test extraction skills depuis offre"""
        skills = matcher.extract_job_skills(MOCK_JOB)
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_extract_job_skills_empty_job(self, matcher):
        """Test extraction depuis offre vide"""
        empty_job = {"job_id": "empty", "title": "Test"}
        skills = matcher.extract_job_skills(empty_job)
        assert isinstance(skills, list)

    def test_calculate_skills_similarity_empty_cv(self, matcher):
        """Test similarité avec CV vide"""
        result = matcher.calculate_skills_similarity([], MOCK_JOB)
        assert result['overall_score'] == 0
        assert result['coverage'] == 0
        assert result['covered_count'] == 0

    def test_calculate_skills_similarity_returns_dict(self, matcher):
        """Test que calculate_skills_similarity retourne le bon format"""
        matcher.model.encode.side_effect = lambda skills, **kwargs: np.random.rand(
            768)

        result = matcher.calculate_skills_similarity(
            ["python", "machine learning"], MOCK_JOB
        )
        assert "overall_score" in result
        assert "coverage" in result
        assert "quality" in result
        assert "covered_count" in result
        assert "total_required" in result
        assert "matches" in result

    def test_calculate_job_match_score_format(self, matcher):
        """Test format de retour de calculate_job_match_score"""
        matcher.model.encode.side_effect = lambda skills, **kwargs: np.random.rand(
            768)

        result = matcher.calculate_job_match_score(["python"], MOCK_JOB)

        assert result['job_id'] == "job_001"
        assert result['title'] == "Junior ML Engineer"
        assert 'score' in result
        assert 'skills_details' in result
        assert 0 <= result['score'] <= 100

    def test_calculate_job_match_score_skills_details(self, matcher):
        """Test structure skills_details"""
        matcher.model.encode.side_effect = lambda skills, **kwargs: np.random.rand(
            768)

        result = matcher.calculate_job_match_score(["python", "sql"], MOCK_JOB)
        details = result['skills_details']

        assert 'coverage' in details
        assert 'quality' in details
        assert 'covered_count' in details
        assert 'total_required' in details
        assert 'top_matches' in details

    def test_rank_jobs_empty_skills(self, matcher):
        """Test ranking avec CV vide"""
        result = matcher.rank_jobs([], [MOCK_JOB])
        assert result == []

    def test_rank_jobs_empty_jobs(self, matcher):
        """Test ranking avec liste d'offres vide"""
        result = matcher.rank_jobs(["python"], [])
        assert result == []

    def test_rank_jobs_sorted(self, matcher):
        """Test que rank_jobs retourne les offres triées"""
        job2 = {**MOCK_JOB, "job_id": "job_002", "title": "Senior ML Engineer"}
        matcher.model.encode.side_effect = lambda skills, **kwargs: np.random.rand(
            768)

        results = matcher.rank_jobs(["python", "sql"], [MOCK_JOB, job2])

        assert isinstance(results, list)
        if len(results) >= 2:
            assert results[0]['score'] >= results[1]['score']
