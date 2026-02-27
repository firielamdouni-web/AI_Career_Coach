"""
Tests unitaires - JobMatcher Approche 4 (Coverage + Quality)
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


MOCK_SKILLS_DB = {
    "technical_skills": ["python", "django", "postgresql", "docker", "git"],
    "soft_skills": ["communication", "teamwork"],
    "variations": {}
}

MOCK_JOB = {
    "job_id": "test_001",
    "title": "Développeur Python/Django",
    "company": "Tech Startup",
    "location": "Paris",
    "description": "Nous recherchons un développeur Python/Django.",
    "requirements": [
        "Python (Django, Flask)",
        "PostgreSQL ou MySQL",
        "Docker",
        "Git"
    ],
    "nice_to_have": ["AWS", "Kubernetes"]
}

CV_SKILLS_GOOD = ["Python", "Django", "PostgreSQL", "Docker", "Git", "Excel"]
CV_SKILLS_POOR = ["Excel", "Word", "PowerPoint"]
CV_SKILLS_EMPTY = []


@pytest.fixture
def matcher():
    """Fixture JobMatcher avec modèle mocké"""
    with patch('src.job_matcher.SentenceTransformer') as mock_st, \
         patch('builtins.open'), \
         patch('json.load', return_value=MOCK_SKILLS_DB):

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        # Simuler encode : retourne des vecteurs normalisés
        def mock_encode(texts, **kwargs):
            np.random.seed(42)
            vectors = np.random.rand(len(texts), 768).astype('float32')
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        mock_model.encode.side_effect = mock_encode
        mock_st.return_value = mock_model

        from src.job_matcher import JobMatcher
        return JobMatcher()


class TestJobMatcherApproach4:

    def test_score_is_between_0_and_100(self, matcher):
        """Le score doit être entre 0 et 100"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        assert 0 <= result['score'] <= 100

    def test_score_has_coverage_and_quality(self, matcher):
        """Le résultat doit contenir coverage et quality"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        assert 'coverage' in result['skills_details']
        assert 'quality' in result['skills_details']

    def test_good_profile_scores_higher_than_poor(self, matcher):
        """Un bon profil doit scorer plus haut qu'un mauvais profil"""
        result_good = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        result_poor = matcher.calculate_job_match_score(CV_SKILLS_POOR, MOCK_JOB)
        assert result_good['score'] >= result_poor['score']

    def test_score_formula_is_coverage_quality_average(self, matcher):
        """Score = (coverage + quality) / 2"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        details = result['skills_details']
        expected_score = (details['coverage'] + details['quality']) / 2
        assert abs(result['score'] - expected_score) < 0.1

    def test_covered_count_not_exceeds_total_required(self, matcher):
        """Le nombre de skills couverts ne doit pas dépasser le total requis"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        details = result['skills_details']
        assert details['covered_count'] <= details['total_required']

    def test_top_matches_is_list(self, matcher):
        """top_matches doit être une liste"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        assert isinstance(result['skills_details']['top_matches'], list)

    def test_top_matches_have_required_keys(self, matcher):
        """Chaque match doit avoir cv_skill, job_skill, similarity"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        for match in result['skills_details']['top_matches']:
            assert 'cv_skill' in match
            assert 'job_skill' in match
            assert 'similarity' in match

    def test_empty_cv_skills_returns_zero_score(self, matcher):
        """CV vide → score à 0"""
        result = matcher.calculate_job_match_score(CV_SKILLS_EMPTY, MOCK_JOB)
        assert result['score'] == 0

    def test_coverage_is_percentage(self, matcher):
        """Coverage doit être un pourcentage entre 0 et 100"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        coverage = result['skills_details']['coverage']
        assert 0 <= coverage <= 100

    def test_quality_is_percentage(self, matcher):
        """Quality doit être un pourcentage entre 0 et 100"""
        result = matcher.calculate_job_match_score(CV_SKILLS_GOOD, MOCK_JOB)
        quality = result['skills_details']['quality']
        assert 0 <= quality <= 100 + 1e-4  # float32 precision tolerance