"""
Tests unitaires - MLPredictor
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.ml_predictor import MLPredictor, get_ml_predictor, FEATURES


MOCK_FEATURES = {
    'coverage': 60.0,
    'quality': 75.0,
    'nb_covered_skills': 12.0,
    'nb_missing_skills': 8.0,
    'skills_ratio': 0.8,
    'similarity_mean': 70.0,
    'similarity_max': 95.0,
    'similarity_std': 15.0,
    'top3_similarity_avg': 90.0,
    'tfidf_similarity': 0.35,
    'embedding_similarity': 0.65,
    'nb_resume_technical': 10.0,
    'nb_resume_soft': 3.0,
    'nb_job_technical': 15.0,
    'nb_job_soft': 5.0
}


@pytest.fixture
def predictor():
    """Fixture MLPredictor avec modèle mocké"""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('joblib.load') as mock_joblib:

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])  # Perfect Fit
        mock_model.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.1] * 15])

        mock_joblib.side_effect = [mock_model, mock_scaler]

        p = MLPredictor()
        p.model = mock_model
        p.scaler = mock_scaler
        p._loaded = True
        return p


class TestMLPredictor:

    def test_features_list_complete(self):
        """Test que les 15 features sont définies"""
        assert len(FEATURES) == 15

    def test_is_loaded_false(self):
        """Test is_loaded quand modèle non chargé"""
        with patch('pathlib.Path.exists', return_value=False):
            p = MLPredictor()
            assert p.is_loaded is False

    def test_predict_not_loaded(self):
        """Test prédiction quand modèle non chargé"""
        with patch('pathlib.Path.exists', return_value=False):
            p = MLPredictor()
            result = p.predict(MOCK_FEATURES)
            assert result['ml_available'] is False
            assert result['ml_label'] == 'N/A'
            assert result['ml_score'] is None

    def test_predict_perfect_fit(self, predictor):
        """Test prédiction Perfect Fit"""
        predictor.model.predict.return_value = np.array([2])
        predictor.model.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])

        result = predictor.predict(MOCK_FEATURES)

        assert result['ml_available'] is True
        assert result['ml_label'] == 'Perfect Fit'
        assert result['ml_score'] == 1.0
        assert result['ml_probabilities']['perfect_fit'] == 0.8

    def test_predict_partial_fit(self, predictor):
        """Test prédiction Partial Fit"""
        predictor.model.predict.return_value = np.array([1])
        predictor.model.predict_proba.return_value = np.array([[0.10, 0.75, 0.15]])

        result = predictor.predict(MOCK_FEATURES)

        assert result['ml_label'] == 'Partial Fit'
        assert result['ml_score'] == 0.5

    def test_predict_no_fit(self, predictor):
        """Test prédiction No Fit"""
        predictor.model.predict.return_value = np.array([0])
        predictor.model.predict_proba.return_value = np.array([[0.85, 0.10, 0.05]])

        result = predictor.predict(MOCK_FEATURES)

        assert result['ml_label'] == 'No Fit'
        assert result['ml_score'] == 0.0

    def test_predict_probabilities_sum(self, predictor):
        """Test que les probabilités somment à ~1"""
        result = predictor.predict(MOCK_FEATURES)
        proba = result['ml_probabilities']
        total = proba['no_fit'] + proba['partial_fit'] + proba['perfect_fit']
        assert abs(total - 1.0) < 0.01

    def test_predict_feature_order(self, predictor):
        """Test que les features sont dans le bon ordre"""
        predictor.predict(MOCK_FEATURES)
        
        call_args = predictor.scaler.transform.call_args[0][0]
        assert len(call_args[0]) == 15
        assert call_args[0][0] == MOCK_FEATURES['coverage']
        assert call_args[0][1] == MOCK_FEATURES['quality']

    def test_get_ml_predictor_singleton(self):
        """Test que get_ml_predictor retourne un singleton"""
        import src.ml_predictor as ml_module
        ml_module._ml_predictor = None

        with patch('pathlib.Path.exists', return_value=False):
            p1 = get_ml_predictor()
            p2 = get_ml_predictor()
            assert p1 is p2

        ml_module._ml_predictor = None