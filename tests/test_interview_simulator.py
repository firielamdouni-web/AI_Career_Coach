"""
Tests unitaires - InterviewSimulator
"""
import pytest
import json
from unittest.mock import patch, MagicMock


MOCK_QUESTIONS_RESPONSE = {
    "rh_questions": [
        {"id": 1, "question": "Parlez-moi de vous.", "type": "présentation"},
        {"id": 2, "question": "Pourquoi ce poste ?", "type": "motivation"}
    ],
    "technical_questions": [
        {"id": 3, "question": "Comment utilisez-vous Python ?", "type": "technique", "skill": "Python"},
        {"id": 4, "question": "Expliquez Docker.", "type": "outil", "skill": "Docker"}
    ]
}

MOCK_EVALUATION_RESPONSE = {
    "score": 75.0,
    "evaluation": "Bonne réponse structurée.",
    "points_forts": ["Clarté", "Exemples concrets"],
    "points_amelioration": ["Plus de détails techniques"],
    "recommandations": ["Préparer des exemples chiffrés"]
}

MOCK_FEEDBACK_RESPONSE = {
    "global_score": 72.5,
    "decision": "Prometteur",
    "synthese": "Bon profil junior avec de bonnes bases.",
    "competences_validees": ["Python", "Communication"],
    "axes_progression": ["Approfondir Docker"],
    "prochaines_etapes": ["Pratiquer les entretiens"]
}


@pytest.fixture
def simulator():
    """Fixture InterviewSimulator avec Groq mocké"""
    with patch('groq.Groq') as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        from src.interview_simulator import InterviewSimulator
        sim = InterviewSimulator(api_key="fake_key_for_testing")
        sim.client = mock_client
        return sim


def _make_groq_response(content: dict) -> MagicMock:
    """Helper pour créer une fausse réponse Groq"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content)
    return mock_response


class TestInterviewSimulatorInit:

    def test_raises_without_api_key(self):
        """Doit lever ValueError sans clé API"""
        with patch.dict('os.environ', {}, clear=True):
            with patch('os.getenv', return_value=None):
                from src.interview_simulator import InterviewSimulator
                with pytest.raises(ValueError, match="Clé API Groq manquante"):
                    InterviewSimulator(api_key=None)

    def test_init_with_api_key(self):
        """Initialisation correcte avec clé API"""
        with patch('groq.Groq'):
            from src.interview_simulator import InterviewSimulator
            sim = InterviewSimulator(api_key="fake_key")
            assert sim.api_key == "fake_key"

    def test_model_is_set(self):
        """Le modèle LLM doit être défini"""
        with patch('groq.Groq'):
            from src.interview_simulator import InterviewSimulator
            sim = InterviewSimulator(api_key="fake_key")
            assert sim.model is not None
            assert len(sim.model) > 0


class TestGenerateQuestions:

    def test_generate_questions_returns_rh_and_technical(self, simulator):
        """Doit retourner rh_questions et technical_questions"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_QUESTIONS_RESPONSE)

        result = simulator.generate_questions(
            cv_skills=["Python", "Docker"],
            job_title="Data Scientist",
            job_description="Analyse de données.",
            job_requirements=["Python", "SQL"],
            num_questions=4
        )

        assert "rh_questions" in result
        assert "technical_questions" in result

    def test_generate_questions_rh_is_list(self, simulator):
        """rh_questions doit être une liste"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_QUESTIONS_RESPONSE)

        result = simulator.generate_questions(
            cv_skills=["Python"],
            job_title="Developer",
            job_description="Dev role.",
            job_requirements=["Python"],
            num_questions=4
        )
        assert isinstance(result["rh_questions"], list)
        assert isinstance(result["technical_questions"], list)

    def test_generate_questions_each_has_id_and_question(self, simulator):
        """Chaque question doit avoir 'id' et 'question'"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_QUESTIONS_RESPONSE)

        result = simulator.generate_questions(
            cv_skills=["Python"],
            job_title="Developer",
            job_description="Dev role.",
            job_requirements=["Python"],
            num_questions=4
        )
        for q in result["rh_questions"] + result["technical_questions"]:
            assert "id" in q
            assert "question" in q

    def test_generate_questions_invalid_json_raises(self, simulator):
        """JSON invalide doit lever une ValueError"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid json {{{"
        simulator.client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="JSON valide"):
            simulator.generate_questions(
                cv_skills=["Python"],
                job_title="Dev",
                job_description="Role.",
                job_requirements=["Python"],
                num_questions=4
            )

    def test_generate_questions_cleans_markdown_json(self, simulator):
        """Doit nettoyer le markdown ```json``` de la réponse"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "```json\n" + json.dumps(MOCK_QUESTIONS_RESPONSE) + "\n```"
        )
        simulator.client.chat.completions.create.return_value = mock_response

        result = simulator.generate_questions(
            cv_skills=["Python"],
            job_title="Dev",
            job_description="Role.",
            job_requirements=["Python"],
            num_questions=4
        )
        assert "rh_questions" in result


class TestEvaluateAnswer:

    def test_evaluate_returns_score(self, simulator):
        """L'évaluation doit retourner un score"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_EVALUATION_RESPONSE)

        result = simulator.evaluate_answer(
            question="Parlez-moi de Python.",
            answer="J'utilise Python depuis 2 ans.",
            question_type="technique"
        )
        assert "score" in result

    def test_evaluate_score_between_0_and_100(self, simulator):
        """Le score doit être entre 0 et 100"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_EVALUATION_RESPONSE)

        result = simulator.evaluate_answer(
            question="Parlez-moi de Python.",
            answer="J'utilise Python depuis 2 ans.",
            question_type="technique"
        )
        assert 0 <= result["score"] <= 100

    def test_evaluate_returns_points_forts(self, simulator):
        """L'évaluation doit retourner des points forts"""
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(MOCK_EVALUATION_RESPONSE)

        result = simulator.evaluate_answer(
            question="Question test.",
            answer="Réponse de test suffisamment longue.",
            question_type="rh"
        )
        assert "points_forts" in result
        assert isinstance(result["points_forts"], list)

    def test_evaluate_normalizes_score_above_100(self, simulator):
        """Un score > 100 doit être ramené à 100"""
        bad_score = {**MOCK_EVALUATION_RESPONSE, "score": 150}
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(bad_score)

        result = simulator.evaluate_answer(
            question="Question.",
            answer="Réponse suffisamment longue pour le test.",
            question_type="technique"
        )
        assert result["score"] <= 100

    def test_evaluate_adds_recommandations_if_missing(self, simulator):
        """Doit ajouter des recommandations si absentes du JSON"""
        response_without_reco = {k: v for k, v in MOCK_EVALUATION_RESPONSE.items()
                                  if k != "recommandations"}
        simulator.client.chat.completions.create.return_value = \
            _make_groq_response(response_without_reco)

        result = simulator.evaluate_answer(
            question="Question.",
            answer="Réponse suffisamment longue pour le test.",
            question_type="rh"
        )
        assert "recommandations" in result


class TestGenerateFinalFeedback:

    def test_empty_evaluations_returns_default(self, simulator):
        """Liste vide → feedback par défaut sans appel API"""
        result = simulator.generate_final_feedback([], "Data Scientist")
        assert result["global_score"] == 0
        assert "decision" in result

    def test_feedback_decision_poor_score(self, simulator):
        """Score < 50 → décision 'À retravailler' (fallback)"""
        simulator.client.chat.completions.create.side_effect = Exception("API error")

        evaluations = [{"score": 30, "points_forts": [], "points_amelioration": []}]
        result = simulator.generate_final_feedback(evaluations, "Data Scientist")
        assert result["decision"] == "À retravailler"

    def test_feedback_decision_good_score(self, simulator):
        """Score > 75 → décision 'Excellent' (fallback)"""
        simulator.client.chat.completions.create.side_effect = Exception("API error")

        evaluations = [{"score": 85, "points_forts": [], "points_amelioration": []}]
        result = simulator.generate_final_feedback(evaluations, "Data Scientist")
        assert result["decision"] == "Excellent"

    def test_feedback_global_score_is_average(self, simulator):
        """global_score = moyenne des scores (fallback)"""
        simulator.client.chat.completions.create.side_effect = Exception("API error")

        evaluations = [
            {"score": 60, "points_forts": [], "points_amelioration": []},
            {"score": 80, "points_forts": [], "points_amelioration": []}
        ]
        result = simulator.generate_final_feedback(evaluations, "Dev")
        assert result["global_score"] == 70.0


class TestGetInterviewSimulatorSingleton:

    def test_singleton_returns_same_instance(self):
        """get_interview_simulator doit retourner la même instance"""
        with patch('groq.Groq'), \
             patch('os.getenv', return_value="fake_key"):
            from src.interview_simulator import get_interview_simulator
            import src.interview_simulator as module
            module._interview_simulator = None  # reset

            sim1 = get_interview_simulator()
            sim2 = get_interview_simulator()
            assert sim1 is sim2
            module._interview_simulator = None  # cleanup