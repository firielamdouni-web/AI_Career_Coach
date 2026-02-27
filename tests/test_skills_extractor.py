"""
Tests unitaires - SkillsExtractor
"""
import pytest
from unittest.mock import patch, MagicMock
from src.skills_extractor import SkillsExtractor


MOCK_SKILLS_DB = {
    "technical_skills": [
        "Python", "Machine Learning", "SQL", "Docker", "TensorFlow"], "soft_skills": [
            "Communication", "Teamwork", "Leadership"], "variations": {
                "python": [
                    "python", "py"], "machine learning": [
                        "machine learning", "ml"]}}


@pytest.fixture
def extractor():
    """Fixture pour créer un SkillsExtractor mocké"""
    with patch('spacy.load') as mock_spacy, \
            patch('builtins.open'), \
            patch('json.load', return_value=MOCK_SKILLS_DB):

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp

        ext = SkillsExtractor.__new__(SkillsExtractor)
        ext.nlp = mock_nlp
        ext.skills_database = MOCK_SKILLS_DB
        return ext


class TestSkillsExtractor:

    def test_extract_skills_from_text_found(self, extractor):
        """Test extraction skill trouvé"""
        text = "I have experience with Python and SQL databases"
        skills = extractor.extract_skills_from_text(
            text, MOCK_SKILLS_DB['technical_skills']
        )
        assert "Python" in skills
        assert "SQL" in skills

    def test_extract_skills_from_text_not_found(self, extractor):
        """Test extraction skill non trouvé"""
        text = "I am a creative person"
        skills = extractor.extract_skills_from_text(
            text, MOCK_SKILLS_DB['technical_skills']
        )
        assert skills == []

    def test_extract_skills_case_insensitive(self, extractor):
        """Test extraction insensible à la casse"""
        text = "I know PYTHON and machine learning"
        skills = extractor.extract_skills_from_text(
            text, MOCK_SKILLS_DB['technical_skills']
        )
        assert "Python" in skills
        assert "Machine Learning" in skills

    def test_extract_skills_empty_text(self, extractor):
        """Test extraction texte vide"""
        skills = extractor.extract_skills_from_text(
            "", MOCK_SKILLS_DB['technical_skills']
        )
        assert skills == []

    def test_extract_skills_empty_list(self, extractor):
        """Test extraction liste vide"""
        skills = extractor.extract_skills_from_text(
            "Python developer", []
        )
        assert skills == []

    def test_extract_from_cv(self, extractor):
        """Test extraction complète depuis CV"""
        cv_text = "Python developer with Machine Learning experience. Good Communication skills."

        mock_doc = MagicMock()
        mock_doc.ents = []
        extractor.nlp.return_value = mock_doc

        result = extractor.extract_from_cv(cv_text)

        assert "technical_skills" in result
        assert "soft_skills" in result
        assert "total_skills" in result
        assert "Python" in result['technical_skills']
        assert "Machine Learning" in result['technical_skills']
        assert "Communication" in result['soft_skills']
        assert result['total_skills'] == len(
            result['technical_skills']) + len(result['soft_skills'])

    def test_extract_from_cv_no_skills(self, extractor):
        """Test extraction sans skills reconnus"""
        cv_text = "I am a student looking for opportunities"

        mock_doc = MagicMock()
        mock_doc.ents = []
        extractor.nlp.return_value = mock_doc

        result = extractor.extract_from_cv(cv_text)

        assert result['technical_skills'] == []
        assert result['soft_skills'] == []
        assert result['total_skills'] == 0
