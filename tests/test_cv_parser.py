"""
Tests unitaires - CVParser
"""
import pytest
from unittest.mock import patch, MagicMock
from src.cv_parser import CVParser


class TestCVParser:

    def test_init_default_method(self):
        """Test initialisation avec méthode par défaut"""
        parser = CVParser()
        assert parser.method == 'pdfplumber'
        assert parser.text == ""

    def test_init_pypdf2_method(self):
        """Test initialisation avec pypdf2"""
        parser = CVParser(method='pypdf2')
        assert parser.method == 'pypdf2'

    def test_invalid_method(self, tmp_path):
        """Test méthode non supportée"""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        parser = CVParser(method='invalid')
        with pytest.raises(ValueError, match="Méthode non supportée"):
            parser.parse(fake_pdf)

    def test_file_not_found(self):
        """Test fichier inexistant"""
        parser = CVParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("non_existent.pdf")

    def test_get_stats_no_text(self):
        """Test stats sans texte extrait"""
        parser = CVParser()
        stats = parser.get_stats()
        assert "error" in stats

    def test_get_stats_with_text(self):
        """Test stats avec texte extrait"""
        parser = CVParser()
        parser.text = "Hello World\nPython Developer"
        stats = parser.get_stats()
        assert stats['words'] == 4
        assert stats['characters'] == len("Hello World\nPython Developer")
        assert stats['lines'] == 2
        assert stats['method'] == 'pdfplumber'

    def test_save_text_no_text(self, tmp_path):
        """Test sauvegarde sans texte"""
        parser = CVParser()
        with pytest.raises(ValueError, match="Aucun texte à sauvegarder"):
            parser.save_text(tmp_path / "output.txt")

    def test_save_text_success(self, tmp_path):
        """Test sauvegarde réussie"""
        parser = CVParser()
        parser.text = "Python Developer with 5 years experience"
        output_path = tmp_path / "output.txt"
        result = parser.save_text(output_path)
        assert result is True
        assert output_path.read_text(encoding='utf-8') == parser.text

    @patch('pdfplumber.open')
    def test_parse_pdfplumber(self, mock_pdfplumber, tmp_path):
        """Test parsing avec pdfplumber"""
        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Python Machine Learning"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf

        # Créer un fichier PDF factice
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        parser = CVParser(method='pdfplumber')
        text = parser.parse(fake_pdf)

        assert "Python Machine Learning" in text
