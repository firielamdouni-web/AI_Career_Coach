"""
Module de parsing de CV PDF
"""
from pathlib import Path
import PyPDF2
import pdfplumber


class CVParser:
    """Parser de CV PDF avec choix de méthode"""

    def __init__(self, method='pdfplumber'):
        """
        Initialiser le parser
        Args:
            method: 'pdfplumber' (par défaut) ou 'pypdf2'
        """
        self.method = method
        self.text = ""

    def parse(self, pdf_path):
        """Parser un fichier PDF"""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {pdf_path}")

        if self.method == 'pypdf2':
            self.text = self._parse_with_pypdf2(pdf_path)
        elif self.method == 'pdfplumber':
            self.text = self._parse_with_pdfplumber(pdf_path)
        else:
            raise ValueError(f"Méthode non supportée : {self.method}")

        return self.text

    def _parse_with_pypdf2(self, pdf_path):
        """Extraire texte avec PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
        return text.strip()

    def _parse_with_pdfplumber(self, pdf_path):
        """Extraire texte avec pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()

    def get_stats(self):
        """Obtenir des statistiques sur le texte extrait"""
        if not self.text:
            return {"error": "Aucun texte extrait"}

        return {
            "characters": len(self.text),
            "words": len(self.text.split()),
            "lines": len(self.text.split('\n')),
            "method": self.method
        }

    def save_text(self, output_path):
        """Sauvegarder le texte extrait"""
        if not self.text:
            raise ValueError("Aucun texte à sauvegarder")

        Path(output_path).write_text(self.text, encoding='utf-8')
        return True
