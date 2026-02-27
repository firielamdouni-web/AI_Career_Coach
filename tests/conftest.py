"""
Configuration pytest - Fixtures partagées
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Retourne la racine du projet"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Retourne le dossier data"""
    return project_root / "data"


@pytest.fixture(scope="session")
def skills_db_path(data_dir):
    """Retourne le chemin vers skills_reference.json"""
    return data_dir / "skills_reference.json"