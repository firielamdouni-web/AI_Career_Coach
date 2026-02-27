"""
Tests unitaires - DatabaseManager (PostgreSQL mocké)
"""
import pytest
from unittest.mock import patch, MagicMock, call
from src.database import DatabaseManager, get_db_manager, close_db_connection
import psycopg2


MOCK_DATABASE_URL = "postgresql://user:password@localhost:5432/testdb"

MOCK_CV_DATA = {
    "cv_filename": "cv_test.pdf",
    "cv_text": "Python developer with 2 years experience.",
    "technical_skills": ["Python", "SQL", "Docker"],
    "soft_skills": ["Communication", "Teamwork"],
    "user_id": 1
}

MOCK_JOB_REC_DATA = {
    "cv_analysis_id": 1,
    "job_id": "job_001",
    "job_title": "Data Scientist",
    "company": "TechCorp",
    "score": 75.5,
    "coverage": 80.0,
    "quality": 71.0,
    "matching_skills": ["Python", "SQL"],
    "missing_skills": ["Docker"]
}

MOCK_INTERVIEW_DATA = {
    "cv_analysis_id": 1,
    "job_id": "job_001",
    "rh_questions": [{"id": 1, "question": "Parlez-moi de vous."}],
    "technical_questions": [{"id": 2, "question": "Expliquez Python."}],
    "answers": [],
    "scores": [],
    "average_score": 0.0
}


@pytest.fixture
def db_manager():
    """Fixture DatabaseManager avec psycopg2 entièrement mocké"""
    with patch('src.database.psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_conn.closed = False
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        manager = DatabaseManager(MOCK_DATABASE_URL)
        manager.conn = mock_conn
        manager.cursor = mock_cursor

        yield manager


# ============================================================================
# TESTS INIT
# ============================================================================

class TestDatabaseManagerInit:

    def test_init_stores_database_url(self, db_manager):
        assert db_manager.database_url == MOCK_DATABASE_URL

    def test_init_connection_established(self, db_manager):
        assert db_manager.conn is not None

    def test_init_cursor_created(self, db_manager):
        assert db_manager.cursor is not None

    def test_init_connection_error_raises(self):
        with patch('src.database.psycopg2.connect',
                   side_effect=psycopg2.Error("Connection refused")):
            with pytest.raises(ConnectionError):
                DatabaseManager(MOCK_DATABASE_URL)


# ============================================================================
# TESTS SAVE CV ANALYSIS
# ============================================================================

class TestSaveCVAnalysis:

    def test_save_cv_returns_id(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 42}

        cv_id = db_manager.save_cv_analysis(**MOCK_CV_DATA)
        assert cv_id == 42

    def test_save_cv_calls_insert(self, db_manager):
        """execute est appelé 2 fois : SELECT 1 (health) + INSERT"""
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_cv_analysis(**MOCK_CV_DATA)

        # Récupérer tous les appels et vérifier qu'un INSERT est présent
        all_calls = db_manager.cursor.execute.call_args_list
        sql_calls = [str(c) for c in all_calls]
        assert any("INSERT INTO cv_analyses" in s for s in sql_calls)

    def test_save_cv_commits(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_cv_analysis(**MOCK_CV_DATA)
        db_manager.conn.commit.assert_called_once()

    def test_save_cv_rollback_on_error(self, db_manager):
        """Simuler une erreur sur INSERT uniquement (pas sur SELECT 1)"""
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")  # INSERT → KO

        db_manager.cursor.execute.side_effect = execute_side_effect

        with pytest.raises(psycopg2.Error):
            db_manager.save_cv_analysis(**MOCK_CV_DATA)

        db_manager.conn.rollback.assert_called_once()

    def test_save_cv_calculates_total_skills(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_cv_analysis(**MOCK_CV_DATA)

        # Trouver l'appel INSERT parmi tous les appels execute
        all_calls = db_manager.cursor.execute.call_args_list
        insert_call = next(
            c for c in all_calls
            if "INSERT INTO cv_analyses" in str(c)
        )
        call_args = insert_call[0][1]  # 2e arg positionnel = tuple de valeurs
        total_expected = len(MOCK_CV_DATA["technical_skills"]) + len(MOCK_CV_DATA["soft_skills"])
        assert total_expected in call_args


# ============================================================================
# TESTS SAVE JOB RECOMMENDATION
# ============================================================================

class TestSaveJobRecommendation:

    def test_save_recommendation_returns_id(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 10}

        rec_id = db_manager.save_job_recommendation(**MOCK_JOB_REC_DATA)
        assert rec_id == 10

    def test_save_recommendation_calls_insert(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_job_recommendation(**MOCK_JOB_REC_DATA)

        all_calls = db_manager.cursor.execute.call_args_list
        assert any("INSERT INTO job_recommendations" in str(c) for c in all_calls)

    def test_save_recommendation_commits(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_job_recommendation(**MOCK_JOB_REC_DATA)
        db_manager.conn.commit.assert_called_once()

    def test_save_recommendation_rollback_on_error(self, db_manager):
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")  # INSERT → KO

        db_manager.cursor.execute.side_effect = execute_side_effect

        with pytest.raises(psycopg2.Error):
            db_manager.save_job_recommendation(**MOCK_JOB_REC_DATA)

        db_manager.conn.rollback.assert_called_once()

    def test_save_recommendation_rounds_score(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        data = {**MOCK_JOB_REC_DATA, "score": 75.123456}
        db_manager.save_job_recommendation(**data)

        all_calls = db_manager.cursor.execute.call_args_list
        insert_call = next(
            c for c in all_calls
            if "INSERT INTO job_recommendations" in str(c)
        )
        call_args = insert_call[0][1]
        assert round(75.123456, 2) in call_args


# ============================================================================
# TESTS GET RECENT CV ANALYSES
# ============================================================================

class TestGetRecentCVAnalyses:

    def test_returns_list(self, db_manager):
        db_manager.cursor.fetchall.return_value = []

        result = db_manager.get_recent_cv_analyses()
        assert isinstance(result, list)

    def test_returns_correct_count(self, db_manager):
        db_manager.cursor.fetchall.return_value = [
            {"id": 1, "cv_filename": "cv1.pdf", "total_skills": 5},
            {"id": 2, "cv_filename": "cv2.pdf", "total_skills": 8}
        ]

        result = db_manager.get_recent_cv_analyses(limit=10)
        assert len(result) == 2

    def test_returns_empty_on_error(self, db_manager):
        """Simuler erreur uniquement sur SELECT query (pas SELECT 1)"""
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")  # vraie query → KO

        db_manager.cursor.execute.side_effect = execute_side_effect

        result = db_manager.get_recent_cv_analyses()
        assert result == []


# ============================================================================
# TESTS GET RECOMMENDATIONS FOR CV
# ============================================================================

class TestGetRecommendationsForCV:

    def test_returns_list(self, db_manager):
        db_manager.cursor.fetchall.return_value = []

        result = db_manager.get_recommendations_for_cv(cv_analysis_id=1)
        assert isinstance(result, list)

    def test_returns_empty_on_error(self, db_manager):
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")

        db_manager.cursor.execute.side_effect = execute_side_effect

        result = db_manager.get_recommendations_for_cv(cv_analysis_id=1)
        assert result == []


# ============================================================================
# TESTS SAVE INTERVIEW SIMULATION
# ============================================================================

class TestSaveInterviewSimulation:

    def test_save_simulation_returns_id(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 5}

        sim_id = db_manager.save_interview_simulation(**MOCK_INTERVIEW_DATA)
        assert sim_id == 5

    def test_save_simulation_calls_insert(self, db_manager):
        db_manager.cursor.fetchone.return_value = {"id": 1}

        db_manager.save_interview_simulation(**MOCK_INTERVIEW_DATA)

        all_calls = db_manager.cursor.execute.call_args_list
        assert any("INSERT INTO interview_simulations" in str(c) for c in all_calls)

    def test_save_simulation_rollback_on_error(self, db_manager):
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")

        db_manager.cursor.execute.side_effect = execute_side_effect

        with pytest.raises(psycopg2.Error):
            db_manager.save_interview_simulation(**MOCK_INTERVIEW_DATA)

        db_manager.conn.rollback.assert_called_once()


# ============================================================================
# TESTS GET STATISTICS
# ============================================================================

class TestGetStatistics:

    def test_returns_dict(self, db_manager):
        db_manager.cursor.fetchone.side_effect = [
            {"count": 10},
            {"count": 50},
            {"count": 5},
            {"avg_score": 72.5}
        ]

        result = db_manager.get_statistics()
        assert isinstance(result, dict)

    def test_contains_required_keys(self, db_manager):
        db_manager.cursor.fetchone.side_effect = [
            {"count": 10},
            {"count": 50},
            {"count": 5},
            {"avg_score": 72.5}
        ]

        result = db_manager.get_statistics()
        assert "total_cv_analyses" in result
        assert "total_recommendations" in result
        assert "total_simulations" in result
        assert "average_score" in result

    def test_returns_empty_dict_on_error(self, db_manager):
        call_count = 0

        def execute_side_effect(sql, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # SELECT 1 → OK
            raise psycopg2.Error("DB error")

        db_manager.cursor.execute.side_effect = execute_side_effect

        result = db_manager.get_statistics()
        assert result == {}


# ============================================================================
# TESTS DISCONNECT
# ============================================================================

class TestDisconnect:

    def test_disconnect_closes_connection(self, db_manager):
        db_manager.disconnect()
        db_manager.conn.close.assert_called_once()
        db_manager.cursor.close.assert_called_once()


# ============================================================================
# TESTS SINGLETON
# ============================================================================

class TestGetDbManagerSingleton:

    def test_raises_without_database_url(self):
        import src.database as module
        module._db_manager = None

        with patch('src.database.os.getenv', return_value=None):
            with pytest.raises(ValueError, match="DATABASE_URL"):
                get_db_manager()

    def test_singleton_returns_same_instance(self):
        import src.database as module
        module._db_manager = None

        with patch('src.database.psycopg2.connect') as mock_connect, \
             patch('src.database.os.getenv', return_value=MOCK_DATABASE_URL):
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_conn.cursor.return_value = MagicMock()
            mock_connect.return_value = mock_conn

            db1 = get_db_manager()
            db2 = get_db_manager()
            assert db1 is db2
            module._db_manager = None

    def test_close_db_resets_singleton(self):
        import src.database as module

        mock_manager = MagicMock()
        module._db_manager = mock_manager

        close_db_connection()

        assert module._db_manager is None
        mock_manager.disconnect.assert_called_once()