-- ============================================================================
-- SCHÉMA BASE DE DONNÉES AI CAREER COACH
-- ============================================================================

-- Table des utilisateurs (optionnel pour l'instant)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des analyses de CV
CREATE TABLE IF NOT EXISTS cv_analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    cv_filename VARCHAR(255),
    cv_text TEXT,
    technical_skills TEXT[],  -- Array de compétences techniques
    soft_skills TEXT[],       -- Array de soft skills
    total_skills INTEGER,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des recommandations
CREATE TABLE IF NOT EXISTS job_recommendations (
    id SERIAL PRIMARY KEY,
    cv_analysis_id INTEGER REFERENCES cv_analyses(id),
    job_id VARCHAR(50),
    job_title VARCHAR(255),
    company VARCHAR(255),
    score DECIMAL(5,2),       -- Score de matching (0-100)
    skills_match DECIMAL(5,2),
    experience_match DECIMAL(5,2),
    location_match DECIMAL(5,2),
    competition_factor DECIMAL(5,2),
    matching_skills TEXT[],   -- Array des compétences matchées
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des simulations d'entretien
CREATE TABLE IF NOT EXISTS interview_simulations (
    id SERIAL PRIMARY KEY,
    cv_analysis_id INTEGER REFERENCES cv_analyses(id),
    job_id VARCHAR(50),
    questions JSONB,          -- Questions générées (format JSON)
    answers JSONB,            -- Réponses du candidat (format JSON)
    scores JSONB,             -- Scores détaillés (format JSON)
    average_score DECIMAL(5,2),
    simulated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances
CREATE INDEX idx_cv_analyses_user_id ON cv_analyses(user_id);
CREATE INDEX idx_recommendations_cv_id ON job_recommendations(cv_analysis_id);
CREATE INDEX idx_recommendations_score ON job_recommendations(score DESC);
CREATE INDEX idx_interviews_cv_id ON interview_simulations(cv_analysis_id);