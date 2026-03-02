-- ============================================================================
-- SCHÉMA BASE DE DONNÉES AI CAREER COACH
-- ============================================================================

-- Table des utilisateurs
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des analyses de CV
CREATE TABLE IF NOT EXISTS cv_analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cv_filename VARCHAR(255) NOT NULL,
    cv_text TEXT,
    technical_skills TEXT[] NOT NULL DEFAULT '{}',
    soft_skills TEXT[] NOT NULL DEFAULT '{}',
    total_skills INTEGER DEFAULT 0,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des recommandations
CREATE TABLE IF NOT EXISTS job_recommendations (
    id SERIAL PRIMARY KEY,
    cv_analysis_id INTEGER NOT NULL REFERENCES cv_analyses(id) ON DELETE CASCADE,
    job_id VARCHAR(50) NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    coverage DECIMAL(5,2) DEFAULT 0,
    quality DECIMAL(5,2) DEFAULT 0,
    matching_skills TEXT[] NOT NULL DEFAULT '{}',
    missing_skills TEXT[] NOT NULL DEFAULT '{}',
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des simulations d'entretien
CREATE TABLE IF NOT EXISTS interview_simulations (
    id SERIAL PRIMARY KEY,
    cv_analysis_id INTEGER NOT NULL REFERENCES cv_analyses(id) ON DELETE CASCADE,
    job_id VARCHAR(50) NOT NULL,
    rh_questions JSONB,
    technical_questions JSONB,
    answers JSONB,
    scores JSONB,
    average_score DECIMAL(5,2),
    simulated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Créer les indexes
CREATE INDEX idx_cv_analyses_user_id ON cv_analyses(user_id);
CREATE INDEX idx_cv_analyses_date ON cv_analyses(analyzed_at DESC);
CREATE INDEX idx_recommendations_cv_id ON job_recommendations(cv_analysis_id);
CREATE INDEX idx_recommendations_score ON job_recommendations(score DESC);
CREATE INDEX idx_recommendations_job_id ON job_recommendations(job_id);
CREATE INDEX idx_interviews_cv_id ON interview_simulations(cv_analysis_id);
CREATE INDEX idx_interviews_job_id ON interview_simulations(job_id);

-- table scraped_jobs

CREATE TABLE IF NOT EXISTS scraped_jobs (
    id                  SERIAL PRIMARY KEY,
    job_id              VARCHAR(255) UNIQUE NOT NULL,
    title               VARCHAR(500) NOT NULL,
    company             VARCHAR(255),
    location            VARCHAR(255),
    description         TEXT,
    url                 VARCHAR(1000),
    source              VARCHAR(50),        -- 'linkedin', 'indeed', etc.
    employment_type     VARCHAR(100),
    is_remote           BOOLEAN DEFAULT FALSE,
    salary_min          NUMERIC(10,2),
    salary_max          NUMERIC(10,2),
    required_skills     JSONB DEFAULT '[]',
    scraped_at          TIMESTAMP DEFAULT NOW(),
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scraped_jobs_title    ON scraped_jobs(title);
CREATE INDEX IF NOT EXISTS idx_scraped_jobs_source   ON scraped_jobs(source);
CREATE INDEX IF NOT EXISTS idx_scraped_jobs_remote   ON scraped_jobs(is_remote);
CREATE INDEX IF NOT EXISTS idx_scraped_jobs_scraped  ON scraped_jobs(scraped_at);

-- Insérer un user anonyme par défaut
INSERT INTO users (username, email) 
VALUES ('anonymous', 'anonymous@app.local')
ON CONFLICT (username) DO NOTHING;

-- Afficher le statut
SELECT 'PostgreSQL initialized successfully' as status;