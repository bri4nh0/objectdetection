-- Enhanced database schema for TRD-UQ system
USE model_evaluation;

-- Add TRD-UQ specific columns to existing experiment_log table
ALTER TABLE experiment_log 
ADD COLUMN uncertainty_total DECIMAL(5,3),
ADD COLUMN uncertainty_epistemic DECIMAL(5,3),
ADD COLUMN uncertainty_aleatoric DECIMAL(5,3),
ADD COLUMN risk_pattern VARCHAR(50),
ADD COLUMN confidence_level VARCHAR(20),
ADD COLUMN temporal_trend VARCHAR(50),
ADD COLUMN trd_uq_score DECIMAL(5,3);

-- Create table for TRD-UQ specific analysis
CREATE TABLE IF NOT EXISTS trd_uq_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    experiment_id VARCHAR(50),
    person_id INT,
    risk_score DECIMAL(5,3),
    adjusted_risk DECIMAL(5,3),
    uncertainty_total DECIMAL(5,3),
    uncertainty_epistemic DECIMAL(5,3),
    uncertainty_aleatoric DECIMAL(5,3),
    risk_pattern VARCHAR(50),
    confidence_level VARCHAR(20),
    object_risk DECIMAL(5,3),
    behavior_risk DECIMAL(5,3),
    proximity_risk DECIMAL(5,3),
    frame_id INT,
    scenario_context VARCHAR(100)
);

-- Create table for experiment configurations
CREATE TABLE IF NOT EXISTS experiment_configs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(50) UNIQUE,
    experiment_name VARCHAR(100),
    description TEXT,
    enabled_modules JSON,
    uq_enabled BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert experiment configurations
INSERT INTO experiment_configs (experiment_id, experiment_name, description, enabled_modules, uq_enabled) VALUES
('E1_PERFORMANCE', 'System Performance Analysis', 'FPS and latency analysis across system components', '["object", "pose", "behavior", "fusion"]', FALSE),
('E2_FUSION_COMPARISON', 'Fusion Method Comparison', 'Compare TRD-UQ fusion vs simple fusion methods', '["object", "pose", "behavior", "fusion"]', TRUE),
('E3_SCENARIO_ANALYSIS', 'Risk Scenario Analysis', 'Analysis of different risk scenarios and patterns', '["object", "pose", "behavior", "fusion"]', TRUE),
('E4_UNCERTAINTY_QUANTIFICATION', 'Uncertainty Quantification', 'Validation of uncertainty estimates in risk assessment', '["object", "pose", "behavior", "fusion"]', TRUE),
('E5_ABLATION_STUDY', 'Component Ablation Study', 'Study the impact of different system components', '["object", "pose", "behavior", "fusion"]', TRUE);

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS system_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    experiment_id VARCHAR(50),
    avg_fps DECIMAL(8,2),
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    object_detection_ms DECIMAL(10,3),
    pose_estimation_ms DECIMAL(10,3),
    behavior_analysis_ms DECIMAL(10,3),
    fusion_ms DECIMAL(10,3),
    total_frames_processed INT
);

-- Create risk pattern analysis table
CREATE TABLE IF NOT EXISTS risk_patterns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    experiment_id VARCHAR(50),
    person_id INT,
    pattern_type VARCHAR(50),
    pattern_duration INT,
    risk_level_start DECIMAL(5,3),
    risk_level_end DECIMAL(5,3),
    uncertainty_avg DECIMAL(5,3)
);