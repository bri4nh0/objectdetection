-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS object_detection_db;

-- Use the database
USE object_detection_db;

-- Create the detections table
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    object_name VARCHAR(255),
    confidence FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
