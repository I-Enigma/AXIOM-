-- ============================================================
--  AXIOM Face Recognition Attendance System
--  MySQL Database Schema
--
--  HOW TO RUN:
--    Option A (terminal):  mysql -u root -p < axiom_database.sql
--    Option B (Workbench):  Open file → Run (Ctrl+Shift+Enter)
-- ============================================================

CREATE DATABASE IF NOT EXISTS axiom_attendance;
USE axiom_attendance;

-- ============================================================
--  TABLE: teachers
-- ============================================================
CREATE TABLE IF NOT EXISTS teachers (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(100) NOT NULL UNIQUE,
    department      VARCHAR(60),
    password_hash   VARCHAR(255) NOT NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  TABLE: students
-- ============================================================
CREATE TABLE IF NOT EXISTS students (
    id              INT AUTO_INCREMENT PRIMARY KEY,

    -- Step 1: Personal Details
    first_name      VARCHAR(50)  NOT NULL,
    last_name       VARCHAR(50)  NOT NULL,
    email           VARCHAR(100) NOT NULL UNIQUE,
    phone           VARCHAR(20),
    dob             DATE,

    -- Step 2: Academic Details
    roll_number     VARCHAR(30)  NOT NULL UNIQUE,
    branch          VARCHAR(60)  NOT NULL,
    year            VARCHAR(20)  NOT NULL,
    section         VARCHAR(5),
    admission_year  YEAR,

    -- Step 3: Credentials
    username        VARCHAR(60)  NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,

    -- Status: pending → teacher verifies → verified/rejected
    status          ENUM('pending','verified','rejected') NOT NULL DEFAULT 'pending',
    verified_by     INT,    -- FK to teachers.id (NULL until approved)

    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_username (username),
    INDEX idx_roll     (roll_number),
    INDEX idx_status   (status)
);

ALTER TABLE students
    ADD CONSTRAINT fk_verified_by
    FOREIGN KEY (verified_by) REFERENCES teachers(id)
    ON DELETE SET NULL;

-- ============================================================
--  TABLE: attendance_sessions  (created by teacher per class)
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance_sessions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id  INT         NOT NULL,
    subject     VARCHAR(80),
    class_date  DATE        NOT NULL,
    started_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at    DATETIME,
    FOREIGN KEY (teacher_id) REFERENCES teachers(id)
);

-- ============================================================
--  TABLE: attendance_records  (filled by face recognition)
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance_records (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    session_id  INT NOT NULL,
    student_id  INT NOT NULL,
    status      ENUM('present','absent','late') NOT NULL DEFAULT 'absent',
    marked_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(id),
    FOREIGN KEY (student_id) REFERENCES students(id),
    UNIQUE KEY uq_session_student (session_id, student_id)
);

-- ============================================================
--  SEED: Default teacher account
--  Email: teacher@college.edu   |   Password: teacher123
--  (Change password in production!)
-- ============================================================
INSERT INTO teachers (name, email, department, password_hash) VALUES (
    'Prof. Rajesh Kumar',
    'teacher@college.edu',
    'Computer Science',
    '$2b$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LzTFONlAWK2'
);

SELECT 'AXIOM database created successfully!' AS message;