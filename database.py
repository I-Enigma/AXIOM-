-- ============================================================
--  AXIOM – Face Recognition Attendance System
--  MySQL Database Schema
--  Version: 1.0
-- ============================================================

-- Create and use the database
CREATE DATABASE IF NOT EXISTS axiom_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE axiom_db;

-- ============================================================
-- TABLE 1: organizations
-- An organization can be a company, institution, or any group
-- ============================================================
CREATE TABLE IF NOT EXISTS organizations (
  id            INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  name          VARCHAR(150)      NOT NULL,
  type          ENUM('institution','company','ngo','other') DEFAULT 'institution',
  contact_email VARCHAR(150)      DEFAULT NULL,
  created_at    TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 2: teachers
-- Teachers / managers who verify students and manage groups
-- ============================================================
CREATE TABLE IF NOT EXISTS teachers (
  id                INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  org_id            INT UNSIGNED      DEFAULT NULL,
  full_name         VARCHAR(100)      NOT NULL,
  email             VARCHAR(150)      NOT NULL UNIQUE,
  password_hash     VARCHAR(255)      NOT NULL,          -- bcrypt hash
  phone             VARCHAR(20)       DEFAULT NULL,
  assigned_dept     VARCHAR(100)      DEFAULT NULL,      -- department/field they manage
  assigned_group    CHAR(5)           DEFAULT NULL,      -- group/section (A, B, C…)
  is_active         TINYINT(1)        NOT NULL DEFAULT 1,
  last_login        TIMESTAMP         DEFAULT NULL,
  created_at        TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at        TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT fk_teacher_org FOREIGN KEY (org_id)
    REFERENCES organizations(id) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 3: students
-- Core user table – everyone who registers via the AXIOM portal
-- ============================================================
CREATE TABLE IF NOT EXISTS students (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  org_id          INT UNSIGNED      DEFAULT NULL,
  mentor_id       INT UNSIGNED      DEFAULT NULL,        -- assigned teacher/manager

  -- Personal info
  first_name      VARCHAR(60)       NOT NULL,
  last_name       VARCHAR(60)       NOT NULL,
  email           VARCHAR(150)      NOT NULL UNIQUE,
  phone           VARCHAR(20)       DEFAULT NULL,
  date_of_birth   DATE              DEFAULT NULL,

  -- Academic / professional info
  roll_no         VARCHAR(30)       NOT NULL UNIQUE,     -- employee/roll/ID number
  department      VARCHAR(100)      DEFAULT NULL,
  year_level      VARCHAR(30)       DEFAULT NULL,        -- "2nd Year", "Senior", "Staff"
  group_section   CHAR(5)           DEFAULT NULL,        -- A / B / C / D

  -- Credentials
  username        VARCHAR(60)       NOT NULL UNIQUE,
  password_hash   VARCHAR(255)      NOT NULL,            -- bcrypt hash

  -- Face recognition
  face_encoding   LONGTEXT          DEFAULT NULL,        -- JSON array of 128-d face vector
  face_registered TINYINT(1)        NOT NULL DEFAULT 0,

  -- Account status
  status          ENUM('pending','verified','rejected','suspended')
                                    NOT NULL DEFAULT 'pending',
  verified_at     TIMESTAMP         DEFAULT NULL,
  verified_by     INT UNSIGNED      DEFAULT NULL,        -- teacher id who approved

  -- Meta
  last_login      TIMESTAMP         DEFAULT NULL,
  created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  CONSTRAINT fk_student_org     FOREIGN KEY (org_id)
    REFERENCES organizations(id) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT fk_student_mentor  FOREIGN KEY (mentor_id)
    REFERENCES teachers(id)      ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT fk_student_verified_by FOREIGN KEY (verified_by)
    REFERENCES teachers(id)      ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 4: verification_requests
-- Every time a student registers, a row is created here.
-- The teacher acts on this row to approve / reject.
-- ============================================================
CREATE TABLE IF NOT EXISTS verification_requests (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  student_id      INT UNSIGNED      NOT NULL,
  teacher_id      INT UNSIGNED      NOT NULL,            -- teacher to be notified
  status          ENUM('pending','approved','rejected')
                                    NOT NULL DEFAULT 'pending',
  teacher_note    TEXT              DEFAULT NULL,        -- optional rejection reason
  notified_at     TIMESTAMP         DEFAULT NULL,        -- when email was sent
  acted_at        TIMESTAMP         DEFAULT NULL,        -- when teacher approved/rejected
  created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  CONSTRAINT fk_vreq_student  FOREIGN KEY (student_id)
    REFERENCES students(id)  ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT fk_vreq_teacher  FOREIGN KEY (teacher_id)
    REFERENCES teachers(id)  ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 5: notifications
-- In-app notifications shown on the teacher dashboard
-- ============================================================
CREATE TABLE IF NOT EXISTS notifications (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  recipient_id    INT UNSIGNED      NOT NULL,            -- teacher id
  type            ENUM('new_registration','system','info')
                                    NOT NULL DEFAULT 'new_registration',
  title           VARCHAR(150)      NOT NULL,
  message         TEXT              NOT NULL,
  is_read         TINYINT(1)        NOT NULL DEFAULT 0,
  reference_id    INT UNSIGNED      DEFAULT NULL,        -- e.g. verification_request id
  created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  CONSTRAINT fk_notif_teacher FOREIGN KEY (recipient_id)
    REFERENCES teachers(id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 6: attendance_sessions
-- A teacher creates a session (e.g. "Morning Lecture – 9 AM")
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance_sessions (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  teacher_id      INT UNSIGNED      NOT NULL,
  org_id          INT UNSIGNED      DEFAULT NULL,
  session_name    VARCHAR(150)      NOT NULL,
  department      VARCHAR(100)      DEFAULT NULL,
  group_section   CHAR(5)           DEFAULT NULL,
  started_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ended_at        TIMESTAMP         DEFAULT NULL,
  is_active       TINYINT(1)        NOT NULL DEFAULT 1,

  PRIMARY KEY (id),
  CONSTRAINT fk_session_teacher FOREIGN KEY (teacher_id)
    REFERENCES teachers(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT fk_session_org     FOREIGN KEY (org_id)
    REFERENCES organizations(id) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 7: attendance_records
-- Each row = one student marked present in a session
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance_records (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  session_id      INT UNSIGNED      NOT NULL,
  student_id      INT UNSIGNED      NOT NULL,
  marked_at       TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  method          ENUM('face','manual','qr') NOT NULL DEFAULT 'face',
  confidence      DECIMAL(5,2)      DEFAULT NULL,        -- face match confidence %
  is_present      TINYINT(1)        NOT NULL DEFAULT 1,

  PRIMARY KEY (id),
  UNIQUE KEY uq_session_student (session_id, student_id),
  CONSTRAINT fk_att_session  FOREIGN KEY (session_id)
    REFERENCES attendance_sessions(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT fk_att_student  FOREIGN KEY (student_id)
    REFERENCES students(id)           ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 8: password_reset_tokens
-- Temporary tokens for "Forgot Password" flow
-- ============================================================
CREATE TABLE IF NOT EXISTS password_reset_tokens (
  id              INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  user_type       ENUM('student','teacher') NOT NULL,
  user_id         INT UNSIGNED      NOT NULL,
  token           VARCHAR(255)      NOT NULL UNIQUE,     -- hashed random token
  expires_at      TIMESTAMP         NOT NULL,
  used            TINYINT(1)        NOT NULL DEFAULT 0,
  created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 9: audit_log
-- Track important actions (verifications, logins, changes)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
  id              BIGINT UNSIGNED   NOT NULL AUTO_INCREMENT,
  actor_type      ENUM('student','teacher','system') NOT NULL,
  actor_id        INT UNSIGNED      DEFAULT NULL,
  action          VARCHAR(80)       NOT NULL,            -- e.g. 'student.verified'
  target_table    VARCHAR(60)       DEFAULT NULL,
  target_id       INT UNSIGNED      DEFAULT NULL,
  details         JSON              DEFAULT NULL,        -- extra context
  ip_address      VARCHAR(45)       DEFAULT NULL,
  created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  INDEX idx_audit_actor  (actor_type, actor_id),
  INDEX idx_audit_action (action),
  INDEX idx_audit_date   (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- INDEXES for performance
-- ============================================================
CREATE INDEX idx_student_status     ON students(status);
CREATE INDEX idx_student_mentor     ON students(mentor_id);
CREATE INDEX idx_student_dept       ON students(department);
CREATE INDEX idx_vreq_teacher_status ON verification_requests(teacher_id, status);
CREATE INDEX idx_notif_recipient    ON notifications(recipient_id, is_read);
CREATE INDEX idx_att_session        ON attendance_records(session_id);
CREATE INDEX idx_att_student        ON attendance_records(student_id);


-- ============================================================
-- SAMPLE SEED DATA (optional – remove in production)
-- ============================================================

-- Default organization
INSERT INTO organizations (name, type, contact_email)
VALUES ('AXIOM Demo Organization', 'institution', 'admin@axiom.app');

-- Default teacher (password: Teacher@123 — bcrypt hashed below)
INSERT INTO teachers (org_id, full_name, email, password_hash, assigned_dept, assigned_group)
VALUES (
  1,
  'Prof. Rajesh Kumar',
  'rajesh@axiom.app',
  '$2b$12$Gv8XNx4Mj3qP5zDHfC2mKuRt7lYsOeWnBpA0cI1dE6hM9Tj3vFXya',
  'Computer Science',
  'A'
);

-- Demo student (password: Student@123 — bcrypt hashed below, status = verified)
INSERT INTO students (
  org_id, mentor_id, first_name, last_name, email, phone,
  roll_no, department, year_level, group_section,
  username, password_hash, status, verified_at, verified_by
) VALUES (
  1, 1, 'Arjun', 'Sharma', 'arjun@axiom.app', '+91-9876543210',
  'CS2024001', 'Computer Science', '2nd Year', 'A',
  'arjun_sh_0012024',
  '$2b$12$Kv9YNx5Lj4rQ6aDIfD3nLvSu8mZtPfXoBqB1dJ2eF7iN0Uk4wGYzb',
  'verified', NOW(), 1
);

-- ============================================================
-- VIEWS
-- ============================================================

-- View: pending students with their assigned teacher info
CREATE OR REPLACE VIEW v_pending_students AS
SELECT
  s.id                            AS student_id,
  CONCAT(s.first_name,' ',s.last_name) AS full_name,
  s.email,
  s.roll_no,
  s.department,
  s.year_level,
  s.group_section,
  s.username,
  s.created_at                    AS registered_at,
  t.id                            AS teacher_id,
  t.full_name                     AS teacher_name,
  t.email                         AS teacher_email,
  vr.id                           AS verification_request_id,
  vr.status                       AS verification_status
FROM students s
LEFT JOIN teachers t              ON t.id = s.mentor_id
LEFT JOIN verification_requests vr ON vr.student_id = s.id
WHERE s.status = 'pending';


-- View: attendance summary per student per session
CREATE OR REPLACE VIEW v_attendance_summary AS
SELECT
  sess.id                          AS session_id,
  sess.session_name,
  sess.started_at,
  s.id                             AS student_id,
  CONCAT(s.first_name,' ',s.last_name) AS student_name,
  s.roll_no,
  s.department,
  ar.marked_at,
  ar.method,
  ar.confidence,
  ar.is_present
FROM attendance_sessions sess
JOIN attendance_records ar   ON ar.session_id = sess.id
JOIN students s              ON s.id = ar.student_id;


-- ============================================================
-- STORED PROCEDURE: approve_student
-- Call: CALL approve_student(student_id, teacher_id);
-- ============================================================
DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS approve_student(
  IN p_student_id  INT UNSIGNED,
  IN p_teacher_id  INT UNSIGNED
)
BEGIN
  -- Update student status
  UPDATE students
  SET
    status      = 'verified',
    verified_at = NOW(),
    verified_by = p_teacher_id
  WHERE id = p_student_id;

  -- Update verification request
  UPDATE verification_requests
  SET
    status   = 'approved',
    acted_at = NOW()
  WHERE student_id = p_student_id
    AND teacher_id = p_teacher_id
    AND status     = 'pending';

  -- Log the action
  INSERT INTO audit_log (actor_type, actor_id, action, target_table, target_id)
  VALUES ('teacher', p_teacher_id, 'student.verified', 'students', p_student_id);

  -- Mark notification as read
  UPDATE notifications
  SET is_read = 1
  WHERE recipient_id  = p_teacher_id
    AND reference_id  = (
      SELECT id FROM verification_requests
      WHERE student_id = p_student_id LIMIT 1
    );
END$$

DELIMITER ;


-- ============================================================
-- STORED PROCEDURE: register_student
-- Call from your backend after hashing the password
-- ============================================================
DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS register_student(
  IN p_org_id        INT UNSIGNED,
  IN p_first_name    VARCHAR(60),
  IN p_last_name     VARCHAR(60),
  IN p_email         VARCHAR(150),
  IN p_phone         VARCHAR(20),
  IN p_dob           DATE,
  IN p_roll_no       VARCHAR(30),
  IN p_department    VARCHAR(100),
  IN p_year_level    VARCHAR(30),
  IN p_group_section CHAR(5),
  IN p_username      VARCHAR(60),
  IN p_password_hash VARCHAR(255)
)
BEGIN
  DECLARE v_student_id  INT UNSIGNED;
  DECLARE v_teacher_id  INT UNSIGNED;
  DECLARE v_vreq_id     INT UNSIGNED;

  -- Insert the new student (status defaults to 'pending')
  INSERT INTO students (
    org_id, first_name, last_name, email, phone, date_of_birth,
    roll_no, department, year_level, group_section,
    username, password_hash
  ) VALUES (
    p_org_id, p_first_name, p_last_name, p_email, p_phone, p_dob,
    p_roll_no, p_department, p_year_level, p_group_section,
    p_username, p_password_hash
  );

  SET v_student_id = LAST_INSERT_ID();

  -- Find the right teacher (same dept + group in the org)
  SELECT id INTO v_teacher_id
  FROM teachers
  WHERE org_id          = p_org_id
    AND assigned_dept   = p_department
    AND assigned_group  = p_group_section
  LIMIT 1;

  -- Fallback: any teacher in the org
  IF v_teacher_id IS NULL THEN
    SELECT id INTO v_teacher_id
    FROM teachers WHERE org_id = p_org_id LIMIT 1;
  END IF;

  -- Assign mentor
  IF v_teacher_id IS NOT NULL THEN
    UPDATE students SET mentor_id = v_teacher_id WHERE id = v_student_id;

    -- Create verification request
    INSERT INTO verification_requests (student_id, teacher_id, notified_at)
    VALUES (v_student_id, v_teacher_id, NOW());

    SET v_vreq_id = LAST_INSERT_ID();

    -- Create in-app notification for teacher
    INSERT INTO notifications (recipient_id, type, title, message, reference_id)
    VALUES (
      v_teacher_id,
      'new_registration',
      CONCAT('New Registration: ', p_first_name, ' ', p_last_name),
      CONCAT('Student ', p_first_name, ' ', p_last_name,
             ' (Roll: ', p_roll_no, ') has registered and needs your verification.'),
      v_vreq_id
    );
  END IF;

  -- Audit log
  INSERT INTO audit_log (actor_type, actor_id, action, target_table, target_id)
  VALUES ('student', v_student_id, 'student.registered', 'students', v_student_id);

  -- Return the new student's id and assigned teacher
  SELECT v_student_id AS new_student_id, v_teacher_id AS assigned_teacher_id;
END$$

DELIMITER ;

-- ============================================================
-- END OF SCHEMA
-- ============================================================