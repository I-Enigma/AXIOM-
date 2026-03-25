import sqlite3
import os
from datetime import datetime

DB_PATH = "axiom.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# ══════════════════════════════════════════
# CREATE ALL TABLES
# ══════════════════════════════════════════
def create_tables():
    conn = get_conn()
    cursor = conn.cursor()

    # ── TEACHERS ──────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS teachers (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id  TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        password    TEXT NOT NULL,
        department  TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )""")

    # ── STUDENTS ──────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id      TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        password        TEXT NOT NULL,
        batch           TEXT,
        semester        TEXT,
        department      TEXT,
        face_registered INTEGER DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now'))
    )""")

    # ── FACE PROFILES ─────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_profiles (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id       TEXT UNIQUE NOT NULL,
        golden_ratio     REAL,
        eye_ratio        REAL,
        inner_eye_ratio  REAL,
        nose_ratio       REAL,
        lip_ratio        REAL,
        jaw_ratio        REAL,
        brow_ratio       REAL,
        nose_width_ratio REAL,
        fwhr             REAL,
        symmetry         REAL,
        total_samples    INTEGER DEFAULT 40,
        registered_on    TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    )""")

    # ── BATCHES ───────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batches (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id   TEXT UNIQUE NOT NULL,
        name       TEXT NOT NULL,
        year       TEXT,
        semester   TEXT,
        strength   INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )""")

    # ── SUBJECTS ──────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT UNIQUE NOT NULL,
        name       TEXT NOT NULL,
        code       TEXT,
        color      TEXT DEFAULT '#00ff88',
        created_at TEXT DEFAULT (datetime('now'))
    )""")

    # ── STUDENT-BATCH MAPPING ─────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS student_batch (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        batch_id   TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (batch_id)   REFERENCES batches(batch_id)
    )""")

    # ── LECTURES ──────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lectures (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        lecture_id    TEXT UNIQUE NOT NULL,
        subject_id    TEXT NOT NULL,
        batch_id      TEXT NOT NULL,
        teacher_id    TEXT NOT NULL,
        day           TEXT NOT NULL,
        start_time    TEXT NOT NULL,
        end_time      TEXT NOT NULL,
        late_after    TEXT,
        room          TEXT,
        status        TEXT DEFAULT 'upcoming',
        cancel_reason TEXT,
        lecture_date  TEXT,
        created_at    TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY (batch_id)   REFERENCES batches(batch_id),
        FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id)
    )""")

    # ── ATTENDANCE SESSIONS ───────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_sessions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        lecture_id TEXT NOT NULL,
        teacher_id TEXT NOT NULL,
        opened_at  TEXT,
        closed_at  TEXT,
        status     TEXT DEFAULT 'open',
        FOREIGN KEY (lecture_id) REFERENCES lectures(lecture_id),
        FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id)
    )""")

    # ── ATTENDANCE ────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id   TEXT NOT NULL,
        lecture_id   TEXT NOT NULL,
        subject_id   TEXT NOT NULL,
        batch_id     TEXT NOT NULL,
        lecture_date TEXT NOT NULL,
        lecture_time TEXT,
        marked_at    TEXT,
        late_by_mins INTEGER DEFAULT 0,
        status       TEXT DEFAULT 'ABSENT',
        match_score  REAL,
        marked_by    TEXT DEFAULT 'student',
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (lecture_id) REFERENCES lectures(lecture_id)
    )""")

    # ── UNKNOWN PERSONS ───────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS unknown_persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at TEXT DEFAULT (datetime('now')),
        lecture_id  TEXT,
        session_id  TEXT,
        note        TEXT
    )""")

    conn.commit()
    conn.close()
    print("✅ All tables created successfully!")

# ══════════════════════════════════════════
# INSERT DEMO DATA
# ══════════════════════════════════════════
def insert_demo_data():
    conn = get_conn()
    cursor = conn.cursor()

    # Teachers
    cursor.execute("""
        INSERT OR IGNORE INTO teachers
        (teacher_id, name, password, department)
        VALUES (?,?,?,?)
    """, ('TCH001','Prof. Sharma','sharma123','Computer Science'))

    # Students
    students = [
        ('CS2024101','Rahul Sharma', 'axiom123','CS-2024-A','Sem 5','Computer Science'),
        ('CS2024102','Priya Patel',  'axiom123','CS-2024-A','Sem 5','Computer Science'),
        ('CS2024103','Amit Kumar',   'axiom123','CS-2024-B','Sem 5','Computer Science'),
        ('CS2024104','Sneha Joshi',  'axiom123','CS-2024-B','Sem 5','Computer Science'),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO students
        (student_id, name, password, batch, semester, department)
        VALUES (?,?,?,?,?,?)
    """, students)

    # Batches
    batches = [
        ('B001','CS-2024-A','3rd Year','Sem 5',60),
        ('B002','CS-2024-B','3rd Year','Sem 5',58),
        ('B003','IT-2024-A','2nd Year','Sem 3',55),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO batches
        (batch_id, name, year, semester, strength)
        VALUES (?,?,?,?,?)
    """, batches)

    # Subjects
    subjects = [
        ('S001','Data Structures',   'CS301','#a29bfe'),
        ('S002','Computer Networks', 'CS401','#74b9ff'),
        ('S003','Operating Systems', 'CS302','#fd79a8'),
        ('S004','Python Programming','IT201','#00b894'),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO subjects
        (subject_id, name, code, color)
        VALUES (?,?,?,?)
    """, subjects)

    # Student-Batch mapping
    mappings = [
        ('CS2024101','B001'),
        ('CS2024102','B001'),
        ('CS2024103','B002'),
        ('CS2024104','B002'),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO student_batch
        (student_id, batch_id) VALUES (?,?)
    """, mappings)

    # Lectures
    from datetime import date
    today = date.today().strftime('%Y-%m-%d')
    lectures = [
        ('LEC001','S001','B001','TCH001','Monday','09:00','10:00','09:10','Room 301','upcoming',today),
        ('LEC002','S002','B001','TCH001','Monday','10:00','11:00','10:10','Lab 2',   'upcoming',today),
        ('LEC003','S003','B001','TCH001','Monday','11:00','12:00','11:10','Room 201','upcoming',today),
        ('LEC004','S004','B002','TCH001','Tuesday','13:00','14:00','13:10','Lab 1',  'upcoming',today),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO lectures
        (lecture_id, subject_id, batch_id, teacher_id,
         day, start_time, end_time, late_after,
         room, status, lecture_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, lectures)

    conn.commit()
    conn.close()
    print("✅ Demo data inserted!")

# ══════════════════════════════════════════
# USEFUL FUNCTIONS
# ══════════════════════════════════════════

# ── Save face profile ──────────────────
def save_face_profile(student_id, profile):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO face_profiles
        (student_id, golden_ratio, eye_ratio, inner_eye_ratio,
         nose_ratio, lip_ratio, jaw_ratio, brow_ratio,
         nose_width_ratio, fwhr, symmetry, total_samples)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        student_id,
        profile['golden_ratio'],
        profile['eye_ratio'],
        profile['inner_eye_ratio'],
        profile['nose_ratio'],
        profile['lip_ratio'],
        profile['jaw_ratio'],
        profile['brow_ratio'],
        profile['nose_width_ratio'],
        profile['fWHR'],
        profile['symmetry'],
        profile.get('total_samples', 40)
    ))
    # Mark student as face registered
    cursor.execute("""
        UPDATE students SET face_registered=1
        WHERE student_id=?
    """, (student_id,))
    conn.commit()
    conn.close()
    print(f"✅ Face profile saved for {student_id}")

# ── Load all face profiles ─────────────
def load_all_profiles():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fp.*, s.name
        FROM face_profiles fp
        JOIN students s ON fp.student_id = s.student_id
    """)
    rows = cursor.fetchall()
    conn.close()
    profiles = {}
    for row in rows:
        profiles[row['student_id']] = {
            'name'            : row['name'],
            'profile'         : {
                'golden_ratio'    : row['golden_ratio'],
                'eye_ratio'       : row['eye_ratio'],
                'inner_eye_ratio' : row['inner_eye_ratio'],
                'nose_ratio'      : row['nose_ratio'],
                'lip_ratio'       : row['lip_ratio'],
                'jaw_ratio'       : row['jaw_ratio'],
                'brow_ratio'      : row['brow_ratio'],
                'nose_width_ratio': row['nose_width_ratio'],
                'fWHR'            : row['fwhr'],
                'symmetry'        : row['symmetry'],
            }
        }
    return profiles

# ── Mark attendance ────────────────────
def mark_attendance(student_id, lecture_id, subject_id,
                    batch_id, status, match_score, late_mins=0):
    conn = get_conn()
    cursor = conn.cursor()
    from datetime import date, datetime
    cursor.execute("""
        INSERT OR REPLACE INTO attendance
        (student_id, lecture_id, subject_id, batch_id,
         lecture_date, marked_at, late_by_mins,
         status, match_score, marked_by)
        VALUES (?,?,?,?,?,?,?,?,?,'student')
    """, (
        student_id, lecture_id, subject_id, batch_id,
        date.today().strftime('%Y-%m-%d'),
        datetime.now().strftime('%H:%M:%S'),
        late_mins, status, match_score
    ))
    conn.commit()
    conn.close()
    print(f"✅ {student_id} marked {status}")

# ── Open session (teacher) ─────────────
def open_session(lecture_id, teacher_id):
    conn = get_conn()
    cursor = conn.cursor()
    import uuid
    session_id = 'SESS_' + str(uuid.uuid4())[:8].upper()
    cursor.execute("""
        INSERT INTO attendance_sessions
        (session_id, lecture_id, teacher_id, opened_at, status)
        VALUES (?,?,?,datetime('now'),'open')
    """, (session_id, lecture_id, teacher_id))
    cursor.execute("""
        UPDATE lectures SET status='open'
        WHERE lecture_id=?
    """, (lecture_id,))
    conn.commit()
    conn.close()
    print(f"✅ Session opened: {session_id}")
    return session_id

# ── Close session (teacher) ────────────
def close_session(session_id, lecture_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE attendance_sessions
        SET status='closed', closed_at=datetime('now')
        WHERE session_id=?
    """, (session_id,))
    cursor.execute("""
        UPDATE lectures SET status='closed'
        WHERE lecture_id=?
    """, (lecture_id,))
    conn.commit()
    conn.close()
    print(f"✅ Session closed: {session_id}")

# ── Get student report ─────────────────
def get_student_report(student_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.name        AS subject,
            s.code,
            COUNT(a.id)   AS total,
            SUM(CASE WHEN a.status IN ('PRESENT','LATE') THEN 1 ELSE 0 END) AS present,
            SUM(CASE WHEN a.status = 'LATE' THEN 1 ELSE 0 END) AS late_count,
            ROUND(
                SUM(CASE WHEN a.status IN ('PRESENT','LATE') THEN 1.0 ELSE 0 END)
                / COUNT(a.id) * 100, 1
            ) AS percentage
        FROM attendance a
        JOIN subjects s ON a.subject_id = s.subject_id
        WHERE a.student_id = ?
        GROUP BY a.subject_id
    """, (student_id,))
    report = cursor.fetchall()
    conn.close()
    return [dict(row) for row in report]

# ── Log unknown person ─────────────────
def log_unknown(lecture_id=None, session_id=None, note=''):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO unknown_persons
        (lecture_id, session_id, note)
        VALUES (?,?,?)
    """, (lecture_id, session_id, note))
    conn.commit()
    conn.close()

# ── Delete everything ──────────────────
def reset_database():
    conn.close() if 'conn' in dir() else None
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️  Database deleted!")
    create_tables()
    print("✅ Fresh database created!")

# ══════════════════════════════════════════
# RUN
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("Setting up AXIOM database...")
    create_tables()
    insert_demo_data()
    print("\n✅ AXIOM database ready!")
    print(f"📁 File: {os.path.abspath(DB_PATH)}")

    # Test report
    print("\n📊 Student Report:")
    report = get_student_report('CS2024101')
    for row in report:
        print(f"  {row['subject']} — {row['percentage']}%")