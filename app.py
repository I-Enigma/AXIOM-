import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector
import bcrypt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Serve Static Files ──
@app.route('/')
def index():
    return send_from_directory('.', 'axiom.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)
def get_db():
    return mysql.connector.connect(
        host     = os.getenv('DB_HOST', 'localhost'),
        user     = os.getenv('DB_USER', 'root'),
        password = os.getenv('DB_PASSWORD', 'root467'),
        database = os.getenv('DB_NAME', 'axiom_db'),
        port     = int(os.getenv('DB_PORT', 3306))
    )

# ────────────────────────────────────────
# REGISTER — saves student to database
# ────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    db   = get_db()
    cur  = db.cursor()

    hashed = bcrypt.hashpw(
        data['password'].encode('utf-8'),
        bcrypt.gensalt()
    )

    try:
        # Fixed column names: roll_no, department, year_level, group_section
        cur.execute("""
            INSERT INTO students
            (first_name, last_name, email, phone, date_of_birth,
             roll_no, department, year_level, group_section,
             username, password_hash, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending')
        """, (
            data['first_name'],
            data['last_name'],
            data['email'],
            data.get('phone',''),
            data.get('dob', None),
            data['roll_number'],
            data['branch'],
            data['year'],
            data.get('section',''),
            data['username'],
            hashed.decode('utf-8')
        ))
        db.commit()
        return jsonify({
            "success": True,
            "message": "Registered successfully! Awaiting teacher verification."
        })

    except mysql.connector.IntegrityError as e:
        return jsonify({
            "success": False,
            "message": f"Integration error: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

    finally:
        cur.close()
        db.close()

# ────────────────────────────────────────
# LOGIN — checks username and password
# ────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    db   = get_db()
    cur  = db.cursor(dictionary=True)

    cur.execute(
        "SELECT * FROM students WHERE username = %s",
        (data['username'],)
    )
    student = cur.fetchone()
    cur.close()
    db.close()

    if not student:
        return jsonify({
            "success": False,
            "message": "Username not found."
        }), 404

    if student['status'] == 'pending':
        return jsonify({
            "success": False,
            "message": "Your account is pending teacher verification."
        }), 403

    if student['status'] == 'rejected':
        return jsonify({
            "success": False,
            "message": "Your account was rejected. Contact your teacher."
        }), 403

    password_match = bcrypt.checkpw(
        data['password'].encode('utf-8'),
        student['password_hash'].encode('utf-8')
    )

    if not password_match:
        return jsonify({
            "success": False,
            "message": "Wrong password."
        }), 401

    return jsonify({
        "success": True,
        "student": {
            "id"              : student['id'],
            "name"            : f"{student['first_name']} {student['last_name']}",
            "roll"            : student['roll_no'],
            "branch"          : student['department'],
            "year"            : student['year_level'],
            "section"         : student['group_section'],
            "email"           : student['email'],
            "username"        : student['username'],
            "faceRegistered" : bool(student['face_registered'])
        }
    })

# ────────────────────────────────────────
# STUDENT DASHBOARD — aggregated data
# ────────────────────────────────────────
@app.route('/api/student/dashboard', methods=['GET'])
def student_dashboard():
    student_id = request.args.get('id')
    if not student_id:
        return jsonify({"success": False, "message": "Missing student ID"}), 400

    db = get_db()
    cur = db.cursor(dictionary=True)

    try:
        # 1. Fetch Enrolled Subjects
        cur.execute("""
            SELECT s.id, s.subject_name AS name, s.subject_code AS code, 
                   s.color_code AS color, s.icon_char AS icon,
                   COALESCE(att.present_count, 0) AS present,
                   COALESCE(att.total_sessions, 0) AS total
            FROM enrollments e
            JOIN subjects s ON e.subject_id = s.id
            LEFT JOIN (
                SELECT sub.id AS subj_id, 
                       COUNT(r.id) AS total_sessions,
                       SUM(CASE WHEN r.is_present = 1 THEN 1 ELSE 0 END) AS present_count
                FROM subjects sub
                JOIN attendance_sessions sess ON sess.session_name LIKE CONCAT('%', sub.subject_name, '%')
                LEFT JOIN attendance_records r ON r.session_id = sess.id AND r.student_id = %s
                GROUP BY sub.id
            ) att ON att.subj_id = s.id
            WHERE e.student_id = %s
        """, (student_id, student_id))
        subjects = cur.fetchall()

        # 2. Fetch Recent History
        cur.execute("""
            SELECT sess.session_name AS subj, 
                   '—' AS code, 
                   DATE(r.marked_at) AS date,
                   TIME_FORMAT(sess.started_at, '%%H:%%i') AS lecTime,
                   TIME_FORMAT(r.marked_at, '%%H:%%i') AS markedAt,
                   TIMESTAMPDIFF(MINUTE, sess.started_at, r.marked_at) AS lateBy,
                   CASE WHEN r.is_present=1 THEN 'PRESENT' ELSE 'ABSENT' END AS status,
                   '100%%' AS score
            FROM attendance_records r
            JOIN attendance_sessions sess ON r.session_id = sess.id
            WHERE r.student_id = %s
            ORDER BY r.marked_at DESC
            LIMIT 10
        """, (student_id,))
        history = cur.fetchall()
        for h in history:
            if h['date']: h['date'] = str(h['date'])

        # 3. Fetch Active Sessions
        cur.execute("""
            SELECT sess.id, s.id AS subjId, s.subject_name AS subjName, 
                   s.subject_code AS code, s.color_code AS color, s.icon_char AS icon,
                   TIME_FORMAT(sess.started_at, '%%H:%%i') AS lecTime,
                   'Active' AS status
            FROM attendance_sessions sess
            JOIN subjects s ON sess.session_name LIKE CONCAT('%%', s.subject_name, '%%')
            WHERE sess.is_active = 1
        """)
        sessions = cur.fetchall()

        return jsonify({
            "success": True,
            "data": {
                "subjects": subjects,
                "history": history,
                "sessions": sessions
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        cur.close()
        db.close()

# ────────────────────────────────────────
# GET ALL STUDENTS — for teacher dashboard
# ────────────────────────────────────────
@app.route('/api/students', methods=['GET'])
def get_students():
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("""
        SELECT id, first_name, last_name, email,
               roll_no, department, year_level, group_section,
               username, status, created_at
        FROM students
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    db.close()

    for r in rows:
        if r.get('created_at'):
            r['created_at'] = str(r['created_at'])

    return jsonify(rows)

# ────────────────────────────────────────
# VERIFY STUDENT — teacher approves/rejects
# ────────────────────────────────────────
@app.route('/api/verify/<int:student_id>', methods=['POST'])
def verify_student(student_id):
    data       = request.json
    new_status = 'verified' if data['action'] == 'approve' else 'rejected'
    db         = get_db()
    cur        = db.cursor()
    cur.execute(
        "UPDATE students SET status = %s WHERE id = %s",
        (new_status, student_id)
    )
    db.commit()
    cur.close()
    db.close()
    return jsonify({"success": True, "status": new_status})

# ────────────────────────────────────────
# TEACHER LOGIN
# ────────────────────────────────────────
@app.route('/api/teacher-login', methods=['POST'])
def teacher_login():
    data = request.json
    db   = get_db()
    cur  = db.cursor(dictionary=True)

    cur.execute(
        "SELECT * FROM teachers WHERE email = %s",
        (data['email'],)
    )
    teacher = cur.fetchone()
    cur.close()
    db.close()

    if not teacher:
        return jsonify({
            "success": False,
            "message": "Teacher email not found."
        }), 404

    try:
        password_match = bcrypt.checkpw(
            data['password'].encode('utf-8'),
            teacher['password_hash'].encode('utf-8')
        )
    except Exception:
        password_match = False

    if not password_match:
        return jsonify({
            "success": False,
            "message": "Wrong password."
        }), 401

    return jsonify({
        "success": True,
        "teacher": {
            "id"         : teacher['id'],
            "name"       : teacher['full_name'], # Corrected to match database.py
            "email"      : teacher['email'],
            "department" : teacher['assigned_dept'] # Corrected to match database.py
        }
    })

# ────────────────────────────────────────
# SEED DEFAULT TEACHER (run once)
# ────────────────────────────────────────
@app.route('/api/seed-teacher', methods=['POST'])
def seed_teacher():
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT id FROM teachers LIMIT 1")
    exists = cur.fetchone()

    if exists:
        # Update existing teacher password so login works
        hashed = bcrypt.hashpw(b'teacher123', bcrypt.gensalt()).decode('utf-8')
        cur.execute("UPDATE teachers SET password_hash = %s WHERE id = %s", (hashed, exists['id']))
        db.commit()
        cur.close()
        db.close()
        return jsonify({"success": True, "message": "Teacher password reset to 'teacher123'."})

    hashed = bcrypt.hashpw(b'teacher123', bcrypt.gensalt()).decode('utf-8')
    cur.execute("""
        INSERT INTO teachers (full_name, email, assigned_dept, password_hash)
        VALUES (%s, %s, %s, %s)
    """, ('Prof. Rajesh Kumar', 'teacher@college.edu', 'Computer Science', hashed))
    db.commit()
    cur.close()
    db.close()
    return jsonify({"success": True, "message": "Default teacher created. Email: teacher@college.edu, Password: teacher123"})

# ────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    print(f"AXIOM Backend running at http://localhost:{port}")
    app.run(debug=True, port=port)