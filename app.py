from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import bcrypt

app = Flask(__name__)
CORS(app)

# ── Change this to your MySQL password ──
DB_PASSWORD = "root467"

def get_db():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = DB_PASSWORD,
        database = "axiom_attendance"
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
        cur.execute("""
            INSERT INTO students
            (first_name, last_name, email, phone, dob,
             roll_number, branch, year, section,
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

    except mysql.connector.IntegrityError:
        return jsonify({
            "success": False,
            "message": "Email, roll number or username already exists."
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
            "name"    : student['first_name'] + ' ' + student['last_name'],
            "roll"    : student['roll_number'],
            "branch"  : student['branch'],
            "year"    : student['year'],
            "section" : student['section'],
            "email"   : student['email'],
            "username": student['username']
        }
    })

# ────────────────────────────────────────
# GET ALL STUDENTS — for teacher dashboard
# ────────────────────────────────────────
@app.route('/api/students', methods=['GET'])
def get_students():
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("""
        SELECT id, first_name, last_name, email,
               roll_number, branch, year, section,
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
            "name"       : teacher['name'],
            "email"      : teacher['email'],
            "department" : teacher['department']
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
        INSERT INTO teachers (name, email, department, password_hash)
        VALUES (%s, %s, %s, %s)
    """, ('Prof. Rajesh Kumar', 'teacher@college.edu', 'Computer Science', hashed))
    db.commit()
    cur.close()
    db.close()
    return jsonify({"success": True, "message": "Default teacher created. Email: teacher@college.edu, Password: teacher123"})

# ────────────────────────────────────────
if __name__ == '__main__':
    print("AXIOM Backend running at http://localhost:5000")
    app.run(debug=True, port=5000)