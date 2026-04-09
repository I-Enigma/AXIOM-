import mysql.connector

db = mysql.connector.connect(
    host='localhost', user='root', password='root467', database='axiom_db'
)
cur = db.cursor()

cur.execute('SHOW TABLES')
tables = [t[0] for t in cur.fetchall()]
print(f"TABLES ({len(tables)}):")
for t in tables:
    print(f"  - {t}")

print("\nTEACHERS:")
cur.execute("SELECT id, email, full_name FROM teachers")
for r in cur.fetchall():
    print(f"  id={r[0]}  email={r[1]}  name={r[2]}")

print("\nSTUDENTS:")
cur.execute("SELECT id, username, first_name, last_name, status FROM students")
rows = cur.fetchall()
if rows:
    for r in rows:
        print(f"  id={r[0]}  user={r[1]}  name={r[2]} {r[3]}  status={r[4]}")
else:
    print("  (none)")

print("\nSUBJECTS:")
try:
    cur.execute("SELECT id, subject_name, subject_code FROM subjects")
    for r in cur.fetchall():
        print(f"  id={r[0]}  {r[1]} ({r[2]})")
except Exception as e:
    print(f"  error: {e}")

cur.close()
db.close()
print("\nDATABASE OK")
