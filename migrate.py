import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def migrate():
    # Load SQL from database.sql
    try:
        with open('database.sql', 'r', encoding='utf-8') as f:
            sql_content = f.read()
    except Exception as e:
        print(f"Error reading database.sql: {e}")
        return

    # Split into individual commands (primitive split by ;)
    # Note: DELIMITER sections like the stored procedure will need better handling
    # For now, we'll just execute the whole file via a connection
    
    db = mysql.connector.connect(
        host     = os.getenv('DB_HOST', 'localhost'),
        user     = os.getenv('DB_USER', 'root'),
        password = os.getenv('DB_PASSWORD', 'root467'),
        database = os.getenv('DB_NAME', 'axiom_db')
    )
    cur = db.cursor()

    print("Executing database.sql migration...")
    
    # We'll use multi=True to execute multiple statements
    try:
        # Note: mysql-connector's execute(multi=True) handles multiple statements
        results = cur.execute(sql_content, multi=True)
        for res in results:
            if res.with_rows:
                res.fetchall() # consuming any stray results
        db.commit()
        print("Migration COMPLETED successfully.")
    except Exception as e:
        print(f"Migration FAILED: {e}")
    finally:
        cur.close()
        db.close()

if __name__ == "__main__":
    migrate()
