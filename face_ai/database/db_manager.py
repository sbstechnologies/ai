import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")


class DatabaseManager:
    def __init__(self, db_type="sqlite", host="localhost", user="root", password="", database="face_ai", sqlite_path=DEFAULT_SQLITE_PATH):
        self.db_type = db_type.lower()
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.sqlite_path = sqlite_path
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        if self.db_type == "mysql" and MYSQL_AVAILABLE:
            try:
                conn = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database
                )
                return conn
            except MySQLError as e:
                print(f"[DB Warning] MySQL connection failed ({e}). Falling back to SQLite.")
                return sqlite3.connect(self.sqlite_path)
        else:
            return sqlite3.connect(self.sqlite_path)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        is_sqlite = isinstance(conn, sqlite3.Connection)

        if is_sqlite:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT,
                    role TEXT,
                    email TEXT,
                    embedding TEXT,
                    photo_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    department TEXT,
                    date TEXT NOT NULL,
                    check_in TEXT,
                    check_out TEXT,
                    status TEXT,
                    confidence REAL,
                    snapshot_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    camera_name TEXT,
                    face_count INTEGER,
                    confidence REAL
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_code VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    department VARCHAR(100),
                    role VARCHAR(100),
                    email VARCHAR(100),
                    embedding LONGTEXT,
                    photo_path VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_code VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    department VARCHAR(100),
                    date DATE NOT NULL,
                    check_in TIME,
                    check_out TIME,
                    status VARCHAR(50),
                    confidence FLOAT,
                    snapshot_path VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    camera_name VARCHAR(100),
                    face_count INT,
                    confidence FLOAT
                )
            """)

        conn.commit()
        cursor.close()
        conn.close()

    def add_user(self, user_code, name, department="", role="", email="", embedding=None, photo_path=""):
        conn = self.get_connection()
        cursor = conn.cursor()
        embedding_json = json.dumps(embedding.tolist()) if embedding is not None else "[]"
        try:
            if isinstance(conn, sqlite3.Connection):
                cursor.execute("""
                    INSERT OR REPLACE INTO users (user_code, name, department, role, email, embedding, photo_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_code, name, department, role, email, embedding_json, photo_path))
            else:
                cursor.execute("""
                    INSERT INTO users (user_code, name, department, role, email, embedding, photo_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name=VALUES(name), department=VALUES(department), role=VALUES(role),
                    email=VALUES(email), embedding=VALUES(embedding), photo_path=VALUES(photo_path)
                """, (user_code, name, department, role, email, embedding_json, photo_path))
            conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error] add_user: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_all_users(self):
        conn = self.get_connection()
        try:
            df = pd.read_sql("SELECT user_code, name, department, role, email, photo_path, created_at FROM users", conn)
            return df
        except Exception as e:
            print(f"[DB Error] get_all_users: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_user_embeddings(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        users = []
        try:
            cursor.execute("SELECT user_code, name, department, embedding FROM users")
            rows = cursor.fetchall()
            for row in rows:
                user_code, name, dept, emb_str = row[0], row[1], row[2], row[3]
                if emb_str:
                    emb = json.loads(emb_str)
                    if len(emb) > 0:
                        users.append({
                            "user_code": user_code,
                            "name": name,
                            "department": dept,
                            "embedding": emb
                        })
            return users
        except Exception as e:
            print(f"[DB Error] get_user_embeddings: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def delete_user(self, user_code):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if isinstance(conn, sqlite3.Connection):
                cursor.execute("DELETE FROM users WHERE user_code = ?", (user_code,))
            else:
                cursor.execute("DELETE FROM users WHERE user_code = %s", (user_code,))
            conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error] delete_user: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def mark_attendance(self, user_code, name, department, date_str, time_str, status="Present", confidence=1.0, snapshot_path=""):
        conn = self.get_connection()
        cursor = conn.cursor()
        is_sqlite = isinstance(conn, sqlite3.Connection)
        try:
            # Check if record exists for today
            if is_sqlite:
                cursor.execute("SELECT id, check_in FROM attendance WHERE user_code = ? AND date = ?", (user_code, date_str))
            else:
                cursor.execute("SELECT id, check_in FROM attendance WHERE user_code = %s AND date = %s", (user_code, date_str))
            
            existing = cursor.fetchone()
            if existing:
                # Already marked check-in today, update check_out
                att_id = existing[0]
                if is_sqlite:
                    cursor.execute("UPDATE attendance SET check_out = ?, snapshot_path = ? WHERE id = ?", (time_str, snapshot_path, att_id))
                else:
                    cursor.execute("UPDATE attendance SET check_out = %s, snapshot_path = %s WHERE id = %s", (time_str, snapshot_path, att_id))
                conn.commit()
                return "updated", existing[1]
            else:
                # New check-in
                if is_sqlite:
                    cursor.execute("""
                        INSERT INTO attendance (user_code, name, department, date, check_in, status, confidence, snapshot_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_code, name, department, date_str, time_str, status, confidence, snapshot_path))
                else:
                    cursor.execute("""
                        INSERT INTO attendance (user_code, name, department, date, check_in, status, confidence, snapshot_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (user_code, name, department, date_str, time_str, status, confidence, snapshot_path))
                conn.commit()
                return "marked", time_str
        except Exception as e:
            print(f"[DB Error] mark_attendance: {e}")
            return "error", None
        finally:
            cursor.close()
            conn.close()

    def get_attendance_records(self, date_from=None, date_to=None, department="All", user_code=""):
        conn = self.get_connection()
        try:
            query = "SELECT id, date, user_code, name, department, check_in, check_out, status, confidence, snapshot_path FROM attendance WHERE 1=1"
            params = []
            is_sqlite = isinstance(conn, sqlite3.Connection)
            p_placeholder = "?" if is_sqlite else "%s"

            if date_from:
                query += f" AND date >= {p_placeholder}"
                params.append(str(date_from))
            if date_to:
                query += f" AND date <= {p_placeholder}"
                params.append(str(date_to))
            if department and department != "All":
                query += f" AND department = {p_placeholder}"
                params.append(department)
            if user_code:
                query += f" AND (user_code LIKE {p_placeholder} OR name LIKE {p_placeholder})"
                params.append(f"%{user_code}%")
                params.append(f"%{user_code}%")

            query += " ORDER BY date DESC, check_in DESC"
            df = pd.read_sql(query, conn, params=params)
            return df
        except Exception as e:
            print(f"[DB Error] get_attendance_records: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_today_summary(self, date_str=None):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        conn = self.get_connection()
        try:
            df_users = pd.read_sql("SELECT COUNT(*) as total_users FROM users", conn)
            total_users = df_users.iloc[0]["total_users"] if not df_users.empty else 0

            is_sqlite = isinstance(conn, sqlite3.Connection)
            p_placeholder = "?" if is_sqlite else "%s"

            df_att = pd.read_sql(f"SELECT status FROM attendance WHERE date = {p_placeholder}", conn, params=[date_str])
            present_count = len(df_att)
            late_count = len(df_att[df_att["status"] == "Late"])
            absent_count = max(0, total_users - present_count)

            return {
                "total_users": total_users,
                "present": present_count,
                "late": late_count,
                "absent": absent_count,
            }
        except Exception as e:
            print(f"[DB Error] get_today_summary: {e}")
            return {"total_users": 0, "present": 0, "late": 0, "absent": 0}
        finally:
            conn.close()

    def save_detection(self, camera_name, face_count, confidence):
        conn = self.get_connection()
        cursor = conn.cursor()
        is_sqlite = isinstance(conn, sqlite3.Connection)
        try:
            if is_sqlite:
                cursor.execute("""
                    INSERT INTO face_detections (camera_name, face_count, confidence)
                    VALUES (?, ?, ?)
                """, (camera_name, face_count, confidence))
            else:
                cursor.execute("""
                    INSERT INTO face_detections (camera_name, face_count, confidence)
                    VALUES (%s, %s, %s)
                """, (camera_name, face_count, confidence))
            conn.commit()
            return True
        except Exception as e:
            print(f"[DB Error] save_detection: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def load_detection_logs(self, limit=100):
        conn = self.get_connection()
        try:
            query = f"SELECT id, detected_at, camera_name, face_count, confidence FROM face_detections ORDER BY detected_at DESC LIMIT {limit}"
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"[DB Error] load_detection_logs: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
