import sqlite3
import json
import hashlib
import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "sentinel.db"

def get_connection():
    # Check_same_thread=False allows Streamlit's multiple threads to share the connection 
    # (safely if we only do basic queries)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT NOT NULL,
            district TEXT,
            constituency TEXT
        )
    ''')
    
    # Create Audit Log table
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            project_id TEXT,
            detail TEXT NOT NULL,
            extra TEXT,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        )
    ''')
    
    # Create Evidence Uploads table
    c.execute('''
        CREATE TABLE IF NOT EXISTS evidence_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sr_no TEXT NOT NULL,
            filename TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL
        )
    ''')
    
    # Seed default users if empty
    c.execute('SELECT COUNT(*) FROM users')
    if c.fetchone()[0] == 0:
        default_users = [
            ("admin", "admin123", "MoSPI Admin", "Dr. Rajesh Kumar", "", ""),
            ("nodal", "nodal123", "District Nodal Officer", "Ms. Priya Sharma", "Ajmer", ""),
            ("mp", "mp123", "Member of Parliament", "Shri Ram Prasad Verma", "", "Agra"),
            ("public", "pub123", "Public Viewer", "Citizen", "", "")
        ]
        c.executemany('INSERT INTO users (username, password, role, name, district, constituency) VALUES (?,?,?,?,?,?)', default_users)
    else:
        # Migrate nodal user if district was Agra (which has 0 records in demo dataset)
        c.execute("UPDATE users SET district = 'Ajmer' WHERE username = 'nodal' AND (district = 'Agra' OR district = '' OR district IS NULL)")
        
    conn.commit()
    conn.close()

def add_evidence(sr_no, filename, uploaded_by, img_type):
    conn = get_connection()
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    c.execute('INSERT INTO evidence_uploads (sr_no, filename, uploaded_by, timestamp, type) VALUES (?,?,?,?,?)',
              (str(sr_no), filename, uploaded_by, ts, img_type))
    conn.commit()
    conn.close()

def get_evidence_for_work(sr_no):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM evidence_uploads WHERE sr_no = ? ORDER BY id DESC', (str(sr_no),))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_all_evidence_counts():
    """Returns a dict mapping sr_no -> count of uploaded evidence"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT sr_no, COUNT(*) FROM evidence_uploads GROUP BY sr_no')
    rows = c.fetchall()
    conn.close()
    return {str(row[0]): row[1] for row in rows}

def hash_record(data):
    # Ensure keys are sorted for consistent hashing
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

def add_audit_entry(user, role, action, project_id, detail, extra=None):
    conn = get_connection()
    c = conn.cursor()
    
    # Get last hash
    c.execute('SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    prev_hash = row[0] if row else "GENESIS"
    
    entry_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user,
        "role": role,
        "action": action,
        "project_id": project_id,
        "detail": detail,
        "extra": extra or {},
        "prev_hash": prev_hash
    }
    
    entry_hash = hash_record(entry_data)
    
    c.execute('''
        INSERT INTO audit_log (timestamp, user, role, action, project_id, detail, extra, prev_hash, hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (entry_data["timestamp"], user, role, action, project_id, detail, json.dumps(extra or {}), prev_hash, entry_hash))
    
    conn.commit()
    conn.close()
    return entry_hash

def get_audit_logs():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM audit_log ORDER BY id DESC')
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def verify_chain():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM audit_log ORDER BY id ASC')
    rows = c.fetchall()
    conn.close()
    
    broken_links = []
    expected_prev = "GENESIS"
    
    for row in rows:
        row_id = row["id"]
        # Reconstruct the dict for hashing
        entry_data = {
            "timestamp": row["timestamp"],
            "user": row["user"],
            "role": row["role"],
            "action": row["action"],
            "project_id": str(row["project_id"]) if row["project_id"] else "—",
            "detail": row["detail"],
            "extra": json.loads(row["extra"]) if row["extra"] else {},
            "prev_hash": row["prev_hash"]
        }
        computed_hash = hash_record(entry_data)
        
        # Check prev_hash matches the actual previous hash
        if row["prev_hash"] != expected_prev:
            broken_links.append({"id": row_id, "issue": "PREV_HASH_MISMATCH", "expected": expected_prev, "actual": row["prev_hash"]})
        
        # Check current hash matches DB hash
        if computed_hash != row["hash"]:
            broken_links.append({"id": row_id, "issue": "TAMPERED_DATA", "expected_hash": computed_hash, "db_hash": row["hash"]})
            
        expected_prev = row["hash"]
        
    return len(broken_links) == 0, broken_links

def tamper_demo_record():
    """Intentionally modifies a record in the DB to demonstrate tampering detection"""
    conn = get_connection()
    c = conn.cursor()
    
    # Change the detail of the most recent 'STATUS_CHANGE' or just the last record
    c.execute("UPDATE audit_log SET detail = 'TAMPERED: CLEARED (Coverup)' WHERE id = (SELECT id FROM audit_log ORDER BY id DESC LIMIT 1)")
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def register_user(username, password, role, name, district="", constituency=""):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, role, name, district, constituency) VALUES (?,?,?,?,?,?)',
                  (username, password, role, name, district, constituency))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
