import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).resolve().parent / "storage/playground.db"

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_db()
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL, -- 'queued', 'prep', 'skeleton', 'skin', 'rig', 'completed', 'failed'
            stage INTEGER DEFAULT 0, -- 1=input, 2=skeleton, 3=skin, 4=rigged
            error_message TEXT,
            input_filename TEXT,
            input_file_path TEXT,
            num_vertices INTEGER DEFAULT 0,
            num_faces INTEGER DEFAULT 0,
            num_bones INTEGER DEFAULT 0,
            duration_sec REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT -- JSON string with extra details (bone hierarchy, stats, etc.)
        );
        """)
    conn.close()

def create_job(job_id: str, title: str, input_filename: str, input_file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    conn = get_db()
    now = time.time()
    meta_json = json.dumps(metadata) if metadata is not None else json.dumps({})
    with conn:
        conn.execute("""
        INSERT INTO jobs (id, title, status, stage, input_filename, input_file_path, created_at, updated_at, metadata)
        VALUES (?, ?, 'queued', 1, ?, ?, ?, ?, ?)
        """, (job_id, title, input_filename, input_file_path, now, now, meta_json))
    conn.close()
    return get_job(job_id)

def update_job(job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    conn = get_db()
    kwargs["updated_at"] = time.time()
    
    if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
        kwargs["metadata"] = json.dumps(kwargs["metadata"])
        
    set_clauses = [f"{k} = ?" for k in kwargs.keys()]
    values = list(kwargs.values()) + [job_id]
    
    with conn:
        conn.execute(f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?", values)
    conn.close()
    return get_job(job_id)

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    if d.get("metadata"):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
    else:
        d["metadata"] = {}
    return d

def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    res = []
    for r in rows:
        d = dict(r)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except Exception:
                d["metadata"] = {}
        else:
            d["metadata"] = {}
        res.append(d)
    return res

def delete_job(job_id: str):
    conn = get_db()
    with conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.close()
