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
        # Standard UniRig jobs table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            stage INTEGER DEFAULT 0,
            error_message TEXT,
            input_filename TEXT,
            input_file_path TEXT,
            num_vertices INTEGER DEFAULT 0,
            num_faces INTEGER DEFAULT 0,
            num_bones INTEGER DEFAULT 0,
            duration_sec REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT
        );
        """)

        # Dedicated Statue 3D Painting Pipeline jobs table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS statue_jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL, -- 'queued', 'processing', 'completed', 'failed'
            input_filename TEXT,
            input_file_path TEXT,
            generator_type TEXT DEFAULT 'trellis',
            mesh_detail TEXT DEFAULT 'high',
            texture_detail TEXT DEFAULT 'high',
            target_faces INTEGER DEFAULT 50000,
            pedestal_shape TEXT DEFAULT 'round',
            enable_rigging INTEGER DEFAULT 0,
            duration_sec REAL DEFAULT 0.0,
            num_vertices INTEGER DEFAULT 0,
            num_faces INTEGER DEFAULT 0,
            num_parts INTEGER DEFAULT 0,
            is_automated INTEGER DEFAULT 0,
            webhook_status TEXT DEFAULT 'idle', -- 'idle', 'sent', 'failed'
            webhook_code INTEGER DEFAULT 0,
            error_message TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT
        );
        """)

        # Automation Configuration table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS automation_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """)

        # Insert default config if not exists
        default_config = {
            "enabled": False,
            "input_folder": str(Path(__file__).resolve().parent / "storage/automation/input"),
            "output_folder": str(Path(__file__).resolve().parent / "storage/automation/output"),
            "poll_interval_sec": 5,
            "generator": "trellis",
            "mesh_detail": "high",
            "texture_detail": "high",
            "target_faces": 50000,
            "pedestal_shape": "round",
            "enable_rigging": False,
            "webhook_url": "",
            "webhook_secret": "",
            "webhook_retry_count": 3
        }
        for k, v in default_config.items():
            conn.execute("""
            INSERT OR IGNORE INTO automation_config (key, value, updated_at)
            VALUES (?, ?, ?)
            """, (k, json.dumps(v), time.time()))

    conn.close()

# --- Standard UniRig Jobs ---

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

# --- Statue 3D Painting Jobs ---

def create_statue_job(
    job_id: str,
    title: str,
    input_filename: str,
    input_file_path: str,
    generator_type: str = "trellis",
    mesh_detail: str = "high",
    texture_detail: str = "high",
    target_faces: int = 50000,
    pedestal_shape: str = "round",
    enable_rigging: bool = False,
    is_automated: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    conn = get_db()
    now = time.time()
    meta_json = json.dumps(metadata) if metadata is not None else json.dumps({})
    with conn:
        conn.execute("""
        INSERT INTO statue_jobs (
            id, title, status, input_filename, input_file_path,
            generator_type, mesh_detail, texture_detail, target_faces,
            pedestal_shape, enable_rigging, is_automated,
            created_at, updated_at, metadata
        ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, title, input_filename, input_file_path,
            generator_type, mesh_detail, texture_detail, target_faces,
            pedestal_shape, 1 if enable_rigging else 0, 1 if is_automated else 0,
            now, now, meta_json
        ))
    conn.close()
    return get_statue_job(job_id)

def update_statue_job(job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    conn = get_db()
    kwargs["updated_at"] = time.time()
    if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
        kwargs["metadata"] = json.dumps(kwargs["metadata"])
    set_clauses = [f"{k} = ?" for k in kwargs.keys()]
    values = list(kwargs.values()) + [job_id]
    with conn:
        conn.execute(f"UPDATE statue_jobs SET {', '.join(set_clauses)} WHERE id = ?", values)
    conn.close()
    return get_statue_job(job_id)

def get_statue_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM statue_jobs WHERE id = ?", (job_id,))
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

def list_statue_jobs(limit: int = 50, automated_only: Optional[bool] = None) -> List[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    if automated_only is True:
        cur.execute("SELECT * FROM statue_jobs WHERE is_automated = 1 ORDER BY created_at DESC LIMIT ?", (limit,))
    elif automated_only is False:
        cur.execute("SELECT * FROM statue_jobs WHERE is_automated = 0 ORDER BY created_at DESC LIMIT ?", (limit,))
    else:
        cur.execute("SELECT * FROM statue_jobs ORDER BY created_at DESC LIMIT ?", (limit,))
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

def delete_statue_job(job_id: str):
    conn = get_db()
    with conn:
        conn.execute("DELETE FROM statue_jobs WHERE id = ?", (job_id,))
    conn.close()

# --- Automation Config Management ---

def get_automation_config() -> Dict[str, Any]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM automation_config")
    rows = cur.fetchall()
    conn.close()
    cfg = {}
    for r in rows:
        try:
            cfg[r["key"]] = json.loads(r["value"])
        except Exception:
            cfg[r["key"]] = r["value"]
    return cfg

def update_automation_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db()
    now = time.time()
    with conn:
        for k, v in new_config.items():
            conn.execute("""
            INSERT INTO automation_config (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (k, json.dumps(v), now))
    conn.close()
    return get_automation_config()
