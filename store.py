"""SQLite persistence: append-only mistakes, recomputed gap profiles, practice log."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY,
    grade TEXT,
    class_id TEXT
);
CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    image_path TEXT,
    question_text TEXT,
    steps TEXT,
    broken_step TEXT,
    error_type TEXT,
    knowledge_point_id TEXT,
    confidence REAL,
    grounded INTEGER,
    provider TEXT
);
CREATE TABLE IF NOT EXISTS practice_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mistake_id INTEGER NOT NULL,
    item_id TEXT,
    due_day INTEGER,
    result TEXT,
    logged_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def add_mistake(conn: sqlite3.Connection, record: Dict[str, Any]) -> int:
    cur = conn.execute(
        """INSERT INTO mistakes
           (student_id, created_at, image_path, question_text, steps, broken_step,
            error_type, knowledge_point_id, confidence, grounded, provider)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            record.get("student_id", "anon"),
            record.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"),
            record.get("image_path"),
            record.get("question_text", ""),
            "\n".join(record.get("steps", []) or []),
            record.get("broken_step"),
            record.get("error_type"),
            record.get("knowledge_point_id"),
            record.get("confidence"),
            1 if record.get("grounded") else 0,
            record.get("provider", ""),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def record_practice(conn: sqlite3.Connection, mistake_id: int, item_id: str, due_day: int, result: str) -> None:
    conn.execute(
        "INSERT INTO practice_log (mistake_id, item_id, due_day, result, logged_at) VALUES (?,?,?,?,?)",
        (mistake_id, item_id, due_day, result, datetime.utcnow().isoformat(timespec="seconds")),
    )
    conn.commit()


def list_mistakes(conn: sqlite3.Connection, student_id: Optional[str] = None, limit: int = 100) -> List[sqlite3.Row]:
    if student_id:
        rows = conn.execute(
            "SELECT * FROM mistakes WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM mistakes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return rows


def gap_profile(conn: sqlite3.Connection, student_id: str) -> List[Dict[str, Any]]:
    """Recomputed from append-only mistakes; never mutated in place."""
    rows = conn.execute(
        "SELECT knowledge_point_id, error_type, created_at FROM mistakes WHERE student_id=? AND knowledge_point_id IS NOT NULL",
        (student_id,),
    ).fetchall()
    now = datetime.utcnow()
    profile: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        kp = row["knowledge_point_id"]
        try:
            seen = datetime.fromisoformat(row["created_at"])
        except ValueError:
            seen = now
        entry = profile.setdefault(
            kp, {"knowledge_point": kp, "repeat_count": 0, "last_seen": seen, "error_type": row["error_type"]}
        )
        entry["repeat_count"] += 1
        if seen > entry["last_seen"]:
            entry["last_seen"] = seen
            entry["error_type"] = row["error_type"]  # latest mistake defines current severity
    for entry in profile.values():
        cleared = conn.execute(
            """SELECT COUNT(*) c FROM practice_log p JOIN mistakes m ON m.id = p.mistake_id
               WHERE m.student_id=? AND m.knowledge_point_id=? AND p.result='pass'""",
            (student_id, entry["knowledge_point"]),
        ).fetchone()["c"]
        entry["cleared_count"] = cleared
        entry["days_since"] = max(0, (now - entry["last_seen"]).days)
        entry["last_seen"] = entry["last_seen"].isoformat(timespec="seconds")
    return sorted(profile.values(), key=lambda e: -e["repeat_count"])


def repeat_error_rate(conn: sqlite3.Connection, student_id: Optional[str] = None, window_days: int = 14) -> Dict[str, Any]:
    since = (datetime.utcnow() - timedelta(days=window_days)).isoformat(timespec="seconds")
    params: List[Any] = [since]
    where = "WHERE created_at >= ?"
    if student_id:
        where += " AND student_id=?"
        params.append(student_id)
    rows = conn.execute(
        f"SELECT knowledge_point_id, COUNT(*) c FROM mistakes {where} AND knowledge_point_id IS NOT NULL GROUP BY knowledge_point_id",
        params,
    ).fetchall()
    total = sum(r["c"] for r in rows)
    repeat_points = sum(1 for r in rows if r["c"] > 1)
    points = len(rows)
    return {
        "window_days": window_days,
        "distinct_knowledge_points": points,
        "repeat_knowledge_points": repeat_points,
        "repeat_error_rate": round(repeat_points / points, 4) if points else 0.0,
        "observations": total,
    }


def class_heatmap(conn: sqlite3.Connection, class_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conditions = ["m.knowledge_point_id IS NOT NULL"]
    params: List[Any] = []
    if class_id:
        conditions.append("s.class_id=?")
        params.append(class_id)
    rows = conn.execute(
        f"""SELECT m.knowledge_point_id kp, COUNT(*) c
            FROM mistakes m LEFT JOIN students s ON s.id = m.student_id
            WHERE {' AND '.join(conditions)}
            GROUP BY m.knowledge_point_id ORDER BY c DESC""",
        params,
    ).fetchall()
    return [{"knowledge_point": r["kp"], "mistakes": r["c"]} for r in rows]
