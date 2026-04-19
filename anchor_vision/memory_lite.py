"""
Lightweight visual memory for Anchor Vision — standalone, no Anchor Memory needed.

Stores perceptual hashes and detection results so Vision can recognize
previously seen images without requiring the full Anchor Memory system.

If Anchor Memory is available, Vision uses it instead (richer Hebbian graph).
This is the fallback for users who only install anchor-vision.
"""

import sqlite3
import os
import json
from datetime import datetime


class VisualMemoryLite:
    """Simple SQLite store for visual observations."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser("~/.anchor-vision/visual_memory.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                obs_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                phash       TEXT,
                description TEXT,
                detections  TEXT,
                intention   TEXT,
                created_at  TEXT,
                last_seen   TEXT,
                seen_count  INTEGER DEFAULT 1
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_phash ON observations(phash)")
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def store(self, phash: str, description: str, detections: list = None,
              intention: str = "") -> int:
        """Store a new visual observation."""
        conn = self._conn()
        # Check if we've seen this phash before
        existing = conn.execute(
            "SELECT obs_id, seen_count FROM observations WHERE phash = ?",
            (phash,)
        ).fetchone()

        if existing:
            # Update existing — increment seen_count
            conn.execute(
                "UPDATE observations SET last_seen = ?, seen_count = seen_count + 1, "
                "description = ?, intention = ? WHERE obs_id = ?",
                (datetime.utcnow().isoformat(), description, intention, existing["obs_id"])
            )
            conn.commit()
            obs_id = existing["obs_id"]
        else:
            # New observation
            cur = conn.execute(
                "INSERT INTO observations (phash, description, detections, intention, "
                "created_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (phash, description, json.dumps(detections or []),
                 intention, datetime.utcnow().isoformat(),
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            obs_id = cur.lastrowid

        conn.close()
        return obs_id

    def find_by_phash(self, phash: str, threshold: float = 0.85) -> dict:
        """Find a previous observation by perceptual hash similarity."""
        conn = self._conn()
        rows = conn.execute("SELECT * FROM observations").fetchall()
        conn.close()

        best = None
        best_sim = 0
        for row in rows:
            sim = self._phash_similarity(phash, row["phash"])
            if sim > threshold and sim > best_sim:
                best = dict(row)
                best_sim = sim

        if best:
            best["similarity"] = best_sim
        return best

    def get_recent(self, limit: int = 10) -> list:
        """Get recent observations."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM observations ORDER BY last_seen DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def forget(self, obs_id: int):
        """Delete an observation."""
        conn = self._conn()
        conn.execute("DELETE FROM observations WHERE obs_id = ?", (obs_id,))
        conn.commit()
        conn.close()

    def forget_all(self):
        """Delete all observations."""
        conn = self._conn()
        conn.execute("DELETE FROM observations")
        conn.commit()
        conn.close()

    @staticmethod
    def _phash_similarity(h1: str, h2: str) -> float:
        if not h1 or not h2 or len(h1) != len(h2):
            return 0.0
        same = sum(a == b for a, b in zip(h1, h2))
        return same / len(h1)
