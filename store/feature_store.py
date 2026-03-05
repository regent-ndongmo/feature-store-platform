"""
Feature Store : sauvegarde et récupération des features.
Stockage via SQLite (métadonnées) + CSV (données).
"""
import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any


STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(STORE_DIR, "feature_store.db")


def _ensure_dirs():
    os.makedirs(STORE_DIR, exist_ok=True)


def _get_conn():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            entity_col TEXT,
            feature_cols TEXT,
            data_path TEXT,
            created_at TEXT,
            version INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT,
            event TEXT,
            details TEXT,
            ts TEXT
        )
    """)
    conn.commit()
    return conn


class FeatureStore:
    """Interface simple de Feature Store local."""

    def save_feature_group(
        self,
        name: str,
        df: pd.DataFrame,
        entity_col: str,
        feature_cols: List[str],
        description: str = "",
    ) -> Dict[str, Any]:
        _ensure_dirs()
        csv_path = os.path.join(STORE_DIR, f"{name}.csv")
        df = df.copy()
        df["event_timestamp"] = datetime.utcnow().isoformat()
        cols_to_save = [entity_col] + feature_cols + ["event_timestamp"]
        cols_to_save = [c for c in cols_to_save if c in df.columns]
        df[cols_to_save].to_csv(csv_path, index=False)

        conn = _get_conn()
        try:
            existing = conn.execute(
                "SELECT version FROM feature_groups WHERE name=?", (name,)
            ).fetchone()
            version = (existing[0] + 1) if existing else 1
            conn.execute(
                "INSERT OR REPLACE INTO feature_groups "
                "(name, description, entity_col, feature_cols, data_path, created_at, version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, description, entity_col, json.dumps(feature_cols),
                 csv_path, datetime.utcnow().isoformat(), version),
            )
            self._log(conn, name, "SAVE", f"version={version}, rows={len(df)}, cols={feature_cols}")
            conn.commit()
        finally:
            conn.close()
        return {"name": name, "version": version, "rows": len(df), "path": csv_path}

    def list_feature_groups(self) -> List[Dict[str, Any]]:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT name, description, entity_col, feature_cols, created_at, version FROM feature_groups"
            ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            result.append({
                "name": r[0], "description": r[1], "entity_col": r[2],
                "feature_cols": json.loads(r[3]), "created_at": r[4], "version": r[5],
            })
        return result

    def get_features(
        self,
        group_name: str,
        features: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT data_path, entity_col, feature_cols FROM feature_groups WHERE name=?",
                (group_name,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"Groupe de features '{group_name}' introuvable dans le store.")
        data_path, entity_col, feature_cols_json = row
        df = pd.read_csv(data_path)
        if entity_ids is not None:
            df = df[df[entity_col].astype(str).isin([str(e) for e in entity_ids])]
        if features is not None:
            valid = [f for f in features if f in df.columns]
            cols = [entity_col] + valid + ["event_timestamp"]
            df = df[[c for c in cols if c in df.columns]]
        return df

    def delete_feature_group(self, name: str) -> bool:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT data_path FROM feature_groups WHERE name=?", (name,)
            ).fetchone()
            if row:
                if os.path.exists(row[0]):
                    os.remove(row[0])
                conn.execute("DELETE FROM feature_groups WHERE name=?", (name,))
                self._log(conn, name, "DELETE", "groupe supprimé")
                conn.commit()
                return True
            return False
        finally:
            conn.close()

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT name, description, entity_col, feature_cols, created_at, version "
                "FROM feature_groups WHERE name=?", (name,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "name": row[0], "description": row[1], "entity_col": row[2],
            "feature_cols": json.loads(row[3]), "created_at": row[4], "version": row[5],
        }

    def _log(self, conn, group_name: str, event: str, details: str):
        conn.execute(
            "INSERT INTO feature_logs (group_name, event, details, ts) VALUES (?, ?, ?, ?)",
            (group_name, event, details, datetime.utcnow().isoformat()),
        )
