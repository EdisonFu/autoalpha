import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict


class FactorDB:
    def __init__(self, db_path: str = "autoalpha.db"):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
CREATE TABLE IF NOT EXISTS factor_library (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name         TEXT NOT NULL,
    description         TEXT,
    formula             TEXT,
    source_paper_title  TEXT,
    source_paper_url    TEXT,
    ic_mean             REAL,
    icir                REAL,
    ic_decay_ratio      REAL,
    coverage            REAL,
    monthly_win_rate    REAL,
    max_correlation     REAL,
    status              TEXT,
    code_path           TEXT,
    variant_of          TEXT,
    extension_type      TEXT,
    ic_series           TEXT,
    created_at          TEXT
);

CREATE TABLE IF NOT EXISTS search_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    research_idea     TEXT NOT NULL,
    papers_found      INTEGER,
    factors_extracted INTEGER,
    factors_passed    INTEGER,
    report_path       TEXT,
    created_at        TEXT
);
""")

    def save_factor(self, result, spec=None) -> int:
        from .pipeline_state import FactorResult, FactorSpec
        with self._conn() as c:
            cur = c.execute("""
INSERT INTO factor_library
  (factor_name, description, formula, source_paper_title, source_paper_url,
   ic_mean, icir, ic_decay_ratio, coverage, monthly_win_rate, max_correlation,
   status, code_path, variant_of, extension_type, ic_series, created_at)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
""", (
                result.factor_name,
                spec.description if spec else "",
                spec.formula if spec else "",
                spec.source_paper.get("title","") if spec else "",
                spec.source_paper.get("url","") if spec else "",
                result.ic_mean, result.icir, result.ic_decay_ratio,
                result.coverage, result.monthly_win_rate, result.max_correlation,
                result.status, result.code_path,
                spec.variant_of if spec else None,
                spec.extension_type if spec else None,
                json.dumps(result.ic_series),
                datetime.now().isoformat(),
            ))
            return cur.lastrowid

    def get_passed_factors(self) -> List[Dict]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT * FROM factor_library WHERE status='PASS' ORDER BY icir DESC"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_all_ic_series(self) -> List[List[float]]:
        """返回所有已通过因子的 IC 序列，用于计算相关性。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ic_series FROM factor_library WHERE status='PASS'"
            ).fetchall()
        result = []
        for (s,) in rows:
            try:
                result.append(json.loads(s or "[]"))
            except Exception:
                pass
        return result

    def save_session(self, idea: str, papers_found: int,
                     factors_extracted: int, factors_passed: int,
                     report_path: str = "") -> int:
        with self._conn() as c:
            cur = c.execute("""
INSERT INTO search_sessions
  (research_idea, papers_found, factors_extracted, factors_passed, report_path, created_at)
VALUES (?,?,?,?,?,?)
""", (idea, papers_found, factors_extracted, factors_passed, report_path,
      datetime.now().isoformat()))
            return cur.lastrowid
