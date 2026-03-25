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

-- 经验记忆表（来自 FactorMiner RALPH 循环设计）
-- 记录挖掘经历，供未来类似任务检索参考
CREATE TABLE IF NOT EXISTS mining_experience (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name      TEXT,
    hypothesis_type  TEXT,          -- 假设类型关键词（动量/反转/波动/流动性等）
    outcome          TEXT,          -- PASS / FAIL
    failure_mode     TEXT,          -- LOW_IC / LOW_ICIR / LOW_COVERAGE / REVERSAL / TIMEOUT / ENV
    repair_applied   TEXT,          -- 尝试的修复措施描述
    repair_outcome   TEXT,          -- 修复后结果：PASS / FAIL / NOT_TRIED
    ic_mean          REAL,
    key_insight      TEXT,          -- 本次挖掘的关键洞察（什么有效/无效）
    created_at       TEXT
);
""")
            # Idempotent migrations for columns added after initial schema
            for col, typedef in [
                ("judge_score", "REAL DEFAULT 0.0"),
                ("judge_comment", "TEXT DEFAULT ''"),
            ]:
                try:
                    c.execute(f"ALTER TABLE factor_library ADD COLUMN {col} {typedef}")
                except Exception:
                    pass  # column already exists

    # ── Factor library ─────────────────────────────────────────────────────────

    def save_factor(self, result, spec=None) -> int:
        with self._conn() as c:
            cur = c.execute("""
INSERT INTO factor_library
  (factor_name, description, formula, source_paper_title, source_paper_url,
   ic_mean, icir, ic_decay_ratio, coverage, monthly_win_rate, max_correlation,
   status, code_path, variant_of, extension_type, ic_series,
   judge_score, judge_comment, created_at)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                getattr(result, "judge_score", 0.0),
                getattr(result, "judge_comment", ""),
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

    # ── Sessions ───────────────────────────────────────────────────────────────

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

    # ── Experience memory（FactorMiner RALPH 循环）────────────────────────────

    def save_experience(self, factor_name: str, hypothesis_type: str,
                        outcome: str, failure_mode: str = "",
                        repair_applied: str = "", repair_outcome: str = "NOT_TRIED",
                        ic_mean: float = 0.0, key_insight: str = "") -> int:
        """
        保存一次因子挖掘经历，供未来类似任务检索。
        outcome: "PASS" | "FAIL"
        failure_mode: "LOW_IC" | "LOW_ICIR" | "LOW_COVERAGE" | "REVERSAL" | "TIMEOUT" | "ENV" | ""
        """
        with self._conn() as c:
            cur = c.execute("""
INSERT INTO mining_experience
  (factor_name, hypothesis_type, outcome, failure_mode, repair_applied,
   repair_outcome, ic_mean, key_insight, created_at)
VALUES (?,?,?,?,?,?,?,?,?)
""", (factor_name, hypothesis_type, outcome, failure_mode, repair_applied,
      repair_outcome, ic_mean, key_insight, datetime.now().isoformat()))
            return cur.lastrowid

    def get_similar_failures(self, hypothesis_keyword: str, limit: int = 5) -> List[Dict]:
        """
        检索相似假设类型的历史失败经验，用于在生成代码前提供上下文。
        hypothesis_keyword: 从假设描述中提取的关键词（如 "momentum", "reversal"）
        """
        with self._conn() as c:
            cur = c.execute("""
SELECT failure_mode, repair_applied, repair_outcome, key_insight, ic_mean
FROM mining_experience
WHERE outcome = 'FAIL'
  AND (hypothesis_type LIKE ? OR hypothesis_type LIKE ?)
ORDER BY created_at DESC
LIMIT ?
""", (f"%{hypothesis_keyword}%", f"%{hypothesis_keyword.split()[0] if hypothesis_keyword else ''}%",
      limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_successful_patterns(self, limit: int = 10) -> List[Dict]:
        """
        检索高质量成功因子的模式，用于指导新因子生成。
        """
        with self._conn() as c:
            cur = c.execute("""
SELECT hypothesis_type, repair_applied, ic_mean, key_insight
FROM mining_experience
WHERE outcome = 'PASS' AND ic_mean > 0.03
ORDER BY ic_mean DESC
LIMIT ?
""", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
