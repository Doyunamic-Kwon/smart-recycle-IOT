"""단일 이벤트 로그 (SQLite).

4개 분석 기능이 전부 이 한 테이블을 바라본다.
'확정된(confirmed) 탐지' 1건마다 한 행이 쌓이고, 각 행은 이미 분석에 필요한
파생 신호(오분류 여부/불확실 여부/신뢰도 밴드)를 함께 들고 있다.
→ 리포트·시계열·오염분석이 별도 재계산 없이 바로 집계 가능.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,   -- ISO8601 지역시각
    ts_epoch      REAL    NOT NULL,   -- 정렬/구간계산용 유닉스시각
    name          TEXT    NOT NULL,   -- 영문 클래스명
    cls_id        INTEGER NOT NULL,
    conf          REAL    NOT NULL,
    band          TEXT    NOT NULL,   -- 'uncertain' | 'trusted'
    bin_target    TEXT,               -- 이 장치가 담당하는 통의 목표 클래스
    misclassified INTEGER NOT NULL,   -- 1=통 목표와 불일치(오분류)
    uncertain     INTEGER NOT NULL,   -- 1=애매구간 또는 flicker
    flickered     INTEGER NOT NULL,
    track_id      INTEGER,
    region        TEXT                -- 장치가 설치된 지역 (대시보드 지역 필터용)
);
CREATE INDEX IF NOT EXISTS idx_events_epoch ON events(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_name  ON events(name);
"""


class EventDB:
    def __init__(self, cfg):
        self.path = cfg.path_of("database.path")
        self.default_region = cfg.get("device.region")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # 기존(구버전) DB 에는 region 컬럼이 없을 수 있으므로 보강
            cols = {row["name"] for row in c.execute("PRAGMA table_info(events)")}
            if "region" not in cols:
                c.execute("ALTER TABLE events ADD COLUMN region TEXT")
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_region ON events(region)")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(
        self,
        *,
        name: str,
        cls_id: int,
        conf: float,
        band: str,
        bin_target: str | None,
        misclassified: bool,
        uncertain: bool,
        flickered: bool,
        track_id: int | None = None,
        ts: datetime | None = None,
        region: str | None = None,
    ) -> None:
        ts = ts or datetime.now()
        region = region if region is not None else self.default_region
        with self._conn() as c:
            c.execute(
                """INSERT INTO events
                   (ts, ts_epoch, name, cls_id, conf, band, bin_target,
                    misclassified, uncertain, flickered, track_id, region)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts.isoformat(timespec="seconds"),
                    ts.timestamp(),
                    name,
                    cls_id,
                    round(conf, 4),
                    band,
                    bin_target,
                    int(misclassified),
                    int(uncertain),
                    int(flickered),
                    track_id,
                    region,
                ),
            )

    # --- 조회 헬퍼 ----------------------------------------------------
    @staticmethod
    def _where(since_epoch: float | None, region: str | None) -> tuple[str, tuple]:
        clauses, args = [], []
        if since_epoch is not None:
            clauses.append("ts_epoch >= ?")
            args.append(since_epoch)
        if region is not None:
            clauses.append("region = ?")
            args.append(region)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", tuple(args)

    def count(self, since_epoch: float | None = None, region: str | None = None) -> int:
        where, args = self._where(since_epoch, region)
        with self._conn() as c:
            return c.execute(f"SELECT COUNT(*) AS n FROM events{where}", args).fetchone()["n"]

    def rows(
        self, since_epoch: float | None = None, region: str | None = None
    ) -> list[sqlite3.Row]:
        where, args = self._where(since_epoch, region)
        with self._conn() as c:
            return c.execute(f"SELECT * FROM events{where} ORDER BY ts_epoch", args).fetchall()

    def regions(self) -> list[str]:
        """지역(region) 값이 채워진 이벤트들의 고유 지역명 목록 (정렬됨)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT region FROM events WHERE region IS NOT NULL ORDER BY region"
            ).fetchall()
        return [r["region"] for r in rows]

    def to_dataframe(self, since_epoch: float | None = None, region: str | None = None):
        """pandas DataFrame 으로 반환 (리포트/시계열/대시보드에서 사용)."""
        import pandas as pd

        where, args = self._where(since_epoch, region)
        with self._conn() as c:
            df = pd.read_sql_query(f"SELECT * FROM events{where}", c, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
