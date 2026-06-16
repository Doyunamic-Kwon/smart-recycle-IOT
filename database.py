"""SQLite event store for Smart Recycle Dashboard."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    ts_epoch      REAL    NOT NULL,
    name          TEXT    NOT NULL,
    cls_id        INTEGER NOT NULL,
    conf          REAL    NOT NULL,
    band          TEXT    NOT NULL,
    misclassified INTEGER NOT NULL DEFAULT 0,
    uncertain     INTEGER NOT NULL DEFAULT 0,
    flickered     INTEGER NOT NULL DEFAULT 0,
    region        TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_epoch  ON events(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_name   ON events(name);
CREATE INDEX IF NOT EXISTS idx_events_region ON events(region);
"""

NAMES_KO = {"plastic": "플라스틱", "can": "캔", "glass": "유리", "paper": "종이"}
CLASS_IDS = {"plastic": 0, "can": 1, "glass": 2, "paper": 3}
ALL_CLASSES = ["plastic", "can", "glass", "paper"]


class EventDB:
    def __init__(self, path: str = "data/events.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

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
        region: str = "unknown",
        band: str = "trusted",
        misclassified: bool = False,
        uncertain: bool = False,
        flickered: bool = False,
        ts: datetime | None = None,
    ) -> None:
        ts = ts or datetime.now()
        with self._conn() as c:
            c.execute(
                """INSERT INTO events
                   (ts, ts_epoch, name, cls_id, conf, band,
                    misclassified, uncertain, flickered, region)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts.isoformat(timespec="seconds"),
                    ts.timestamp(),
                    name,
                    cls_id,
                    round(conf, 4),
                    band,
                    int(misclassified),
                    int(uncertain),
                    int(flickered),
                    region,
                ),
            )

    def recent_events(self, limit: int = 50, region: str | None = None) -> list[dict]:
        where = "WHERE region = ?" if region else ""
        args = (region,) if region else ()
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM events {where} ORDER BY ts_epoch DESC LIMIT ?",
                args + (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, region: str | None = None) -> dict:
        where = "WHERE region = ?" if region else ""
        args: tuple = (region,) if region else ()

        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        week_start = (datetime.now() - timedelta(days=7)).timestamp()

        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) AS n FROM events {where}", args).fetchone()["n"]
            today = c.execute(
                f"SELECT COUNT(*) AS n FROM events {where} {'AND' if where else 'WHERE'} ts_epoch >= ?",
                args + (today_start,),
            ).fetchone()["n"]

            by_class_rows = c.execute(
                f"SELECT name, COUNT(*) AS cnt FROM events {where} GROUP BY name", args
            ).fetchall()
            by_class = {r["name"]: r["cnt"] for r in by_class_rows}

            recent = c.execute(
                f"SELECT name, conf, ts, region FROM events {where} ORDER BY ts_epoch DESC LIMIT 1",
                args,
            ).fetchone()

            hourly_rows = c.execute(
                f"""SELECT CAST(strftime('%H', ts) AS INTEGER) AS hour, COUNT(*) AS cnt
                    FROM events {where} {'AND' if where else 'WHERE'} ts_epoch >= ?
                    GROUP BY hour ORDER BY hour""",
                args + (week_start,),
            ).fetchall()
            hourly = {r["hour"]: r["cnt"] for r in hourly_rows}

                contam_row = c.execute(
                f"""SELECT
                    SUM(misclassified) AS mis,
                    SUM(uncertain)     AS unc,
                    SUM(CASE WHEN misclassified=1 OR uncertain=1 THEN 1 ELSE 0 END) AS contam_count
                    FROM events {where}""",
                args,
            ).fetchone()
            mis = int(contam_row["mis"] or 0)
            unc = int(contam_row["unc"] or 0)
            contam_count = int(contam_row["contam_count"] or 0)

            # --- Rolling window (최근 50건) ---
            rolling_rows = c.execute(
                f"""SELECT misclassified, uncertain, conf FROM events {where}
                    ORDER BY ts_epoch DESC LIMIT 50""",
                args,
            ).fetchall()
            rolling_n = len(rolling_rows)
            rolling_contam = sum(1 for r in rolling_rows if r["misclassified"] or r["uncertain"])
            rolling_conf = sum(r["conf"] for r in rolling_rows) / rolling_n if rolling_n else 0.0

            # --- Anomaly: 이벤트 급증 (현재 시간 vs 최근 7일 시간당 평균) ---
            hour_start = datetime.now().replace(minute=0, second=0, microsecond=0).timestamp()
            cur_count = c.execute(
                f"""SELECT COUNT(*) AS n FROM events {where}
                    {'AND' if where else 'WHERE'} ts_epoch >= ?""",
                args + (hour_start,),
            ).fetchone()["n"]

            avg_row = c.execute(
                f"""SELECT AVG(cnt) AS avg FROM (
                        SELECT COUNT(*) AS cnt FROM events {where}
                        {'AND' if where else 'WHERE'} ts_epoch >= ? AND ts_epoch < ?
                        GROUP BY CAST(strftime('%Y%m%d%H', ts) AS TEXT)
                    )""",
                args + (week_start, hour_start),
            ).fetchone()
            avg_hourly = float(avg_row["avg"] or 0)
            spike = bool(avg_hourly > 0 and cur_count >= 2 * avg_hourly)

            # --- Anomaly: 신뢰도 하락 (최근 20건 vs 전체 평균) ---
            recent_conf_rows = c.execute(
                f"SELECT conf FROM events {where} ORDER BY ts_epoch DESC LIMIT 20",
                args,
            ).fetchall()
            overall_conf = c.execute(
                f"SELECT AVG(conf) AS avg FROM events {where}", args
            ).fetchone()["avg"] or 0.0
            recent_conf = (
                sum(r["conf"] for r in recent_conf_rows) / len(recent_conf_rows)
                if recent_conf_rows else 0.0
            )
            conf_drop = bool(len(recent_conf_rows) >= 10 and (overall_conf - recent_conf) > 0.1)

            regions_rows = c.execute(
                "SELECT DISTINCT region FROM events WHERE region IS NOT NULL ORDER BY region"
            ).fetchall()
            regions = [r["region"] for r in regions_rows]

        contam_rate = round(contam_count / total, 4) if total else 0.0
        accuracy = round(1 - mis / total, 4) if total else 0.0

        return {
            "total": total,
            "today": today,
            "misclassified": mis,
            "uncertain": unc,
            "contamination_rate": contam_rate,
            "accuracy": accuracy,
            "rolling": {
                "total": rolling_n,
                "contamination_rate": round(rolling_contam / rolling_n, 4) if rolling_n else 0.0,
                "avg_conf": round(rolling_conf, 4),
            },
            "anomalies": {
                "spike": spike,
                "spike_current": cur_count,
                "spike_avg": round(avg_hourly, 1),
                "conf_drop": conf_drop,
                "conf_recent": round(recent_conf, 4),
                "conf_overall": round(overall_conf, 4),
            },
            "by_class": {k: by_class.get(k, 0) for k in ALL_CLASSES},
            "by_class_ko": {NAMES_KO.get(k, k): by_class.get(k, 0) for k in ALL_CLASSES},
            "last": dict(recent) if recent else None,
            "hourly": [hourly.get(h, 0) for h in range(24)],
            "regions": regions,
        }
