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
                    SUM(uncertain)     AS unc
                    FROM events {where}""",
                args,
            ).fetchone()
            mis = int(contam_row["mis"] or 0)
            unc = int(contam_row["unc"] or 0)

            regions_rows = c.execute(
                "SELECT DISTINCT region FROM events WHERE region IS NOT NULL ORDER BY region"
            ).fetchall()
            regions = [r["region"] for r in regions_rows]

        contam_rate = round((mis + unc) / total, 4) if total else 0.0
        accuracy = round(1 - mis / total, 4) if total else 0.0

        return {
            "total": total,
            "today": today,
            "misclassified": mis,
            "uncertain": unc,
            "contamination_rate": contam_rate,
            "accuracy": accuracy,
            "by_class": {k: by_class.get(k, 0) for k in ALL_CLASSES},
            "by_class_ko": {NAMES_KO.get(k, k): by_class.get(k, 0) for k in ALL_CLASSES},
            "last": dict(recent) if recent else None,
            "hourly": [hourly.get(h, 0) for h in range(24)],
            "regions": regions,
        }
