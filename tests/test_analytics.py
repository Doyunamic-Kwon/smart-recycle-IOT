"""분석 로직 테스트 — 카메라/YOLO 모델 없이 EventDB(특히 지역(region) 기능) 검증.

실행:
  python tests/test_analytics.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.database import EventDB  # noqa: E402
from src.analytics import timeseries  # noqa: E402


class FakeConfig:
    """EventDB 가 사용하는 두 메서드(path_of, get)만 제공하는 최소 스텁.

    실제 config.yaml / data/events.db 를 건드리지 않고
    임시 디렉터리에 독립된 DB로 테스트한다.
    """

    def __init__(self, db_path: Path, region: str | None = None):
        self._db_path = db_path
        self._region = region

    def path_of(self, dotted: str) -> Path:
        assert dotted == "database.path"
        return self._db_path

    def get(self, dotted: str, default=None):
        if dotted == "device.region":
            return self._region
        return default


def _log(db: EventDB, **over):
    base = dict(
        name="plastic", cls_id=0, conf=0.9, band="trusted",
        bin_target="plastic", misclassified=False, uncertain=False,
        flickered=False,
    )
    base.update(over)
    db.log(**base)


# --- 테스트 --------------------------------------------------------------

def test_new_db_has_region_column(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db", region="AI공학관")
    db = EventDB(cfg)
    with db._conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(events)")}
    assert "region" in cols


def test_log_uses_default_region_from_config(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db", region="가천관")
    db = EventDB(cfg)
    _log(db)
    rows = db.rows()
    assert len(rows) == 1
    assert rows[0]["region"] == "가천관"


def test_log_explicit_region_overrides_default(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db", region="가천관")
    db = EventDB(cfg)
    _log(db, name="can", cls_id=1, bin_target="can", region="1기숙사")
    rows = db.rows()
    assert rows[0]["region"] == "1기숙사"


def test_regions_returns_distinct_sorted(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db")
    db = EventDB(cfg)
    for region in ["3기숙사", "1기숙사", "1기숙사", "2기숙사"]:
        _log(db, region=region)
    assert db.regions() == ["1기숙사", "2기숙사", "3기숙사"]


def test_count_and_rows_filter_by_region(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db")
    db = EventDB(cfg)
    for region, n in [("AI공학관", 3), ("가천관", 2)]:
        for _ in range(n):
            _log(db, name="paper", cls_id=3, bin_target="paper", region=region)
    assert db.count(region="AI공학관") == 3
    assert db.count(region="가천관") == 2
    assert db.count() == 5
    assert len(db.rows(region="AI공학관")) == 3


def test_to_dataframe_filters_by_region(tmpdir: Path):
    cfg = FakeConfig(tmpdir / "events.db")
    db = EventDB(cfg)
    _log(db, name="glass", cls_id=2, bin_target="glass", region="2기숙사")
    _log(db, name="can", cls_id=1, bin_target="can", region="1기숙사")

    df_all = db.to_dataframe()
    df_one = db.to_dataframe(region="2기숙사")
    assert len(df_all) == 2
    assert len(df_one) == 1
    assert df_one.iloc[0]["region"] == "2기숙사"


def test_migration_adds_region_column_to_old_db(tmpdir: Path):
    db_path = tmpdir / "legacy.db"
    # region 컬럼이 없는 구버전 스키마를 직접 생성
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, ts_epoch REAL NOT NULL, name TEXT NOT NULL,
            cls_id INTEGER NOT NULL, conf REAL NOT NULL, band TEXT NOT NULL,
            bin_target TEXT, misclassified INTEGER NOT NULL,
            uncertain INTEGER NOT NULL, flickered INTEGER NOT NULL,
            track_id INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO events (ts, ts_epoch, name, cls_id, conf, band, "
        "bin_target, misclassified, uncertain, flickered, track_id) "
        "VALUES ('2024-01-01T00:00:00', 0, 'plastic', 0, 0.9, 'trusted', "
        "'plastic', 0, 0, 0, NULL)"
    )
    conn.commit()
    conn.close()

    cfg = FakeConfig(db_path, region="AI공학관")
    db = EventDB(cfg)  # 여기서 ALTER TABLE 로 region 컬럼 보강

    with db._conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(events)")}
    assert "region" in cols

    rows = db.rows()
    assert len(rows) == 1
    assert rows[0]["region"] is None  # 기존 행은 region 없음

    _log(db, name="can", cls_id=1, bin_target="can")  # 신규 행은 기본 region 적용
    assert db.regions() == ["AI공학관"]


def test_dashboard_aggregation_pipeline(tmpdir: Path):
    """seed_demo 와 같은 패턴으로 여러 지역 데이터를 넣고
    대시보드가 쓰는 집계(시간대 패턴/지역별 그룹)가 정상 동작하는지 확인."""
    cfg = FakeConfig(tmpdir / "events.db")
    db = EventDB(cfg)
    ts = datetime(2025, 1, 1, 9, 0, 0)
    for region in ["가천관", "AI공학관", "1기숙사"]:
        for _ in range(5):
            _log(db, region=region, ts=ts)

    df = db.to_dataframe()
    assert len(df) == 15
    assert sorted(df["region"].unique()) == ["1기숙사", "AI공학관", "가천관"]

    hp = timeseries.hourly_pattern(df)
    assert hp[9] == 15  # 모두 9시에 기록됨

    by_region = df.groupby("region").size()
    assert by_region["가천관"] == 5


TESTS = [
    test_new_db_has_region_column,
    test_log_uses_default_region_from_config,
    test_log_explicit_region_overrides_default,
    test_regions_returns_distinct_sorted,
    test_count_and_rows_filter_by_region,
    test_to_dataframe_filters_by_region,
    test_migration_adds_region_column_to_old_db,
    test_dashboard_aggregation_pipeline,
]


def main() -> int:
    passed = 0
    for fn in TESTS:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
            except AssertionError as e:
                print(f"FAIL  {fn.__name__}: {e}")
                continue
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {fn.__name__}: {e!r}")
                continue
            print(f"  ok  {fn.__name__}")
            passed += 1

    total = len(TESTS)
    print(f"\n{passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
