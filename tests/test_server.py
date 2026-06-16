"""중앙 수집 서버(scripts/server.py) + RemoteLogger 테스트.

실행:
  python tests/test_server.py
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from werkzeug.serving import make_server  # noqa: E402

from server import create_app  # noqa: E402
from src.analytics.database import EventDB  # noqa: E402
from src.analytics.remote_logger import RemoteLogger  # noqa: E402


class FakeConfig:
    """실제 Config 와 동일한 dotted-get/path_of 인터페이스를 가진 스텁."""

    def __init__(self, data: dict, db_path: Path):
        self._data = data
        self._db_path = db_path

    def get(self, dotted: str, default=None):
        node = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path_of(self, dotted: str) -> Path:
        assert dotted == "database.path"
        return self._db_path


BASE_EVENT = dict(
    name="plastic", cls_id=0, conf=0.9, band="trusted",
    bin_target="plastic", misclassified=False, uncertain=False, flickered=False,
)


def _run_in_thread(app):
    httpd = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# --- 테스트 --------------------------------------------------------------

def test_health(tmpdir: Path):
    app = create_app(FakeConfig({}, tmpdir / "events.db"))
    r = app.test_client().get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_regions_empty_initially(tmpdir: Path):
    app = create_app(FakeConfig({}, tmpdir / "events.db"))
    r = app.test_client().get("/api/regions")
    assert r.get_json() == []


def test_post_event_missing_required_field(tmpdir: Path):
    app = create_app(FakeConfig({}, tmpdir / "events.db"))
    r = app.test_client().post("/api/events", json={"name": "plastic"})
    assert r.status_code == 400


def test_post_event_missing_region(tmpdir: Path):
    app = create_app(FakeConfig({}, tmpdir / "events.db"))
    r = app.test_client().post("/api/events", json=BASE_EVENT)
    assert r.status_code == 400


def test_post_event_success_and_query(tmpdir: Path):
    cfg = FakeConfig({}, tmpdir / "events.db")
    app = create_app(cfg)
    client = app.test_client()

    payload = {**BASE_EVENT, "region": "가천관"}
    r = client.post("/api/events", json=payload)
    assert r.status_code == 201

    assert client.get("/api/regions").get_json() == ["가천관"]
    db = EventDB(cfg)
    assert db.count(region="가천관") == 1


def test_remote_logger_disabled_does_nothing(tmpdir: Path):
    cfg = FakeConfig({"server": {"enabled": False}}, tmpdir / "events.db")
    remote = RemoteLogger(cfg)
    assert remote.enabled is False
    assert remote.send(**BASE_EVENT) is False


def test_remote_logger_enabled_without_url_disables(tmpdir: Path):
    cfg = FakeConfig({"server": {"enabled": True}}, tmpdir / "events.db")  # url 없음
    remote = RemoteLogger(cfg)
    assert remote.enabled is False


def test_remote_logger_unreachable_server_returns_false(tmpdir: Path):
    cfg = FakeConfig(
        {"server": {"enabled": True, "url": "http://127.0.0.1:1/api/events", "timeout_sec": 1},
         "device": {"region": "가천관"}},
        tmpdir / "events.db",
    )
    remote = RemoteLogger(cfg)
    assert remote.send(**BASE_EVENT) is False


def test_remote_logger_sends_to_running_server(tmpdir: Path):
    cfg_server = FakeConfig({}, tmpdir / "server.db")
    app = create_app(cfg_server)
    httpd = _run_in_thread(app)
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/api/events"
        cfg_edge = FakeConfig(
            {"server": {"enabled": True, "url": url, "timeout_sec": 2},
             "device": {"region": "1기숙사"}},
            tmpdir / "edge.db",
        )
        remote = RemoteLogger(cfg_edge)
        assert remote.send(**BASE_EVENT) is True

        db_server = EventDB(cfg_server)
        assert db_server.regions() == ["1기숙사"]
    finally:
        httpd.shutdown()


TESTS = [
    test_health,
    test_regions_empty_initially,
    test_post_event_missing_required_field,
    test_post_event_missing_region,
    test_post_event_success_and_query,
    test_remote_logger_disabled_does_nothing,
    test_remote_logger_enabled_without_url_disables,
    test_remote_logger_unreachable_server_returns_false,
    test_remote_logger_sends_to_running_server,
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
