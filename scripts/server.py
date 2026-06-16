"""중앙 수집 서버 — 여러 지역(장치)의 이벤트를 한 DB로 모은다.

각 장치는 `config.yaml`에서 `server.enabled: true` 로 설정하면, 로컬 DB
기록과 동시에 이 서버의 `/api/events` 로 이벤트를 전송한다(src/analytics/
remote_logger.py). 서버는 받은 이벤트를 region과 함께 자신의 events.db에
저장하고, 같은 머신에서 `streamlit run scripts/dashboard.py` 를 실행하면
지역별로 모인 데이터를 바로 볼 수 있다.

실행:
  python scripts/server.py                # 0.0.0.0:8000 에서 대기
  python scripts/server.py --port 9000

엔드포인트:
  POST /api/events   - 이벤트 1건 수집 (JSON)
  GET  /api/regions  - 현재까지 수집된 지역 목록
  GET  /api/health   - 헬스 체크
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from src.analytics.database import EventDB  # noqa: E402
from src.config import Config  # noqa: E402

REQUIRED_FIELDS = ["name", "cls_id", "conf", "band", "misclassified", "uncertain", "flickered"]


def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or Config()
    db = EventDB(cfg)

    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/regions")
    def regions():
        return jsonify(db.regions())

    @app.post("/api/events")
    def post_event():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid or missing JSON body"}), 400

        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            return jsonify({"error": f"missing fields: {missing}"}), 400
        if not data.get("region"):
            return jsonify({"error": "missing field: region"}), 400

        ts = None
        if data.get("ts"):
            try:
                ts = datetime.fromisoformat(data["ts"])
            except ValueError:
                return jsonify({"error": "invalid ts (ISO8601 expected)"}), 400

        db.log(
            name=data["name"],
            cls_id=int(data["cls_id"]),
            conf=float(data["conf"]),
            band=data["band"],
            bin_target=data.get("bin_target"),
            misclassified=bool(data["misclassified"]),
            uncertain=bool(data["uncertain"]),
            flickered=bool(data["flickered"]),
            track_id=data.get("track_id"),
            region=data["region"],
            ts=ts,
        )
        return jsonify({"status": "ok"}), 201

    return app


def main():
    ap = argparse.ArgumentParser(description="Yollo 중앙 수집 서버")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    app = create_app(Config(args.config))
    print(f"[Yollo 서버] {args.host}:{args.port} 에서 대기 중 (POST /api/events)")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
