"""Smart Recycle Dashboard — Flask API + Frontend.

엔드포인트:
  GET  /                  대시보드 HTML
  POST /api/events        라즈베리파이에서 인식 결과 수신
  GET  /api/events        최근 이벤트 목록 (JSON)
  GET  /api/stats         통계 (JSON)
  GET  /api/health        헬스체크

실행:
  python app.py                         # 개발용
  gunicorn -w 2 -b 0.0.0.0:8000 app:app  # AWS 운영용
"""
from __future__ import annotations

import base64
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import EventDB

DB_PATH = os.environ.get("DB_PATH", "data/events.db")
PORT = int(os.environ.get("PORT", 8000))
HEARTBEAT_TIMEOUT = 60  # 이 시간(초) 동안 heartbeat 없으면 오프라인

app = Flask(__name__, static_folder="static")
CORS(app)
db = EventDB(DB_PATH)

# region → 마지막 heartbeat 시각 (메모리 저장, 재시작 시 초기화)
_heartbeats: dict[str, float] = {}

CAPTURES_DIR = Path("static/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
LATEST_IMG = CAPTURES_DIR / "latest.jpg"
UNCERTAIN_IMG = CAPTURES_DIR / "uncertain_latest.jpg"

# 검토 대기 중인 uncertain 이벤트 (메모리, 재시작 시 초기화)
_pending_review: dict | None = None

REQUIRED = ["name", "cls_id", "conf"]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/events")
def post_event():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400
    if missing := [k for k in REQUIRED if k not in data]:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    ts = None
    if raw_ts := data.get("ts"):
        try:
            ts = datetime.fromisoformat(raw_ts)
        except ValueError:
            return jsonify({"error": "invalid ts (ISO8601 expected)"}), 400

    conf = float(data["conf"])
    band = "trusted" if conf >= 0.50 else "uncertain"
    is_uncertain = bool(data.get("uncertain", band == "uncertain"))

    db.log(
        name=str(data["name"]),
        cls_id=int(data["cls_id"]),
        conf=conf,
        region=str(data.get("region", "unknown")),
        band=data.get("band", band),
        misclassified=bool(data.get("misclassified", False)),
        uncertain=is_uncertain,
        flickered=bool(data.get("flickered", False)),
        ts=ts,
    )

    # uncertain 이벤트면 이미지 유무와 관계없이 검토 대기에 등록
    if is_uncertain:
        global _pending_review
        _pending_review = {
            "id": str(uuid.uuid4()),
            "name": str(data["name"]),
            "conf": conf,
            "region": str(data.get("region", "unknown")),
            "ts": (ts or datetime.now()).isoformat(timespec="seconds"),
            "has_image": False,
        }

    if img_b64 := data.get("image_b64"):
        try:
            img_bytes = base64.b64decode(img_b64)
            LATEST_IMG.write_bytes(img_bytes)
            if is_uncertain and _pending_review:
                UNCERTAIN_IMG.write_bytes(img_bytes)
                _pending_review["has_image"] = True
        except Exception:
            pass

    return jsonify({"status": "ok"}), 201


@app.get("/api/events")
def get_events():
    limit = request.args.get("limit", 50, type=int)
    region = request.args.get("region") or None
    return jsonify(db.recent_events(limit=limit, region=region))


@app.get("/api/stats")
def get_stats():
    region = request.args.get("region") or None
    return jsonify(db.stats(region=region))


@app.post("/api/heartbeat")
def post_heartbeat():
    data = request.get_json(silent=True) or {}
    region = str(data.get("region", "unknown"))
    _heartbeats[region] = time.time()
    return jsonify({"status": "ok"}), 200


@app.get("/api/heartbeat")
def get_heartbeat():
    now = time.time()
    status = {
        region: {
            "online": (now - ts) < HEARTBEAT_TIMEOUT,
            "seconds_ago": int(now - ts),
            "last_seen": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
        }
        for region, ts in _heartbeats.items()
    }
    return jsonify(status)


@app.get("/api/uncertain")
def get_uncertain():
    return jsonify(_pending_review)


@app.post("/api/label")
def post_label():
    global _pending_review
    data = request.get_json(silent=True) or {}
    review_id = data.get("review_id")
    if _pending_review is None or _pending_review.get("id") != review_id:
        return jsonify({"error": "no matching pending review"}), 400
    corrected = data.get("label")
    if corrected:
        # 보정 라벨로 DB에 추가 기록
        cls_map = {"plastic": 0, "can": 1, "glass": 2, "paper": 3}
        db.log(
            name=corrected,
            cls_id=cls_map.get(corrected, -1),
            conf=_pending_review["conf"],
            region=_pending_review["region"],
            band="trusted",
            misclassified=corrected != _pending_review["name"],
            uncertain=False,
            flickered=False,
        )
    _pending_review = None
    return jsonify({"status": "ok"})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
