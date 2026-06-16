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
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import EventDB

DB_PATH = os.environ.get("DB_PATH", "data/events.db")
PORT = int(os.environ.get("PORT", 8000))

app = Flask(__name__, static_folder="static")
CORS(app)
db = EventDB(DB_PATH)

CAPTURES_DIR = Path("static/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
LATEST_IMG = CAPTURES_DIR / "latest.jpg"

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
    band = "trusted" if conf >= 0.65 else "uncertain"

    db.log(
        name=str(data["name"]),
        cls_id=int(data["cls_id"]),
        conf=conf,
        region=str(data.get("region", "unknown")),
        band=data.get("band", band),
        misclassified=bool(data.get("misclassified", False)),
        uncertain=bool(data.get("uncertain", band == "uncertain")),
        flickered=bool(data.get("flickered", False)),
        ts=ts,
    )

    if img_b64 := data.get("image_b64"):
        try:
            LATEST_IMG.write_bytes(base64.b64decode(img_b64))
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


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
