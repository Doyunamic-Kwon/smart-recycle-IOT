#!/usr/bin/env python3
"""pi_companion.py — smart_recycle.py stdout 파싱 → 대시보드 서버 전송.

smart_recycle.py 를 수정하지 않고, stdout 을 파이프해서 인식 결과를 서버로 보냄.

사용법:
  python3 -u smart_recycle.py 2>&1 | python3 pi_companion.py

환경변수:
  DASHBOARD_URL   서버 URL  (예: https://smart-recycle-iot-production.up.railway.app)
  DEVICE_REGION   지역명    (예: AI공학관)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import threading
from datetime import datetime

try:
    import requests
except ImportError:
    print("[pi_companion] ERROR: requests 미설치. pip install requests", file=sys.stderr)
    sys.exit(1)

DETECT_RE = re.compile(r"detected:\s*(\w+)\s*\(conf:\s*([\d.]+)\)")

CLASS_IDS = {"plastic": 0, "can": 1, "glass": 2, "paper": 3}
TRUSTED_THRESHOLD = 0.50
DEDUP_COOLDOWN = 5.0      # 같은 클래스 재전송 차단 시간(초)
HEARTBEAT_INTERVAL = 30   # 헬스비트 전송 주기(초)


# ── Dedup: 클래스별 마지막 전송 시각 추적 ──────────────────────────────────
_last_sent: dict[str, float] = {}

def is_duplicate(name: str) -> bool:
    now = time.time()
    if now - _last_sent.get(name, 0) < DEDUP_COOLDOWN:
        return True
    _last_sent[name] = now
    return False


# ── 이미지 읽기 ────────────────────────────────────────────────────────────
def get_latest_image_b64() -> str | None:
    """현재 디렉터리의 가장 최근 waste_*.jpg 를 base64로 반환.
    smart_recycle.py 가 print 후 imwrite 하므로 0.3초 대기 후 읽음."""
    import base64
    import glob as _glob
    time.sleep(0.3)
    files = sorted(_glob.glob("waste_*.jpg"))
    if not files:
        return None
    try:
        with open(files[-1], "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


# ── 이벤트 전송 ────────────────────────────────────────────────────────────
def send_event(api_url: str, name: str, conf: float, region: str) -> bool:
    cls_id = CLASS_IDS.get(name, -1)
    band = "trusted" if conf >= TRUSTED_THRESHOLD else "uncertain"
    payload = {
        "name": name,
        "cls_id": cls_id,
        "conf": conf,
        "region": region,
        "band": band,
        "misclassified": False,
        "uncertain": band == "uncertain",
        "flickered": False,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "image_b64": get_latest_image_b64(),
    }
    try:
        r = requests.post(api_url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"[pi_companion] 서버 연결 실패", file=sys.stderr)
    except requests.exceptions.Timeout:
        print("[pi_companion] 요청 타임아웃", file=sys.stderr)
    except requests.exceptions.HTTPError as e:
        print(f"[pi_companion] HTTP 오류: {e}", file=sys.stderr)
    return False


# ── 헬스비트 (백그라운드 스레드) ───────────────────────────────────────────
def heartbeat_loop(heartbeat_url: str, region: str, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            requests.post(
                heartbeat_url,
                json={"region": region, "ts": datetime.now().isoformat(timespec="seconds")},
                timeout=5,
            )
        except Exception:
            pass
        stop.wait(HEARTBEAT_INTERVAL)


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Smart Recycle → Dashboard 연결")
    parser.add_argument("--server", default=os.environ.get("DASHBOARD_URL", ""))
    parser.add_argument("--region", default=os.environ.get("DEVICE_REGION", "Pi"))
    args = parser.parse_args()

    if not args.server:
        print("[pi_companion] ERROR: DASHBOARD_URL 환경변수 또는 --server 필요", file=sys.stderr)
        sys.exit(1)

    base = args.server.rstrip("/")
    api_url = base + "/api/events"
    heartbeat_url = base + "/api/heartbeat"

    print(f"[pi_companion] 시작 → {api_url} (region={args.region})", file=sys.stderr)

    # 헬스비트 스레드 시작
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=heartbeat_loop, args=(heartbeat_url, args.region, stop_event), daemon=True
    )
    hb_thread.start()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            m = DETECT_RE.search(line)
            if not m:
                continue

            name, conf = m.group(1), float(m.group(2))

            # 중복 제거: 같은 클래스 5초 이내 재감지 무시
            if is_duplicate(name):
                print(f"[pi_companion] dup skip → {name}", file=sys.stderr)
                continue

            ok = send_event(api_url, name, conf, args.region)
            print(f"[pi_companion] {'✓' if ok else '✗'} {name} conf={conf:.2f}", file=sys.stderr)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
