#!/usr/bin/env python3
"""pi_companion.py — smart_recycle.py stdout 파싱 → 대시보드 서버 전송.

smart_recycle.py 를 수정하지 않고, stdout 을 파이프해서 인식 결과를 서버로 보냄.

사용법:
  python smart_recycle.py 2>&1 | python pi_companion.py --server http://<AWS_IP>:8000 --region "AI공학관"

환경변수로도 설정 가능:
  export DASHBOARD_URL=http://<AWS_IP>:8000
  export DEVICE_REGION=AI공학관
  python smart_recycle.py 2>&1 | python pi_companion.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("[pi_companion] ERROR: requests 미설치. pip install requests", file=sys.stderr)
    sys.exit(1)

DETECT_RE = re.compile(r"detected:\s*(\w+)\s*\(conf:\s*([\d.]+)\)")

CLASS_IDS = {"plastic": 0, "can": 1, "glass": 2, "paper": 3}
TRUSTED_THRESHOLD = 0.65


def send_event(url: str, name: str, conf: float, region: str) -> bool:
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
    }
    try:
        r = requests.post(url, json=payload, timeout=5)
        r.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"[pi_companion] 서버 연결 실패: {url}", file=sys.stderr)
    except requests.exceptions.Timeout:
        print("[pi_companion] 요청 타임아웃", file=sys.stderr)
    except requests.exceptions.HTTPError as e:
        print(f"[pi_companion] HTTP 오류: {e}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Smart Recycle → Dashboard 연결")
    parser.add_argument(
        "--server",
        default=os.environ.get("DASHBOARD_URL", ""),
        help="대시보드 서버 URL (예: http://1.2.3.4:8000)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("DEVICE_REGION", "Pi"),
        help="장치 지역명 (예: AI공학관)",
    )
    args = parser.parse_args()

    if not args.server:
        print("[pi_companion] ERROR: --server 또는 DASHBOARD_URL 환경변수 필요", file=sys.stderr)
        sys.exit(1)

    api_url = args.server.rstrip("/") + "/api/events"
    print(f"[pi_companion] 시작 → {api_url} (region={args.region})", file=sys.stderr)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        m = DETECT_RE.search(line)
        if not m:
            continue

        name, conf = m.group(1), float(m.group(2))
        ok = send_event(api_url, name, conf, args.region)
        status = "✓" if ok else "✗"
        print(f"[pi_companion] {status} {name} conf={conf:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
