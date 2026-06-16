"""중앙 서버로 확정 이벤트 전송 (선택 기능).

여러 지역에 배포된 장치들이 각자 로컬 DB에 기록하는 것과 **동시에**,
`config.yaml`의 `server.enabled`가 true 이면 중앙 서버(scripts/server.py)의
`/api/events` 로 같은 이벤트를 HTTP POST 한다. 중앙 서버는 이를 region과
함께 하나의 DB에 모아 저장하고, 대시보드(scripts/dashboard.py)가 그 DB를
읽어 지역별로 비교한다.

네트워크 문제로 전송에 실패해도 실시간 인식 루프(로컬 기록)에는
영향을 주지 않도록 짧은 timeout 으로 예외를 흡수한다.
표준 라이브러리(urllib)만 사용해 Pi5 측에 추가 의존성이 없다.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime


class RemoteLogger:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("server.enabled", False))
        self.url = cfg.get("server.url")
        self.timeout = float(cfg.get("server.timeout_sec", 2))
        self.region = cfg.get("device.region")
        if self.enabled and not self.url:
            self.enabled = False

    def send(self, **event) -> bool:
        """이벤트 1건을 중앙 서버로 전송. 비활성/실패 시 False (예외를 던지지 않음)."""
        if not self.enabled:
            return False

        payload = dict(event)
        payload.setdefault("region", self.region)
        ts = payload.get("ts")
        if isinstance(ts, datetime):
            payload["ts"] = ts.isoformat(timespec="seconds")

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False
