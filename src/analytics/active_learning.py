"""자동 데이터 수집 (능동학습, Active Learning).

모델이 '헷갈린' 순간만 골라 저장해 재학습 데이터셋을 스스로 키운다.
무엇이 헷갈린 순간인가 → contamination 모듈이 만든 needs_review 신호:
  · 신뢰도 애매구간(detect~trusted)
  · 통 목표와 불일치(오분류)
  · 프레임 간 클래스 흔들림(flicker)

저장물:
  images/<ts>.jpg              원본 프레임
  labels/<ts>.txt              YOLO 형식 의사라벨(pseudo-label) — 검수 후 학습에 투입
  crops/<ts>_<i>_<name>.jpg    객체 crop (검수 편의)

스토리지 폭주를 막기 위해 cooldown(초)과 하루 상한(max_per_day)을 둔다.
이렇게 모은 데이터를 사람이 라벨 검수 → train.py 재학습 → 모델이 점점 똑똑해지는
self-improving 루프가 완성된다.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path


class ActiveLearningCollector:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("active_learning.enabled", True))
        self.save_dir = cfg.path_of("active_learning.save_dir")
        self.save_crops = bool(cfg.get("active_learning.save_crops", True))
        self.cooldown = float(cfg.get("active_learning.cooldown_sec", 5))
        self.max_per_day = int(cfg.get("active_learning.max_per_day", 500))

        self.img_dir = self.save_dir / "images"
        self.lbl_dir = self.save_dir / "labels"
        self.crop_dir = self.save_dir / "crops"
        for d in (self.img_dir, self.lbl_dir, self.crop_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._last_save = 0.0
        self._day = date.today()
        self._count_today = self._count_existing_today()

    def _count_existing_today(self) -> int:
        prefix = datetime.now().strftime("%Y%m%d")
        return len(list(self.img_dir.glob(f"{prefix}_*.jpg")))

    def _roll_day(self):
        today = date.today()
        if today != self._day:
            self._day = today
            self._count_today = 0

    def maybe_save(self, frame, items: list[tuple]) -> bool:
        """검수 후보 프레임을 조건부 저장.

        items: [(detection, reason), ...]  reason 예: 'uncertain'|'misclass'|'flicker'
        반환: 저장했으면 True.
        """
        if not self.enabled or not items:
            return False
        self._roll_day()
        now = time.time()
        if now - self._last_save < self.cooldown:
            return False
        if self._count_today >= self.max_per_day:
            return False

        import cv2

        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        reasons = "-".join(sorted({r for _, r in items}))
        stem = f"{ts}_{reasons}"

        # 1) 원본 프레임
        cv2.imwrite(str(self.img_dir / f"{stem}.jpg"), frame)

        # 2) YOLO 의사라벨 (검수용)
        lines = []
        for det, _ in items:
            x1, y1, x2, y2 = det.xyxy
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{det.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (self.lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 3) crop
        if self.save_crops:
            for i, (det, _) in enumerate(items):
                x1, y1, x2, y2 = det.xyxy
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    cv2.imwrite(
                        str(self.crop_dir / f"{stem}_{i}_{det.name}.jpg"), crop
                    )

        self._last_save = now
        self._count_today += 1
        return True

    @property
    def saved_today(self) -> int:
        return self._count_today
