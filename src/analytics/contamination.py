"""오염 / 오분류 탐지 — 캡스톤에서 가장 차별화되는 분석.

4-클래스 객체탐지만으로 '음식물 오염'을 직접 보긴 어렵다. 그래서 두 개의
관측 가능한 대리 신호(proxy)로 문제를 정의한다:

  1) 오분류(misclassification) : 통(bin)의 목표 클래스와 다른 물체가 들어옴
        예) '플라스틱' 통에 캔/유리/종이가 잡힘
  2) 오염/불확실(contamination·uncertain) : 신뢰도가 애매구간(detect~trusted)
        깨끗하고 전형적인 재활용품은 모델이 자신있게(>trusted) 맞힌다.
        젖었거나·찌그러졌거나·라벨 붙은 '오염된' 물체는 신뢰도가 떨어진다.
        → 낮은 신뢰도를 '오염 의심' 대리 지표로 활용한다. (+ flicker)

이 모듈은 탐지 1건을 받아 위 신호로 분류(classify)하고,
실시간 화면용 한글 안내 메시지와, 최근 구간 오염률(rolling rate)을 만든다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Signal:
    name: str            # 영문 클래스명
    conf: float
    band: str            # 'uncertain' | 'trusted'
    misclassified: bool  # 통 목표와 불일치
    uncertain: bool      # 애매구간 또는 flicker
    flickered: bool
    message: str         # 화면 표시용 한글 안내
    severity: str        # 'ok' | 'warn' | 'alert'

    @property
    def needs_review(self) -> bool:
        """라벨 검수/오염 검토가 필요한 건인가 = 능동학습 후보."""
        return self.uncertain or self.misclassified


class ContaminationDetector:
    def __init__(self, cfg):
        self.bin_target = cfg.get("bin.target")          # None 가능(키오스크 모드)
        self.t_detect = float(cfg.get("confidence.detect", 0.35))
        self.t_trust = float(cfg.get("confidence.trusted", 0.65))
        self.cfg = cfg
        # 최근 N건으로 실시간 오염률 추정
        self._recent: deque[int] = deque(maxlen=50)   # 1=문제건, 0=정상

    def classify(self, name: str, conf: float, flickered: bool = False) -> Signal:
        band = "trusted" if conf >= self.t_trust else "uncertain"
        uncertain = (band == "uncertain") or flickered
        misclassified = self.bin_target is not None and name != self.bin_target

        ko = self.cfg.ko(name)
        if misclassified:
            severity = "alert"
            tgt = self.cfg.ko(self.bin_target)
            msg = f"⚠ 오분류: 여기는 '{tgt}' 통 — '{ko}' 빼주세요"
        elif uncertain:
            severity = "warn"
            reason = "흔들림" if flickered else f"낮은확신 {conf:.0%}"
            msg = f"? 확인필요: {ko} ({reason}) — 오염/검수 후보"
        else:
            severity = "ok"
            msg = f"✓ {ko} {conf:.0%}"

        self._recent.append(1 if (misclassified or uncertain) else 0)
        return Signal(name, conf, band, misclassified, uncertain, flickered, msg, severity)

    @property
    def rolling_contamination_rate(self) -> float:
        """최근 구간 오염+오분류 비율 (0~1). 실시간 표시용."""
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)
