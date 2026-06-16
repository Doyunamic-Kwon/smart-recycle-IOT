"""분리수거 분류 표시 라벨 변환.

캔/고철류, 병(유리), 플라스틱 → 해당 품목명
그 외 모든 클래스              → 일반쓰레기
"""
from __future__ import annotations

_RECYCLE_LABELS: dict[str, str] = {
    "can":     "캔/고철류",
    "glass":   "병(유리)",
    "plastic": "플라스틱",
}


def to_display_label(class_name: str) -> str:
    """클래스명 → 표시 라벨. 분리 불가능한 품목은 '일반쓰레기'로 반환."""
    return _RECYCLE_LABELS.get(class_name, "일반쓰레기")
