"""config.yaml 로더.

어디서 실행해도(PC, Pi5) 프로젝트 루트의 config.yaml 을 찾아 읽고,
점 표기(dot-notation)로 값을 꺼낼 수 있는 얇은 래퍼를 제공한다.

    cfg = Config()
    cfg.get("confidence.detect")        # -> 0.35
    cfg.classes                         # -> {0:'plastic', ...}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 = 이 파일(src/config.py)의 부모의 부모
ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else ROOT / "config.yaml"
        with open(self.path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

    def get(self, dotted: str, default: Any = None) -> Any:
        """'confidence.detect' 같은 점 표기로 중첩 값 조회."""
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path_of(self, dotted: str) -> Path:
        """설정에 적힌 상대경로를 프로젝트 루트 기준 절대경로로 변환."""
        raw = self.get(dotted)
        if raw is None:
            raise KeyError(f"config 경로 키 없음: {dotted}")
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    # 자주 쓰는 항목은 프로퍼티로 노출 ---------------------------------
    @property
    def classes(self) -> dict[int, str]:
        # YAML 키가 문자열일 수 있으니 int 로 정규화
        return {int(k): v for k, v in self.get("classes", {}).items()}

    @property
    def names(self) -> list[str]:
        """class_id 순서대로 영문 이름 리스트."""
        c = self.classes
        return [c[i] for i in sorted(c)]

    @property
    def labels_ko(self) -> dict[str, str]:
        return self.get("labels_ko", {})

    def ko(self, name: str) -> str:
        """영문 클래스명 -> 한글 라벨 (없으면 원문)."""
        return self.labels_ko.get(name, name)


if __name__ == "__main__":
    c = Config()
    print("classes:", c.classes)
    print("bin.target:", c.get("bin.target"))
    print("conf band:", c.get("confidence.detect"), "~", c.get("confidence.trusted"))
