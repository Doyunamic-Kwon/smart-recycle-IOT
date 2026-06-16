"""YOLO 추론 래퍼.

학습 결과(.pt), Pi5 배포용 NCNN 디렉터리(*_ncnn_model), ONNX(.onnx) 를
경로만 보고 자동 판별해 동일한 방식으로 추론한다. (ultralytics 가 내부 처리)

    det = YoloDetector(cfg)
    detections = det.infer(frame)   # -> list[Detection]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    cls_id: int
    name: str            # 영문 클래스명
    conf: float
    xyxy: tuple[int, int, int, int]   # (x1, y1, x2, y2) 픽셀

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class YoloDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.weights = cfg.path_of("model.weights")
        self.imgsz = int(cfg.get("model.imgsz", 640))
        self.device = cfg.get("model.device", "cpu")
        self.conf_detect = float(cfg.get("confidence.detect", 0.35))
        self.names = cfg.names

        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"모델 가중치를 찾을 수 없습니다: {self.weights}\n"
                f"  · PC 학습:  python -m src.train\n"
                f"  · Pi5 변환: python scripts/export_ncnn.py"
            )
        # NCNN/ONNX 는 task 메타가 없어 경고가 뜨므로 명시
        self.model = YOLO(str(self.weights), task="detect")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """BGR 프레임 1장 -> Detection 리스트."""
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_detect,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None:
            return out
        for b in r.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            name = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
            out.append(Detection(cls_id, name, conf, (x1, y1, x2, y2)))
        return out
