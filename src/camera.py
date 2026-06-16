"""카메라 입력 추상화.

세 가지 소스를 동일한 인터페이스로 제공한다:
  · 정수(0,1...)     -> USB/내장 웹캠 (OpenCV)
  · "picamera2"     -> 라즈베리 파이 카메라 모듈 (picamera2, Pi5 권장)
  · 파일 경로        -> 동영상 파일 (개발/테스트용)

picamera2 는 Pi 에서만 설치되므로 import 를 지연(lazy)시켜
PC 에서도 이 모듈을 문제없이 불러올 수 있게 한다.

    with Camera(cfg) as cam:
        for frame in cam.frames():   # frame: numpy BGR ndarray
            ...
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


class Camera:
    def __init__(self, cfg):
        self.source = cfg.get("camera.source", 0)
        self.width = int(cfg.get("camera.width", 1280))
        self.height = int(cfg.get("camera.height", 720))
        self.fps = int(cfg.get("camera.fps", 15))
        self._backend = None      # "cv2" | "picamera2"
        self._cap = None

    # --- 라이프사이클 -------------------------------------------------
    def open(self):
        if self.source == "picamera2":
            self._open_picamera2()
        else:
            self._open_cv2()
        return self

    def _open_picamera2(self):
        from picamera2 import Picamera2  # Pi 에서만 존재

        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        self._cap = cam
        self._backend = "picamera2"

    def _open_cv2(self):
        import cv2

        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"카메라/동영상을 열 수 없습니다: source={self.source}")
        self._cap = cap
        self._backend = "cv2"

    def read(self) -> np.ndarray | None:
        """프레임 1장(BGR) 반환. 더 없으면 None."""
        if self._backend == "picamera2":
            import cv2

            rgb = self._cap.capture_array()       # picamera2 는 RGB
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        """프레임을 계속 내보내는 제너레이터."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def close(self):
        if self._cap is None:
            return
        if self._backend == "picamera2":
            self._cap.stop()
        else:
            self._cap.release()
        self._cap = None

    # with 문 지원
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
