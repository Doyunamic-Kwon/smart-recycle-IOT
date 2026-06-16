"""아주 가벼운 중심점(centroid) 트래커.

목적은 두 가지 '야무진' 신호를 만드는 것:

1) 안정 카운트(stable count)
   같은 물체가 stable_frames 프레임 연속 같은 클래스로 잡혀야 1개로 '확정'.
   → 한 물체가 카메라 앞에 5초 있다고 5번 세는 중복 카운트를 막는다.

2) 흔들림(flicker) = 불확실성 신호
   추적되는 동안 예측 클래스가 바뀌면(plastic↔glass 등) 모델이 헷갈리는 것.
   → 능동학습 1순위 후보로 표시.

외부 라이브러리 없이 동작하도록 IoU/거리 매칭만 사용한다. (Pi5 가벼움)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


@dataclass
class Track:
    track_id: int
    cls_id: int
    name: str
    center: tuple[float, float]
    xyxy: tuple[int, int, int, int]
    conf: float
    hits: int = 1                       # 연속 매칭된 프레임 수
    misses: int = 0                     # 연속 미매칭 프레임 수
    confirmed: bool = False             # stable_frames 도달해 카운트 확정됐는가
    flickered: bool = False             # 추적 중 클래스가 바뀐 적 있는가
    seen_classes: set = field(default_factory=set)


class CentroidTracker:
    def __init__(self, cfg):
        self.stable_frames = int(cfg.get("tracking.stable_frames", 3))
        self.flicker_on = bool(cfg.get("tracking.flicker_is_uncertain", True))
        self._max_dist = 120.0          # 같은 물체로 볼 최대 중심 이동(px)
        self._max_misses = 8            # 이만큼 안 보이면 트랙 폐기
        self._next_id = 0
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def update(self, detections: list[Detection]) -> list[tuple[Track, bool]]:
        """현재 프레임 탐지로 트랙 갱신.

        반환: (track, just_confirmed) 리스트.
        just_confirmed=True 인 트랙이 이번 프레임에 '새로 확정된 1개'.
        """
        unmatched = set(self.tracks.keys())
        results: list[tuple[Track, bool]] = []

        for det in detections:
            # 가장 가까운 기존 트랙 찾기
            best_id, best_d = None, self._max_dist
            for tid in unmatched:
                d = self._dist(self.tracks[tid].center, det.center)
                if d < best_d:
                    best_id, best_d = tid, d

            if best_id is None:
                # 새 트랙 생성
                t = Track(
                    track_id=self._next_id,
                    cls_id=det.cls_id,
                    name=det.name,
                    center=det.center,
                    xyxy=det.xyxy,
                    conf=det.conf,
                    seen_classes={det.cls_id},
                )
                self.tracks[self._next_id] = t
                self._next_id += 1
                results.append((t, False))
                continue

            # 기존 트랙 갱신
            t = self.tracks[best_id]
            unmatched.discard(best_id)
            if det.cls_id != t.cls_id and self.flicker_on:
                t.flickered = True
            t.cls_id, t.name, t.conf = det.cls_id, det.name, det.conf
            t.center, t.xyxy = det.center, det.xyxy
            t.seen_classes.add(det.cls_id)
            t.hits += 1
            t.misses = 0

            just_confirmed = False
            if not t.confirmed and t.hits >= self.stable_frames:
                t.confirmed = True
                just_confirmed = True
            results.append((t, just_confirmed))

        # 매칭 안 된 트랙은 miss 증가, 오래되면 폐기
        for tid in list(unmatched):
            self.tracks[tid].misses += 1
            if self.tracks[tid].misses > self._max_misses:
                del self.tracks[tid]

        return results
