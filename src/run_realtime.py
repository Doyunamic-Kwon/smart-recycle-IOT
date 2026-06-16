"""Pi5 실시간 메인 루프 — 모든 조각을 하나로 묶는다.

흐름:
  카메라 → YOLO 추론 → 트래커(안정카운트/흔들림) → 오염·오분류 분류
        → 확정건 DB 로깅 → 불확실건 능동학습 저장 → 화면/HUD

실행:
  python -m src.run_realtime                 # 헤드리스(Pi5/SSH)
  python -m src.run_realtime --show          # 창 띄워 디버그(PC)
  python -m src.run_realtime --source 0      # 카메라 소스 덮어쓰기
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from .analytics.active_learning import ActiveLearningCollector
from .analytics.contamination import ContaminationDetector
from .analytics.database import EventDB
from .analytics.remote_logger import RemoteLogger
from .config import Config
from .detector import YoloDetector
from .lcd import LCD, to_display_label
from .tracker import CentroidTracker

# severity → BGR 색상
COLORS = {"ok": (90, 200, 90), "warn": (40, 180, 240), "alert": (60, 60, 235)}


def draw(frame, track, signal):
    import cv2

    x1, y1, x2, y2 = track.xyxy
    color = COLORS.get(signal.severity, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{track.name} {signal.conf:.0%}"
    cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_hud(frame, *, total, contam_rate, saved):
    import cv2

    h = frame.shape[0]
    lines = [
        f"confirmed: {total}",
        f"contamination(rolling): {contam_rate:.0%}",
        f"active-learning saved today: {saved}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Yollo 분리수거 실시간 인식")
    ap.add_argument("--config", default=None, help="config.yaml 경로")
    ap.add_argument("--source", default=None, help="카메라 소스 덮어쓰기 (0 / picamera2 / 파일)")
    ap.add_argument("--show", action="store_true", help="디버그 창 표시")
    args = ap.parse_args()

    cfg = Config(args.config)
    if args.source is not None:
        # 정수면 int 로
        cfg._data.setdefault("camera", {})["source"] = (
            int(args.source) if args.source.isdigit() else args.source
        )

    from .camera import Camera  # cv2 의존 → 지연 import

    detector = YoloDetector(cfg)
    tracker = CentroidTracker(cfg)
    contam = ContaminationDetector(cfg)
    collector = ActiveLearningCollector(cfg)
    db = EventDB(cfg)
    remote = RemoteLogger(cfg)
    bin_target = cfg.get("bin.target")
    if remote.enabled:
        print(f"[Yollo] 중앙 서버로 이벤트 전송 활성화: {remote.url}")

    confirmed_total = db.count()
    print(f"[Yollo] 시작. 누적 확정 {confirmed_total}건. (종료: Ctrl+C 또는 q)")

    t_last_stat = time.time()
    try:
        with Camera(cfg) as cam:
            for frame in cam.frames():
                detections = detector.infer(frame)
                tracked = tracker.update(detections)

                review_items = []
                for track, just_confirmed in tracked:
                    sig = contam.classify(track.name, track.conf, track.flickered)
                    if args.show:
                        draw(frame, track, sig)

                    # 확정된 1건만 DB 로깅 (중복 카운트 방지)
                    if just_confirmed:
                        confirmed_total += 1
                        ts = datetime.now()
                        event = dict(
                            name=track.name,
                            cls_id=track.cls_id,
                            conf=track.conf,
                            band=sig.band,
                            bin_target=bin_target,
                            misclassified=sig.misclassified,
                            uncertain=sig.uncertain,
                            flickered=track.flickered,
                            track_id=track.track_id,
                        )
                        db.log(**event, ts=ts)
                        remote.send(**event, ts=ts)
                        print(f"detected: {to_display_label(track.name)} (conf: {track.conf:.2f})")
                        if sig.severity != "ok":
                            print(f"  · {sig.message}")

                    # 검수 후보 → 능동학습 수집
                    if sig.needs_review:
                        reason = ("misclass" if sig.misclassified
                                  else "flicker" if track.flickered else "uncertain")
                        review_items.append((track, reason))

                if collector.maybe_save(frame, review_items):
                    print(f"  · [능동학습] 검수후보 저장 (오늘 {collector.saved_today}장)")

                if args.show:
                    import cv2

                    draw_hud(frame, total=confirmed_total,
                             contam_rate=contam.rolling_contamination_rate,
                             saved=collector.saved_today)
                    cv2.imshow("Yollo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # 주기적 콘솔 상태(헤드리스)
                if not args.show and time.time() - t_last_stat > 30:
                    print(f"[상태] 확정 {confirmed_total} · "
                          f"오염률(rolling) {contam.rolling_contamination_rate:.0%} · "
                          f"능동학습 {collector.saved_today}장")
                    t_last_stat = time.time()
    except KeyboardInterrupt:
        print("\n[Yollo] 종료 요청.")
    finally:
        if args.show:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"[Yollo] 종료. 총 확정 {confirmed_total}건. "
              f"리포트: python scripts/make_report.py")


if __name__ == "__main__":
    main()
