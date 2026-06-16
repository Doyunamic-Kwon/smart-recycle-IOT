"""모델·카메라 없이 분석/대시보드를 체험하기 위한 데모 데이터 생성.

여러 '지역'에 설치된 장치가 있다고 가정하고, 각 지역마다 다른 통(bin) 목표와
오염률 특성을 가진 가짜 이벤트를 events.db 에 채워 넣는다.
대시보드의 지역별 비교 기능을 바로 확인할 수 있다.

실행:
  python scripts/seed_demo.py                # 최근 14일치 데모 데이터 생성
  python scripts/seed_demo.py --days 30      # 최근 30일치
  python scripts/seed_demo.py --reset        # 기존 events.db 를 비우고 새로 생성
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.database import EventDB  # noqa: E402
from src.config import Config  # noqa: E402

# 지역별 데모 프로필: 통(bin) 목표 / 활동량(하루 평균 배출 수) / 오염·오분류율
REGIONS = [
    {"name": "가천관",   "bin_target": "paper",   "daily_avg": 70,  "problem_rate": 0.10},
    {"name": "AI공학관", "bin_target": "plastic", "daily_avg": 130, "problem_rate": 0.08},
    {"name": "1기숙사",  "bin_target": "can",     "daily_avg": 100, "problem_rate": 0.15},
    {"name": "2기숙사",  "bin_target": "glass",   "daily_avg": 60,  "problem_rate": 0.05},
    {"name": "3기숙사",  "bin_target": "plastic", "daily_avg": 90,  "problem_rate": 0.20},
]

ALL_CLASSES = ["plastic", "can", "glass", "paper"]

# 시간대별 배출 가중치 (출근/점심/퇴근 시간대에 피크)
HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2,        # 0~5시
    4, 8, 10, 6, 5, 6,       # 6~11시
    9, 6, 4, 4, 5, 7,        # 12~17시
    10, 9, 6, 4, 3, 2,       # 18~23시
]


def make_event(cfg, region: dict, ts: datetime) -> dict:
    t_detect = float(cfg.get("confidence.detect", 0.35))
    t_trust = float(cfg.get("confidence.trusted", 0.65))
    bin_target = region["bin_target"]

    if random.random() < region["problem_rate"]:
        if random.random() < 0.5:
            # 오분류: 다른 재질이 들어옴 (자신있게 인식되는 경우가 많음)
            name = random.choice([c for c in ALL_CLASSES if c != bin_target])
            conf = round(random.uniform(t_trust, 0.97), 4)
        else:
            # 오염/불확실: 맞는 재질이지만 신뢰도가 애매함
            name = bin_target
            conf = round(random.uniform(t_detect, t_trust), 4)
    else:
        name = bin_target
        conf = round(random.uniform(t_trust, 0.99), 4)

    flickered = random.random() < 0.03
    band = "trusted" if conf >= t_trust else "uncertain"
    uncertain = band == "uncertain" or flickered
    misclassified = name != bin_target

    return dict(
        name=name,
        cls_id=ALL_CLASSES.index(name),
        conf=conf,
        band=band,
        bin_target=bin_target,
        misclassified=misclassified,
        uncertain=uncertain,
        flickered=flickered,
        ts=ts,
        region=region["name"],
    )


def seed(cfg, days: int) -> int:
    db = EventDB(cfg)
    now = datetime.now()
    total = 0

    for region in REGIONS:
        for day_offset in range(days, 0, -1):
            day = now - timedelta(days=day_offset)
            # 요일 효과: 주말은 활동량 70%
            day_factor = 0.7 if day.weekday() >= 5 else 1.0
            day_count = max(1, int(random.gauss(region["daily_avg"] * day_factor, 8)))

            for _ in range(day_count):
                hour = random.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                ts = day.replace(hour=hour, minute=minute, second=second, microsecond=0)
                db.log(**make_event(cfg, region, ts))
                total += 1

    return total


def main():
    ap = argparse.ArgumentParser(description="Yollo 데모 데이터 생성 (지역별)")
    ap.add_argument("--days", type=int, default=14, help="생성할 기간(일)")
    ap.add_argument("--reset", action="store_true", help="생성 전 기존 events.db 비우기")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config(args.config)

    if args.reset:
        db_path = cfg.path_of("database.path")
        if db_path.exists():
            db_path.unlink()
            print(f"[데모] 기존 DB 삭제: {db_path}")

    total = seed(cfg, args.days)
    regions = ", ".join(r["name"] for r in REGIONS)
    print(f"[데모] {len(REGIONS)}개 지역({regions}), 최근 {args.days}일치 총 {total}건 생성 완료.")
    print("       대시보드 실행: streamlit run scripts/dashboard.py")


if __name__ == "__main__":
    main()
