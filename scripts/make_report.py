"""events.db 로부터 분리수거 데이터 리포트를 생성.

실행:
  python scripts/make_report.py                # 최근 7일
  python scripts/make_report.py --days 30      # 최근 30일

cron 등록 예 (매일 23:50 일일 리포트):
  50 23 * * *  cd /home/<id>/Yollo && python scripts/make_report.py --days 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.report import ReportGenerator  # noqa: E402
from src.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Yollo 데이터 리포트 생성")
    ap.add_argument("--days", type=int, default=7, help="대상 기간(일)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config(args.config)
    md = ReportGenerator(cfg).generate(period_days=args.days)
    print(f"[리포트 생성 완료] {md}")


if __name__ == "__main__":
    main()
