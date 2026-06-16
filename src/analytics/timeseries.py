"""배출 패턴 시계열 분석 + 통(bin) 가득참 예측.

events.db 의 확정 탐지들을 시간축으로 집계해:
  · 시간대(0~23시)별 / 요일별 배출량 패턴
  · 피크 시간대
  · 최근 처리율(items/hour) 기반 '통이 언제 가득 차는가' 예측

'야무진' 포인트: 단순 누적이 아니라 최근 구간의 배출 속도로 ETA(가득참 예상
시각)를 추정 → 수거 타이밍을 미리 알려주는 운영 신호로 쓸 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


def hourly_pattern(df):
    """시간대(0~23)별 배출 건수 Series."""
    import pandas as pd

    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby(df["ts"].dt.hour).size().reindex(range(24), fill_value=0)


def weekday_pattern(df):
    """요일별(월=0 ~ 일=6) 배출 건수 Series."""
    import pandas as pd

    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby(df["ts"].dt.dayofweek).size().reindex(range(7), fill_value=0)


def peak_hours(df, top: int = 3) -> list[tuple[int, int]]:
    """배출이 가장 많은 시간대 top N -> [(hour, count), ...]."""
    hp = hourly_pattern(df)
    if hp.empty or hp.sum() == 0:
        return []
    return [(int(h), int(n)) for h, n in hp.sort_values(ascending=False).head(top).items()]


def throughput_per_hour(df, window_hours: float = 3.0) -> float:
    """최근 window_hours 동안의 평균 배출 속도(items/hour)."""
    if df.empty:
        return 0.0
    now = df["ts"].max()
    recent = df[df["ts"] >= now - timedelta(hours=window_hours)]
    if recent.empty:
        return 0.0
    span = max((recent["ts"].max() - recent["ts"].min()).total_seconds() / 3600.0, 1e-6)
    # 데이터가 짧으면 window 로 나눠 과대추정 방지
    span = max(span, min(window_hours, 1.0))
    return len(recent) / span


@dataclass
class FillForecast:
    current: int          # 마지막 비움 이후 누적 개수
    capacity: int
    rate_per_hour: float
    fill_ratio: float     # 0~1
    hours_to_full: float | None
    eta: datetime | None

    def summary(self) -> str:
        pct = f"{self.fill_ratio:.0%}"
        if self.hours_to_full is None:
            if self.eta is not None:        # remaining==0 → 이미 가득
                return f"통 채움 {pct} · 이미 가득 참 (수거 필요)"
            return f"통 채움 {pct} (배출 없음 → ETA 산정 불가)"
        return (
            f"통 채움 {pct} · 최근 {self.rate_per_hour:.1f}개/시간 · "
            f"약 {self.hours_to_full:.1f}시간 뒤 가득참 "
            f"(예상 {self.eta:%m/%d %H:%M})"
        )


def predict_fill(df, capacity: int, since_epoch_last_empty: float | None = None) -> FillForecast:
    """최근 배출 속도로 통이 가득 찰 시점을 예측."""
    sub = df
    if since_epoch_last_empty is not None and not df.empty:
        sub = df[df["ts_epoch"] >= since_epoch_last_empty]
    current = len(sub)
    rate = throughput_per_hour(sub if not sub.empty else df)
    fill_ratio = min(current / capacity, 1.0) if capacity else 0.0
    remaining = max(capacity - current, 0)

    if rate <= 0 or remaining == 0:
        hours = None
        eta = None if rate <= 0 else datetime.now()
    else:
        hours = remaining / rate
        eta = datetime.now() + timedelta(hours=hours)
    return FillForecast(current, capacity, rate, fill_ratio, hours, eta)
