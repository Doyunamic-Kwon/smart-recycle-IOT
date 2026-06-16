"""지역별 분리수거 데이터 대시보드 (Streamlit).

events.db 에 쌓인 모든 지역(region)의 이벤트를 한 화면에서 비교·조회한다.
실제 장치가 여러 지역에 배포된 경우, 각 장치의 events.db 를 한 곳에
모아두면(또는 동기화하면) 이 대시보드가 region 컬럼으로 필터링해 보여준다.

실행:
  streamlit run scripts/dashboard.py

데모 데이터가 없다면 먼저:
  python scripts/seed_demo.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics import timeseries  # noqa: E402
from src.analytics.database import EventDB  # noqa: E402
from src.config import Config  # noqa: E402

st.set_page_config(page_title="Yollo 분리수거 대시보드", page_icon="♻️", layout="wide")


@st.cache_resource
def _load_db():
    cfg = Config()
    return cfg, EventDB(cfg)


@st.cache_data(ttl=30)
def _load_df(_db) -> pd.DataFrame:
    return _db.to_dataframe()


PERIOD_OPTIONS = {"오늘": 1, "최근 7일": 7, "최근 14일": 14, "최근 30일": 30, "전체 기간": None}


def main():
    cfg, db = _load_db()
    df_all = _load_df(db)

    st.title("♻ Yollo 분리수거 대시보드")

    if df_all.empty:
        st.warning(
            "아직 기록된 데이터가 없습니다.\n\n"
            "데모 데이터를 생성하려면 터미널에서 "
            "`python scripts/seed_demo.py` 를 실행하세요."
        )
        return

    regions = sorted(r for r in df_all["region"].dropna().unique().tolist())

    with st.sidebar:
        st.header("🔎 필터")
        region_sel = st.selectbox("지역", ["전체"] + regions)
        period_label = st.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=1)
        if st.button("🔄 새로고침"):
            _load_df.clear()
            st.rerun()

    period_days = PERIOD_OPTIONS[period_label]
    df = df_all
    if period_days is not None:
        since = (datetime.now() - timedelta(days=period_days)).timestamp()
        df = df[df["ts_epoch"] >= since]
    if region_sel != "전체":
        df = df[df["region"] == region_sel]

    st.caption(
        f"지역: **{region_sel}** · 기간: **{period_label}** · "
        f"표시 중 {len(df)}건 / 전체 {len(df_all)}건"
    )

    if df.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    _kpi_section(cfg, df)
    st.divider()

    if region_sel == "전체" and len(regions) > 1:
        _region_comparison(cfg, df)
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        _class_distribution(cfg, df)
    with col2:
        _hourly_pattern(df)

    _daily_trend(df)
    st.divider()

    if region_sel != "전체":
        _bin_fill(cfg, df)
        st.divider()

    _recent_events(cfg, df)


def _kpi_section(cfg, df: pd.DataFrame):
    total = len(df)
    mis = int(df["misclassified"].sum())
    unc = int(df["uncertain"].sum())
    correct = total - mis
    accuracy = correct / total if total else 0.0
    contamination = (mis + unc) / total if total else 0.0
    avg_conf = df["conf"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 인식 건수", f"{total:,}건")
    c2.metric("정분류율", f"{accuracy:.1%}")
    c3.metric("오염·오분류율", f"{contamination:.1%}")
    c4.metric("평균 신뢰도", f"{avg_conf:.0%}")


def _region_comparison(cfg, df: pd.DataFrame):
    st.subheader("🗺️ 지역별 비교")

    def _agg(g: pd.DataFrame) -> pd.Series:
        total = len(g)
        mis = int(g["misclassified"].sum())
        unc = int(g["uncertain"].sum())
        return pd.Series(
            {
                "총 건수": total,
                "정분류율": (total - mis) / total if total else 0.0,
                "오염·오분류율": (mis + unc) / total if total else 0.0,
                "평균 신뢰도": g["conf"].mean(),
            }
        )

    summary = df.groupby("region").apply(_agg, include_groups=False)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("지역별 총 인식 건수")
        st.bar_chart(summary["총 건수"])
    with col2:
        st.caption("지역별 오염·오분류율")
        st.bar_chart(summary["오염·오분류율"])

    st.dataframe(
        summary.style.format(
            {"정분류율": "{:.1%}", "오염·오분류율": "{:.1%}", "평균 신뢰도": "{:.0%}"}
        ),
        use_container_width=True,
    )


def _class_distribution(cfg, df: pd.DataFrame):
    st.subheader("🧴 재질별 분포")
    vc = df["name"].value_counts()
    vc.index = [cfg.ko(n) for n in vc.index]
    st.bar_chart(vc)


def _hourly_pattern(df: pd.DataFrame):
    st.subheader("🕒 시간대별 배출 패턴")
    hp = timeseries.hourly_pattern(df)
    st.line_chart(hp)


def _daily_trend(df: pd.DataFrame):
    st.subheader("📈 일자별 추세")
    daily = df.copy()
    daily["date"] = daily["ts"].dt.date
    dc = daily.groupby("date").size()
    st.line_chart(dc)


def _bin_fill(cfg, df: pd.DataFrame):
    st.subheader("🗑️ 통(bin) 채움 예측")
    capacity = int(cfg.get("bin.capacity_items", 200))
    today_start = pd.Timestamp.now().normalize().timestamp()
    forecast = timeseries.predict_fill(df, capacity, since_epoch_last_empty=today_start)

    c1, c2, c3 = st.columns(3)
    c1.metric("현재 채움률", f"{forecast.fill_ratio:.0%}")
    c2.metric("최근 처리율", f"{forecast.rate_per_hour:.1f}개/시간")
    eta_str = f"{forecast.eta:%m/%d %H:%M}" if forecast.eta else "산정 불가"
    c3.metric("가득 예상 시각", eta_str)
    st.progress(min(forecast.fill_ratio, 1.0))
    st.caption(forecast.summary())


def _recent_events(cfg, df: pd.DataFrame):
    st.subheader("🕘 최근 이벤트")
    cols = ["ts", "region", "name", "conf", "band", "misclassified", "uncertain"]
    view = df.sort_values("ts_epoch", ascending=False).head(50)[cols].copy()
    view["name"] = view["name"].map(cfg.ko)
    view["misclassified"] = view["misclassified"].astype(bool)
    view["uncertain"] = view["uncertain"].astype(bool)
    view.columns = ["시각", "지역", "재질", "신뢰도", "밴드", "오분류", "오염/불확실"]
    st.dataframe(view, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
