"""재질별 통계 + 리포트 자동 생성.

events.db 를 읽어 한 번에:
  · 재질(클래스)별 누적/비율
  · 정분류율(올바르게 들어온 비율) · 오분류율 · 오염(불확실)률
  · 시간대 패턴 그래프, 일자별 추세 그래프, 재질 분포 그래프 (PNG)
  · 요약 지표 CSV
  · 사람이 읽는 Markdown 리포트
를 reports/ 에 떨군다.

matplotlib 은 헤드리스(Pi5/SSH)에서도 되도록 Agg 백엔드 사용.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import timeseries  # noqa: E402
from .database import EventDB  # noqa: E402


def _korean_font() -> str | None:
    """설치돼 있으면 한글 가능 폰트명을 반환, 없으면 None."""
    from matplotlib import font_manager

    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Malgun Gothic", "AppleGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Pretendard"):
        if name in avail:
            return name
    return None


class ReportGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db = EventDB(cfg)
        self.out_dir = cfg.path_of("report.out_dir")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 한글 폰트가 있으면 그래프에 한글 사용, 없으면 영문 라벨로 폴백
        self.kfont = _korean_font()
        if self.kfont:
            plt.rcParams["font.family"] = self.kfont
            plt.rcParams["axes.unicode_minus"] = False

    def generate(self, period_days: int = 7) -> Path:
        since = (datetime.now() - timedelta(days=period_days)).timestamp()
        df = self.db.to_dataframe(since_epoch=since)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_dir / f"report_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            md = run_dir / "report.md"
            md.write_text(
                f"# Yollo 분리수거 리포트\n\n최근 {period_days}일간 기록된 데이터가 없습니다.\n",
                encoding="utf-8",
            )
            return md

        metrics = self._metrics(df)
        self._write_csv(df, run_dir)
        charts = self._charts(df, run_dir)
        md = self._write_markdown(metrics, charts, run_dir, period_days)
        return md

    # --- 지표 계산 ----------------------------------------------------
    def _metrics(self, df) -> dict:
        import pandas as pd

        total = len(df)
        mis = int(df["misclassified"].sum())
        unc = int(df["uncertain"].sum())
        correct = total - mis
        per_class = df["name"].value_counts().to_dict()
        # 통 가득참 예측은 '오늘 0시 이후' 누적으로 (마지막 비움 대용)
        today_start = pd.Timestamp.now().normalize().timestamp()
        capacity = int(self.cfg.get("bin.capacity_items", 200))
        return {
            "total": total,
            "per_class": per_class,
            "misclassified": mis,
            "uncertain": unc,
            "accuracy": correct / total if total else 0.0,       # 정분류율
            "contamination_rate": (mis + unc) / total if total else 0.0,
            "bin_target": self.cfg.get("bin.target"),
            "fill": timeseries.predict_fill(df, capacity, since_epoch_last_empty=today_start),
            "peaks": timeseries.peak_hours(df, top=3),
        }

    # --- CSV ----------------------------------------------------------
    def _write_csv(self, df, run_dir: Path):
        import pandas as pd

        by_class = (
            df.groupby("name")
            .agg(
                count=("id", "size"),
                avg_conf=("conf", "mean"),
                misclassified=("misclassified", "sum"),
                uncertain=("uncertain", "sum"),
            )
            .reset_index()
        )
        by_class.to_csv(run_dir / "summary_by_class.csv", index=False, encoding="utf-8-sig")

        daily = df.copy()
        daily["date"] = daily["ts"].dt.date
        daily.groupby("date").size().to_csv(
            run_dir / "daily_counts.csv", header=["count"], encoding="utf-8-sig"
        )

    # --- 그래프 -------------------------------------------------------
    def _charts(self, df, run_dir: Path) -> dict:
        charts = {}
        ko = self.cfg.labels_ko

        # 1) 재질 분포 (한글 폰트 있으면 한글, 없으면 영문 라벨)
        vc = df["name"].value_counts()
        labels = [ko.get(n, n) for n in vc.index] if self.kfont else list(vc.index)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(labels, vc.values, color="#4C9F70")
        ax.set_title("재질별 분포" if self.kfont else "recycle by material")
        ax.set_ylabel("count")
        fig.tight_layout()
        p = run_dir / "chart_class.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        charts["class"] = p.name

        # 2) 시간대 패턴
        hp = timeseries.hourly_pattern(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hp.index, hp.values, marker="o", color="#3A6EA5")
        ax.set_title("hourly pattern")
        ax.set_xlabel("hour")
        ax.set_ylabel("count")
        ax.set_xticks(range(0, 24, 2))
        fig.tight_layout()
        p = run_dir / "chart_hourly.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        charts["hourly"] = p.name

        # 3) 일자별 추세
        daily = df.copy()
        daily["date"] = daily["ts"].dt.date
        dc = daily.groupby("date").size()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([str(d) for d in dc.index], dc.values, marker="s", color="#C46210")
        ax.set_title("daily trend")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = run_dir / "chart_daily.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        charts["daily"] = p.name

        return charts

    # --- Markdown -----------------------------------------------------
    def _write_markdown(self, m: dict, charts: dict, run_dir: Path, days: int) -> Path:
        ko = self.cfg.ko
        lines = [
            "# ♻ Yollo 분리수거 데이터 리포트",
            "",
            f"- 생성: {datetime.now():%Y-%m-%d %H:%M}",
            f"- 대상 기간: 최근 {days}일",
            f"- 담당 통(bin): **{ko(m['bin_target']) if m['bin_target'] else '전체(키오스크)'}**",
            "",
            "## 핵심 지표",
            "",
            f"- 총 배출 인식: **{m['total']}건**",
            f"- 정분류율(통 목표 일치): **{m['accuracy']:.1%}**",
            f"- 오분류: **{m['misclassified']}건**",
            f"- 오염/불확실(검수 후보): **{m['uncertain']}건**",
            f"- 종합 오염·오분류율: **{m['contamination_rate']:.1%}**",
            f"- 통 채움 예측: {m['fill'].summary()}",
        ]
        if m["peaks"]:
            peak_str = ", ".join(f"{h}시({n}건)" for h, n in m["peaks"])
            lines.append(f"- 배출 피크 시간대: {peak_str}")
        lines += [
            "",
            "## 재질별 분포",
            "",
        ]
        for name, cnt in m["per_class"].items():
            share = cnt / m["total"] if m["total"] else 0
            lines.append(f"- {ko(name)}: {cnt}건 ({share:.0%})")
        lines += [
            "",
            "## 그래프",
            "",
            f"![재질 분포]({charts['class']})",
            "",
            f"![시간대 패턴]({charts['hourly']})",
            "",
            f"![일자별 추세]({charts['daily']})",
            "",
            "## 첨부",
            "",
            "- `summary_by_class.csv` — 재질별 집계",
            "- `daily_counts.csv` — 일자별 건수",
            "",
        ]
        md = run_dir / "report.md"
        md.write_text("\n".join(lines), encoding="utf-8")
        return md
