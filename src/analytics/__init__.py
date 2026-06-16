"""데이터 분석 모듈 묶음.

설계 핵심: 모든 분석은 단 하나의 이벤트 로그(events.db)와
하나의 신뢰도 신호를 공유한다.

  database       - 모든 탐지 이벤트를 쌓는 단일 SQLite
  contamination  - 오염/오분류 탐지 (신뢰도 애매구간 + 통(bin) 불일치)
  timeseries     - 배출 패턴 시계열 + 통 가득참 예측
  active_learning- 불확실 탐지 자동 저장 (능동학습)
  report         - 재질별 통계 + CSV/그래프 리포트
"""
