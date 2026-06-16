# ♻ Smart Recycle IOT

> AIoT 기반 스마트 분리수거 시스템 — Raspberry Pi 5 + YOLO11n + 실시간 대시보드

라즈베리 파이 5에 카메라와 초음파 센서를 달아 쓰레기를 자동 분류하고,  
인식 결과를 AWS(Railway)에 배포된 웹 대시보드로 실시간 전송해 모니터링하는 AIoT 프로젝트입니다.

---

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [Raspberry Pi](#-raspberry-pi)
3. [Model (YOLO11n)](#-model-yolo11n)
4. [Dashboard](#-dashboard)
5. [폴더 구조](#폴더-구조)

---

## 시스템 아키텍처

```
┌─────────────────────────────────┐        ┌──────────────────────────────┐
│         Raspberry Pi 5          │        │     Railway (Cloud Server)   │
│                                 │        │                              │
│  초음파 센서 → 물체 감지         │        │  Flask API  ←  POST /events  │
│       ↓                         │  HTTP  │      ↓                       │
│  카메라 → YOLO11n 추론           │ ──────▶│  SQLite DB                   │
│       ↓                         │        │      ↓                       │
│  LCD + LED + 서보(뚜껑 개폐)     │        │  웹 대시보드 (5초 자동갱신)   │
│       ↓                         │        └──────────────────────────────┘
│  pi_companion.py → API 전송     │                      ▲
└─────────────────────────────────┘               브라우저 접속
```

---

## 🍓 Raspberry Pi

### 하드웨어 구성

| 부품 | 연결 | 역할 |
|------|------|------|
| 카메라 (USB) | GPIO | 쓰레기 촬영 |
| 초음파 센서 | TRIG=GPIO23, ECHO=GPIO24 | 30cm 이내 물체 감지 |
| 서보 모터 | GPIO18 | 뚜껑 개폐 (0°=닫힘, 90°=열림) |
| I2C LCD 16×2 | I2C 0x27 (PCF8574) | 분류 결과 표시 |
| LED × 3 | GPIO17(플라스틱), GPIO27(캔), GPIO22(종이) | 해당 분류 깜빡임 |

### 동작 흐름

```
1. 초음파 센서가 30cm 이내 물체 감지
2. 서보 모터로 뚜껑 자동 오픈
3. 카메라가 5초 간격으로 프레임 촬영
4. YOLO11n(NCNN)으로 쓰레기 분류 (plastic / can / glass / paper)
5. LCD에 분류 결과 출력 + 해당 LED 점등
6. pi_companion.py → 대시보드 서버로 결과 POST 전송
7. 물체가 없어지면 뚜껑 자동 닫힘
```

### 분류별 안내 메시지

| 클래스 | LCD 표시 | 수거함 | LED |
|--------|----------|--------|-----|
| plastic | Plastic Bottle | Green bin | GPIO17 |
| can | Can / Metal | Red bin | GPIO27 |
| paper | Paper | Yellow bin | GPIO22 |
| glass | Glass | Glass bin | (없음) |

### Pi 실행 방법

```bash
# 기본 실행 (smart_recycle.py 수정 없이 대시보드와 연동)
python3 smart_recycle.py 2>&1 | python3 pi_companion.py \
  --server https://smart-recycle-iot-production.up.railway.app \
  --region "AI공학관"

# 환경변수로도 설정 가능
export DASHBOARD_URL=https://smart-recycle-iot-production.up.railway.app
export DEVICE_REGION=AI공학관
python3 smart_recycle.py 2>&1 | python3 pi_companion.py
```

### Pi 의존성

```bash
pip3 install lgpio gpiozero RPLCD ultralytics opencv-python requests
```

---

## 🧠 Model (YOLO11n)

### 현재 모델 성능

- 백본: **YOLO11n** (Pi5 CPU에서 가장 빠름), NCNN 변환 배포
- 학습 데이터: **TACO**(길거리 쓰레기) + **AI-Hub 생활폐기물 140**(한국 가정환경) 병합
- 최종 실험: `recycle_aihub2` (강한 증강 + AI-Hub oversample)

**혼합 검증셋 성능 (TACO + AI-Hub val, 600장)**

| 클래스 | mAP50 | 비고 |
|--------|-------|------|
| **can** | **0.882** | 금속 광택·원통 형태로 특징 명확 — 최고 성능 |
| **paper** | 0.813 | 형태가 다양해도 안정적 |
| **plastic** | 0.718 | 형태·색상 분산이 크고 TACO에서 병·용기 경계가 애매 |
| **glass** | 0.613 | 투명성·반사 아티팩트 + 데이터 희소 — 최하위 |
| **전체** | **0.756** | Precision 0.884 / Recall 0.679 / mAP50-95 0.640 |

**평가 환경별 성능 차이**

| 평가 환경 | glass | can | paper | plastic | 전체 mAP50 |
|----------|-------|-----|-------|---------|-----------|
| 단일물체·깔끔 (재활용 스테이션, 실제 사용처) | 0.98 | 0.99 | 0.99 | 0.96 | **~0.98** |
| 어수선하게 쌓인 장면 (TACO풍) | 0.01 | 0.37 | 0.26 | 0.51 | 0.29 |

> ⚠️ 단일물체 점수(~0.98)는 AI-Hub가 같은 물체를 여러 각도로 찍어 train/val에 같은 물체가 섞이는 **낙관적(누수 가능)** 수치입니다. 실제 성능은 Pi 카메라로 직접 찍은 데이터로 평가해야 진짜 숫자가 나옵니다.

**용도 적합성**: 물체를 하나씩 카메라 앞에 놓는 방식(투입구/선별대)이면 잘 맞습니다.

---

### 1. 왜 YOLO11n인가

Pi5 CPU에서 실시간(≤300ms/frame)을 맞추려면 모델이 작아야 한다.

| 모델 | 파라미터 | COCO mAP50 | Pi5 추론(NCNN) |
|------|---------|-----------|----------------|
| YOLOv8n | 3.2M | 37.3% | ~250ms/frame |
| **YOLO11n** | **2.6M** | **39.5%** | **~200ms/frame** |
| YOLO11s | 9.4M | 47.0% | ~600ms+ |

YOLO11n은 전작 YOLOv8n 대비 파라미터 17% 감소, mAP 2.2%p 향상이다. 4-클래스 분류에 Small 이상은 과설계고, 실시간성을 포기할 이유도 없다.

---

### 2. 데이터 구성 — "한국 데이터가 핵심이었다"

**가장 중요한 결론: 증강보다 데이터 도메인이 훨씬 중요했다.**

| 실험 | 추가 내용 | best mAP50 | 의미 |
|------|---------|-----------|------|
| `recycle` | TACO만, 기본 증강 | 0.30 | 도메인 불일치의 하한선 |
| `recycle_n` | TACO + 약한 증강 추가 | 0.31 | 증강만으로 도메인 갭을 못 메움 |
| `recycle_aihub` | TACO + AI-Hub 140 추가 | 0.61 | **데이터 도메인 매칭으로 2배 점프** |
| `recycle_aihub2` | + 강한 증강 전체 적용 | **0.76** | 증강이 일반화를 추가 향상 |

TACO는 길거리에 버려진 쓰레기를 다각도로 찍은 데이터다. 재활용 선별대에 물체를 하나 올려놓고 촬영하는 우리 환경과 도메인이 전혀 다르다. 증강을 아무리 해도 0.30 → 0.31로 거의 개선되지 않은 이유다.

AI-Hub 생활폐기물(한국 가정·재활용 환경)을 추가하자 0.30 → 0.61로 뛰었다. **증강 몇 달치 효과가 데이터 도메인 한 번 바꾼 것보다 못했다.**

AI-Hub에 `--oversample 2` 가중치를 준 이유: TACO가 수량 면에서 더 많고 도메인은 덜 맞다. AI-Hub를 2배 복제해 학습 중 노출 빈도를 높여 실제 환경에 더 적응하게 했다.

---

### 3. 클래스 불균형 해결

원본 데이터는 plastic이 다수, glass가 극소로 분포가 치우쳐 있다.

- **오버샘플링**: glass×4, can×2, paper×2 복제 (Undersample 대신 — 데이터 절대량 부족)
- **Copy-paste 증강** (`copy_paste=0.2`): 기존 이미지에 glass 객체를 잘라붙여 가상 샘플 생성

glass가 최하위 성능(mAP50 0.613)인 이유는 투명성 자체의 어려움도 있지만 데이터 희소가 주원인이다.

---

### 4. 증강 파라미터 설계 — 각 항목의 근거

증강의 목표는 학습 데이터 분포를 실제 현장 분포에 근사시키는 것이다.

```python
aug = dict(
    degrees=30.0,       # 투입구에 넣을 때 물체가 임의 각도로 낙하
    flipud=0.5,         # 뒤집힌 채 투입되는 캔·병류
    fliplr=0.5,         # 방향성 없는 객체의 데이터 다양성
    scale=0.8,          # 카메라와 물체 거리 변동(가까이/멀리)
    translate=0.2,      # 물체가 프레임 가장자리에 걸리는 경우
    shear=5.0,          # 카메라 각도 오차에 의한 형태 왜곡
    perspective=0.0005, # 고정 마운트 카메라의 하향 촬영 원근감
    mosaic=1.0,         # 복수 객체 동시 등장 대응 + 소형 객체 탐지력
    mixup=0.15,         # 클래스 경계 일반화, 모호한 중간 케이스 학습
    copy_paste=0.2,     # glass 희소 클래스 데이터 보완
    close_mosaic=10,    # 마지막 10 epoch mosaic 끔 → bbox 정밀도 fine-tuning
)
```

`perspective`가 특히 중요한 이유: Pi5 카메라는 상단 고정·하향이다. 정면 촬영 데이터만 학습하면 이 각도에서 성능이 떨어진다. `perspective=0.0005`는 이 왜곡을 학습 중에 미리 경험시킨다.

---

### 5. 학습 하이퍼파라미터

```yaml
epochs: 100       # recycle_aihub2는 97 epoch에서 best 달성 → patience 여유가 필요
patience: 25      # plateau 구간 진동에도 25 epoch 여유 → 조기 종료 방지
batch: -1         # GPU 메모리에 맞게 자동 결정
lr0: 0.01         # cosine decay 시작값
lrf: 0.01         # 종료 lr (시작과 같게 — 후반 과도한 lr 감소 방지)
optimizer: auto   # warm-up SGD → AdamW 자동 전환
amp: true         # FP16 혼합 정밀도 (2× 속도)
close_mosaic: 10  # 마지막 10 epoch mosaic 꺼서 bbox 회귀 안정화
```

Patience=25로 설정한 이유: validation mAP는 plateau 구간에서도 epoch마다 진동한다. 실제로 `recycle_aihub2`는 97 epoch에서 best가 나왔다 — 70 epoch 이후에도 개선이 있었다는 의미다.

---

### 6. NCNN 변환 — Pi5에서 실시간이 되는 이유

PyTorch `.pt` 모델을 Pi5 CPU에서 그대로 실행하면 500ms+가 나온다. NCNN 변환 후 ~200ms로 2.5배 빨라지는 이유:

- **ARM NEON SIMD**: 벡터 연산 병렬화 (ARM Cortex-A76 내장)
- **Winograd 알고리즘**: 3×3 컨볼루션 연산 횟수 감소
- **fp16 양자화**: 연산량 절반, 메모리 대역폭 절약

```python
from ultralytics import YOLO
model = YOLO("models/yolo11n_recycle.pt")
model.export(format="ncnn", imgsz=640)
# → models/yolo11n_recycle_ncnn_model/ (약 11MB)
```

---

### 7. 신뢰도 임계값 3구간 설계

단순 이진(탐지/비탐지) 대신 3구간으로 분리했다.

```
conf < 0.35          → 탐지 폐기      배경 오인식, 모션 블러
0.35 ≤ conf < 0.65  → 불확실 구간   → 능동학습 후보로 자동 저장
conf ≥ 0.65          → 신뢰 탐지    → 분류 기록, 서보 동작
```

0.65 기준의 근거: 초기 현장 테스트에서 conf 0.5~0.65 구간 예측의 오분류율이 눈에 띄게 높았다. 0.65 이상에서 precision이 안정적으로 88% 이상 유지됨을 확인하고 설정했다.

불확실 구간을 버리지 않는 이유: 모델이 헷갈리는 케이스가 학습 효과가 가장 큰 케이스다. 이를 `captured/active_learning/`에 의사라벨과 함께 저장하고 수동 검수 후 재학습에 투입하면 동일한 라벨링 비용으로 더 큰 성능 향상을 얻는다.

### 전처리 파이프라인

카메라 프레임이 DB에 기록되기까지 5단계를 거칩니다.

```
카메라 (1280×720, 15fps)
    │
    ▼ 1) 리사이즈·정규화 (ultralytics 내부)
YOLO11n NCNN 추론 (640×640 입력)
    │  conf < 0.35 → 버림
    ▼ 2) 신뢰도 필터링
탐지 후보 (Detection 리스트)
    │
    ▼ 3) 중심점 트래킹 (CentroidTracker)
       · 같은 물체가 3프레임 연속 같은 클래스 → '확정'
       · 프레임 간 클래스 바뀌면 flickered=True (흔들림)
    │
    ▼ 4) 오염·오분류 분류 (ContaminationDetector)
       · conf 0.35~0.65 → uncertain (낮은 확신 = 오염 의심 대리 지표)
       · conf ≥ 0.65    → trusted
       · 통 목표 클래스와 다른 클래스 → misclassified
    │
    ▼ 5) 이벤트 기록
       · 확정 건만 SQLite DB 로깅 (중복 카운트 방지)
       · uncertain/misclassified 건 → 능동학습 이미지 자동 저장
```

### 데이터 전송 방식

Pi → 대시보드 서버로 데이터를 보내는 구조입니다.  
**`smart_recycle.py`는 수정하지 않고**, stdout 파이프로 연결합니다.

```
[Raspberry Pi]

  python3 -u smart_recycle.py 2>&1
       │  stdout: "detected: plastic (conf: 0.87)"
       │
       ▼ 파이프
  python3 pi_companion.py --server <URL> --region "AI공학관"
       │
       │  1) "detected:" 라인 파싱 → name, conf 추출
       │  2) 0.3초 대기 (imwrite 완료 보장)
       │  3) waste_*.jpg 중 최신 파일 → base64 인코딩
       │
       ▼ HTTP POST /api/events (JSON)
  {
    "name": "plastic",
    "cls_id": 0,
    "conf": 0.87,
    "region": "AI공학관",
    "band": "trusted",
    "misclassified": false,
    "uncertain": false,
    "flickered": false,
    "ts": "2026-06-16T14:30:00",
    "image_b64": "iVBORw0KGgoAAAANS..."   ← 인식 장면 이미지
  }

[Railway 서버]

  Flask POST /api/events
       │
       ├─ SQLite events 테이블에 이벤트 저장
       └─ image_b64 디코딩 → static/captures/latest.jpg 저장

[브라우저]

  5초마다 GET /api/stats, /api/events 폴링
  오염률 ≥ 70% → 경고 카드 + latest.jpg 표시
```

**전송 페이로드 필드 설명**

| 필드 | 타입 | 설명 |
|------|------|------|
| name | string | 클래스명 (plastic/can/glass/paper) |
| cls_id | int | 클래스 ID (0~3) |
| conf | float | YOLO 신뢰도 (0~1) |
| region | string | 장치 설치 지역명 |
| band | string | trusted(≥0.65) / uncertain(<0.65) |
| misclassified | bool | 통 목표 클래스와 불일치 여부 |
| uncertain | bool | 낮은 신뢰도 또는 flicker 여부 |
| flickered | bool | 프레임 간 클래스 흔들림 여부 |
| image_b64 | string | 인식 장면 JPG (Base64) |

### 능동 학습 (Active Learning) 루프

현장 운영 중 모델이 헷갈린 순간(불확실 구간 0.35~0.65, 오분류, 프레임 간 흔들림)을 자동으로 저장해 재학습 데이터로 활용하는 **self-improving 루프**를 구현했습니다.

```
현장 운영 → 불확실 프레임 자동 저장 → 라벨 검수 → 재학습 → 성능 향상
```

### 설계 핵심

> **하나의 신뢰도(confidence) 신호와 하나의 이벤트 로그(SQLite)로 4개 분석을 전부 엮는다.**

| 분석 기능 | 무엇을 / 어떻게 |
|---|---|
| **오염·오분류 탐지** | 통 목표와 다른 품목=오분류, 낮은 신뢰도=오염 의심. 실시간 경고 + rolling 오염률 |
| **배출 패턴 시계열** | 시간대/요일 패턴, 피크, 최근 배출 속도로 통 가득참 ETA 예측 |
| **자동 데이터 수집(능동학습)** | 헷갈린 프레임만 의사라벨과 저장 → 검수 → 재학습(self-improving) |
| **재질별 통계·리포트** | 정분류율/오염률, CSV + 그래프 + Markdown 자동 생성 |

---

### 라이프사이클 관리 (`pi_companion.py`)

`smart_recycle.py`를 **전혀 수정하지 않고** stdout 파이프만으로 아래 기능을 추가했다.

**중복 이벤트 제거 (Dedup)**

같은 물체가 카메라 앞에 오래 있어도 DB에 1건만 기록된다.

```
같은 클래스 감지 → 마지막 전송 시각 확인
  5초 이내 재감지 → 전송 차단 (dup skip)
  5초 이후       → 정상 전송
```

**헬스비트 (Pi 온라인/오프라인 감지)**

백그라운드 스레드가 30초마다 서버에 ping을 보낸다. 서버는 마지막 ping 시각을 region별로 메모리에 저장하고, 60초 이상 ping이 없으면 해당 region을 오프라인으로 판정한다. 대시보드 헤더에 실시간으로 표시된다.

```
pi_companion 시작
  └─ 백그라운드 스레드: 30초마다 POST /api/heartbeat

서버: region별 last_seen 시각 저장
  └─ GET /api/heartbeat → { "AI공학관": { "online": true, "seconds_ago": 12 } }

대시보드 헤더: ● AI공학관 (12s 전)   ← 10초마다 갱신
              ● 부산캠퍼스 오프라인  ← 60초 이상 무응답 시 빨간색
```

| 설정 | 값 | 의미 |
|------|----|------|
| 헬스비트 전송 주기 | 30초 | 네트워크 부하 최소화 |
| 오프라인 판정 기준 | 60초 | 전송 주기 × 2 — 1회 실패는 무시 |
| 대시보드 갱신 주기 | 10초 | 상태 변화를 빠르게 반영 |

---

## 📊 Dashboard

### 아키텍처

```
Pi (smart_recycle.py + pi_companion.py)
    ↓ POST /api/events
Flask 서버 (Railway)
    ↓ SQLite 저장
브라우저 → GET /api/stats, /api/events (5초 폴링)
    ↓
HTML/JS 대시보드 (Chart.js)
```

### 주요 화면

**KPI 카드 (6종)**

| 카드 | 설명 |
|------|------|
| 오늘 인식 | 당일 누적 이벤트 수 |
| 전체 누적 | DB 전체 이벤트 수 |
| 정분류율 | 오분류 건 제외한 정확도 |
| 오염·오분류율 | misclassified OR uncertain 비율 |
| 최근 50건 오염률 | 전체 누적이 아닌 최근 N건 기준 실시간 오염 추이 |
| 최근 평균 신뢰도 | 최근 20건 conf 평균 — 카메라·모델 상태 모니터링 |

**차트 & 테이블**
- **재질별 분포**: plastic / can / glass / paper 가로 바 차트
- **시간대별 패턴**: 0~23시 배출량 막대 그래프 (최근 7일)
- **최근 이벤트 테이블**: 시각 / 지역 / 재질 / 신뢰도 / 판정

**이상 감지 알림 (Anomaly Detection)**

| 감지 항목 | 조건 | 의미 |
|-----------|------|------|
| 이벤트 급증 | 현재 시간 이벤트 수 ≥ 최근 7일 시간당 평균 × 2 | 비정상적으로 많은 투입 발생 |
| 신뢰도 하락 | 최근 20건 평균 conf가 전체 평균 대비 10%p↓ | 카메라 오염 또는 모델 드리프트 의심 |

이상 감지 조건 충족 시 대시보드 상단에 주황색 경고 카드 자동 표시.

**오염 경고**
- 오염·오분류율 ≥ 70% 시 빨간 경고 카드 + 마지막 인식 장면 이미지 표시

### Data Aggregation 구조

```
[전체 누적 집계]
  total, misclassified, uncertain → contamination_rate, accuracy

[Rolling Window (최근 50건)]
  최근 N건만 슬라이딩 → rolling_contamination_rate, rolling_avg_conf
  전체 누적과 비교해 현재 추이 파악 가능

[시간대별 집계]
  GROUP BY hour → 0~23시 배출 패턴 (최근 7일)

[이상 감지용 집계]
  현재 시간 count vs AVG(count per hour, 최근 7일) → 급증 여부
  AVG(conf, 최근 20건) vs AVG(conf, 전체) → 신뢰도 하락 여부
```

### API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/events` | Pi에서 인식 결과 + 이미지 수신 |
| `GET` | `/api/events?limit=50&region=AI공학관` | 최근 이벤트 목록 |
| `GET` | `/api/stats?region=AI공학관` | KPI·Rolling·Anomaly 통계 전체 |
| `GET` | `/api/health` | 헬스체크 |

**GET /api/stats 응답 구조**
```json
{
  "total": 120,
  "today": 30,
  "contamination_rate": 0.25,
  "accuracy": 0.85,
  "rolling": {
    "total": 50,
    "contamination_rate": 0.32,
    "avg_conf": 0.71
  },
  "anomalies": {
    "spike": false,
    "spike_current": 4,
    "spike_avg": 3.2,
    "conf_drop": false,
    "conf_recent": 0.68,
    "conf_overall": 0.72
  },
  "by_class": { "plastic": 60, "can": 30, "glass": 15, "paper": 15 },
  "hourly": [0, 0, 0, ..., 12, 18, 0]
}

**POST /api/events 요청 예시:**
```json
{
  "name": "plastic",
  "cls_id": 0,
  "conf": 0.87,
  "region": "AI공학관",
  "band": "trusted",
  "misclassified": false,
  "uncertain": false,
  "flickered": false
}
```

### Railway 배포

1. [railway.app](https://railway.app) → **Deploy from GitHub repo** → `Doyunamic-Kwon/smart-recycle-IOT`
2. Dockerfile 자동 감지 → 빌드 & 배포
3. **Settings → Networking → Generate Domain** 으로 공개 URL 발급
4. (권장) **Volumes → Add Volume**, Mount Path: `/app/data` 로 DB 영구 저장

**배포된 대시보드:** https://smart-recycle-iot-production.up.railway.app

### 로컬 실행

```bash
git clone https://github.com/Doyunamic-Kwon/smart-recycle-IOT.git
cd smart-recycle-IOT
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

---

## 폴더 구조

```
smart-recycle-IOT/
├── app.py                  # Flask API 서버 + 프론트엔드 서빙
├── database.py             # SQLite 이벤트 저장소
├── pi_companion.py         # Pi용 — smart_recycle.py stdout 파싱 → API 전송
├── static/
│   └── index.html          # 대시보드 프론트엔드 (Chart.js, 5초 자동갱신)
├── data/
│   └── events.db           # 인식 이벤트 DB (자동 생성)
├── requirements.txt        # Flask, gunicorn, flask-cors, requests
├── Dockerfile              # Railway / AWS 배포용
└── .gitignore
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| AI 모델 | YOLO11n (Ultralytics), NCNN |
| 엣지 디바이스 | Raspberry Pi 5 |
| 백엔드 | Python / Flask |
| 프론트엔드 | HTML / CSS / JavaScript / Chart.js |
| DB | SQLite |
| 배포 | Railway (Docker) |
| Pi 인터페이스 | lgpio, gpiozero, RPLCD, OpenCV |
