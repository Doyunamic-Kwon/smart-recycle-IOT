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

### 모델 선택 이유

**YOLO11n** (Nano — 가장 작은 YOLO11 모델)을 선택한 이유:

- Raspberry Pi 5 CPU에서 **실시간 추론**이 가능하려면 모델이 작아야 함
- 4종 분류(plastic/can/glass/paper)는 단순한 태스크 → 대형 모델 불필요
- NCNN 포맷으로 변환 시 Pi5에서 안정적인 추론 속도 확보

### 학습 설정

| 항목 | 값 |
|------|----|
| 기반 모델 | yolo11n.pt (ImageNet 사전학습) |
| 학습 에폭 | 80 |
| 배치 크기 | 16 |
| 입력 해상도 | 640×640 |
| 조기 종료 | patience=30 에폭 |
| 추론 임계값 | conf ≥ 0.35 (탐지) / ≥ 0.65 (확실) |

### 문제점 발견

초기 모델 학습 후 실제 분리수거함에 테스트했을 때 **인식률이 크게 떨어지는 문제**가 발생했습니다.

**원인 분석:**

- 데이터셋의 대부분이 **정면·정립** 상태의 쓰레기 이미지
- 현실에서는 물체가 **뒤집히거나, 기울거나, 화면 모서리에 걸쳐** 들어옴
- 특히 `glass`(유리) 클래스는 **데이터 수가 적고** 반사가 심해 오인식률이 높음
- 카메라 설치 각도로 인한 **원근 왜곡** 발생

```
[문제 상황 예시]
- 캔을 옆으로 눕혀서 넣으면 "unknown" 처리
- 종이를 구겨서 넣으면 plastic으로 오분류
- 카메라 정중앙이 아닌 모서리에 물체가 들어오면 신뢰도 급락
```

### 데이터 어규멘테이션 (해결책)

현실 분리수거 환경에 특화된 **강한 증강 프로파일**을 적용했습니다.

```python
aug = dict(
    degrees=30.0,       # 회전 ±30°  → 눕거나 기운 물체 대응
    flipud=0.5,         # 상하 반전 50% → 뒤집힌 물체 대응
    fliplr=0.5,         # 좌우 반전 50%
    scale=0.8,          # 크기 변동 ±80% → 물체가 가까이/멀리 있는 경우
    translate=0.2,      # 위치 이동 → 화면 끝에 걸친 물체 대응
    shear=5.0,          # 기울임 변환
    perspective=0.0005, # 원근 왜곡 → 카메라 설치 각도 보정
    mosaic=1.0,         # 4장 합성 → 다양한 배경 조합
    mixup=0.15,         # 이미지 혼합 → 일반화 성능 향상
    copy_paste=0.2,     # 객체 복붙 → glass 희소 클래스 보완
    close_mosaic=10,    # 마지막 10에폭 mosaic 비활성 → 안정적 수렴
)
```

**각 증강의 효과:**

| 증강 기법 | 해결한 문제 |
|-----------|------------|
| `degrees`, `flipud` | 뒤집히거나 기울어진 물체 |
| `translate`, `scale` | 화면 끝에 걸치거나 크기가 다른 물체 |
| `perspective` | 카메라 설치 각도로 인한 왜곡 |
| `copy_paste` | glass 클래스 데이터 부족 문제 |
| `mosaic` | 배경에 대한 과적합 방지 |

### 학습 결과 (성능)

> 학습은 PC/GPU 환경에서 진행, NCNN 변환 후 Pi5에 배포

| 지표 | 값 |
|------|-----|
| **mAP50** | **~90%** |
| 추론 속도 (Pi5 NCNN, CPU) | ~200ms/frame |
| 추론 속도 (Mac M-series) | ~30ms/frame |

**신뢰도 구간 정책**

| 구간 | 판정 |
|------|------|
| conf < 0.35 | 탐지 버림 |
| 0.35 ≤ conf < 0.65 | 불확실 — 능동학습 저장 대상 |
| conf ≥ 0.65 | 확실한 탐지로 DB 기록 |

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

### Pi5 배포 (NCNN 변환)

YOLO11n을 Raspberry Pi5에서 최적 속도로 돌리기 위해 **NCNN 포맷**으로 변환합니다.

```bash
# 1) PC에서 학습
python -m src.train --strong-aug --epochs 80

# 2) NCNN 변환 (ultralytics 자동 처리)
from ultralytics import YOLO
model = YOLO("models/yolo11n_recycle.pt")
model.export(format="ncnn", imgsz=640)
# → models/yolo11n_recycle_ncnn_model/ 생성

# 3) Pi5에 models/ 폴더 복사 후 실행
```

**NCNN을 쓰는 이유:** Raspberry Pi5는 GPU가 없어 CPU 추론을 하는데, NCNN은 ARM CPU에 최적화된 경량 추론 엔진으로 `.pt` 대비 속도가 크게 향상됩니다.

### 능동 학습 (Active Learning) 루프

현장 운영 중 모델이 헷갈린 순간(불확실 구간 0.35~0.65, 오분류, 프레임 간 흔들림)을 자동으로 저장해 재학습 데이터로 활용하는 **self-improving 루프**를 구현했습니다.

```
현장 운영 → 불확실 프레임 자동 저장 → 라벨 검수 → 재학습 → 성능 향상
```

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

- **KPI 카드**: 오늘 인식 건수 / 전체 누적 / 최다 재질 / 마지막 인식
- **재질별 분포**: 4종 가로 바 차트
- **시간대별 패턴**: 0~23시 배출량 막대 그래프 (최근 7일)
- **최근 이벤트 테이블**: 시각 / 지역 / 재질 / 신뢰도 / 판정

### API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/events` | Pi에서 인식 결과 수신 |
| `GET` | `/api/events?limit=50&region=AI공학관` | 최근 이벤트 목록 |
| `GET` | `/api/stats?region=AI공학관` | 통계 (KPI, 분포, 시간대) |
| `GET` | `/api/health` | 헬스체크 |

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
