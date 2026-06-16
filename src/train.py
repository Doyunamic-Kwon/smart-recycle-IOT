"""모델 학습 (PC/GPU에서 실행 권장 — Pi5 아님).

YOLO11n(가장 작은 모델)을 4종 분리수거 데이터로 파인튜닝한다.
n 모델을 쓰는 이유: Pi5 CPU 에서 실시간이 나오려면 작아야 함.

실행:
  python -m src.train                     # 기본값으로 학습
  python -m src.train --epochs 100 --model yolo11s.pt

결과:
  runs/detect/train*/weights/best.pt  → models/yolo11n_recycle.pt 로 복사 권장
  이후 scripts/export_ncnn.py 로 Pi5 배포용 NCNN 변환.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ROOT


def main():
    ap = argparse.ArgumentParser(description="Yollo YOLO11 학습")
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="사전학습 가중치 (n/s/m...)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0=GPU / cpu")
    ap.add_argument("--patience", type=int, default=30,
                    help="N에폭 동안 개선 없으면 조기 종료 (시간 절약)")
    ap.add_argument("--name", default="recycle", help="runs/ 하위 결과 폴더명")
    ap.add_argument("--strong-aug", action="store_true",
                    help="강한 증강 프로파일 (회전/상하반전/원근/mixup/copy_paste) — "
                         "물체가 임의 방향으로 놓이는 분리수거함에 적합")
    args = ap.parse_args()

    from ultralytics import YOLO
    import yaml

    # ultralytics 는 dataset.yaml 의 상대 path 를 ~/datasets 기준으로 풀어버려
    # 경로를 못 찾는 경우가 많다 → yaml 위치 기준 절대경로로 바꾼 사본을 만들어 사용.
    data_path = Path(args.data).resolve()
    with open(data_path, encoding="utf-8") as f:
        dd = yaml.safe_load(f)
    p = Path(dd.get("path", "."))
    if not p.is_absolute():
        dd["path"] = str((data_path.parent / p).resolve())
    resolved = ROOT / "runs" / "dataset_resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True)
    print(f"[데이터] 경로 확정: {dd['path']}")

    # 강한 증강 프로파일: 물체가 똑바로 놓이지 않는 현실 반영
    aug = {}
    if args.strong_aug:
        aug = dict(
            degrees=30.0,       # 회전 ±30° (눕거나 기운 물체)
            flipud=0.5,         # 상하 반전 (뒤집힌 물체)
            fliplr=0.5,         # 좌우 반전
            scale=0.8,          # 크기 변동 ±80%
            translate=0.2,      # 위치 이동
            shear=5.0,          # 기울임
            perspective=0.0005, # 약한 원근 왜곡 (카메라 각도)
            mosaic=1.0,         # 4장 합성
            mixup=0.15,         # 이미지 혼합
            copy_paste=0.2,     # 객체 복붙 — 희소 클래스(glass)에 도움
            close_mosaic=10,    # 마지막 10에폭은 mosaic 끔 (안정 수렴)
        )
        print("[증강] 강한 프로파일 적용:", aug)

    model = YOLO(args.model)
    results = model.train(
        data=str(resolved),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        **aug,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[학습 완료] best 가중치: {best}")
    print(f"  배포 준비:  cp {best} {ROOT/'models'/'yolo11n_recycle.pt'}")
    print(f"  Pi5 변환:   python scripts/export_ncnn.py --weights models/yolo11n_recycle.pt")


if __name__ == "__main__":
    main()
