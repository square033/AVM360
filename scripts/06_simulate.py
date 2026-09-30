#!/usr/bin/env python3
"""가상 씬으로 전체 파이프라인 검증 (하드웨어 불필요).

지면 텍스처를 만들고 → 4대의 카메라 모델로 렌더링해서 "촬영된 것처럼" 왜곡시킨 뒤
→ AVM 파이프라인으로 다시 합성한다. 합성 결과가 원본 지면 텍스처와 같아야 한다.

    python scripts/06_simulate.py --config configs/rig_desk.yaml --out out/sim

만들어지는 파일:
    cam_*.png       각 카메라가 본 (왜곡된) 이미지
    bev.png         합성된 360° BEV
    bev_seams.png   이음선 표시
    ground_truth.png 정답 지면 텍스처
    coverage.png    카메라별 담당 영역 (색상 구분)
    error.png       정답 대비 오차 (밝을수록 큼)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import AVMStitcher, RigConfig
from avm360.synth import GroundScene, default_boxes, make_ground_texture

CAM_COLORS = {
    "front": (60, 90, 235),
    "left": (80, 200, 90),
    "rear": (235, 160, 60),
    "right": (200, 90, 220),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--out", default="out/sim")
    ap.add_argument("--no-boxes", action="store_true", help="수직 장애물 없이 순수 지면만 렌더링")
    ap.add_argument("--seams", action="store_true", help="이음선을 표시한 결과도 저장")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rig = RigConfig.load(args.config)
    print(f"리그: {rig.names}, BEV {rig.bev.height}x{rig.bev.width} @ {rig.bev.resolution*1000:.1f}mm/px")

    # 1) 정답 지면 만들기
    texture = make_ground_texture(rig.bev, tile=0.10, with_aruco=True)
    cv2.imwrite(str(out / "ground_truth.png"), texture)
    scene = GroundScene(texture, rig.bev, boxes=[] if args.no_boxes else default_boxes())

    # 2) 각 카메라가 본 장면 렌더링
    frames = {}
    for cam in rig.cameras:
        print(f"  렌더링 {cam.name} ... ", end="", flush=True)
        frames[cam.name] = scene.render(cam)
        cv2.imwrite(str(out / f"cam_{cam.name}.png"), frames[cam.name])
        print("완료")

    # 3) AVM 합성
    stitcher = AVMStitcher(rig)
    cov = stitcher.coverage_report()
    print("\n커버리지 (BEV 픽셀 비율):")
    for cam in rig.cameras:
        print(f"  {cam.name:>6}: {cov[cam.name]*100:5.1f}%")
    print(f"  {'합계':>6}: {cov['_any']*100:5.1f}%   사각지대: {cov['_blind']*100:5.1f}%")

    bev = stitcher(frames)
    cv2.imwrite(str(out / "bev.png"), bev)
    if args.seams:
        cv2.imwrite(str(out / "bev_seams.png"), stitcher(frames, draw_seams=True))

    # 4) 어느 카메라가 어디를 맡았는지 시각화
    coverage_vis = np.zeros((*rig.bev.shape, 3), dtype=np.float32)
    for name, w in stitcher.weights.items():
        color = np.array(CAM_COLORS.get(name, (200, 200, 200)), dtype=np.float32)
        coverage_vis += w[..., None] * color
    coverage_vis[stitcher.vehicle_mask] = (60, 60, 60)
    cv2.imwrite(str(out / "coverage.png"), np.clip(coverage_vis, 0, 255).astype(np.uint8))

    # 5) 정답과 비교 — 지면만 있는 경우 이 오차가 곧 기하 오차다
    valid = stitcher.coverage & ~stitcher.vehicle_mask
    diff = np.abs(bev.astype(np.float32) - texture.astype(np.float32)).mean(axis=2)
    diff[~valid] = 0
    mae = float(diff[valid].mean())
    cv2.imwrite(str(out / "error.png"), np.clip(diff * 3, 0, 255).astype(np.uint8))

    print(f"\n정답 대비 평균 절대오차: {mae:.2f} / 255")
    if args.no_boxes:
        verdict = "양호 (기하 파이프라인 정상)" if mae < 12 else "오차 큼 — K/R/t 또는 LUT 확인 필요"
        print(f"  판정: {verdict}")
    else:
        print("  (수직 장애물이 IPM 가정을 깨서 늘어나 보이는 부분은 정상적인 오차다)")
    print(f"\n결과 저장: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
