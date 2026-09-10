#!/usr/bin/env python3
"""BEV LUT 생성 및 커버리지 점검 (BEV.md 7.2절).

    python scripts/04_build_lut.py --config configs/rig_desk.yaml

카메라 파라미터가 고정인 한 LUT 는 한 번만 만들면 되므로, 미리 계산해서 저장해 두고
실행 시에는 remap 만 한다. 커버리지가 낮거나 사각지대가 크면 카메라 배치(높이·틸트)를
조정해야 한다는 신호다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import AVMStitcher, RigConfig

CAM_COLORS = {
    "front": (60, 90, 235),
    "left": (80, 200, 90),
    "rear": (235, 160, 60),
    "right": (200, 90, 220),
}
BLIND_WARN = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--out", default="out/lut.npz")
    ap.add_argument("--vis", default="out/coverage.png")
    args = ap.parse_args()

    rig = RigConfig.load(args.config)
    bev = rig.bev
    print(f"BEV 캔버스: {bev.height} x {bev.width} px")
    print(f"  X(전후) {bev.x_min:+.2f} ~ {bev.x_max:+.2f} m,  Y(좌우) {bev.y_min:+.2f} ~ {bev.y_max:+.2f} m")
    print(f"  해상도 {bev.resolution*1000:.1f} mm/px\n")

    stitcher = AVMStitcher(rig)
    cov = stitcher.coverage_report()
    for cam in rig.cameras:
        c = cam.center
        print(f"  {cam.name:>6}: 커버리지 {cov[cam.name]*100:5.1f}%   "
              f"위치 ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})m")
    print(f"\n  전체 커버리지 : {cov['_any']*100:5.1f}%")
    print(f"  사각지대      : {cov['_blind']*100:5.1f}%  (차량 본체 영역 제외)")

    if cov["_blind"] > BLIND_WARN:
        print("\n  경고: 사각지대가 큽니다. 카메라를 더 높이 달거나 틸트를 키우거나,")
        print("        화각이 더 넓은 렌즈를 쓰거나, BEV 범위를 줄이세요.")

    stitcher.save_luts(args.out)

    vis = np.zeros((*bev.shape, 3), dtype=np.float32)
    for name, w in stitcher.weights.items():
        vis += w[..., None] * np.array(CAM_COLORS.get(name, (200, 200, 200)), dtype=np.float32)
    vis[stitcher.vehicle_mask] = (60, 60, 60)
    Path(args.vis).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.vis, np.clip(vis, 0, 255).astype(np.uint8))

    print(f"\nLUT 저장  : {Path(args.out).resolve()}")
    print(f"커버리지도: {Path(args.vis).resolve()}")
    print("\n다음 단계: python scripts/05_run_avm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
