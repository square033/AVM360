#!/usr/bin/env python3
"""전체 왕복 자체 검증 (하드웨어 불필요).

    python scripts/08_selftest.py

설계값 R,t 로 가상 이미지를 렌더링 → 그 이미지만 보고 외부 파라미터를 다시 추정 →
복원된 R,t 가 원래 설계값과 일치하는지, 그리고 그 값으로 만든 BEV 가 정답 지면과
일치하는지 확인한다. 캘리브레이션·LUT·블렌딩 전 과정이 한 번에 검증된다.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import AVMStitcher, RigConfig
from avm360.calib_extrinsic import GroundMarker, solve_extrinsic
from avm360.synth import GroundScene, default_marker_layout, make_ground_texture

POS_TOL_M = 0.01  # 위치 오차 허용 (10mm)
ANG_TOL_DEG = 1.0  # 각도 오차 허용
MAE_TOL = 12.0  # 평지 BEV 재구성 오차 허용 (0~255)


def angle_diff(a: np.ndarray, b: np.ndarray) -> float:
    """두 회전행렬 사이의 각도 차 (도)."""
    cos = (np.trace(a @ b.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--marker-size", type=float, default=0.06)
    ap.add_argument("--out", default=None, help="중간 산출물 저장 폴더 (기본: 임시폴더)")
    args = ap.parse_args()

    rig = RigConfig.load(args.config)
    layout = default_marker_layout(rig.bev)
    markers = {mid: GroundMarker(id=mid, x=x, y=y, size=args.marker_size) for mid, x, y in layout}

    print("1) 정답 지면 생성 및 가상 촬영")
    texture = make_ground_texture(rig.bev, tile=0.10, with_aruco=True, marker_size=args.marker_size)
    scene = GroundScene(texture, rig.bev, boxes=[])  # 기하 검증이므로 장애물 없이
    frames = {cam.name: scene.render(cam) for cam in rig.cameras}

    print("2) 렌더링된 이미지만으로 외부 파라미터 재추정")
    failures: list[str] = []
    solved_cams = []
    for cam in rig.cameras:
        try:
            solved, rms, used = solve_extrinsic(cam, frames[cam.name], markers)
        except RuntimeError as exc:
            failures.append(f"{cam.name}: {exc}")
            solved_cams.append(cam)
            continue

        d_pos = float(np.linalg.norm(solved.center - cam.center))
        d_ang = angle_diff(solved.R, cam.R)
        ok = d_pos <= POS_TOL_M and d_ang <= ANG_TOL_DEG
        print(f"   {cam.name:>6}: 위치오차 {d_pos*1000:5.1f}mm  각도오차 {d_ang:4.2f}°  "
              f"RMS {rms:4.2f}px  마커 {len(used)}개  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{cam.name}: 위치오차 {d_pos*1000:.1f}mm, 각도오차 {d_ang:.2f}°")
        solved_cams.append(solved)

    print("3) 재추정된 파라미터로 BEV 재구성")
    rig.cameras = solved_cams
    stitcher = AVMStitcher(rig)
    bev = stitcher(frames, draw_vehicle=False)

    valid = stitcher.coverage & ~stitcher.vehicle_mask
    diff = np.abs(bev.astype(np.float32) - texture.astype(np.float32)).mean(axis=2)
    mae = float(diff[valid].mean())
    cov = stitcher.coverage_report()
    print(f"   커버리지 {cov['_any']*100:.1f}%, 사각지대 {cov['_blind']*100:.1f}%")
    print(f"   정답 대비 평균 절대오차 {mae:.2f} / 255 (허용 {MAE_TOL})")
    if mae > MAE_TOL:
        failures.append(f"BEV 재구성 오차 {mae:.2f} > {MAE_TOL}")

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="avm_selftest_"))
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "bev_from_solved.png"), bev)
    cv2.imwrite(str(out / "ground_truth.png"), texture)
    cv2.imwrite(str(out / "error.png"), np.clip(diff * 3, 0, 255).astype(np.uint8))
    print(f"   산출물: {out}")

    if failures:
        print("\n=== FAIL ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n=== PASS === 캘리브레이션 → LUT → 블렌딩 전 과정 정상")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
