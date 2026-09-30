#!/usr/bin/env python3
"""가상 씬 파이프라인을 단계별로 프린트하며 실행 — 다른 상황(카메라 각도/장애물)도 테스트.

06_simulate.py 와 같은 파이프라인(가상 지면 → 카메라 렌더 → LUT → 블렌딩 → 합성)을
실행하되, 각 단계의 중간 결과(통계값)와 카메라 한 대·픽셀 한 개에 대한 상세 연산
과정(월드좌표 → 카메라좌표 → 정규화 → 어안 왜곡 → 픽셀)을 그대로 프린트한다.

    # 기본 시나리오
    python scripts/09_debug_trace.py

    # 장애물 없이 (순수 지면 IPM 오차만 확인)
    python scripts/09_debug_trace.py --no-boxes

    # front 카메라를 더 눕혀서(60도) 각도 변화 영향 확인
    python scripts/09_debug_trace.py --camera front --pitch 60

    # front 카메라를 앞으로 5cm, 위로 3cm 옮겨서 위치 변화 영향 확인
    python scripts/09_debug_trace.py --camera front --position 0.20 0.0 0.25

    # 특정 BEV 픽셀/카메라를 지정해서 상세 추적
    python scripts/09_debug_trace.py --trace-row 120 --trace-col 350 --trace-camera front

    # 장애물 위치를 직접 지정 (여러 번 줄 수 있음): x_min x_max y_min y_max height
    python scripts/09_debug_trace.py --box 0.20 0.30 -0.05 0.05 0.25

    # front 카메라 화각을 100도로 좁혀서(K도 재계산됨) 커버리지 변화 확인
    python scripts/09_debug_trace.py --camera front --hfov 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import AVMStitcher, RigConfig
from avm360.blend import build_weights
from avm360.config import make_K
from avm360.lut import build_ground_lut
from avm360.model import R_AXIS_VEH2CAM, CameraModel, R_to_euler
from avm360.synth import SKY_COLOR, Box, GroundScene, default_boxes, make_ground_texture

SEP = "-" * 60
BOX_PALETTE = [(70, 70, 220), (80, 200, 90), (230, 180, 60), (200, 90, 220), (60, 200, 220)]


def hd(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def describe_camera(cam: CameraModel) -> str:
    roll, pitch, yaw = R_to_euler(cam.R.T @ R_AXIS_VEH2CAM)
    c = cam.center
    return (
        f"{cam.name:>6}: position=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})  "
        f"roll/pitch/yaw=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})  hfov={cam.hfov_deg:.0f}"
    )


def apply_mount_override(rig: RigConfig, name: str, roll, pitch, yaw, position, hfov) -> None:
    """카메라 하나의 mount(위치/각도)와 화각을 덮어써서 다른 배치/렌즈 시나리오를 만든다.

    RigConfig 는 R,t 로 이미 굳어진 CameraModel 만 갖고 있으므로, 먼저 현재 R,t 에서
    mount 값(roll,pitch,yaw,position)을 역산한 뒤, 지정된 값만 바꿔 다시 R,t 를 만든다.
    화각(hfov)을 바꾸면 유효 시야각 마스크만 바뀌는 게 아니라 렌즈 자체가 달라지는
    것이므로, make_K() 로 초점거리(K)도 새 화각에 맞게 다시 계산한다.
    """
    cam = rig.camera(name)
    R_mount = cam.R.T @ R_AXIS_VEH2CAM
    cur_roll, cur_pitch, cur_yaw = R_to_euler(R_mount)
    cur_pos = cam.center
    cur_hfov = cam.hfov_deg

    new_roll = cur_roll if roll is None else roll
    new_pitch = cur_pitch if pitch is None else pitch
    new_yaw = cur_yaw if yaw is None else yaw
    new_pos = cur_pos if position is None else np.array(position, dtype=float)
    new_hfov = cur_hfov if hfov is None else hfov
    new_K = cam.K if hfov is None else make_K(cam.image_size[0], cam.image_size[1], new_hfov, cam.model)

    hd(f"카메라 오버라이드: {name}")
    print(f"  이전: position=({cur_pos[0]:+.3f},{cur_pos[1]:+.3f},{cur_pos[2]:+.3f})  "
          f"rpy=({cur_roll:+.1f},{cur_pitch:+.1f},{cur_yaw:+.1f})  hfov={cur_hfov:.1f}  fx={cam.K[0,0]:.2f}")
    print(f"  이후: position=({new_pos[0]:+.3f},{new_pos[1]:+.3f},{new_pos[2]:+.3f})  "
          f"rpy=({new_roll:+.1f},{new_pitch:+.1f},{new_yaw:+.1f})  hfov={new_hfov:.1f}  fx={new_K[0,0]:.2f}")

    new_cam = CameraModel.from_mount(
        name=cam.name,
        image_size=cam.image_size,
        K=new_K,
        D=cam.D,
        position=tuple(new_pos),
        roll=new_roll,
        pitch=new_pitch,
        yaw=new_yaw,
        model=cam.model,
        hfov_deg=new_hfov,
    )
    idx = rig.cameras.index(cam)
    rig.cameras[idx] = new_cam
    print(f"  → R,t,K 재계산 완료. 새 center={new_cam.center.round(4)}, forward={new_cam.forward.round(4)}")


def frame_hit_stats(frame: np.ndarray) -> tuple[float, float, float]:
    """render() 결과에서 지면/박스 적중, 화각밖(비네팅), 하늘 배경 비율을 추정."""
    total = frame.shape[0] * frame.shape[1]
    vignette = float(np.all(frame == 0, axis=-1).sum()) / total
    sky = float(np.all(frame == np.array(SKY_COLOR), axis=-1).sum()) / total
    hit = 1.0 - vignette - sky
    return hit, sky, vignette


def trace_project(cam: CameraModel, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
    """model.py 의 project() 연산을 그대로 한 점에 대해 풀어서 프린트한다."""
    pt = np.array([x, y, z])
    print(f"    world point (x,y,z)       = ({x:+.4f}, {y:+.4f}, {z:+.4f})")
    pc = cam.R @ pt + cam.t
    print(f"    camera coord X_c = R·X_w+t = ({pc[0]:+.4f}, {pc[1]:+.4f}, {pc[2]:+.4f})")
    if pc[2] <= 1e-9:
        print("    Z_c <= 0 → 카메라 뒤쪽, 투영 불가")
        return float("nan"), float("nan")
    xn, yn = pc[0] / pc[2], pc[1] / pc[2]
    print(f"    normalized (xn,yn) = Xc/Zc  = ({xn:+.5f}, {yn:+.5f})")
    xd, yd, theta = cam._distort(np.array([xn]), np.array([yn]))
    xd, yd, theta = float(xd[0]), float(yd[0]), float(theta[0])
    print(f"    theta = arctan(|xn,yn|)     = {np.degrees(theta):.3f}°  (hfov/2={cam.hfov_deg/2:.1f}°)")
    print(f"    distorted (xd,yd)           = ({xd:+.5f}, {yd:+.5f})")
    u = cam.K[0, 0] * xd + cam.K[0, 1] * yd + cam.K[0, 2]
    v = cam.K[1, 1] * yd + cam.K[1, 2]
    print(f"    pixel (u,v) = fx·xd+cx, ... = ({u:.2f}, {v:.2f})   image={cam.image_size}")
    return u, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--out", default="out/debug")
    ap.add_argument("--no-boxes", action="store_true", help="장애물 없이 순수 지면만 렌더링")
    ap.add_argument(
        "--box",
        action="append",
        type=float,
        nargs=5,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "HEIGHT"),
        help="장애물(기둥) 위치를 직접 지정 (미터, 여러 번 줄 수 있음). 지정하면 기본 배치 대신 이것만 사용",
    )
    ap.add_argument("--camera", default=None, help="각도/위치를 바꿔볼 카메라 이름 (front/left/rear/right)")
    ap.add_argument("--roll", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--hfov", type=float, default=None, help="화각(도) 변경. K(초점거리)도 함께 재계산됨")
    ap.add_argument("--trace-row", type=int, default=None)
    ap.add_argument("--trace-col", type=int, default=None)
    ap.add_argument("--trace-camera", default=None, help="상세 추적에 쓸 카메라 (기본: 가중치 최대인 카메라)")
    args = ap.parse_args()
    if args.no_boxes and args.box:
        ap.error("--no-boxes 와 --box 는 함께 쓸 수 없습니다")
    if not args.camera and any(v is not None for v in (args.roll, args.pitch, args.yaw, args.position, args.hfov)):
        ap.error("--roll/--pitch/--yaw/--position/--hfov 를 쓰려면 --camera 로 대상 카메라를 지정해야 합니다")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 0) 리그 로드
    hd("0) 리그 설정")
    rig = RigConfig.load(args.config)
    print(f"BEV: x=[{rig.bev.x_min},{rig.bev.x_max}] y=[{rig.bev.y_min},{rig.bev.y_max}] "
          f"res={rig.bev.resolution*1000:.1f}mm/px → shape={rig.bev.shape}")
    print(f"blend mode={rig.blend.mode}  feather_m={rig.blend.feather_m}  photometric={rig.blend.photometric}")
    for cam in rig.cameras:
        print("  " + describe_camera(cam))

    # ------------------------------------------------------------ 1) 시나리오: 카메라 오버라이드
    if args.camera:
        apply_mount_override(rig, args.camera, args.roll, args.pitch, args.yaw, args.position, args.hfov)

    # ------------------------------------------------------------ 2) 가상 지면 텍스처
    hd("2) 가상 지면 텍스처 생성")
    texture = make_ground_texture(rig.bev, tile=0.10, with_aruco=True)
    cv2.imwrite(str(out / "ground_truth.png"), texture)
    print(f"texture shape={texture.shape} dtype={texture.dtype} → {out/'ground_truth.png'}")

    if args.no_boxes:
        boxes = []
    elif args.box:
        boxes = [
            Box(x=(x_min, x_max), y=(y_min, y_max), height=h, color=BOX_PALETTE[i % len(BOX_PALETTE)])
            for i, (x_min, x_max, y_min, y_max, h) in enumerate(args.box)
        ]
    else:
        boxes = default_boxes()
    hd(f"3) 씬 구성 (장애물 {len(boxes)}개, --no-boxes={args.no_boxes}, --box 지정={bool(args.box)})")
    for b in boxes:
        print(f"  box x={tuple(b.lo[:1].tolist()+b.hi[:1].tolist())} "
              f"y=({b.lo[1]:.2f},{b.hi[1]:.2f}) h={b.hi[2]:.2f} color={tuple(b.color)}")
    scene = GroundScene(texture, rig.bev, boxes=boxes)

    # ------------------------------------------------------------ 4) 카메라별 렌더 + LUT
    hd("4) 카메라별 렌더링 & LUT")
    frames: dict[str, np.ndarray] = {}
    luts = {}
    for cam in rig.cameras:
        frame = scene.render(cam)
        frames[cam.name] = frame
        cv2.imwrite(str(out / f"cam_{cam.name}.png"), frame)
        hit, sky, vig = frame_hit_stats(frame)
        lut = build_ground_lut(cam, rig.bev)
        luts[cam.name] = lut
        print(f"  {cam.name:>6}: render 지면/박스 적중={hit*100:5.1f}%  하늘={sky*100:5.1f}%  "
              f"비네팅={vig*100:5.1f}%   |  LUT BEV 커버리지={lut.coverage*100:5.1f}%")

    # ------------------------------------------------------------ 5) 블렌딩 가중치
    hd("5) 블렌딩 가중치")
    weights = build_weights(luts, rig.cameras, rig.bev, rig.blend)
    for name, w in weights.items():
        nz = w[w > 1e-6]
        print(f"  {name:>6}: 기여 픽셀 비율={float((w>1e-6).mean())*100:5.1f}%  "
              f"평균 가중치(기여 구간)={nz.mean() if nz.size else 0:.3f}")

    # ------------------------------------------------------------ 6) 합성
    hd("6) AVM 합성")
    stitcher = AVMStitcher(rig, luts=luts)
    cov = stitcher.coverage_report()
    for name in rig.names:
        print(f"  커버리지 {name:>6}: {cov[name]*100:5.1f}%")
    print(f"  합계(any): {cov['_any']*100:5.1f}%   사각지대: {cov['_blind']*100:5.1f}%")

    bev = stitcher(frames)
    cv2.imwrite(str(out / "bev.png"), bev)

    valid = stitcher.coverage & ~stitcher.vehicle_mask
    diff = np.abs(bev.astype(np.float32) - texture.astype(np.float32)).mean(axis=2)
    mae = float(diff[valid].mean()) if valid.any() else float("nan")
    print(f"\n  정답(가상 지면) 대비 평균 절대오차 MAE = {mae:.2f} / 255"
          f"  (박스 없음일 때 이 값이 순수 기하 오차)")
    print(f"  결과 저장: {out.resolve()}")

    # ------------------------------------------------------------ 7) 픽셀 1개 상세 추적
    hd("7) 샘플 1개 상세 추적")
    if args.trace_row is None or args.trace_col is None:
        col, row = rig.bev.world_to_pixel(0.25, 0.0)  # 차량 앞쪽 25cm 지점
        row, col = int(round(row)), int(round(col))
    else:
        row, col = args.trace_row, args.trace_col
    row = int(np.clip(row, 0, rig.bev.height - 1))
    col = int(np.clip(col, 0, rig.bev.width - 1))

    xs, ys = rig.bev.grid()
    wx, wy = float(xs[row, col]), float(ys[row, col])
    print(f"  추적 BEV 픽셀 (row={row}, col={col}) → world (x={wx:+.4f}, y={wy:+.4f})")

    if args.trace_camera:
        cam_name = args.trace_camera
    else:
        cam_name = max(rig.names, key=lambda n: weights[n][row, col])
    cam = rig.camera(cam_name)
    print(f"  추적 카메라: {cam_name}  (이 픽셀에서의 블렌딩 가중치={weights[cam_name][row, col]:.3f})")

    print("\n  [model.py project() 연산 과정]")
    u, v = trace_project(cam, wx, wy, 0.0)

    lut = luts[cam_name]
    print(f"\n  LUT에 저장된 값과 비교: map_x={lut.map_x[row, col]:.2f}, map_y={lut.map_y[row, col]:.2f}, "
          f"mask={lut.mask[row, col]}  (project() 결과와 일치해야 함)")

    if lut.mask[row, col] and 0 <= int(round(v)) < cam.image_size[1] and 0 <= int(round(u)) < cam.image_size[0]:
        sample = frames[cam_name][int(round(v)), int(round(u))]
        print(f"  카메라 원본 프레임에서 샘플한 색(BGR, nearest) = {tuple(int(c) for c in sample)}")
    else:
        print("  이 픽셀은 해당 카메라의 유효 화각 밖입니다 (mask=False)")

    final_color = bev[row, col]
    gt_color = texture[row, col]
    print(f"  최종 BEV 픽셀 색(BGR)   = {tuple(int(c) for c in final_color)}")
    print(f"  정답 지면 텍스처 색(BGR) = {tuple(int(c) for c in gt_color)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
