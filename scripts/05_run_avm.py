#!/usr/bin/env python3
"""실시간 360° AVM 실행.

    python scripts/05_run_avm.py --devices 0 2 4 6
    python scripts/05_run_avm.py --images out/extrinsic     # 저장된 이미지로 1회 합성

조작:
    s   이음선 표시 켜기/끄기
    p   밝기 보정(photometric) 켜기/끄기
    r   원본 4분할 화면 보기 전환
    c   현재 화면 저장
    q   종료
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import AVMStitcher, RigConfig

CAM_NAMES = ["front", "left", "rear", "right"]
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def make_stitcher(rig: RigConfig, lut_path: str | None) -> AVMStitcher:
    if lut_path and Path(lut_path).exists():
        print(f"LUT 캐시 사용: {lut_path}")
        return AVMStitcher.from_cached(rig, lut_path)
    print("LUT 계산 중 ...")
    return AVMStitcher(rig)


def run_still(args, rig: RigConfig, stitcher: AVMStitcher) -> int:
    folder = Path(args.images)
    frames = {}
    for cam in rig.cameras:
        path = next((folder / f"{cam.name}{s}" for s in SUFFIXES if (folder / f"{cam.name}{s}").exists()), None)
        if path is None:
            print(f"오류: {folder}/{cam.name}.png 가 없습니다")
            return 1
        frames[cam.name] = cv2.imread(str(path))

    bev = stitcher(frames, draw_seams=args.seams)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bev)
    print(f"저장: {out.resolve()}")
    if not args.headless:
        cv2.imshow("AVM 360", bev)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def tile_raw(frames: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    tiles = [cv2.resize(frames[n], (480, 270)) for n in names]
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])


def run_live(args, rig: RigConfig, stitcher: AVMStitcher) -> int:
    from avm360.capture import CameraSpec, MultiCamCapture

    names = rig.names
    specs = [
        CameraSpec(name=names[i], device=dev, width=cam.image_size[0], height=cam.image_size[1], fps=args.fps)
        for i, (dev, cam) in enumerate(zip(args.devices, rig.cameras))
    ]
    show_seams, show_raw = args.seams, False
    fps_ema = 0.0

    with MultiCamCapture(specs) as cap:
        for name, s in cap.settings().items():
            if s.get("fourcc") != "MJPG":
                print(f"경고: {name} 이 {s.get('fourcc')} 로 열렸습니다. 대역폭 문제가 생길 수 있습니다.")
        print("실행 중 — s:이음선  p:밝기보정  r:원본  c:저장  q:종료")

        while True:
            t0 = time.monotonic()
            frames, skew = cap.read()
            view = tile_raw(frames, names) if show_raw else stitcher(frames, draw_seams=show_seams)

            dt = time.monotonic() - t0
            fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
            hud = f"{fps_ema:4.1f} fps   skew {skew:4.0f}ms   photometric {'ON' if rig.blend.photometric else 'OFF'}"
            cv2.putText(view, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(view, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.imshow("AVM 360", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                show_seams = not show_seams
            elif key == ord("r"):
                show_raw = not show_raw
            elif key == ord("p"):
                rig.blend.photometric = not rig.blend.photometric
            elif key == ord("c"):
                path = Path(args.out).with_name(f"avm_{int(time.time())}.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), view)
                print(f"  저장: {path}")

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--lut", default="out/lut.npz")
    ap.add_argument("--devices", type=int, nargs="+", default=None, help="실시간 실행할 장치 번호 4개")
    ap.add_argument("--images", default=None, help="저장된 이미지 폴더로 1회 합성")
    ap.add_argument("--out", default="out/avm.png")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seams", action="store_true")
    ap.add_argument("--headless", action="store_true", help="창을 띄우지 않고 저장만")
    args = ap.parse_args()

    if args.devices is None and args.images is None:
        ap.error("--devices 또는 --images 중 하나가 필요합니다")

    rig = RigConfig.load(args.config)
    stitcher = make_stitcher(rig, args.lut)
    cov = stitcher.coverage_report()
    print(f"커버리지 {cov['_any']*100:.1f}%, 사각지대 {cov['_blind']*100:.1f}%")

    return run_still(args, rig, stitcher) if args.images else run_live(args, rig, stitcher)


if __name__ == "__main__":
    raise SystemExit(main())
