#!/usr/bin/env python3
"""캘리브레이션용 이미지 촬영.

내부 파라미터용 (카메라 한 대씩, 체커보드를 여러 자세로 20~30장):
    python scripts/01_capture.py --mode intrinsic --device 0 --name front

외부 파라미터용 (4대 동시, 지면 마커가 보이는 상태로 각 1장):
    python scripts/01_capture.py --mode extrinsic --devices 0 2 4 6

조작:
    SPACE  현재 프레임 저장
    a      (intrinsic) 자동 촬영 켜기/끄기 — 체커보드가 검출될 때만 일정 간격으로 저장
    q/ESC  종료
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360.capture import CameraSpec, MultiCamCapture

CAM_NAMES = ["front", "left", "rear", "right"]


def draw_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return out


def capture_intrinsic(args) -> int:
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    pattern = tuple(args.pattern)

    spec = CameraSpec(name=args.name, device=args.device, width=args.width, height=args.height, fps=args.fps)
    saved = len(list(out.glob("*.png")))
    auto, last_auto = False, 0.0

    print(f"[{args.name}] 체커보드 {pattern[0]}x{pattern[1]} 내부코너. 저장 위치: {out}")
    print("보드를 화면 곳곳(특히 가장자리)과 여러 기울기로 옮겨가며 20~30장 찍으세요.")

    with MultiCamCapture([spec]) as cap:
        while True:
            frames, _ = cap.read()
            frame = frames[args.name]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
            )
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)

            vis = draw_hud(vis, [
                f"{args.name}  저장 {saved}장  {'보드 검출됨' if found else '보드 없음'}",
                f"자동촬영: {'ON' if auto else 'OFF'}  (a 키로 전환)",
                "SPACE 저장   q 종료",
            ])
            cv2.imshow("intrinsic capture", vis)

            now = time.monotonic()
            do_save = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                do_save = True
            elif key == ord("a"):
                auto = not auto
            elif key in (ord("q"), 27):
                break
            if auto and found and now - last_auto > args.interval:
                do_save, last_auto = True, now

            if do_save:
                cv2.imwrite(str(out / f"{saved:03d}.png"), frame)
                saved += 1
                print(f"  저장 {saved}장" + ("" if found else "  (보드 미검출 — 캘리브레이션에서 무시됨)"))

    cv2.destroyAllWindows()
    print(f"\n총 {saved}장. 다음: python scripts/02_calib_intrinsic.py --name {args.name}")
    return 0


def capture_extrinsic(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        CameraSpec(name=CAM_NAMES[i], device=dev, width=args.width, height=args.height, fps=args.fps)
        for i, dev in enumerate(args.devices)
    ]
    try:
        adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    except AttributeError:
        detector = None

    print(f"{len(specs)}대 동시 촬영. 저장 위치: {out}")
    print("모든 카메라가 지면 마커를 2개 이상 보도록 배치한 뒤 SPACE 를 누르세요.")

    with MultiCamCapture(specs) as cap:
        while True:
            frames, skew = cap.read()
            tiles, counts = [], []
            for name in [s.name for s in specs]:
                vis = frames[name].copy()
                n = 0
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(vis)
                    if ids is not None:
                        n = len(ids)
                        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                counts.append(f"{name}:{n}")
                vis = draw_hud(vis, [f"{name}  마커 {n}개" + ("" if n >= 2 else "  <-- 부족")])
                tiles.append(cv2.resize(vis, (640, 360)))

            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]) if len(tiles) == 4 else np.hstack(tiles)
            grid = draw_hud(grid, [f"스큐 {skew:.0f}ms   검출: {' '.join(counts)}   SPACE 저장 / q 종료"])
            cv2.imshow("extrinsic capture", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                for name, frame in frames.items():
                    cv2.imwrite(str(out / f"{name}.png"), frame)
                print(f"  저장 완료: {', '.join(f'{n}.png' for n in frames)}")
            elif key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    print("\n다음: python scripts/03_calib_extrinsic.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["intrinsic", "extrinsic"], required=True)
    ap.add_argument("--device", type=int, default=0, help="intrinsic: 촬영할 장치 번호")
    ap.add_argument("--name", default="front", help="intrinsic: 카메라 이름")
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 2, 4, 6], help="extrinsic: 장치 번호 4개")
    ap.add_argument("--pattern", type=int, nargs=2, default=[9, 6], help="체커보드 내부코너 (cols rows)")
    ap.add_argument("--interval", type=float, default=1.2, help="자동촬영 간격(초)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = "out/intrinsic" if args.mode == "intrinsic" else "out/extrinsic"
    return capture_intrinsic(args) if args.mode == "intrinsic" else capture_extrinsic(args)


if __name__ == "__main__":
    raise SystemExit(main())
