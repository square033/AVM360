#!/usr/bin/env python3
"""지면 캘리브레이션용 ArUco 마커 생성 + 배치도(YAML) 작성.

    python scripts/07_make_markers.py --config configs/rig_desk.yaml --size 0.06

만들어지는 것:
    out/markers/marker_XX.png   인쇄용 마커 이미지 (여백 포함)
    out/markers/sheet.png       A4 한 장에 모아 찍은 시트
    configs/markers.yaml        마커 id 와 차량 좌표계 위치 배치도

인쇄 후 배치 방법:
    1. sheet.png 를 "실제 크기(100%, 배율 조정 없음)"로 인쇄한다. 인쇄된 마커의 검은
       사각형 한 변을 자로 재서 --size 로 준 값과 같은지 반드시 확인한다.
    2. configs/markers.yaml 의 좌표대로 바닥에 붙인다. 원점은 차량(상자) 중심,
       +X 가 전방, +Y 가 좌측이다. 마커 이미지의 위쪽이 전방을 향하게 놓는다.
    3. 마커는 평평하게 붙어야 한다. 들뜨면 그만큼 R,t 오차가 된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from avm360 import RigConfig
from avm360.calib_extrinsic import GroundMarker, save_marker_map
from avm360.synth import default_marker_layout

DPI = 300
MM_PER_INCH = 25.4


def render_marker(adict, mid: int, side_px: int, quiet_ratio: float = 0.25) -> np.ndarray:
    try:
        patch = cv2.aruco.generateImageMarker(adict, mid, side_px)
    except AttributeError:
        patch = cv2.aruco.drawMarker(adict, mid, side_px)
    pad = int(side_px * quiet_ratio)
    canvas = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, dtype=np.uint8)
    canvas[pad : pad + side_px, pad : pad + side_px] = patch
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--size", type=float, default=0.06, help="마커 한 변 길이 (미터)")
    ap.add_argument("--out", default="out/markers")
    ap.add_argument("--map-out", default="configs/markers.yaml")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rig = RigConfig.load(args.config)
    layout = default_marker_layout(rig.bev)
    markers = {mid: GroundMarker(id=mid, x=x, y=y, size=args.size) for mid, x, y in layout}

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    side_px = int(round(args.size * 1000 / MM_PER_INCH * DPI))

    tiles = []
    for mid in sorted(markers):
        img = render_marker(adict, mid, side_px)
        labeled = cv2.copyMakeBorder(img, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=255)
        m = markers[mid]
        cv2.putText(
            labeled, f"id={mid}  x={m.x:+.2f} y={m.y:+.2f}",
            (5, labeled.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2,
        )
        cv2.imwrite(str(out / f"marker_{mid:02d}.png"), labeled)
        tiles.append(labeled)

    # A4 세로 (210x297mm) 시트에 격자로 배치
    cols = max(1, int((210 - 20) / (args.size * 1000 + 12)))
    rows = int(np.ceil(len(tiles) / cols))
    th, tw = tiles[0].shape
    gap = int(12 / MM_PER_INCH * DPI)
    sheet = np.full((rows * (th + gap) + gap, cols * (tw + gap) + gap), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0, x0 = gap + r * (th + gap), gap + c * (tw + gap)
        sheet[y0 : y0 + th, x0 : x0 + tw] = tile
    cv2.imwrite(str(out / "sheet.png"), sheet)

    save_marker_map(markers, args.map_out, args.size)

    print(f"마커 {len(markers)}개 생성 ({args.size*1000:.0f}mm, {args.dictionary}, {DPI}dpi)")
    print(f"  인쇄용 시트 : {(out / 'sheet.png').resolve()}")
    print(f"  배치도      : {Path(args.map_out).resolve()}")
    print("\n바닥 배치 좌표 (차량 중심 원점, +X 전방 / +Y 좌측):")
    for mid in sorted(markers):
        m = markers[mid]
        print(f"  id={mid:>2}  x={m.x:+.3f}m  y={m.y:+.3f}m")
    print("\n주의: 반드시 100% 배율로 인쇄하고, 인쇄된 마커 변 길이를 자로 검증하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
