#!/usr/bin/env python3
"""외부 파라미터 캘리브레이션 — 지면 마커로 각 카메라의 R, t 구하기 (BEV.md 3.5절).

    python scripts/03_calib_extrinsic.py \
        --config configs/rig_desk.yaml \
        --markers configs/markers.yaml \
        --images out/extrinsic

--images 폴더에는 카메라 이름과 같은 파일이 있어야 한다 (front.png, left.png, ...).
scripts/01_capture.py --mode extrinsic 로 찍으면 그 이름으로 저장된다.

각 카메라는 자기 눈에 보이는 마커만 쓰지만, 모든 마커의 좌표가 같은 차량 좌표계로
정의되어 있으므로 결과 R, t 도 자동으로 같은 기준을 갖게 된다. 이것이 BEV.md 6-(B)절의
"정렬은 캘리브레이션 단계에서 이미 끝나 있다"는 말의 실체다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from avm360 import RigConfig
from avm360.calib_extrinsic import describe_pose, load_marker_map, solve_extrinsic

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
RMS_WARN_PX = 3.0


def find_image(folder: Path, name: str) -> Path | None:
    for suffix in SUFFIXES:
        p = folder / f"{name}{suffix}"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--markers", default="configs/markers.yaml")
    ap.add_argument("--images", default="out/extrinsic")
    ap.add_argument("--out", default=None, help="결과 저장 경로 (기본: --config 덮어쓰기)")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--min-markers", type=int, default=2)
    args = ap.parse_args()

    rig = RigConfig.load(args.config)
    markers = load_marker_map(args.markers)
    folder = Path(args.images)
    print(f"마커 배치도: {len(markers)}개 (id {sorted(markers)})\n")

    updated = []
    failures = []
    for cam in rig.cameras:
        path = find_image(folder, cam.name)
        if path is None:
            failures.append(f"[{cam.name}] {folder}/{cam.name}.png 이미지가 없습니다")
            updated.append(cam)
            continue

        image = cv2.imread(str(path))
        if image is None:
            failures.append(f"[{cam.name}] {path} 를 읽을 수 없습니다")
            updated.append(cam)
            continue
        if (image.shape[1], image.shape[0]) != cam.image_size:
            failures.append(
                f"[{cam.name}] 해상도 불일치: 이미지 {image.shape[1]}x{image.shape[0]} "
                f"vs 설정 {cam.image_size[0]}x{cam.image_size[1]}"
            )
            updated.append(cam)
            continue

        try:
            solved, rms, used = solve_extrinsic(
                cam, image, markers, dictionary=args.dictionary, min_markers=args.min_markers
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            updated.append(cam)
            continue

        flag = "  <-- 오차 큼" if rms > RMS_WARN_PX else ""
        print(f"{describe_pose(solved)}")
        print(f"         마커 {len(used)}개(id {used}), 재투영 RMS = {rms:.2f}px{flag}")
        updated.append(solved)

    rig.cameras = updated
    out_path = args.out or args.config
    rig.save(out_path)
    print(f"\n저장: {Path(out_path).resolve()}")

    if failures:
        print("\n실패/경고:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\n마커가 안 잡히면: 조명 반사 줄이기, 마커를 카메라에 더 가깝게, "
            "초점 확인, 마커 주변 흰 여백 확보를 점검하세요."
        )
        return 1

    print("\n다음 단계: python scripts/04_build_lut.py --config", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
