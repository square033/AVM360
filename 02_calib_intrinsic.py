#!/usr/bin/env python3
"""내부 파라미터 캘리브레이션 — 체커보드로 K, D 구하기.

    python scripts/02_calib_intrinsic.py --name front --square 0.025
    python scripts/02_calib_intrinsic.py --all --square 0.025   # 4대 한 번에

out/intrinsic/<name>/ 안의 이미지를 읽어 K, D 를 구하고 --config 파일에 써넣는다.

체커보드 준비:
  - 9x6 내부코너(= 10x7 칸) 보드를 딱딱한 판에 평평하게 붙인다. 종이가 휘면 그대로 오차다.
  - --square 는 한 칸의 실제 한 변 길이(미터). 인쇄 후 자로 재서 넣는다.
  - 20~30장, 보드가 화면 중앙뿐 아니라 네 귀퉁이에도 오도록, 기울기도 다양하게.
    어안 렌즈는 가장자리 왜곡이 심해서 가장자리 샘플이 없으면 D 가 제대로 안 나온다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avm360 import RigConfig
from avm360.calib_intrinsic import calibrate_folder

RMS_WARN_PX = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rig_desk.yaml")
    ap.add_argument("--images", default="out/intrinsic")
    ap.add_argument("--name", default=None, help="캘리브레이션할 카메라 이름")
    ap.add_argument("--all", action="store_true", help="설정에 있는 모든 카메라 처리")
    ap.add_argument("--pattern", type=int, nargs=2, default=[9, 6], help="내부코너 (cols rows)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸 실제 변 길이 (미터)")
    ap.add_argument("--out", default=None, help="결과 저장 경로 (기본: --config 덮어쓰기)")
    args = ap.parse_args()

    if not args.all and args.name is None:
        ap.error("--name 또는 --all 중 하나가 필요합니다")

    rig = RigConfig.load(args.config)
    targets = rig.names if args.all else [args.name]
    root = Path(args.images)

    failures = []
    for name in targets:
        cam = rig.camera(name)
        folder = root / name
        if not folder.is_dir():
            failures.append(f"[{name}] {folder} 폴더가 없습니다")
            continue
        try:
            solved, rms, used = calibrate_folder(
                folder, tuple(args.pattern), args.square,
                model=cam.model, name=name, hfov_deg=cam.hfov_deg,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            failures.append(f"[{name}] {exc}")
            continue

        cam.K, cam.D, cam.image_size = solved.K, solved.D, solved.image_size
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        cx, cy = cam.K[0, 2], cam.K[1, 2]
        flag = "  <-- 오차 큼, 흐린 이미지 제외 후 재시도 권장" if rms > RMS_WARN_PX else ""
        print(f"[{name}] {used}장 사용, RMS = {rms:.3f}px{flag}")
        print(f"        fx={fx:.1f} fy={fy:.1f}  cx={cx:.1f} cy={cy:.1f}  D={cam.D.round(4).tolist()}")
        # 주점이 중심에서 크게 벗어나면 보통 샘플이 한쪽에 몰린 것이다.
        w, h = cam.image_size
        if abs(cx - w / 2) > w * 0.15 or abs(cy - h / 2) > h * 0.15:
            print("        경고: 주점이 이미지 중심에서 많이 벗어났습니다. 보드를 화면 전 영역에 고루 놓고 재촬영하세요.")

    out_path = args.out or args.config
    rig.save(out_path)
    print(f"\n저장: {Path(out_path).resolve()}")

    if failures:
        print("\n실패:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n다음 단계: python scripts/01_capture.py --mode extrinsic  →  03_calib_extrinsic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
