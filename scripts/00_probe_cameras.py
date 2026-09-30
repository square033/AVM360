#!/usr/bin/env python3
"""연결된 USB 카메라 탐색 + 4대 동시 구동 대역폭 테스트.

    python scripts/00_probe_cameras.py                 # 장치 목록만
    python scripts/00_probe_cameras.py --test 0 2 4 6  # 4대 동시 구동 테스트

USB 4대를 물릴 때 가장 흔한 실패는 대역폭 부족이다. 이 스크립트로 실제 달성 FPS 와
프레임 스큐를 먼저 재본 뒤에 캘리브레이션으로 넘어가는 것이 좋다.

FPS 가 목표치의 절반 이하로 떨어지면:
  - fourcc 가 MJPG 로 잡혔는지 확인 (YUYV 로 잡히면 대역폭이 10배 이상 든다)
  - 해상도를 640x480 으로 낮춰본다
  - 카메라를 서로 다른 USB 컨트롤러(루트 허브)에 나눠 꽂는다
    확인: lsusb -t  →  같은 Bus 아래 4대가 몰려 있으면 대역폭을 나눠 쓴다
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avm360.capture import CameraSpec, MultiCamCapture, probe_devices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-index", type=int, default=10)
    ap.add_argument("--test", type=int, nargs="+", metavar="IDX", help="동시 구동 테스트할 장치 번호들")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    if not args.test:
        print(f"장치 탐색 (index 0..{args.max_index-1}) ...\n")
        found = probe_devices(args.max_index)
        if not found:
            print("카메라를 찾지 못했습니다. 연결과 권한(Linux: video 그룹)을 확인하세요.")
            return 1
        for d in found:
            status = "OK  " if d["ok"] else "실패"
            print(f"  /dev/video{d['index']}  {status}  {d['shape']}  fourcc={d['fourcc']}  fps={d['fps']:.0f}")
        print(f"\n총 {len(found)}대 발견. 4대 동시 테스트:")
        idxs = " ".join(str(d["index"]) for d in found[:4])
        print(f"  python scripts/00_probe_cameras.py --test {idxs}")
        return 0

    names = ["front", "left", "rear", "right"]
    specs = [
        CameraSpec(name=names[i] if i < len(names) else f"cam{i}", device=dev,
                   width=args.width, height=args.height, fps=args.fps)
        for i, dev in enumerate(args.test)
    ]
    print(f"{len(specs)}대 동시 구동 테스트 ({args.width}x{args.height} @ {args.fps}fps, {args.seconds:.0f}초)\n")

    with MultiCamCapture(specs) as cap:
        for name, s in cap.settings().items():
            warn = "  <-- MJPG 아님! 대역폭 문제 발생" if s.get("fourcc") != "MJPG" else ""
            print(f"  {name:>6}: {s['width']}x{s['height']} {s['fourcc']} {s['fps']:.0f}fps{warn}")

        print("\n측정 중 ...")
        start = time.monotonic()
        reads, skews = 0, []
        while time.monotonic() - start < args.seconds:
            _, skew = cap.read()
            skews.append(skew)
            reads += 1
            time.sleep(0.001)
        elapsed = time.monotonic() - start
        counts = {n: t.frame_count for n, t in cap.threads.items()}

    print(f"\n결과 ({elapsed:.1f}초):")
    ok = True
    for name, count in counts.items():
        fps = count / elapsed
        flag = ""
        if fps < args.fps * 0.5:
            flag, ok = "  <-- 목표의 절반 미만", False
        print(f"  {name:>6}: {fps:5.1f} fps ({count} 프레임){flag}")
    if skews:
        import statistics

        print(f"\n프레임 스큐: 중앙값 {statistics.median(skews):.1f}ms, 최대 {max(skews):.1f}ms")
        print("  (UVC 는 하드웨어 동기가 없어 스큐가 존재한다. 정지 상태 검증에는 무방하다)")

    if not ok:
        print("\n대역폭 부족으로 보입니다. 상단 주석의 대처법을 참고하세요.")
        return 1
    print("\n정상입니다. 다음 단계: scripts/01_capture.py 로 캘리브레이션 이미지 촬영")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
