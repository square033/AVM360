"""USB 카메라 다중 캡처.

USB 4대를 한 PC 에 물릴 때 실패하는 원인은 거의 항상 둘 중 하나다.

1. 대역폭 — UVC 기본 포맷인 YUYV 는 무압축이라 720p30 한 대가 이미 약 1.3 Gbps 를
   먹는다. USB 3.0 이라도 4대는 못 버틴다. 반드시 MJPG 로 강제해야 한다.
2. 오토 기능 — 오토포커스가 돌면 초점거리가 변해 캘리브레이션한 K 가 무효가 되고,
   오토 노출/화이트밸런스가 돌면 카메라마다 밝기가 계속 달라져 이음선이 뛴다.
   캘리브레이션 전에 전부 수동으로 고정해야 한다.

UVC 에는 하드웨어 동기 신호가 없으므로 프레임 동기는 타임스탬프 기반 근사다.
정지 상태 검증에는 문제없지만, 움직이는 플랫폼에서는 스큐를 확인해야 한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraSpec:
    """카메라 한 대의 캡처 설정."""

    name: str
    device: int | str  # /dev/videoN 의 N, 또는 경로/URL
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    exposure: float | None = None  # None 이면 현재 값 유지
    gain: float | None = None
    wb_temperature: float | None = None


class _CameraThread(threading.Thread):
    """카메라 한 대를 계속 읽어 최신 프레임만 보관한다."""

    def __init__(self, spec: CameraSpec, backend: int | None = None):
        super().__init__(daemon=True, name=f"cam-{spec.name}")
        self.spec = spec
        self.backend = backend
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp: float = 0.0
        self._running = False
        self.frame_count = 0
        self.cap = None

    def open(self) -> None:
        import cv2

        backend = self.backend if self.backend is not None else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.spec.device, backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.spec.name} (device={self.spec.device})")

        s = self.spec
        # 순서 중요: fourcc 를 먼저 설정해야 해상도/fps 가 MJPG 기준으로 잡힌다.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        cap.set(cv2.CAP_PROP_FPS, s.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연 누적 방지

        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # K 가 변하지 않도록 고정
        if s.exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1=manual, 3=auto
            cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure)
        if s.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, s.gain)
        if s.wb_temperature is not None:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s.wb_temperature)

        self.cap = cap

    def actual_settings(self) -> dict:
        import cv2

        if self.cap is None:
            return {}
        raw = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc,
        }

    def run(self) -> None:
        self._running = True
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
                self.frame_count += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None, 0.0) if self._frame is None else (self._frame.copy(), self._stamp)

    def stop(self) -> None:
        self._running = False
        self.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class MultiCamCapture:
    """카메라 여러 대를 스레드로 동시에 읽는다."""

    def __init__(self, specs: list[CameraSpec], backend: int | None = None):
        self.specs = specs
        self.threads = {s.name: _CameraThread(s, backend) for s in specs}

    def __enter__(self) -> "MultiCamCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self, warmup: float = 1.0) -> None:
        for t in self.threads.values():
            t.open()
            t.start()
        deadline = time.monotonic() + warmup
        while time.monotonic() < deadline:  # 오토 기능 안정화 + 첫 프레임 대기
            if all(t.latest()[0] is not None for t in self.threads.values()):
                break
            time.sleep(0.02)

    def read(self) -> tuple[dict[str, np.ndarray], float]:
        """모든 카메라의 최신 프레임과 스큐(ms)를 반환한다."""
        frames, stamps = {}, []
        for name, t in self.threads.items():
            frame, stamp = t.latest()
            if frame is None:
                raise RuntimeError(f"no frame from camera {name!r} yet")
            frames[name] = frame
            stamps.append(stamp)
        skew_ms = (max(stamps) - min(stamps)) * 1000.0
        return frames, skew_ms

    def settings(self) -> dict[str, dict]:
        return {name: t.actual_settings() for name, t in self.threads.items()}

    def stop(self) -> None:
        for t in self.threads.values():
            t.stop()


def probe_devices(max_index: int = 10) -> list[dict]:
    """연결된 카메라를 훑어 지원 해상도/포맷을 확인한다."""
    import cv2

    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        raw = int(cap.get(cv2.CAP_PROP_FOURCC))
        found.append(
            {
                "index": idx,
                "ok": bool(ok),
                "shape": None if frame is None else frame.shape,
                "fourcc": "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
            }
        )
        cap.release()
    return found
