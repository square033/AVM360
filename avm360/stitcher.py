"""4채널 AVM 합성 파이프라인 (BEV.md 6절).

각 카메라의 R, t 가 같은 차량 좌표계로 캘리브레이션되어 있으므로, 각자 자기 LUT 로
워핑하기만 하면 결과가 이미 같은 BEV 캔버스에 정렬되어 나온다. 특징점 스티칭이
필요 없고, 남은 일은 밝기 보정과 이음선 블렌딩뿐이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .blend import build_weights, seam_overlay
from .config import RigConfig
from .lut import GroundLUT, build_rig_luts, load_luts, save_luts
from .photometric import apply_gain, solve_gains


class AVMStitcher:
    """LUT + 블렌딩 가중치를 미리 만들어두고 매 프레임 합성만 수행한다."""

    def __init__(self, rig: RigConfig, luts: dict[str, GroundLUT] | None = None, gain_smooth: float = 0.85):
        self.rig = rig
        self.luts = luts if luts is not None else build_rig_luts(rig.cameras, rig.bev)
        self.weights = build_weights(self.luts, rig.cameras, rig.bev, rig.blend)
        self.masks = {name: lut.mask for name, lut in self.luts.items()}
        self.gain_smooth = float(gain_smooth)
        self._gains: dict[str, np.ndarray] = {}
        self.vehicle_mask = self._build_vehicle_mask()

    # ------------------------------------------------------------ 준비

    def _build_vehicle_mask(self) -> np.ndarray:
        """차량 본체가 차지하는 BEV 영역 (어느 카메라도 볼 수 없는 사각지대)."""
        v, bev = self.rig.vehicle, self.rig.bev
        xs, ys = bev.grid()
        return (
            (np.abs(xs - v.x_offset) <= v.length / 2.0) & (np.abs(ys) <= v.width / 2.0)
        )

    @property
    def coverage(self) -> np.ndarray:
        """어느 카메라든 한 대 이상 커버하는 BEV 영역."""
        return np.stack(list(self.masks.values())).any(axis=0)

    def coverage_report(self) -> dict[str, float]:
        report = {name: float(m.mean()) for name, m in self.masks.items()}
        report["_any"] = float(self.coverage.mean())
        report["_blind"] = float((~self.coverage & ~self.vehicle_mask).mean())
        return report

    # ------------------------------------------------------------ 합성

    def warp_all(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: self.luts[name].remap(frames[name]) for name in self.luts}

    def __call__(
        self,
        frames: dict[str, np.ndarray],
        draw_vehicle: bool = True,
        draw_seams: bool = False,
    ) -> np.ndarray:
        """카메라별 원본 프레임 → 합성된 360° BEV 이미지."""
        missing = set(self.luts) - set(frames)
        if missing:
            raise KeyError(f"missing frames for camera(s): {sorted(missing)}")

        warped = self.warp_all(frames)

        if self.rig.blend.photometric:
            gains = solve_gains(warped, self.masks)
            for name, g in gains.items():  # 프레임 간 깜빡임 방지용 지수평활
                prev = self._gains.get(name)
                self._gains[name] = g if prev is None else self.gain_smooth * prev + (1 - self.gain_smooth) * g
            warped = {name: apply_gain(img, self._gains[name]) for name, img in warped.items()}

        sample = next(iter(warped.values()))
        out = np.zeros(sample.shape, dtype=np.float32)
        for name, img in warped.items():
            w = self.weights[name]
            out += (w[..., None] if img.ndim == 3 else w) * img.astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)

        if draw_seams:
            out[seam_overlay(self.weights)] = (0, 255, 255)
        if draw_vehicle:
            out[self.vehicle_mask] = (60, 60, 60)
        return out

    # ------------------------------------------------------------ 캐시

    def save_luts(self, path: str | Path) -> None:
        save_luts(self.luts, path)

    @classmethod
    def from_cached(cls, rig: RigConfig, lut_path: str | Path, **kwargs) -> "AVMStitcher":
        return cls(rig, luts=load_luts(lut_path), **kwargs)
