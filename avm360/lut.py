"""BEV LUT 생성 — backward mapping (BEV.md 7.2 / 7.3절).

BEV 캔버스의 각 픽셀이 나타내는 지면 좌표(Z=0)를 원본 이미지의 어느 픽셀에서
가져와야 하는지 미리 계산해 둔다. 카메라 파라미터가 고정인 한 한 번만 계산하면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BevConfig
from .model import CameraModel


class GroundLUT:
    """한 카메라에 대한 지면(Z=0) 역방향 매핑 테이블."""

    def __init__(self, map_x: np.ndarray, map_y: np.ndarray, mask: np.ndarray, name: str):
        self.map_x = map_x.astype(np.float32)
        self.map_y = map_y.astype(np.float32)
        self.mask = mask.astype(bool)
        self.name = name

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def coverage(self) -> float:
        """이 카메라가 커버하는 BEV 픽셀 비율."""
        return float(self.mask.mean())

    def remap(self, image: np.ndarray, interpolation: int | None = None) -> np.ndarray:
        """원본 이미지를 BEV 캔버스로 워핑한다."""
        import cv2

        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        out = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[~self.mask] = 0
        return out


def build_ground_lut(cam: CameraModel, bev: BevConfig) -> GroundLUT:
    """카메라 하나에 대한 지면 LUT 생성.

    BEV.md 7.2절 의사코드를 그대로 벡터화한 것이다:
        P_w = (world_x, world_y, 0, 1)  →  P_c = [R|t]·P_w  →  (u,v) = project(P_c, K)
    """
    xs, ys = bev.grid()
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    uv, valid = cam.project(pts)

    h, w = bev.shape
    map_x = uv[:, 0].reshape(h, w)
    map_y = uv[:, 1].reshape(h, w)
    mask = valid.reshape(h, w)
    # 무효 픽셀은 remap 이 borderValue 로 채우도록 범위 밖 좌표를 넣는다.
    map_x = np.where(mask, map_x, -1.0)
    map_y = np.where(mask, map_y, -1.0)
    return GroundLUT(map_x, map_y, mask, cam.name)


def build_rig_luts(cams: list[CameraModel], bev: BevConfig) -> dict[str, GroundLUT]:
    return {cam.name: build_ground_lut(cam, bev) for cam in cams}


def save_luts(luts: dict[str, GroundLUT], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, lut in luts.items():
        payload[f"{name}__map_x"] = lut.map_x
        payload[f"{name}__map_y"] = lut.map_y
        payload[f"{name}__mask"] = lut.mask
    np.savez_compressed(path, **payload)


def load_luts(path: str | Path) -> dict[str, GroundLUT]:
    data = np.load(path)
    names = sorted({k.split("__")[0] for k in data.files})
    return {
        n: GroundLUT(data[f"{n}__map_x"], data[f"{n}__map_y"], data[f"{n}__mask"], n)
        for n in names
    }
