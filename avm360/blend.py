"""이음선 블렌딩 가중치 (BEV.md 6-(B)절).

각 카메라가 커버하는 BEV 영역은 서로 겹친다. 겹치는 구간에서 어느 카메라 픽셀을
얼마나 쓸지 결정하는 가중치 맵을 만든다. 모든 픽셀에서 가중치 합은 1이 된다.

  feather : 유효영역 경계로부터의 거리(distance transform)를 가중치로 사용.
            카메라 배치에 무관하게 동작하고, 경계에서 자연스럽게 0으로 수렴한다.
  wedge   : 차량 중심 기준 각도 섹터로 담당 구역을 나누고 경계만 부드럽게 섞는다.
            상용 AVM 의 고정 이음선 방식에 가깝고 겹침 구간 고스팅이 적다.
"""

from __future__ import annotations

import numpy as np

from .config import BevConfig, BlendConfig
from .lut import GroundLUT
from .model import CameraModel


def _distance_weight(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """유효영역 내부에서 경계까지의 거리를 [0,1] 로 정규화한 가중치."""
    import cv2

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(dist / max(feather_px, 1e-6), 0.0, 1.0).astype(np.float32)


def _wedge_weight(
    cam: CameraModel, bev: BevConfig, cam_yaws: np.ndarray, band_deg: float
) -> np.ndarray:
    """차량 중심에서 본 각도를 기준으로 이 카메라가 담당하는 섹터 가중치.

    각 카메라는 자기 방위각을 중심으로, 이웃 카메라와의 중간선까지를 담당한다.
    중간선 부근 band_deg 폭에서만 선형으로 섞인다.
    """
    xs, ys = bev.grid()
    phi = np.arctan2(ys, xs)  # 차량 중심 기준 방위각 (전방=0, 좌측=+90°)

    yaw = np.arctan2(cam.forward[1], cam.forward[0])
    # 이웃 카메라들과의 각도 차 중 가장 가까운 경계까지의 거리를 구한다.
    others = cam_yaws[~np.isclose(cam_yaws, yaw)]
    if others.size == 0:
        return np.ones(bev.shape, dtype=np.float32)

    d_self = np.abs(np.arctan2(np.sin(phi - yaw), np.cos(phi - yaw)))
    d_other = np.min(
        np.stack([np.abs(np.arctan2(np.sin(phi - o), np.cos(phi - o))) for o in others]), axis=0
    )
    # 자기 방위각에 더 가까우면 +, 경계에서 0, 남의 구역이면 -
    margin = d_other - d_self
    band = np.deg2rad(max(band_deg, 1e-3))
    return np.clip(0.5 + margin / band, 0.0, 1.0).astype(np.float32)


def build_weights(
    luts: dict[str, GroundLUT],
    cams: list[CameraModel],
    bev: BevConfig,
    cfg: BlendConfig,
) -> dict[str, np.ndarray]:
    """카메라별 블렌딩 가중치 맵 (합=1). 유효영역 밖은 0."""
    feather_px = cfg.feather_m / bev.resolution
    cam_yaws = np.array([np.arctan2(c.forward[1], c.forward[0]) for c in cams])

    weights: dict[str, np.ndarray] = {}
    for cam in cams:
        lut = luts[cam.name]
        w = _distance_weight(lut.mask, feather_px)
        if cfg.mode == "wedge":
            w = w * _wedge_weight(cam, bev, cam_yaws, cfg.wedge_band_deg)
        elif cfg.mode != "feather":
            raise ValueError(f"unknown blend mode {cfg.mode!r} (use 'feather' or 'wedge')")
        weights[cam.name] = w

    stack = np.stack(list(weights.values()))
    total = stack.sum(axis=0)
    # wedge 모드에서 섹터 경계가 유효영역과 어긋나 합이 0이 되는 픽셀이 생길 수 있어,
    # 그런 곳은 거리 가중치만으로 되돌린다.
    if cfg.mode == "wedge":
        empty = (total <= 1e-6) & np.stack([l.mask for l in luts.values()]).any(axis=0)
        if empty.any():
            for cam in cams:
                fallback = _distance_weight(luts[cam.name].mask, feather_px)
                weights[cam.name] = np.where(empty, fallback, weights[cam.name])
            stack = np.stack(list(weights.values()))
            total = stack.sum(axis=0)

    safe = np.maximum(total, 1e-6)
    return {name: (w / safe).astype(np.float32) for name, w in weights.items()}


def seam_overlay(weights: dict[str, np.ndarray], threshold: float = 0.05) -> np.ndarray:
    """이음선(2대 이상이 기여하는 구간) 시각화용 마스크."""
    stack = np.stack(list(weights.values()))
    contributors = (stack > threshold).sum(axis=0)
    return contributors >= 2
