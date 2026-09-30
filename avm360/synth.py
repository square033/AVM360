"""가상 씬 렌더러 — 카메라 없이 전체 파이프라인을 검증한다.

지면 텍스처와 수직 박스로 이루어진 간단한 씬을 각 카메라 모델로 레이캐스팅해서
"카메라가 찍었을 법한" 왜곡된 이미지를 만든다. 이 이미지를 AVM 파이프라인에 넣으면
합성 결과가 원본 지면 텍스처와 일치해야 하므로, 하드웨어 없이도 K/R/t/LUT/블렌딩이
맞는지 확인할 수 있다.

수직 박스는 BEV.md 2절이 지적한 IPM 의 한계(지면에서 떨어진 물체가 늘어나 보임)를
눈으로 확인하기 위한 것이다.
"""

from __future__ import annotations

import numpy as np

from .config import BevConfig
from .model import CameraModel

SKY_COLOR = (150, 150, 150)


class Box:
    """지면에 세워진 축 정렬 직육면체 (IPM 왜곡 관찰용)."""

    def __init__(self, x: tuple[float, float], y: tuple[float, float], height: float, color):
        self.lo = np.array([min(x), min(y), 0.0])
        self.hi = np.array([max(x), max(y), float(height)])
        self.color = np.array(color, dtype=np.float32)


class GroundScene:
    """지면 텍스처 + 수직 박스로 구성된 씬."""

    def __init__(self, texture: np.ndarray, bounds: BevConfig, boxes: list[Box] | None = None):
        self.texture = texture
        self.bounds = bounds
        self.boxes = boxes or []

    # -------------------------------------------------------------- 샘플링

    def sample_ground(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """지면 좌표 → 텍스처 색. 텍스처 범위를 벗어나면 valid=False."""
        b = self.bounds
        col = (b.y_max - y) / b.resolution - 0.5
        row = (b.x_max - x) / b.resolution - 0.5
        h, w = self.texture.shape[:2]
        ci = np.round(col).astype(int)
        ri = np.round(row).astype(int)
        valid = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        colors = np.zeros((x.size, self.texture.shape[2]), dtype=np.float32)
        colors[valid] = self.texture[ri[valid], ci[valid]].astype(np.float32)
        return colors, valid

    # -------------------------------------------------------------- 렌더링

    def render(self, cam: CameraModel) -> np.ndarray:
        """카메라 모델로 씬을 레이캐스팅해 이미지를 만든다."""
        w, h = cam.image_size
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(float)

        origin = cam.center
        dirs = cam.unproject(uv)  # (N,3) 단위벡터

        img = np.tile(np.array(SKY_COLOR, dtype=np.float32), (uv.shape[0], 1))
        depth = np.full(uv.shape[0], np.inf)

        # 지면(Z=0) 교차
        dz = dirs[:, 2]
        going_down = dz < -1e-9
        t_g = np.where(going_down, -origin[2] / np.where(going_down, dz, -1.0), np.inf)
        hit_g = going_down & (t_g > 0)
        if hit_g.any():
            pts = origin + dirs[hit_g] * t_g[hit_g, None]
            colors, ok = self.sample_ground(pts[:, 0], pts[:, 1])
            idx = np.flatnonzero(hit_g)[ok]
            img[idx] = colors[ok]
            depth[idx] = t_g[idx]

        # 박스 교차 (더 가까우면 덮어쓴다)
        for box in self.boxes:
            t_hit, face = _ray_box(origin, dirs, box.lo, box.hi)
            closer = np.isfinite(t_hit) & (t_hit < depth)
            if not closer.any():
                continue
            shade = np.array([1.0, 0.82, 0.62])[face[closer]]  # 면별 명암
            img[closer] = np.clip(box.color[None, :] * shade[:, None], 0, 255)
            depth[closer] = t_hit[closer]

        # 화각 밖(투영 불가능한 픽셀)은 검게 비네팅 처리
        _, valid = cam.project(origin + dirs * 1.0)
        img[~valid] = 0

        return np.clip(img.reshape(h, w, -1), 0, 255).astype(np.uint8)


def _ray_box(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """슬랩 방식 ray-AABB 교차. (t, 히트한 축) 반환, 미충돌은 inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo[None, :] - origin[None, :]) * inv
        t1 = (hi[None, :] - origin[None, :]) * inv
    tmin_ax = np.minimum(t0, t1)
    tmax_ax = np.maximum(t0, t1)
    tmin_ax = np.nan_to_num(tmin_ax, nan=-np.inf)
    tmax_ax = np.nan_to_num(tmax_ax, nan=np.inf)

    axis = np.argmax(tmin_ax, axis=1)
    tmin = np.max(tmin_ax, axis=1)
    tmax = np.min(tmax_ax, axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0)
    t = np.where(hit & (tmin > 0), tmin, np.inf)
    return t, axis


# --------------------------------------------------------------- 텍스처 생성


def make_ground_texture(
    bounds: BevConfig,
    tile: float = 0.10,
    with_aruco: bool = True,
    marker_positions: list[tuple[int, float, float]] | None = None,
    marker_size: float = 0.06,
) -> np.ndarray:
    """체커 타일 + 주차선 + (선택) ArUco 마커가 그려진 지면 텍스처.

    ArUco 마커를 함께 그려두면 시뮬레이션 이미지로 외부 파라미터 캘리브레이션까지
    검증할 수 있다 (scripts/03_calib_extrinsic.py 와 같은 마커 배치를 쓴다).
    """
    import cv2

    h, w = bounds.shape
    xs, ys = bounds.grid()

    # 체커 타일 바닥
    checker = (np.floor(xs / tile).astype(int) + np.floor(ys / tile).astype(int)) % 2
    img = np.where(checker[..., None] == 0, np.array([95, 95, 100]), np.array([130, 130, 138]))
    img = img.astype(np.uint8)

    # 타일 경계선을 살짝 어둡게 해서 정렬 오차가 눈에 띄게 한다
    edge = (np.abs((xs / tile) - np.round(xs / tile)) < 0.03) | (
        np.abs((ys / tile) - np.round(ys / tile)) < 0.03
    )
    img[edge] = (70, 70, 75)

    # 차량 좌우 주차선 (흰색)
    for y0 in (bounds.y_min * 0.62, bounds.y_max * 0.62):
        line = np.abs(ys - y0) < 0.012
        img[line] = (245, 245, 245)

    # 전/후방 기준선 (색으로 방향 구분)
    img[np.abs(xs - bounds.x_max * 0.7) < 0.012] = (60, 90, 235)  # 전방: 빨강 계열(BGR)
    img[np.abs(xs - bounds.x_min * 0.7) < 0.012] = (235, 160, 60)  # 후방: 파랑 계열(BGR)

    if with_aruco:
        positions = marker_positions if marker_positions is not None else default_marker_layout(bounds)
        _draw_aruco(img, bounds, positions, marker_size)

    return img


def default_marker_layout(bounds: BevConfig) -> list[tuple[int, float, float]]:
    """전/후/좌/우 각 방향에 마커를 배치한 기본 레이아웃. (id, x, y)."""
    x_f, x_r = bounds.x_max * 0.72, bounds.x_min * 0.72
    y_l, y_r = bounds.y_max * 0.72, bounds.y_min * 0.72
    return [
        (0, x_f, y_l * 0.45),
        (1, x_f, y_r * 0.45),
        (2, x_r, y_l * 0.45),
        (3, x_r, y_r * 0.45),
        (4, x_f * 0.45, y_l),
        (5, x_r * 0.45, y_l),
        (6, x_f * 0.45, y_r),
        (7, x_r * 0.45, y_r),
    ]


def _draw_aruco(img, bounds: BevConfig, positions, marker_size: float) -> None:
    import cv2

    try:
        adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:  # aruco 미포함 빌드
        return

    side_px = max(int(round(marker_size / bounds.resolution)), 8)
    for mid, x, y in positions:
        try:
            patch = cv2.aruco.generateImageMarker(adict, int(mid), side_px)
        except AttributeError:
            patch = cv2.aruco.drawMarker(adict, int(mid), side_px)
        col, row = bounds.world_to_pixel(x, y)
        c0, r0 = int(round(col - side_px / 2)), int(round(row - side_px / 2))
        if c0 < 0 or r0 < 0 or c0 + side_px > img.shape[1] or r0 + side_px > img.shape[0]:
            continue
        # 마커 주변 흰 여백(quiet zone)이 있어야 검출이 안정적이다
        pad = max(side_px // 6, 2)
        img[max(r0 - pad, 0) : r0 + side_px + pad, max(c0 - pad, 0) : c0 + side_px + pad] = 255
        img[r0 : r0 + side_px, c0 : c0 + side_px] = patch[..., None]


def default_boxes() -> list[Box]:
    """IPM 왜곡 관찰용 기본 장애물 배치."""
    return [
        Box(x=(0.34, 0.42), y=(0.10, 0.18), height=0.12, color=(70, 70, 220)),
        Box(x=(-0.42, -0.34), y=(-0.20, -0.12), height=0.09, color=(80, 200, 90)),
        Box(x=(-0.05, 0.05), y=(0.36, 0.44), height=0.15, color=(230, 180, 60)),
    ]
