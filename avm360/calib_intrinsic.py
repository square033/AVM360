"""내부 파라미터 캘리브레이션 — 체커보드로 K, D 구하기.

study/BEV.md 에서 정리한 대로, 체커보드 칸의 절대 크기는 K 자체에는 영향을 주지 않고
(칸이 정사각형이라는 사실만 중요하다) t 를 실측 단위로 얻을 때 필요하다. 다만 여기서
구한 K 를 나중에 외부 파라미터 캘리브레이션에 그대로 쓰므로, 실제 mm 값을 넣어두면
전체 파이프라인의 단위가 자연스럽게 미터로 통일된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import FISHEYE, PINHOLE, CameraModel

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def find_corners(
    image_paths: list[Path],
    pattern: tuple[int, int],
    square_size: float,
    refine: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[Path]]:
    """여러 장의 이미지에서 체커보드 내부 코너를 검출한다.

    Args:
        pattern: 내부 코너 개수 (cols, rows). 8x6 격자면 (8,6). 칸 수가 아니라 코너 수다.
        square_size: 한 칸의 실제 한 변 길이 (미터).

    Returns:
        (objpoints, imgpoints, image_size, used_paths)
    """
    import cv2

    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= square_size

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used: list[Path] = []
    image_size: tuple[int, int] | None = None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            raise ValueError(f"{path.name}: 해상도가 다른 이미지가 섞여 있습니다")

        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok:
            continue
        if refine:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        objpoints.append(grid.copy())
        imgpoints.append(corners.reshape(-1, 2).astype(np.float32))
        used.append(path)

    if image_size is None:
        raise RuntimeError("읽을 수 있는 이미지가 없습니다")
    return objpoints, imgpoints, image_size, used


def calibrate(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
    model: str = FISHEYE,
    name: str = "cam",
    hfov_deg: float = 180.0,
) -> tuple[CameraModel, float]:
    """검출된 코너로부터 K, D 를 구한다. (CameraModel, RMS 재투영오차) 반환."""
    import cv2

    if len(objpoints) < 5:
        raise RuntimeError(f"최소 5장 이상 필요합니다 (현재 {len(objpoints)}장)")

    if model == FISHEYE:
        obj = [p.reshape(-1, 1, 3).astype(np.float64) for p in objpoints]
        img = [p.reshape(-1, 1, 2).astype(np.float64) for p in imgpoints]
        K = np.eye(3)
        D = np.zeros(4)
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        )
        # 초기값을 주지 않으면 수렴 실패가 잦다.
        f0 = image_size[0] / 2.0
        K = np.array([[f0, 0, image_size[0] / 2.0], [0, f0, image_size[1] / 2.0], [0, 0, 1]])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, image_size, K, D, flags=flags, criteria=criteria)
        except cv2.error as exc:
            raise RuntimeError(
                "fisheye 캘리브레이션 실패. 흐릿하거나 보드가 화면 가장자리에만 있는 "
                f"이미지를 빼고 다시 시도해 보세요. (원인: {exc})"
            ) from exc
        D = np.asarray(D).ravel()[:4]
    elif model == PINHOLE:
        obj = [p.astype(np.float32) for p in objpoints]
        img = [p.reshape(-1, 1, 2).astype(np.float32) for p in imgpoints]
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img, image_size, None, None)
        D = np.asarray(D).ravel()[:5]
    else:
        raise ValueError(f"unknown model {model!r}")

    cam = CameraModel(name, image_size, np.asarray(K), D, model=model, hfov_deg=hfov_deg)
    return cam, float(rms)


def calibrate_folder(
    folder: str | Path,
    pattern: tuple[int, int],
    square_size: float,
    model: str = FISHEYE,
    name: str = "cam",
    hfov_deg: float = 180.0,
) -> tuple[CameraModel, float, int]:
    """폴더 안의 체커보드 이미지로 한 번에 캘리브레이션한다."""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"{folder} 안에 이미지가 없습니다")
    objp, imgp, size, used = find_corners(paths, pattern, square_size)
    cam, rms = calibrate(objp, imgp, size, model=model, name=name, hfov_deg=hfov_deg)
    return cam, rms, len(used)
