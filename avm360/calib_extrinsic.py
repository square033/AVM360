"""외부 파라미터 캘리브레이션 — 지면 ArUco 마커로 R, t 구하기 (BEV.md 3.5절).

AVM 은 카메라 4대가 서로 다른 방향의 지면을 보므로, 체커보드 한 장을 모두가 동시에
보게 하기 어렵다. 대신 지면에 여러 개의 ArUco 마커를 "차량 좌표계 기준 알려진 위치"에
깔아두고, 각 카메라가 자기 눈에 보이는 마커만으로 solvePnP 를 푼다.

이렇게 하면 4대 모두가 같은 차량 좌표계를 기준으로 캘리브레이션되므로, BEV.md 6-(B)절
대로 각자 워핑한 결과가 자동으로 같은 캔버스에 정렬된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .model import FISHEYE, CameraModel


@dataclass
class GroundMarker:
    """지면에 놓인 정사각형 마커 하나.

    차량 좌표계에서 중심이 (x, y, 0) 이고 한 변이 size 이며, yaw 만큼 회전해 있다.
    yaw=0 이면 마커 이미지의 위쪽이 차량 전방(+X), 오른쪽이 차량 우측(-Y)을 향한다.
    """

    id: int
    x: float
    y: float
    size: float
    yaw: float = 0.0  # degrees

    def corners(self) -> np.ndarray:
        """ArUco 검출 순서(좌상, 우상, 우하, 좌하)에 맞춘 3D 코너 (4,3)."""
        h = self.size / 2.0
        local = np.array([[h, h], [h, -h], [-h, -h], [-h, h]])  # (X앞, Y좌)
        th = np.deg2rad(self.yaw)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        world = local @ rot.T + np.array([self.x, self.y])
        return np.hstack([world, np.zeros((4, 1))])


def load_marker_map(path: str | Path) -> dict[int, GroundMarker]:
    """마커 배치 YAML 을 읽는다."""
    d = yaml.safe_load(Path(path).read_text())
    default_size = d.get("marker_size", 0.06)
    markers = {}
    for m in d["markers"]:
        markers[int(m["id"])] = GroundMarker(
            id=int(m["id"]),
            x=float(m["x"]),
            y=float(m["y"]),
            size=float(m.get("size", default_size)),
            yaw=float(m.get("yaw", 0.0)),
        )
    return markers


def save_marker_map(markers: dict[int, GroundMarker], path: str | Path, marker_size: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "marker_size": marker_size,
        "dictionary": "DICT_4X4_50",
        "markers": [
            {"id": m.id, "x": round(m.x, 4), "y": round(m.y, 4), "yaw": m.yaw}
            for m in sorted(markers.values(), key=lambda m: m.id)
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def detect_markers(image: np.ndarray, dictionary: str = "DICT_4X4_50"):
    """이미지에서 ArUco 마커를 검출한다. {id: (4,2) 코너} 반환."""
    import cv2

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    try:  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:  # OpenCV < 4.7
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, adict, parameters=params)

    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def solve_extrinsic(
    cam: CameraModel,
    image: np.ndarray,
    markers: dict[int, GroundMarker],
    dictionary: str = "DICT_4X4_50",
    min_markers: int = 2,
) -> tuple[CameraModel, float, list[int]]:
    """한 카메라의 이미지에서 R, t 를 추정한다.

    Returns:
        (R, t 가 채워진 CameraModel 복사본, RMS 재투영오차(px), 사용한 마커 id 목록)
    """
    import cv2

    detected = detect_markers(image, dictionary)
    usable = sorted(set(detected) & set(markers))
    if len(usable) < min_markers:
        raise RuntimeError(
            f"[{cam.name}] 마커가 부족합니다. 검출 {sorted(detected)}, "
            f"배치도에 있는 것 중 사용 가능 {usable} (최소 {min_markers}개 필요)"
        )

    obj = np.vstack([markers[i].corners() for i in usable]).astype(np.float64)
    img = np.vstack([detected[i] for i in usable]).astype(np.float64)

    # 왜곡을 먼저 제거해 정규화 좌표로 만든 뒤, K=I 로 PnP 를 푼다.
    pts = img.reshape(-1, 1, 2)
    if cam.model == FISHEYE:
        norm = cv2.fisheye.undistortPoints(pts, cam.K, cam.D.reshape(4, 1))
    else:
        norm = cv2.undistortPoints(pts, cam.K, cam.D)
    norm = norm.reshape(-1, 1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj.reshape(-1, 1, 3), norm, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError(f"[{cam.name}] solvePnP 실패")
    rvec, tvec = cv2.solvePnPRefineLM(obj.reshape(-1, 1, 3), norm, np.eye(3), None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    out = CameraModel(
        cam.name, cam.image_size, cam.K, cam.D, model=cam.model,
        R=R, t=tvec.ravel(), hfov_deg=cam.hfov_deg,
    )
    uv, _ = out.project(obj)
    rms = float(np.sqrt(np.mean(np.sum((uv - img) ** 2, axis=1))))
    return out, rms, usable


def describe_pose(cam: CameraModel) -> str:
    """캘리브레이션 결과를 사람이 검산할 수 있는 형태로 출력."""
    from .model import R_AXIS_VEH2CAM, R_to_euler

    c = cam.center
    # R = R_AXIS · R_mountᵀ  이므로  R_mount = (R_AXISᵀ · R)ᵀ
    R_mount = (R_AXIS_VEH2CAM.T @ cam.R).T
    roll, pitch, yaw = R_to_euler(R_mount)
    return (
        f"{cam.name:>6}  위치(x,y,z)=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})m  "
        f"roll={roll:+.1f}° pitch={pitch:+.1f}° yaw={yaw:+.1f}°"
    )
