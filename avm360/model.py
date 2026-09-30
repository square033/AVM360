"""카메라 모델 — K, D, [R|t] 와 투영/역투영.

좌표계 정의는 study/BEV.md 3.0절을 그대로 따른다.

  Vehicle(World) : X 전방, Y 좌측, Z 상향. 원점은 차량 중심의 지면(Z=0).
  Camera         : X 우측, Y 하단, Z 전방.
  Image          : 픽셀 (u, v).

투영식은 BEV.md 3.2절 / 4.1절:

  X_c = R · X_w + t                       (world → camera)
  x_un = X_c/Z_c,  y_un = Y_c/Z_c         (정규화 좌표)
  θ = arctan(r_un)                        (피쉬아이)
  r_dn = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹
  u = f_x·x_dn + s·y_dn + c_x,  v = f_y·y_dn + c_y
"""

from __future__ import annotations

import numpy as np

# 차량 좌표계 → 카메라 좌표계 축 변환 (BEV.md 3.0절: X_c=-Y_w, Y_c=-Z_w, Z_c=X_w).
# 마운트 회전이 0일 때, 즉 카메라가 차량 전방(+X)을 수평으로 바라볼 때의 기본 회전.
R_AXIS_VEH2CAM = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

FISHEYE = "fisheye"
PINHOLE = "pinhole"


def euler_to_R(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """오일러 각 → 회전 행렬 (BEV.md 3.6절, R = Rz·Ry·Rx).

    차량 좌표계에서 카메라가 향하는 자세를 나타낸다.
    yaw>0 은 좌회전(+Z 축), pitch>0 은 아래를 향함, roll>0 은 시계방향 기울임.
    """
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """euler_to_R 의 역변환. (roll, pitch, yaw) 반환."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) < 1e-8:  # 짐벌락
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    out = (roll, pitch, yaw)
    return tuple(np.rad2deg(out)) if degrees else out


class CameraModel:
    """단일 카메라의 내부/외부 파라미터와 투영 연산.

    Attributes:
        K: 3x3 내재 행렬.
        D: 왜곡 계수. fisheye 는 [k1,k2,k3,k4], pinhole 은 [k1,k2,p1,p2,k3].
        R: 3x3 회전 (world → camera).
        t: 3벡터 평행이동 (world → camera).
        hfov_deg: 유효 화각. θ > hfov/2 인 광선은 투영 대상에서 제외 (BEV.md 4.1절).
    """

    def __init__(
        self,
        name: str,
        image_size: tuple[int, int],
        K: np.ndarray,
        D: np.ndarray,
        model: str = FISHEYE,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
        hfov_deg: float = 180.0,
    ):
        if model not in (FISHEYE, PINHOLE):
            raise ValueError(f"model must be {FISHEYE!r} or {PINHOLE!r}, got {model!r}")
        self.name = name
        self.image_size = (int(image_size[0]), int(image_size[1]))  # (width, height)
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.D = np.asarray(D, dtype=float).ravel()
        self.model = model
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=float).ravel()
        self.hfov_deg = float(hfov_deg)

    # ------------------------------------------------------------------ 생성

    @classmethod
    def from_mount(
        cls,
        name: str,
        image_size: tuple[int, int],
        K: np.ndarray,
        D: np.ndarray,
        position: tuple[float, float, float],
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        model: str = FISHEYE,
        hfov_deg: float = 180.0,
    ) -> "CameraModel":
        """장착 위치/자세(차량 좌표계 기준)로부터 R, t 를 구성한다.

        position 은 차량 좌표계에서의 카메라 중심 C.
        BEV.md 3.7절과 같은 관계이며, 정확히는 t = -R·C 이다.
        """
        R_mount = euler_to_R(roll, pitch, yaw)
        R = R_AXIS_VEH2CAM @ R_mount.T
        t = -R @ np.asarray(position, dtype=float)
        return cls(name, image_size, K, D, model=model, R=R, t=t, hfov_deg=hfov_deg)

    # ------------------------------------------------------------- 직렬화

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "image_size": list(self.image_size),
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "hfov_deg": self.hfov_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(
            name=d["name"],
            image_size=tuple(d["image_size"]),
            K=np.array(d["K"], dtype=float),
            D=np.array(d["D"], dtype=float),
            model=d.get("model", FISHEYE),
            R=np.array(d["R"], dtype=float) if "R" in d else None,
            t=np.array(d["t"], dtype=float) if "t" in d else None,
            hfov_deg=d.get("hfov_deg", 180.0),
        )

    # ------------------------------------------------------------- 기하 정보

    @property
    def center(self) -> np.ndarray:
        """차량 좌표계에서의 카메라 중심 C = -Rᵀ·t."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """차량 좌표계에서의 광축 방향 (카메라 +Z 를 월드로)."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    # --------------------------------------------------------------- 투영

    def _distort(self, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """정규화 좌표 → 왜곡된 정규화 좌표. theta(피쉬아이 입사각)도 함께 반환."""
        r = np.hypot(xn, yn)
        if self.model == FISHEYE:
            theta = np.arctan(r)
            k1, k2, k3, k4 = (list(self.D) + [0.0] * 4)[:4]
            th2 = theta * theta
            # r_dn = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹  (BEV.md 4.1절)
            rd = theta * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
            # r→0 에서 rd/r → 1 (θ≈r) 이므로 특이점 없음.
            scale = np.where(r > 1e-12, rd / np.where(r > 1e-12, r, 1.0), 1.0)
            return xn * scale, yn * scale, theta
        k1, k2, p1, p2, k3 = (list(self.D) + [0.0] * 5)[:5]
        r2 = r * r
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return xd, yd, np.arctan(r)

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """월드 3D 점 → 이미지 픽셀.

        Args:
            pts_world: (N,3) 차량 좌표계 점.

        Returns:
            uv: (N,2) 픽셀 좌표.
            valid: (N,) bool. 카메라 앞(Z_c>0) & 화각 이내 & 이미지 범위 이내.
        """
        pts = np.asarray(pts_world, dtype=float).reshape(-1, 3)
        pc = pts @ self.R.T + self.t  # world → camera
        z = pc[:, 2]
        in_front = z > 1e-9
        zs = np.where(in_front, z, 1.0)
        xn, yn = pc[:, 0] / zs, pc[:, 1] / zs

        xd, yd, theta = self._distort(xn, yn) 
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]

        w, h = self.image_size
        valid = (
            in_front
            & (theta < np.deg2rad(self.hfov_deg) / 2.0)
            & (u >= 0)
            & (u <= w - 1)
            & (v >= 0)
            & (v <= h - 1)
            & np.isfinite(u)
            & np.isfinite(v)
        )
        return np.stack([u, v], axis=1), valid

    # ------------------------------------------------------------- 역투영

    def _undistort(self, xd: np.ndarray, yd: np.ndarray, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """왜곡된 정규화 좌표 → 왜곡 없는 정규화 좌표 (반복법)."""
        if self.model == FISHEYE:
            rd = np.hypot(xd, yd)
            k1, k2, k3, k4 = (list(self.D) + [0.0] * 4)[:4]
            theta = rd.copy()  # 초기값
            for _ in range(iters):  # 뉴턴법으로 r_dn(θ) = rd 를 푼다
                th2 = theta * theta
                f = theta * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4)))) - rd
                df = 1.0 + th2 * (3 * k1 + th2 * (5 * k2 + th2 * (7 * k3 + th2 * 9 * k4)))
                theta = theta - f / np.maximum(df, 1e-9)
            theta = np.clip(theta, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
            r = np.tan(theta)
            scale = np.where(rd > 1e-12, r / np.where(rd > 1e-12, rd, 1.0), 1.0)
            return xd * scale, yd * scale
        xn, yn = xd.copy(), yd.copy()
        for _ in range(iters):  # 고정점 반복
            xdd, ydd, _ = self._distort(xn, yn)
            xn = xn + (xd - xdd)
            yn = yn + (yd - ydd)
        return xn, yn

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """이미지 픽셀 → 차량 좌표계에서의 시선 방향(단위벡터, (N,3)).

        카메라 중심은 self.center 이므로, 광선은 center + s·dir (s>0) 이다.
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        fx, s, cx = self.K[0, 0], self.K[0, 1], self.K[0, 2]
        fy, cy = self.K[1, 1], self.K[1, 2]
        yd = (uv[:, 1] - cy) / fy
        xd = (uv[:, 0] - cx - s * yd) / fx
        xn, yn = self._undistort(xd, yd)
        ray_c = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        ray_w = ray_c @ self.R  # Rᵀ·ray_c 를 행벡터 형태로
        return ray_w / np.linalg.norm(ray_w, axis=1, keepdims=True)

    def __repr__(self) -> str:
        c = self.center
        return (
            f"CameraModel(name={self.name!r}, model={self.model!r}, "
            f"size={self.image_size}, center=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}))"
        )
