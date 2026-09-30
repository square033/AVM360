"""BEV 캔버스 정의와 리그(카메라 4대) 설정 로드/저장."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .model import CameraModel


@dataclass
class BevConfig:
    """BEV 출력 캔버스 정의 (BEV.md 7.1절).

    월드(차량) 좌표 범위와 해상도로부터 출력 이미지 크기가 결정된다.
    화면 배치 관례: 위쪽이 차량 전방(+X), 왼쪽이 차량 좌측(+Y).

        world_x = x_max - row * resolution
        world_y = y_max - col * resolution
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float  # meter per pixel

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """각 BEV 픽셀이 나타내는 월드 좌표 (xs, ys). 각각 (H,W)."""
        rows = np.arange(self.height)
        cols = np.arange(self.width)
        xs = self.x_max - (rows + 0.5) * self.resolution
        ys = self.y_max - (cols + 0.5) * self.resolution
        return np.meshgrid(xs, ys, indexing="ij")

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """월드 좌표 → BEV 픽셀 (col, row). 시각화용."""
        col = (self.y_max - y) / self.resolution - 0.5
        row = (self.x_max - x) / self.resolution - 0.5
        return col, row

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "resolution": self.resolution,
        }


@dataclass
class BlendConfig:
    """이음선 블렌딩 설정 (BEV.md 6-(B)절)."""

    mode: str = "feather"  # "feather" | "wedge"
    feather_m: float = 0.06  # 유효영역 경계에서 페더링할 폭 (미터)
    wedge_band_deg: float = 20.0  # wedge 모드에서 섹터 경계 블렌딩 폭
    photometric: bool = True  # 겹침 영역 기반 밝기/색 보정 사용 여부


@dataclass
class VehicleConfig:
    """차량(또는 리그 본체) 외형. BEV 중앙에 가림막을 그리는 데 쓴다."""

    length: float = 0.30
    width: float = 0.20
    # 차량 중심 기준 전방 오프셋. 0이면 직사각형이 원점 대칭.
    x_offset: float = 0.0


@dataclass
class RigConfig:
    """카메라 여러 대 + BEV 캔버스 + 블렌딩 설정 묶음."""

    cameras: list[CameraModel]
    bev: BevConfig
    blend: BlendConfig = field(default_factory=BlendConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    def camera(self, name: str) -> CameraModel:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"camera {name!r} not in rig ({[c.name for c in self.cameras]})")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cameras]

    # ------------------------------------------------------------ 직렬화

    def to_dict(self) -> dict:
        return {
            "bev": self.bev.to_dict(),
            "blend": vars(self.blend),
            "vehicle": vars(self.vehicle),
            "cameras": [c.to_dict() for c in self.cameras],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))

    @classmethod
    def load(cls, path: str | Path) -> "RigConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "RigConfig":
        cams = [_camera_from_entry(e) for e in d["cameras"]]
        return cls(
            cameras=cams,
            bev=BevConfig(**d["bev"]),
            blend=BlendConfig(**d.get("blend", {})),
            vehicle=VehicleConfig(**d.get("vehicle", {})),
        )


def _camera_from_entry(e: dict) -> CameraModel:
    """카메라 항목을 읽는다.

    R, t 를 직접 주거나(캘리브레이션 결과), mount(위치+오일러각)로 줄 수 있다.
    mount 방식은 아직 캘리브레이션 전인 설계값/시뮬레이션용이다.
    """
    if "R" in e and "t" in e:
        return CameraModel.from_dict(e)
    m = e["mount"]
    return CameraModel.from_mount(
        name=e["name"],
        image_size=tuple(e["image_size"]),
        K=np.array(e["K"], dtype=float),
        D=np.array(e["D"], dtype=float),
        position=tuple(m["position"]),
        roll=m.get("roll", 0.0),
        pitch=m.get("pitch", 0.0),
        yaw=m.get("yaw", 0.0),
        model=e.get("model", "fisheye"),
        hfov_deg=e.get("hfov_deg", 180.0),
    )


def make_K(width: int, height: int, hfov_deg: float, model: str = "fisheye") -> np.ndarray:
    """화각과 해상도로부터 대략적인 K 를 만든다 (시뮬레이션/초기값용).

    실제 카메라에서는 반드시 체커보드 캘리브레이션으로 얻은 K 를 써야 한다.
    """
    half = np.deg2rad(hfov_deg) / 2.0
    if model == "fisheye":  # 등거리 투영: r = f·θ
        f = (width / 2.0) / half
    else:  # 핀홀: r = f·tanθ
        f = (width / 2.0) / np.tan(half)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
