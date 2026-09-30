"""avm360 — 카메라 4대로 만드는 360° 서라운드 뷰 (AVM/SVM).

study/BEV.md 에 정리한 IPM/호모그래피 이론을 그대로 코드로 옮긴 패키지다.

기본 사용법:

    from avm360 import RigConfig, AVMStitcher

    rig = RigConfig.load("configs/rig_desk.yaml")
    stitcher = AVMStitcher(rig)
    bev = stitcher({"front": img_f, "rear": img_r, "left": img_l, "right": img_g})
"""

from .blend import build_weights
from .config import BevConfig, BlendConfig, RigConfig, VehicleConfig, make_K
from .lut import GroundLUT, build_ground_lut, build_rig_luts, load_luts, save_luts
from .model import FISHEYE, PINHOLE, CameraModel, euler_to_R
from .stitcher import AVMStitcher

__all__ = [
    "AVMStitcher",
    "BevConfig",
    "BlendConfig",
    "CameraModel",
    "FISHEYE",
    "PINHOLE",
    "GroundLUT",
    "RigConfig",
    "VehicleConfig",
    "build_ground_lut",
    "build_rig_luts",
    "build_weights",
    "euler_to_R",
    "load_luts",
    "make_K",
    "save_luts",
]

__version__ = "0.1.0"
