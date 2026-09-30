# avm360 — 카메라 4대로 만드는 360° 서라운드 뷰

[`study/BEV.md](https://github.com/square033/study/blob/main/BEV.md) 에 정리한 IPM·호모그래피 이론을 그대로 코드로 옮긴 패키지다.
전/후/좌/우 카메라 4대를 캘리브레이션해서 하나의 탑다운(BEV) 화면으로 합성한다.

**카메라가 없어도 지금 바로 전체를 돌려볼 수 있다** — 가상 씬 렌더러가 들어 있어서
캘리브레이션부터 합성까지 왕복 검증이 가능하다.

<!-- 시뮬레이션 결과: out/sim/bev.png -->

---

## 빠른 시작 (하드웨어 불필요)

```bash
cd study/avm360
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/08_selftest.py      # 전체 왕복 검증
.venv/bin/python scripts/06_simulate.py --seams   # 결과 이미지 생성
```

`08_selftest.py` 는 설계값으로 가상 카메라 영상을 렌더링한 뒤, **그 영상만 보고** 외부
파라미터를 다시 추정해서 원래 값과 비교한다. 실행하면 이런 결과가 나온다.

```
1) 정답 지면 생성 및 가상 촬영
2) 렌더링된 이미지만으로 외부 파라미터 재추정
    front: 위치오차   1.9mm  각도오차 0.13°  RMS 1.22px  마커 4개  OK
     left: 위치오차   1.7mm  각도오차 0.31°  RMS 1.34px  마커 4개  OK
     rear: 위치오차   1.7mm  각도오차 0.05°  RMS 1.40px  마커 4개  OK
    right: 위치오차   1.6mm  각도오차 0.17°  RMS 1.40px  마커 4개  OK
3) 재추정된 파라미터로 BEV 재구성
   커버리지 95.0%, 사각지대 1.4%
   정답 대비 평균 절대오차 5.43 / 255 (허용 12.0)

=== PASS === 캘리브레이션 → LUT → 블렌딩 전 과정 정상
```

`out/sim/` 에 생성되는 이미지:

| 파일 | 내용 |
| ---- | ---- |
| `cam_front.png` 등 | 각 카메라가 본 어안 왜곡 영상 |
| `bev.png` | 합성된 360° BEV |
| `coverage.png` | 카메라별 담당 영역 (색상 구분) |
| `ground_truth.png` | 정답 지면 |
| `error.png` | 정답 대비 오차 |

`bev.png` 를 보면 지면 타일과 주차선이 4대에 걸쳐 어긋남 없이 이어지고, 세워둔 박스만
바깥쪽으로 길게 늘어나 있다. 이 늘어남이 BEV.md 2절에서 말한 **IPM 의 평면 가정 위반**이며,
버그가 아니라 원리적 한계다.

---

## 실제 카메라로 하기

카메라 선정과 배치는 [`docs/HARDWARE.md`](docs/HARDWARE.md), 캘리브레이션 절차는
[`docs/CALIBRATION.md`](docs/CALIBRATION.md) 에 자세히 정리되어 있다.

```bash
# 0. 장치 확인 + 4대 동시 대역폭 테스트
.venv/bin/python scripts/00_probe_cameras.py --test 0 2 4 6

# 1~2. 내부 파라미터 (카메라 한 대씩 체커보드 20~30장)
.venv/bin/python scripts/01_capture.py --mode intrinsic --device 0 --name front
.venv/bin/python scripts/02_calib_intrinsic.py --all --square 0.025

# 3. 지면 마커 인쇄 → 배치 → 외부 파라미터 (4대 동시 1장)
.venv/bin/python scripts/07_make_markers.py --size 0.06
.venv/bin/python scripts/01_capture.py --mode extrinsic --devices 0 2 4 6
.venv/bin/python scripts/03_calib_extrinsic.py

# 4~5. LUT 생성 후 실시간 실행
.venv/bin/python scripts/04_build_lut.py
.venv/bin/python scripts/05_run_avm.py --devices 0 2 4 6
```

### 카메라 요약 (총 20만원 내외)

모델명보다 **스펙이 먼저**다. 아래 조건 중 하나라도 어긋나면 AVM 이 제대로 안 나온다.

| 필수 스펙 | 요구값 | 이유 |
| --------- | ------ | ---- |
| 화각(대각) | **150° 이상** | 4대로 360°를 덮고 코너에서 겹치려면 필요. 일반 웹캠(60~78°)은 불가 |
| 초점 | **고정초점** | 오토포커스가 돌면 $f$ 가 변해 캘리브레이션한 $K$ 가 무효가 된다 |
| 압축 | **MJPEG** | 무압축 YUYV 는 720p30 한 대가 ~440Mbps. 4대 동시 구동 불가 |
| 제어 | UVC 수동 노출/WB | 오토가 돌면 카메라마다 밝기가 달라 이음선이 튄다 |

**추천**: ELP 어안 USB 모듈 720p 170° 계열 × 4 (대당 3.5만~5.5만원).
**피할 것**: 로지텍 C270/C310(화각 좁음), C920/C922(화각 좁음 + 오토포커스).

---

## 이론 ↔ 코드 대응

BEV.md 의 어느 절이 어느 코드가 되었는지:

| BEV.md | 내용 | 코드 |
| ------ | ---- | ---- |
| 3.0 | 좌표계 정의 (X 전방, Y 좌측, Z 상향) | `model.py: R_AXIS_VEH2CAM` |
| 3.2 | $P = K[R\|t]$ 투영 행렬 | `model.py: CameraModel.project()` |
| 3.5 | 외부 파라미터 $R, t$ 추정 | `calib_extrinsic.py: solve_extrinsic()` |
| 3.5.2 | 상대 변환 $R_{rel}, t_{rel}$ | 마커 기반이라 불필요 — 모두 같은 차량 좌표계로 직접 풀린다 |
| 3.6 | 오일러 각 → $R$ | `model.py: euler_to_R()` |
| 4.1 | 어안 왜곡 모델 | `model.py: CameraModel._distort()` |
| 6-(B) | 다중 카메라 합성 | `stitcher.py: AVMStitcher` |
| 7.2 | LUT 생성 (backward mapping) | `lut.py: build_ground_lut()` |
| 7.4 | 보간 | `lut.py: GroundLUT.remap()` (`cv2.INTER_LINEAR`) |

### 구조

```
avm360/
├── model.py            CameraModel — K, D, R, t + 투영/역투영
├── config.py           BEV 캔버스, 리그 설정 YAML 로드/저장
├── lut.py              지면 LUT 생성 (BEV 픽셀 → 원본 픽셀)
├── blend.py            이음선 블렌딩 가중치 (feather / wedge)
├── photometric.py      겹침 영역 기반 밝기·색 보정
├── stitcher.py         AVMStitcher — 4채널 합성 파이프라인
├── capture.py          USB 다중 카메라 캡처 (MJPEG 강제, 오토 기능 차단)
├── calib_intrinsic.py  체커보드 → K, D
├── calib_extrinsic.py  지면 ArUco 마커 → R, t
└── synth.py            가상 씬 렌더러 (하드웨어 없이 검증)
```

### 라이브러리로 쓰기

```python
from avm360 import RigConfig, AVMStitcher

rig = RigConfig.load("configs/rig_desk.yaml")
stitcher = AVMStitcher(rig)

bev = stitcher({"front": img_f, "left": img_l, "rear": img_r, "right": img_g})
```

`AVMStitcher` 는 생성 시 LUT 와 블렌딩 가중치를 한 번 계산해 두고, 이후에는 `remap` +
가중합만 수행한다 (BEV.md 5절의 통합 호모그래피와 같은 목적).

---

## 설정 파일

`configs/rig_desk.yaml` 하나로 전부 조정한다.

```yaml
bev: # 출력 캔버스: 위쪽이 전방, 왼쪽이 좌측
  x_min: -0.70
  x_max: 0.70 # X 전방
  y_min: -0.60
  y_max: 0.60 # Y 좌측
  resolution: 0.002 # 2mm/px → 700 x 600

blend:
  mode: feather # feather(거리 기반) | wedge(각도 섹터)
  feather_m: 0.06 # 이음선 페더링 폭
  photometric: true # 겹침 영역 밝기 보정

cameras:
  - name: front
    hfov_deg: 150.0
    K: [[489.0, 0.0, 640.0], [0.0, 489.0, 360.0], [0.0, 0.0, 1.0]]
    D: [0.0, 0.0, 0.0, 0.0]
    mount: # 캘리브레이션 전 설계값. 이후 R, t 로 대체된다
      position: [0.15, 0.0, 0.22]
      pitch: 40.0
      yaw: 0.0
```

실차/로봇으로 옮길 때는 `bev` 범위와 `vehicle` 치수, `mount` 위치만 바꾸면 된다.

---

## 알려진 한계

- **평면 가정**: 지면에서 떨어진 물체는 늘어나 보인다 (IPM 의 원리적 한계)
- **UVC 동기화**: USB 카메라는 하드웨어 동기 신호가 없어 프레임 스큐가 존재한다.
  정지 상태 검증에는 무방하지만, 움직이는 플랫폼에서는 스큐만큼 어긋난다
- **USB 케이블 길이**: 5m 를 넘으면 액티브 리피터가 필요하고, 실차 규모에서는
  GMSL2 + 디시리얼라이저로 넘어가는 것이 정석이다
