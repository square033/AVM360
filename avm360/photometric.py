"""겹침 영역 기반 밝기/색 보정 (BEV.md 6-(B)절 3번).

카메라마다 노출·화이트밸런스가 달라 같은 지면인데도 밝기와 색이 다르게 찍힌다.
겹치는 영역에서 두 카메라가 같은 밝기를 갖도록 카메라별 이득(gain)을 구한다.

  겹침 (i,j) 에서:  g_i · m_i ≈ g_j · m_j
  로그를 취하면:     a_i - a_j = log(m_j) - log(m_i),   a = log g

이 선형 연립식을 최소제곱으로 풀되, 전체 밝기가 떠내려가지 않도록 Σa = 0 을 함께 건다.
"""

from __future__ import annotations

import numpy as np

MIN_OVERLAP_PX = 200


def solve_gains(
    warped: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    gain_limit: tuple[float, float] = (0.6, 1.7),
    per_channel: bool = True,
) -> dict[str, np.ndarray]:
    """카메라별 이득을 구한다.

    Args:
        warped: BEV 로 워핑된 카메라별 이미지 (H,W,3) 또는 (H,W).
        masks: 카메라별 유효영역.
        gain_limit: 이득 클리핑 범위. 노출 실패 시 폭주를 막는다.
        per_channel: True 면 채널별 이득(화이트밸런스까지 보정), False 면 밝기만.

    Returns:
        카메라 이름 → 이득 배열. per_channel 이면 (3,), 아니면 (1,).
    """
    names = list(warped.keys())
    n = len(names)
    sample = warped[names[0]]
    n_ch = sample.shape[2] if (per_channel and sample.ndim == 3) else 1

    gains = {name: np.ones(n_ch, dtype=np.float32) for name in names}
    if n < 2:
        return gains

    for ch in range(n_ch):
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        wts: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                mi, mj = masks[names[i]], masks[names[j]]
                overlap = mi & mj
                count = int(overlap.sum())
                if count < MIN_OVERLAP_PX:
                    continue

                a_img, b_img = warped[names[i]], warped[names[j]]
                if a_img.ndim == 3 and n_ch > 1:
                    a_vals, b_vals = a_img[..., ch][overlap], b_img[..., ch][overlap]
                elif a_img.ndim == 3:
                    a_vals, b_vals = a_img[overlap].mean(axis=1), b_img[overlap].mean(axis=1)
                else:
                    a_vals, b_vals = a_img[overlap], b_img[overlap]

                mean_i = float(np.mean(a_vals)) + 1e-3
                mean_j = float(np.mean(b_vals)) + 1e-3
                # 한쪽이 포화되었거나 새까만 구간은 신뢰할 수 없다.
                if not (1.0 < mean_i < 250.0 and 1.0 < mean_j < 250.0):
                    continue

                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                rows.append(row)
                rhs.append(np.log(mean_j) - np.log(mean_i))
                wts.append(np.sqrt(count))

        if not rows:
            continue

        # Σa = 0 제약을 하나의 식으로 추가 (가중치는 데이터 항과 비슷한 스케일로).
        rows.append(np.ones(n))
        rhs.append(0.0)
        wts.append(float(np.mean(wts)))

        A = np.asarray(rows) * np.asarray(wts)[:, None]
        b = np.asarray(rhs) * np.asarray(wts)
        a, *_ = np.linalg.lstsq(A, b, rcond=None)
        g = np.clip(np.exp(a), *gain_limit)
        for idx, name in enumerate(names):
            gains[name][ch] = g[idx]

    return gains


def apply_gain(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """이득을 적용하고 8비트로 클리핑한다."""
    out = image.astype(np.float32)
    if out.ndim == 3 and gain.size == out.shape[2]:
        out *= gain.reshape(1, 1, -1)
    else:
        out *= float(gain.reshape(-1)[0])
    return np.clip(out, 0, 255).astype(np.uint8)
