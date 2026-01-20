from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import eigs


def find_closest_value(E: float, values: Sequence[float]) -> Tuple[float, float]:
    """E に最も近い値とその誤差を返す。"""
    # 最小差の要素を取得
    abs_diffs = [abs(E - v) for v in values]
    i_min = int(np.argmin(abs_diffs))
    return values[i_min], abs_diffs[i_min]


def error_cal_multi(
    t_list: Sequence[float],
    terms_list: Sequence[Any],
    ori_vec: np.ndarray,
    E: float,
    num_eig: int,
) -> Tuple[List[float], List[float]]:
    """各 t での固有値誤差を計算（シフト付き反復法の既存ロジックを保持）。"""
    # 各時刻で固有値近似を評価
    r_tlist, error_list = [], []
    for t, terms in zip(t_list, terms_list):
        Et = E * t
        n_wrap = int((-Et) // (2 * np.pi)) + 1
        # シフトは E の1点のみ（既存挙動）
        sigma = np.exp(1j * E * t)
        eigenvalues = eigs(
            terms, sigma=sigma, k=num_eig, v0=ori_vec, return_eigenvectors=False
        )
        # 位相を [-2π, 0] 側に補正
        phases = np.angle(eigenvalues)
        phases = np.where(phases > 0, phases - 2 * np.pi, phases) - (
            2 * (n_wrap - 1) * np.pi
        )
        # エネルギーに換算して誤差を算出
        en = sorted([ph.real / t for ph in phases])
        approx, _err = find_closest_value(E, en)
        error = abs(E - approx)
        r_tlist.append(t)
        error_list.append(error)
    return r_tlist, error_list
