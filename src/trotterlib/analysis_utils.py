from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayLike: TypeAlias = Sequence[float] | NDArray[Any]


@dataclass(frozen=True)
class LogLogFitResult:
    slope: float
    intercept: float
    coeff: float
    r2: Optional[float] = None


def loglog_fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    mask_nonpositive: bool = True,
    compute_r2: bool = False,
) -> LogLogFitResult:
    """
    Fit log10(y) = slope * log10(x) + intercept and return slope/intercept/coeff.
    coeff is 10**intercept.
    """
    # 入力を1次元配列に整形
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same number of elements")

    # 0以下の値を除外（log を取るため）
    if mask_nonpositive:
        mask = (x_arr > 0) & (y_arr > 0)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]

    if x_arr.size < 2:
        raise ValueError("Need at least two points for log-log fit")

    # log-log 空間で線形回帰
    logx = np.log10(x_arr)
    logy = np.log10(y_arr)
    slope, intercept = np.polyfit(logx, logy, 1)
    coeff = 10 ** intercept

    # 必要なら決定係数を算出
    r2 = None
    if compute_r2:
        corr = np.corrcoef(logx.astype(float), logy.astype(float))[0, 1]
        r2 = float(corr**2)

    return LogLogFitResult(
        slope=float(slope),
        intercept=float(intercept),
        coeff=float(coeff),
        r2=r2,
    )


def loglog_average_coeff(
    x: ArrayLike,
    y: ArrayLike,
    exponent: float,
    *,
    mask_nonpositive: bool = False,
) -> float:
    """Return average coefficient C in y ~= C * x**exponent (log-log average)."""
    # 入力を1次元配列に整形
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same number of elements")

    # 必要なら 0以下の値を除外
    if mask_nonpositive:
        mask = (x_arr > 0) & (y_arr > 0)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]

    # log-log 空間で係数の平均を計算
    logx = np.log10(x_arr)
    logy = np.log10(y_arr)
    set_expo_error = logy - exponent * logx
    return float(10 ** (np.mean(set_expo_error)))


def print_loglog_fit(
    fit: LogLogFitResult,
    *,
    ave_coeff: Optional[float] = None,
) -> None:
    """log-log フィットの係数や決定係数を表示する。"""
    print("error exponent :" + str(fit.slope))
    print("error coefficient :" + str(fit.coeff))
    if fit.r2 is not None:
        print("r^2 (log-log):", fit.r2)
    if ave_coeff is not None:
        print(f"average_coeff:{ave_coeff}")


def loglog_linear_fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    mask_nonpositive: bool = True,
    compute_r2: bool = False,
) -> Dict[str, float]:
    """Legacy dict-based wrapper for log-log fit."""
    # 新実装の loglog_fit を呼び出す
    fit = loglog_fit(
        x,
        y,
        mask_nonpositive=mask_nonpositive,
        compute_r2=compute_r2,
    )
    result: Dict[str, float] = {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "coeff": fit.coeff,
    }
    if fit.r2 is not None:
        result["r2"] = fit.r2
    return result
