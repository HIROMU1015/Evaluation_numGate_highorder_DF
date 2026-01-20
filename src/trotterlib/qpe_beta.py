from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def eval_Fejer_kernel(T: int, x: np.ndarray) -> np.ndarray:
    """
    Generate the kernel of QPE
    """
    x_avoid = np.abs(x % (2 * np.pi)) < 1e-8
    numer = np.sin(0.5 * T * x) ** 2
    denom = np.sin(0.5 * x) ** 2
    denom += x_avoid
    ret = numer / denom
    ret = (1 - x_avoid) * ret + (T**2) * x_avoid
    return ret / T


def generate_QPE_distribution(
    spectrum: Sequence[float], population: Sequence[float], T: int
) -> np.ndarray:
    """
    Generate the index distribution of QPE
    """
    T = int(T)
    N = len(spectrum)
    dist = np.zeros(T)
    j_arr = 2 * np.pi * np.arange(T) / T - np.pi
    for k in range(N):
        dist += population[k] * eval_Fejer_kernel(T, j_arr - spectrum[k]) / T
    return dist


def draw_with_prob(measure: np.ndarray, N: int) -> np.ndarray:
    """
    Draw N indices independently from a given measure
    """
    L = measure.shape[0]
    cdf_measure = np.cumsum(measure)  # 累積和dis
    normal_fac = cdf_measure[-1]
    U = np.random.rand(N) * normal_fac  # 0-1のランダムな数をN個作製
    index = np.searchsorted(cdf_measure, U)
    return index


def estimate_phase(k: int, T: int) -> float:
    estimate = 2 * np.pi * k / (T) - np.pi
    return estimate


def QPE(
    spectrum: Sequence[float], population: Sequence[float], T: int, N: int
) -> float:
    """
    QPE Main routine
    """
    discrete_energies = 2 * np.pi * np.arange(T) / (T) - np.pi
    index_dist = generate_QPE_distribution(
        spectrum, population, T
    )  # Generate QPE samples
    index_samp = draw_with_prob(index_dist, N)
    values, counts = np.unique(index_samp, return_counts=True)
    index_sort = np.argsort(counts)
    estimate_1 = estimate_phase(values[index_sort[-1]], T)
    ground_state_energy = estimate_1
    return ground_state_energy


def beta_plt(
    T_list_QPE: np.ndarray = np.array(
        [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ),
    N_rep: int = 10,  # 繰り返し回数
    N_QPE: int = 100,  # サンプリング数
    spectrum_F: Sequence[float] = [-1.5],
) -> None:
    error_QPE = np.zeros(len(T_list_QPE), dtype="float")
    T_total_QPE = np.zeros(len(T_list_QPE), dtype="float")

    error_QPE_timeevo_all = np.zeros((N_rep, len(T_list_QPE)), dtype="float")
    for n in range(N_rep):
        for k in range(len(T_list_QPE)):
            T_max = T_list_QPE[k]
            output_energy = QPE([spectrum_F[0]], [1], T_max, N_QPE)
            T_total_QPE[k] += T_max * N_QPE
            ##---measure error--##
            error_QPE[k] += np.abs(spectrum_F[0] - output_energy)
            error_QPE_timeevo_all[n][k] += np.abs(spectrum_F[0] - output_energy)
    T_total_QPE = T_total_QPE / N_rep
    error_QPE = error_QPE / N_rep
    error_QPE_std = np.std(error_QPE_timeevo_all, axis=0)

    C = T_list_QPE  # コスト軸に合わせる
    eps = error_QPE

    # ----- α=1 固定フィット -----
    logC, logEps = np.log(C), np.log(eps)

    log_beta = np.average(logEps + logC)
    beta_fix = np.exp(log_beta)

    print(f"α (fixed) = 1.0")
    print(f"β (fitted) = {beta_fix:.3f}")

    # ----- プロット -----
    plt.figure(dpi=150)
    plt.xscale("log")
    plt.yscale("log")

    # データ点
    plt.errorbar(C, eps, yerr=error_QPE_std, fmt="^-", label="QPE")

    # フィット直線
    plt.loglog(
        C, beta_fix / C, "--", lw=3, label=rf"fit: $\varepsilon = {beta_fix:.2f}/M$"
    )

    plt.xlabel("$T$", fontsize=20)
    plt.ylabel(r"$\varepsilon$(T)", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(1e2, 1e5)
    plt.tight_layout()
    plt.legend(fontsize=15, loc="best")
    plt.tight_layout()
    plt.show()


def beta_scaling(
    T_list_QPE: np.ndarray = np.array(
        [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ),  # M
    N_rep: int = 10,  # 繰り返し回数
    N_QPE: int = 100,  # サンプリング数
) -> None:
    # ---- 100 回の trial で beta_fix を求めて平均 ----
    N_trials = 100
    beta_fix_list = []

    np.random.seed(42)
    for trial in range(N_trials):
        # [-pi, pi] から一様ランダムに 1 点選択
        spectrum_F = [np.random.uniform(-np.pi, np.pi)]

        error_QPE = np.zeros(len(T_list_QPE), dtype="float")
        T_total_QPE = np.zeros(len(T_list_QPE), dtype="float")
        error_QPE_timeevo_all = np.zeros((N_rep, len(T_list_QPE)), dtype="float")

        for n in range(N_rep):
            for k in range(len(T_list_QPE)):
                T_max = T_list_QPE[k]
                output_energy = QPE([spectrum_F[0]], [1], T_max, N_QPE)
                T_total_QPE[k] += T_max * N_QPE

                # --- measure error ---
                err = np.abs(spectrum_F[0] - output_energy)
                error_QPE[k] += err
                error_QPE_timeevo_all[n][k] += err

        T_total_QPE = T_total_QPE / N_rep
        error_QPE = error_QPE / N_rep
        error_QPE_std = np.std(error_QPE_timeevo_all, axis=0)

        # ---- α=1 固定フィット（元の式を踏襲）----
        C = T_list_QPE  # コスト軸に合わせる
        eps = error_QPE

        # 数値安定化（log(0)回避）。元の式に +tiny を足す以外は変更しない
        tiny = 1e-300
        logC, logEps = np.log(C), np.log(eps + tiny)
        log_beta = np.average(logEps + logC)
        beta_fix = np.exp(log_beta)

        beta_fix_list.append(beta_fix)

    beta_fix_array = np.array(beta_fix_list, dtype=float)

    print("Mean beta_fix over 100 trials:", beta_fix_array.mean())
    print("Std  beta_fix over 100 trials:", beta_fix_array.std())
