from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from .config import (
    CA,
    BETA,
    TARGET_ERROR,
    COLOR_MAP,
    MARKER_MAP,
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    PFLabel,
    P_DIR,
    DECOMPO_NUM,
    PF_RZ_LAYER,
    PICKLE_DIR_DF_PATH,
    get_df_rank_fraction_for_molecule,
    normalize_pf_label,
    pickle_dir,
    SURFACE_CODE_AUTO_POPULATE,
    SURFACE_CODE_A_EFF_CASES,
    SURFACE_CODE_A_EFF_DEFAULT,
    SURFACE_CODE_CACHE_DIR,
    SURFACE_CODE_CODE_DISTANCE_CANDIDATES,
    SURFACE_CODE_COMPILE_MODE,
    SURFACE_CODE_COMPILE_SKIP_OUTPUT,
    SURFACE_CODE_COMPILE_SKIP_REDUNDANT_IR_PREPROCESS,
    SURFACE_CODE_DELTA_FAIL_CASES,
    SURFACE_CODE_ENTANGLEMENT_GENERATION_PERIOD,
    SURFACE_CODE_MACHINE_TYPE,
    SURFACE_CODE_MAGIC_GENERATION_PERIOD,
    SURFACE_CODE_MAX_ENTANGLED_STATE_STOCK,
    SURFACE_CODE_MAX_MAGIC_STATE_STOCK,
    SURFACE_CODE_P_PHYS_CASES,
    SURFACE_CODE_P_TH,
    SURFACE_CODE_QASM_BASIS_GATES,
    SURFACE_CODE_QASM_DECOMPOSE_REPS,
    SURFACE_CODE_QCSF_PATH,
    SURFACE_CODE_GRIDSYNTH_PATH,
    SURFACE_CODE_STEP_DF_PATH,
    SURFACE_CODE_STEP_GROUPED_ORIGINAL_PATH,
    SURFACE_CODE_STEP_GROUPED_PATH,
    SURFACE_CODE_DF_ROTATION_LAYER_PREFERRED_KEY,
    SURFACE_CODE_DECOMPOSE_CHUNK_MAX_OPS,
    SURFACE_CODE_DECOMPOSE_MAX_WORKERS,
    SURFACE_CODE_PROXY_DEPTH_MULTIPLIER,
    SURFACE_CODE_PROXY_GATE_COUNT_PER_MAGIC,
    SURFACE_CODE_PROXY_RUNTIME_PER_MAGIC,
    SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION,
    SURFACE_CODE_ROTATION_PRECISION_FLOOR,
    SURFACE_CODE_ROTATION_PRECISION_MODE,
    SURFACE_CODE_FIXED_ROTATION_PRECISION,
    SURFACE_CODE_REACTION_TIME,
    SURFACE_CODE_RUNTIME_METRIC,
    SURFACE_CODE_TOPOLOGY_PATH,
    surface_code_step_dir,
)

from .io_cache import load_data, label_replace
from .chemistry_hamiltonian import jw_hamiltonian_maker
from .product_formula import _get_w_list
from .analysis_utils import LogLogFitResult, loglog_fit
from .plot_utils import set_loglog_axes, unique_legend_entries
from .plots_timeevo_error import plot_trotter_error_curve, trotter_error_qc_gr_curve



def calculation_cost(
    clique_list: Sequence[Sequence[Any]],
    num_w: PFLabel | int,
    ham_name: str,
) -> Tuple[int, Dict[int, int]]:
    """
    分解数計算用(decompo_num) clique_num_dir にそのクリークに含まれる項をインデックス付きで登録
    """
    _ = ham_name  # kept for compatibility; not used

    clique_num_dir = {}
    for i, clique in enumerate(clique_list):
        num_terms = 0
        for terms in clique:
            # term_list = [term for term in terms]
            term_list = [term for term in terms if list(term.terms.keys())[0] != ()]
            num_terms += len(term_list)
        clique_num_dir[i] = num_terms

    if num_w == 0:
        total_exp = sum(clique_num_dir.values())
        return total_exp, clique_num_dir

    if isinstance(num_w, int):
        raise ValueError(f"Unsupported num_w: {num_w}")

    w_list = _get_w_list(num_w)
    m = len(w_list)

    total_exp = 0
    J = len(clique_list)

    if m == 1:
        for i in range(J - 1):
            total_exp += clique_num_dir[i]
        total_exp += clique_num_dir[J - 1]
        for k in reversed(range(0, J - 1)):
            total_exp += clique_num_dir[k]
        return total_exp, clique_num_dir

    # S2_left
    for i in range(J - 1):
        total_exp += clique_num_dir[i]
    total_exp += clique_num_dir[J - 1]
    for k in reversed(range(1, J - 1)):
        total_exp += clique_num_dir[k]
    total_exp += clique_num_dir[0]

    # S2
    for _ in reversed(range(1, m - 1)):
        for i in range(1, J - 1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J - 1]
        # 終端 - 1 個目まで
        for k in reversed(range(1, J - 1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]
    for _ in range(0, m - 1):
        for i in range(1, J - 1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J - 1]
        # 終端 - 1 個目まで
        for k in reversed(range(1, J - 1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]

    # S2right
    for i in range(1, J - 1):
        total_exp += clique_num_dir[i]
    # 折り返し
    total_exp += clique_num_dir[J - 1]
    # 終端 まで
    for k in reversed(range(0, J - 1)):
        total_exp += clique_num_dir[k]

    return total_exp, clique_num_dir


def _hchain_series(Hchain: int) -> Tuple[List[int], List[str], List[int]]:
    """H-chain の系列と対応するラベル・量子ビット数を返す。"""
    chain_list = [i for i in range(2, Hchain + 1)]
    chain_str = [f"H{i}" for i in chain_list]
    num_qubits = [i for i in range(4, (Hchain * 2) + 1, 2)]
    return chain_list, chain_str, num_qubits


def _compute_min_f(eps: float, expo: float, coeff: float) -> float:
    """外挿の最小コスト係数 min_f を計算する。"""
    return (
        BETA
        * (eps ** (-(1 + (1 / expo))))
        * (1 / expo)
        * (coeff ** (1 / expo))
        * (expo + 1) ** (1 + (1 / expo))
    )


def _apply_loglog_fit_with_bands(
    ax: Any,
    series: Dict[str, Dict[str, List[float]]],
    *,
    show_bands: bool,
    band_height: float,
    band_alpha: float,
) -> Dict[str, Dict[str, float]]:
    """log-log フィットと色帯描画、凡例の付与を共通化する。"""
    # ---- フィット ----
    fit_params: Dict[str, Dict[str, float]] = {}
    for lb, d in series.items():
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        m = (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        fit = loglog_fit(x, y, mask_nonpositive=True)
        A = fit.coeff
        B = fit.slope
        fit_params[lb] = {"A": A, "B": B}

        x_right = ax.get_xlim()[1]
        xfit = np.logspace(np.log10(x.min()), np.log10(x_right), 400)
        yfit = A * (xfit**B)
        ax.plot(xfit, yfit, "-", color=COLOR_MAP.get(lb), alpha=0.9, linewidth=1.5)

    winners_in_order: List[str] = []
    # ---- 色帯の表示（トグル）----
    if show_bands and fit_params:
        # 表示範囲に合わせて評価用グリッドを作成
        x_left, x_right = ax.get_xlim()
        xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)

        labels_fit = list(fit_params.keys())
        # 各ラベルの y(x)
        Y = np.vstack(
            [fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit]
        )
        imin = np.argmin(Y, axis=0)  # 各 x で最小の系列のインデックス

        # 勝者切替の境界
        switch_idx = np.where(np.diff(imin) != 0)[0] + 1
        bounds = np.r_[0, switch_idx, len(xgrid) - 1]

        for s, e in zip(bounds[:-1], bounds[1:]):
            lb = labels_fit[imin[s]]
            if lb not in winners_in_order:
                winners_in_order.append(lb)
            ax.axvspan(
                xgrid[s],
                xgrid[e],
                ymin=0.0,
                ymax=band_height,
                color=COLOR_MAP.get(lb, "0.6"),
                alpha=band_alpha,
                ec="none",
                zorder=0,
            )

        # 縦点線（任意）
        for i in switch_idx:
            ax.axvline(
                xgrid[i],
                linestyle=":",
                linewidth=1.0,
                color="k",
                alpha=0.5,
                zorder=1,
            )

    # ---- 凡例 ----
    handles_exist, labels_exist = ax.get_legend_handles_labels()
    handles_u, labels_u = unique_legend_entries(handles_exist, labels_exist)
    seen = set(labels_u)

    if show_bands and winners_in_order:
        proxies = [
            Patch(
                facecolor=COLOR_MAP[lb],
                alpha=0.6,
                edgecolor="none",
                label=f"{lb} (lowest)",
            )
            for lb in winners_in_order
            if f"{lb} (lowest)" not in seen
        ]
        handles_u += proxies
        labels_u += [str(p.get_label()) for p in proxies]

    ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)
    return fit_params


NestedPFMapping: TypeAlias = Mapping[str, Mapping[PFLabel, float]]


def _lookup_nested_value(
    table: NestedPFMapping | None, mol_label: str, pf_label: PFLabel
) -> float | None:
    """Read a float from a nested {mol_label: {pf_label: value}} mapping."""

    if table is None:
        return None
    pf_map = table.get(mol_label)
    if pf_map is None:
        return None
    value = pf_map.get(pf_label)
    if value is None:
        return None
    return float(value)


def total_pauli_rotations_from_scaling(
    mol_type: int,
    pf_label: PFLabel,
    *,
    target_error: float = TARGET_ERROR,
    scaling_data: NestedPFMapping | None = None,
    decompo_counts: NestedPFMapping | None = None,
    use_original: bool = False,
) -> float:
    """Return total Pauli-rotation cost for one PF label using streamed scaling data.

    If `scaling_data`/`decompo_counts` are provided, they override the artifacts; otherwise
    the helper falls back to loading the grouped coefficient and N_0 (DECOMPO_NUM).
    """

    mol_label = f"H{mol_type}"
    coeff = _lookup_nested_value(scaling_data, mol_label, pf_label)
    if coeff is None:
        _, _, ham_name, _ = jw_hamiltonian_maker(mol_type, 1.0)
        grouped = f"{ham_name}_grouping"
        target_path = f"{grouped}_Operator_{pf_label}_ave"
        coeff = float(load_data(target_path, use_original=use_original))

    expo = P_DIR[pf_label]
    min_f = _compute_min_f(target_error, expo, coeff)

    unit_expo = _lookup_nested_value(decompo_counts, mol_label, pf_label)
    if unit_expo is None:
        unit_expo = DECOMPO_NUM[mol_label][pf_label]

    return float(unit_expo) * min_f


def df_trotter_total_rotations(
    t_start: float,
    t_end: float,
    t_step: float,
    *,
    molecule_type: int,
    pf_label: PFLabel,
    target_error: float = TARGET_ERROR,
    use_original: bool = False,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    reference: str = "exact",
    calibrate_phase: bool = True,
    show_plot: bool = True,
    fit: bool = True,
) -> tuple[float, float, int, list[float], list[float]]:
    """Run df_trotter energy sweep, optionally show its plot, and return cost info."""

    from scripts.df_trotter_energy_plot_perturb import (
        df_trotter_energy_error_curve_perturb,
        df_trotter_fixed_order_coeff,
        df_trotter_pauli_rotation_count,
        plot_df_trotter_error_curve,
    )

    times, errors, setup = df_trotter_energy_error_curve_perturb(
        t_start,
        t_end,
        t_step,
        molecule_type=molecule_type,
        pf_label=pf_label,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
        reference=reference,
        calibrate_phase=calibrate_phase,
        return_setup=True,
    )

    alpha = df_trotter_fixed_order_coeff(times, errors, pf_label)
    pauli_rotations = df_trotter_pauli_rotation_count(setup)
    if show_plot:
        plot_df_trotter_error_curve(
            times,
            errors,
            molecule_type=molecule_type,
            pf_label=pf_label,
            fit=fit,
        )
    total_rotations = total_pauli_rotations_from_scaling(
        mol_type=molecule_type,
        pf_label=pf_label,
        target_error=target_error,
        scaling_data={f"H{molecule_type}": {pf_label: alpha}},
        use_original=use_original,
        decompo_counts={f"H{molecule_type}": {pf_label: pauli_rotations}},
    )
    print(f"total_rotations ({pf_label}, H{molecule_type}): {total_rotations:.3e}")
    print(f"pauli rotations (DF circuit): {pauli_rotations}")
    return total_rotations, alpha, pauli_rotations, times, errors


def trotter_qc_gr_total_rotations(
    t_start: float,
    t_end: float,
    t_step: float,
    *,
    molecule_type: int,
    pf_label: PFLabel,
    target_error: float = TARGET_ERROR,
    use_original: bool = False,
    scaling_data: NestedPFMapping | None = None,
    decompo_counts: NestedPFMapping | None = None,
    show_plot: bool = True,
    fit: bool = True,
) -> tuple[float, float, list[float], list[float], LogLogFitResult]:
    """Compute total rotations from trotter_error_qc_gr perturbation scaling using avg coeff and measured Pauli rotations."""

    times, errors, counts, ham_name, avg_coeff, fit_result = trotter_error_qc_gr_curve(
        t_start, t_end, t_step, molecule_type, pf_label
    )
    if show_plot:
        plot_trotter_error_curve(
            times,
            errors,
            title=f"{ham_name}_{pf_label}",
            fit_result=fit_result if fit else None,
        )

    alpha = avg_coeff
    measured_pauli = int(round(np.mean(counts))) if counts else 0

    scaling_data_to_use = scaling_data if scaling_data is not None else {f"H{molecule_type}": {pf_label: alpha}}
    decompo_counts_to_use = decompo_counts if decompo_counts is not None else (
        {f"H{molecule_type}": {pf_label: measured_pauli}} if measured_pauli > 0 else None
    )
    total_rotations = total_pauli_rotations_from_scaling(
        mol_type=molecule_type,
        pf_label=pf_label,
        target_error=target_error,
        scaling_data=scaling_data_to_use,
        decompo_counts=decompo_counts_to_use,
        use_original=use_original,
    )
    print(f"total_rotations ({pf_label}, H{molecule_type}) from QC-GR: {total_rotations:.3e}")
    if measured_pauli > 0:
        print(f"pauli rotations (QC-GR circuit): {measured_pauli}")
    return total_rotations, alpha, measured_pauli, times, errors, fit_result


def exp_extrapolation(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    show_bands: bool = True,
    band_height: float = 0.06,
    band_alpha: float = 0.28,
    use_original: bool = False,
    scaling_data: NestedPFMapping | None = None,
    decompo_counts: NestedPFMapping | None = None,
) -> None:
    """PF 別に総コストの外挿をプロットする（use_original=True で original を参照）。

    scaling_data を与えると df_trotter などのスケーリング結果（α）を使って min_f を計算し、
    decompo_counts を与えると 1 ステップあたりのパウリ回転数を上書きできます。
    """
    _, Hchain_str, num_qubits = _hchain_series(Hchain)

    target_error = TARGET_ERROR

    total_dir: Dict[int, Dict[str, float]] = {}

    plt.figure(figsize=(8, 6), dpi=200)

    # 係数データを読み込んで総回転数を算出
    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        total_dir[qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:  # 10th は H15 で未評価
                continue

            coeff = _lookup_nested_value(scaling_data, mol, n_w)
            if coeff is None:
                target_path = f"{ham_name}_Operator_{n_w}_ave"
                data = load_data(target_path, use_original=use_original)
                coeff = float(data)

            expo = P_DIR[n_w]

            min_f = _compute_min_f(target_error, expo, coeff)

            unit_expo = _lookup_nested_value(decompo_counts, mol, n_w)
            if unit_expo is None:
                unit_expo = DECOMPO_NUM[mol][n_w]
            total_expo = float(unit_expo) * min_f

            total_dir[qubits][n_w] = total_expo

    # プロット用の系列を構築
    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(
                qubit,
                gate,
                ls="None",
                marker=MARKER_MAP[pf],
                color=COLOR_MAP[pf],
                label=lb if qubit == num_qubits[0] else None,
            )
            # フィット用
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    set_loglog_axes(ax)

    XMAX = 100
    xmin_current, xmax_current = ax.get_xlim()
    ax.set_xlim(xmin_current, max(xmax_current, XMAX))

    _apply_loglog_fit_with_bands(
        ax,
        series,
        show_bands=show_bands,
        band_height=band_height,
        band_alpha=band_alpha,
    )

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    plt.show()


def exp_extrapolation_diff(
    Hchain: int,
    n_w_list: Sequence[PFLabel] = ("4th(new_2)", "8th(Morales)"),
    MIN_POS: float = 1e-18,
    X_MIN_CALC: float = 4,
    X_MAX_DISPLAY: float = 100,
    use_original: bool = False,
    scaling_data: NestedPFMapping | None = None,
    decompo_counts: NestedPFMapping | None = None,
) -> None:
    """
    単一図（左右Y軸）:
      左Y: 総パウリ回転数（散布 + log–log フィット）
      右Y: 2本のフィット直線の絶対差 |Δ|
    依存：decompo_num, load_data, label_replace,
         MARKER_MAP, COLOR_MAP, P_DIR
    use_original=True で trotter_expo_coeff_gr_original を参照する。

    scaling_data と decompo_counts によって artifacts のデータにアクセスせず
    df_trotter などの現地結果 → α / パウリ回転数で min_f を構築できます。
    """

    # 対象 H チェーン
    _, Hchain_str, num_qubits = _hchain_series(Hchain)

    target_error = TARGET_ERROR


    # 総回転数の算出
    total_dir: Dict[int, Dict[str, float]] = {}
    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"
        total_dir[qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:  # 10th は H15 で未評価
                continue

            coeff = _lookup_nested_value(scaling_data, mol, n_w)
            if coeff is None:
                target_path = f"{ham_name}_Operator_{n_w}_ave"
                data = load_data(target_path, use_original=use_original)
                coeff = float(data)

            expo = P_DIR[n_w]

            min_f = _compute_min_f(target_error, expo, coeff)

            unit_expo = _lookup_nested_value(decompo_counts, mol, n_w)
            if unit_expo is None:
                unit_expo = DECOMPO_NUM[mol][n_w]
            total_expo = float(unit_expo) * min_f

            total_dir[qubits][n_w] = total_expo

    # ---- プロット（単一図・双Y軸）----
    plt.figure(figsize=(8, 6), dpi=200)

    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(
                qubit,
                gate,
                ls="None",
                marker=MARKER_MAP[pf],
                color=COLOR_MAP[pf],
                label=lb if qubit == num_qubits[0] else None,
            )
            # フィット用
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    ax2 = ax.twinx()

    # 軸
    set_loglog_axes(ax)

    # x の右端を固定
    x_left_auto, _ = ax.get_xlim()
    ax.set_xlim(x_left_auto, X_MAX_DISPLAY)

    xfit_lo = max(X_MIN_CALC, x_left_auto)
    xfit_hi = X_MAX_DISPLAY
    fit_params = _apply_loglog_fit_with_bands(
        ax,
        series,
        show_bands=False,
        band_height=0.0,
        band_alpha=0.0,
    )

    # ---- 右軸：差分 |Δ| ----
    # フィット区間と同じ x 範囲 [xfit_lo, xfit_hi] で評価
    # キーは pf 文字列（例: "4th(new_2)", "8(Mauro)"）で揃える
    if xfit_hi > xfit_lo:
        ax2.set_yscale("log")
        if len(n_w_list) == 2 and all(pf in fit_params for pf in n_w_list):
            pf_a, pf_b = n_w_list  # 表示順を n_w_list に揃える
            xx = np.logspace(np.log10(xfit_lo), np.log10(xfit_hi), 1200)

            A_a, B_a = fit_params[pf_a]["A"], fit_params[pf_a]["B"]
            A_b, B_b = fit_params[pf_b]["A"], fit_params[pf_b]["B"]
            ya = A_a * (xx**B_a)
            yb = A_b * (xx**B_b)

            diff = np.maximum(np.abs(yb - ya), MIN_POS)
            ax2.plot(
                xx,
                diff,
                "--",
                lw=2.0,
                alpha=0.9,
                color=COLOR_MAP.get(pf_a, None),
                label=f"|Δ|: {label_replace(pf_b)} − {label_replace(pf_a)}",
            )
            # 左軸とレンジを一致させて目盛位置をそろえる
            ax2.set_ylim(ax.get_ylim())

        # 凡例（右軸）
        hr, lr = ax2.get_legend_handles_labels()
        if hr:
            ax2.legend(hr, lr, loc="upper right", framealpha=0.9)

    # 軸ラベル・グリッド
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax2.set_ylabel("Difference in number of Pauli rotations", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


def t_depth_extrapolation(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    rz_layer: Optional[bool] = None,
    show_bands: bool = True,
    band_height: float = 0.06,
    band_alpha: float = 0.28,
    use_original: bool = False,
) -> None:
    """H-chain サイズに対する T-depth / RZ レイヤー数を外挿する（use_original=True で original を参照）。"""

    target_error = TARGET_ERROR

    total_dir: Dict[int, Dict[str, float]] = {}

    _, Hchain_str, num_qubits = _hchain_series(Hchain)

    plt.figure(figsize=(8, 6), dpi=200)

    # 係数データを読み込んで T-depth を算出
    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + '_sto-3g_singlet_distance_100_charge_0_grouping'
        else:
            ham_name = mol + '_sto-3g_triplet_1+_distance_100_charge_1_grouping'

        total_dir[qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:  # 10th は H15 で未評価
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            data = load_data(target_path, use_original=use_original)

            coeff = data
            expo = P_DIR[n_w]

            # N_0 PF による分解に含まれる RZ 数 T-depth 計算なら実質パウリ回転数
            N_0 = DECOMPO_NUM[mol][n_w]

            # L_Z RZ のレイヤー数
            pf_layer_rz = PF_RZ_LAYER[mol][n_w]

            t = (target_error / coeff * (expo + 1))**(1/expo)
            qpe_factor = _qpe_iteration_factor(
                float(coeff),
                float(expo),
                float(target_error),
            )

            # RZ の近似誤差は許容誤差の 1 パーセント
            eps_rot = (t * 0.01 * target_error) / (N_0 * qpe_factor)
            
            # RZ 近似誤差 T = 3log2(1/eps_rot)
            T_rot = 3 * np.log2(1/eps_rot)

            # PF のユニタリ１回分の T-depth
            D_T = pf_layer_rz * T_rot

            # QPE QC 全体での T-depth
            tot_dt = qpe_factor * D_T

            # QPE QC 全体での RZ レイヤー数
            tot_rz_layer = qpe_factor * pf_layer_rz

            if qubits == 30:
                print(f'T_rot:{T_rot} PF:{n_w}')

            if rz_layer:
                total_dir[qubits][n_w] =  tot_rz_layer
            else:
                total_dir[qubits][n_w] = tot_dt

    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(qubit, gate, ls='None', marker=MARKER_MAP[pf], color=COLOR_MAP[pf],
                     label=lb if qubit == num_qubits[0] else None)
            # フィット用
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    set_loglog_axes(ax)

    X_MAX_DISPLAY = 100
    x_min = float(min(num_qubits))
    ax.set_xlim(x_min, X_MAX_DISPLAY)

    _apply_loglog_fit_with_bands(
        ax,
        series,
        show_bands=show_bands,
        band_height=band_height,
        band_alpha=band_alpha,
    )

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    if rz_layer:
        ax.set_ylabel("Num RZ layer", fontsize=15)
    else:
        ax.set_ylabel("T-depth", fontsize=15)
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)
    plt.show()


def _load_df_artifact_payload(ham_name: str, pf_label: PFLabel) -> Dict[str, Any]:
    """DF 外挿用の保存バイナリを読み込む。"""
    path = PICKLE_DIR_DF_PATH / f"{ham_name}_Operator_{pf_label}"
    data = load_data(str(path), gr=None)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid DF artifact payload: {path}")
    return data


def _artifact_nonnegative_int(value: Any, *, field: str, context: str) -> int:
    """artifact 値を 0 以上の int に正規化する。"""
    try:
        scalar = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}={value!r} in {context}") from exc
    if scalar < 0:
        raise ValueError(f"Negative {field}={scalar} in {context}")
    return scalar


def _normalize_positive_scalar_list(
    values: Iterable[Any],
    *,
    field: str,
) -> List[float]:
    """候補列を正の float リストへ正規化する。"""
    normalized: List[float] = []
    for raw in values:
        try:
            scalar = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field}={raw!r}") from exc
        if (not np.isfinite(scalar)) or scalar <= 0:
            raise ValueError(f"Invalid {field}={scalar!r}")
        normalized.append(float(scalar))
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    return normalized


def _normalize_code_distance_candidates(
    code_distances: Sequence[int] | None,
) -> List[int]:
    """解析対象の code distance 列を正規化する。"""
    raw_values = (
        SURFACE_CODE_CODE_DISTANCE_CANDIDATES
        if code_distances is None
        else code_distances
    )
    normalized: Set[int] = set()
    for raw in raw_values:
        try:
            dist = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid code distance={raw!r}") from exc
        if dist <= 0:
            raise ValueError(f"Code distance must be positive: {dist}")
        if dist % 2 == 0:
            raise ValueError(
                f"Code distance must be odd for surface-code estimates: {dist}"
            )
        normalized.add(dist)
    if not normalized:
        raise ValueError("code_distances must not be empty.")
    return sorted(normalized)


def normalize_surface_code_step_metrics(
    metrics: Mapping[str, Any],
    *,
    context: str = "surface_code_step",
) -> Dict[str, Any]:
    """surface_code の compile_info 由来 step metrics を正規化する。"""
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Invalid {context}: expected mapping.")

    required_int_fields = (
        "magic_state_consumption_count",
        "magic_state_consumption_depth",
        "runtime",
        "runtime_without_topology",
        "qubit_volume",
    )
    optional_int_fields = (
        "gate_count",
        "gate_depth",
        "measurement_feedback_count",
        "measurement_feedback_depth",
        "magic_factory_count",
        "chip_cell_count",
        "t_count",
        "t_depth",
    )

    normalized: Dict[str, Any] = {}
    for field in required_int_fields:
        normalized[field] = _artifact_nonnegative_int(
            metrics.get(field),
            field=field,
            context=context,
        )
    for field in optional_int_fields:
        if field not in metrics:
            continue
        normalized[field] = _artifact_nonnegative_int(
            metrics.get(field),
            field=field,
            context=context,
        )

    for meta_key in ("source", "compile_info_json", "compile_mode"):
        value = metrics.get(meta_key)
        if value is not None:
            normalized[meta_key] = str(value)
    for meta_key in ("target_error", "step_time", "rotation_precision"):
        value = metrics.get(meta_key)
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {meta_key}={value!r} in {context}") from exc
        if (not np.isfinite(scalar)) or scalar <= 0:
            raise ValueError(f"Invalid {meta_key}={scalar!r} in {context}")
        normalized[meta_key] = float(scalar)
    for meta_key in ("generator", "cache_key"):
        value = metrics.get(meta_key)
        if value is not None:
            normalized[meta_key] = str(value)
    if "auto_generated" in metrics:
        normalized["auto_generated"] = bool(metrics.get("auto_generated"))
    compile_mode = normalized.get("compile_mode")
    generator = normalized.get("generator")
    if (
        "t_count" not in normalized
        and (
            compile_mode == "decompose_only"
            or generator == "decompose_only_ir"
        )
    ):
        normalized["t_count"] = normalized["magic_state_consumption_count"]
    if (
        "t_depth" not in normalized
        and (
            compile_mode == "decompose_only"
            or generator == "decompose_only_ir"
        )
    ):
        normalized["t_depth"] = normalized["magic_state_consumption_depth"]
    return normalized


def surface_code_step_metrics_from_compile_info_json(
    compile_info_path: str | Path,
) -> Dict[str, Any]:
    """surface_code の compile_info.json から step metrics を抽出する。"""
    path = Path(compile_info_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        compile_info = json.load(f)
    metrics = normalize_surface_code_step_metrics(
        compile_info,
        context=str(path),
    )
    metrics["compile_info_json"] = str(path)
    return metrics


def _surface_code_cache_key(target_error: float) -> str:
    """target_error 用の cache key を返す。"""
    mode = str(SURFACE_CODE_ROTATION_PRECISION_MODE)
    if mode == "fixed":
        precision_tag = f"rot_fixed_{SURFACE_CODE_FIXED_ROTATION_PRECISION:.3e}"
    elif mode == "task_budget":
        precision_tag = (
            "rot_task_budget"
            f"_eta_{SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION:.3e}"
        )
    elif mode == "layer_linear_floor":
        precision_tag = (
            "rot_layer_linear_floor"
            f"_eta_{SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION:.3e}"
            f"_floor_{SURFACE_CODE_ROTATION_PRECISION_FLOOR:.3e}"
        )
    else:
        precision_tag = f"rot_{mode}"
    return (
        f"{SURFACE_CODE_COMPILE_MODE}:{precision_tag}:eps_{target_error:.12e}"
    )


def _surface_code_rotation_count_for_precision(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> float:
    """task-level rotation precision 用の N_0 を返す。"""
    if source == "gr":
        mol_label = f"H{_parse_molecule_type_from_ham_name(ham_name)}"
        try:
            return float(DECOMPO_NUM[mol_label][pf_label])
        except KeyError as exc:
            raise ValueError(
                f"Missing grouped DECOMPO_NUM for {ham_name}_Operator_{pf_label}"
            ) from exc

    if source == "df":
        payload = _load_df_artifact_payload(ham_name, pf_label)
        rz_layers_raw = payload.get("rz_layers")
        if not isinstance(rz_layers_raw, Mapping):
            raise ValueError(
                f"DF artifact missing rz_layers: {ham_name}_Operator_{pf_label}"
            )
        candidate_keys = (
            "total_nonclifford_rz_count",
            "total_nonclifford_z_count",
            "total_ref_rz_count",
            "ref_rz_count",
            "total_nonclifford_z_coloring_depth",
            "total_nonclifford_z_depth",
            "total_nonclifford_rz_depth",
        )
        for key in candidate_keys:
            if key not in rz_layers_raw:
                continue
            try:
                value = float(rz_layers_raw[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        raise ValueError(
            f"DF artifact missing usable rotation-count proxy: {ham_name}_Operator_{pf_label}"
        )

    raise ValueError(f"Unsupported source: {source}")


def _surface_code_rotation_layer_depth_for_precision(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> float:
    """task-level rotation precision 用の L_eff を返す。"""
    if source == "gr":
        mol_label = f"H{_parse_molecule_type_from_ham_name(ham_name)}"
        try:
            return float(PF_RZ_LAYER[mol_label][pf_label])
        except KeyError as exc:
            raise ValueError(
                f"Missing grouped PF_RZ_LAYER for {ham_name}_Operator_{pf_label}"
            ) from exc

    if source == "df":
        payload = _load_df_artifact_payload(ham_name, pf_label)
        rz_layers_raw = payload.get("rz_layers")
        if not isinstance(rz_layers_raw, Mapping):
            raise ValueError(
                f"DF artifact missing rz_layers: {ham_name}_Operator_{pf_label}"
            )
        _metric_key, rz_layer_value = _pick_df_rz_layer_value(
            rz_layers_raw,
            preferred_key=SURFACE_CODE_DF_ROTATION_LAYER_PREFERRED_KEY,
        )
        return float(rz_layer_value)

    raise ValueError(f"Unsupported source: {source}")


def _surface_code_proxy_t_per_rotation(rotation_precision: float) -> int:
    """1 回転あたりの魔法状態数 proxy を返す。"""
    if (not np.isfinite(rotation_precision)) or rotation_precision <= 0:
        raise ValueError(f"Invalid rotation_precision={rotation_precision!r}")
    return max(1, int(math.ceil(3.0 * math.log2(1.0 / float(rotation_precision)))))


def _surface_code_proxy_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
    step_time: float,
    rotation_precision: float,
    use_original: bool = False,
) -> Dict[str, Any]:
    """surface_code full compile を使わない高速 proxy step metrics。"""
    rotation_count = _surface_code_rotation_count_for_precision(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    rotation_layer_depth = _surface_code_rotation_layer_depth_for_precision(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    t_per_rotation = _surface_code_proxy_t_per_rotation(rotation_precision)
    magic_count = max(1, int(round(float(rotation_count) * float(t_per_rotation))))
    magic_depth = min(
        magic_count,
        max(
            1,
            int(
                math.ceil(
                    float(SURFACE_CODE_PROXY_DEPTH_MULTIPLIER)
                    * float(rotation_layer_depth)
                    * float(t_per_rotation)
                )
            ),
        ),
    )
    gate_count = max(
        magic_count,
        int(math.ceil(float(SURFACE_CODE_PROXY_GATE_COUNT_PER_MAGIC) * magic_count)),
    )
    runtime_wo = max(
        magic_depth,
        int(math.ceil(float(SURFACE_CODE_PROXY_RUNTIME_PER_MAGIC) * magic_count)),
    )
    metrics = {
        "magic_state_consumption_count": magic_count,
        "magic_state_consumption_depth": magic_depth,
        "t_count": magic_count,
        "t_depth": magic_depth,
        "runtime": runtime_wo,
        "runtime_without_topology": runtime_wo,
        "qubit_volume": 0,
        "gate_count": gate_count,
        "gate_depth": runtime_wo,
        "measurement_feedback_count": magic_count,
        "measurement_feedback_depth": magic_depth,
        "magic_factory_count": 0,
        "chip_cell_count": 0,
        "target_error": float(target_error),
        "step_time": float(step_time),
        "rotation_precision": float(rotation_precision),
        "cache_key": _surface_code_cache_key(float(target_error)),
        "generator": "proxy_formula",
        "auto_generated": True,
        "source": str(source),
        "compile_mode": SURFACE_CODE_COMPILE_MODE,
    }
    return normalize_surface_code_step_metrics(
        metrics,
        context=f"{ham_name}_Operator_{pf_label}.proxy_surface_code_step",
    )


_SURFACE_CODE_MAXPLUS_NEG_INF = -(10**15)


def _surface_code_maxplus_identity(size: int) -> List[List[int]]:
    return [
        [0 if i == j else _SURFACE_CODE_MAXPLUS_NEG_INF for j in range(size)]
        for i in range(size)
    ]


def _surface_code_maxplus_compose(
    lhs: Sequence[Sequence[int]],
    rhs: Sequence[Sequence[int]],
) -> List[List[int]]:
    size = len(lhs)
    out = [
        [_SURFACE_CODE_MAXPLUS_NEG_INF for _ in range(size)]
        for _ in range(size)
    ]
    for i in range(size):
        for k in range(size):
            lhs_ik = int(lhs[i][k])
            if lhs_ik <= _SURFACE_CODE_MAXPLUS_NEG_INF // 2:
                continue
            row = rhs[k]
            for j in range(size):
                rhs_kj = int(row[j])
                if rhs_kj <= _SURFACE_CODE_MAXPLUS_NEG_INF // 2:
                    continue
                candidate = lhs_ik + rhs_kj
                if candidate > out[i][j]:
                    out[i][j] = candidate
    return out


def _surface_code_maxplus_depth(matrix: Sequence[Sequence[int]]) -> int:
    best = 0
    for row in matrix:
        for value in row:
            if value > best:
                best = int(value)
    return best


def _surface_code_one_qubit_transfer(
    size: int,
    qubit: int,
    *,
    weight: int,
) -> List[List[int]]:
    out = _surface_code_maxplus_identity(size)
    out[int(qubit)][int(qubit)] = int(weight)
    return out


def _surface_code_two_qubit_clifford_transfer(
    size: int,
    q0: int,
    q1: int,
    *,
    weight: int,
) -> List[List[int]]:
    q0 = int(q0)
    q1 = int(q1)
    out = _surface_code_maxplus_identity(size)
    for out_qubit in (q0, q1):
        for row in range(size):
            out[row][out_qubit] = _SURFACE_CODE_MAXPLUS_NEG_INF
        out[q0][out_qubit] = int(weight)
        out[q1][out_qubit] = int(weight)
    return out


def _surface_code_embed_call_transfer(
    size: int,
    operate: Sequence[int],
    submatrix: Sequence[Sequence[int]],
) -> List[List[int]]:
    out = _surface_code_maxplus_identity(size)
    local = [int(q) for q in operate]
    for local_out, outer_out in enumerate(local):
        for row in range(size):
            out[row][outer_out] = _SURFACE_CODE_MAXPLUS_NEG_INF
        for local_in, outer_in in enumerate(local):
            out[outer_in][outer_out] = int(submatrix[local_in][local_out])
    return out


def _surface_code_ir_summary_from_opt_json(
    opt_ir_path: str | Path,
    *,
    circuit_name: str = "main",
) -> Dict[str, Any]:
    """qcsf opt 後の IR から分解済み T-count/T-depth を高速計算する。"""
    path = Path(opt_ir_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        module = json.load(f)

    circuits_raw = module.get("circuit_list")
    if not isinstance(circuits_raw, list):
        raise ValueError(f"Invalid IR JSON: missing circuit_list in {path}")

    circuit_map: Dict[str, Mapping[str, Any]] = {}
    for circuit in circuits_raw:
        if isinstance(circuit, Mapping) and isinstance(circuit.get("name"), str):
            circuit_map[str(circuit["name"])] = circuit

    if circuit_name not in circuit_map:
        raise ValueError(f"Circuit '{circuit_name}' not found in {path}")

    summaries: Dict[str, Dict[str, Any]] = {}

    def _summarize(name: str) -> Dict[str, Any]:
        cached = summaries.get(name)
        if cached is not None:
            return cached

        circuit = circuit_map.get(name)
        if circuit is None:
            raise ValueError(f"Unknown callee '{name}' in {path}")
        argument = circuit.get("argument")
        if not isinstance(argument, Mapping):
            raise ValueError(f"Invalid IR JSON: missing argument for circuit '{name}'")
        num_qubits = int(argument.get("num_qubits", 0))

        magic_matrix = _surface_code_maxplus_identity(num_qubits)
        gate_matrix = _surface_code_maxplus_identity(num_qubits)
        magic_count = 0
        gate_count = 0

        bb_list = circuit.get("bb_list")
        if not isinstance(bb_list, list):
            raise ValueError(f"Invalid IR JSON: missing bb_list for circuit '{name}'")

        for bb in bb_list:
            if not isinstance(bb, Mapping):
                continue
            inst_list = bb.get("inst_list")
            if not isinstance(inst_list, list):
                continue
            for inst in inst_list:
                if not isinstance(inst, Mapping):
                    continue
                opcode = str(inst.get("opcode"))
                if opcode in {"Return", "I"}:
                    continue
                if opcode == "Call":
                    callee = str(inst.get("callee"))
                    child = _summarize(callee)
                    operate = inst.get("operate")
                    if not isinstance(operate, list):
                        raise ValueError(
                            f"Invalid Call instruction in circuit '{name}': missing operate"
                        )
                    magic_matrix = _surface_code_maxplus_compose(
                        magic_matrix,
                        _surface_code_embed_call_transfer(
                            num_qubits,
                            operate,
                            child["magic_matrix"],
                        ),
                    )
                    gate_matrix = _surface_code_maxplus_compose(
                        gate_matrix,
                        _surface_code_embed_call_transfer(
                            num_qubits,
                            operate,
                            child["gate_matrix"],
                        ),
                    )
                    magic_count += int(child["magic_count"])
                    gate_count += int(child["gate_count"])
                    continue
                if opcode in {"H", "S", "SDag", "X", "Z"}:
                    q = int(inst["q"])
                    magic_matrix = _surface_code_maxplus_compose(
                        magic_matrix,
                        _surface_code_one_qubit_transfer(num_qubits, q, weight=0),
                    )
                    gate_matrix = _surface_code_maxplus_compose(
                        gate_matrix,
                        _surface_code_one_qubit_transfer(num_qubits, q, weight=1),
                    )
                    gate_count += 1
                    continue
                if opcode in {"T", "TDag"}:
                    q = int(inst["q"])
                    magic_matrix = _surface_code_maxplus_compose(
                        magic_matrix,
                        _surface_code_one_qubit_transfer(num_qubits, q, weight=1),
                    )
                    gate_matrix = _surface_code_maxplus_compose(
                        gate_matrix,
                        _surface_code_one_qubit_transfer(num_qubits, q, weight=1),
                    )
                    magic_count += 1
                    gate_count += 1
                    continue
                if opcode == "CX":
                    q0 = int(inst["q0"])
                    q1 = int(inst["q1"])
                    magic_matrix = _surface_code_maxplus_compose(
                        magic_matrix,
                        _surface_code_two_qubit_clifford_transfer(
                            num_qubits,
                            q0,
                            q1,
                            weight=0,
                        ),
                    )
                    gate_matrix = _surface_code_maxplus_compose(
                        gate_matrix,
                        _surface_code_two_qubit_clifford_transfer(
                            num_qubits,
                            q0,
                            q1,
                            weight=1,
                        ),
                    )
                    gate_count += 1
                    continue
                raise ValueError(
                    f"Unsupported opcode '{opcode}' in decomposed IR circuit '{name}'"
                )

        summary = {
            "num_qubits": num_qubits,
            "magic_count": int(magic_count),
            "magic_depth": int(_surface_code_maxplus_depth(magic_matrix)),
            "magic_matrix": magic_matrix,
            "gate_count": int(gate_count),
            "gate_depth": int(_surface_code_maxplus_depth(gate_matrix)),
            "gate_matrix": gate_matrix,
        }
        summaries[name] = summary
        return summary

    return _summarize(circuit_name)


def _surface_code_compose_ir_summaries(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    """2 つの逐次 IR summary を合成する。"""
    num_qubits = int(left["num_qubits"])
    if int(right["num_qubits"]) != num_qubits:
        raise ValueError("Cannot compose IR summaries with different num_qubits.")

    magic_matrix = _surface_code_maxplus_compose(
        left["magic_matrix"],
        right["magic_matrix"],
    )
    gate_matrix = _surface_code_maxplus_compose(
        left["gate_matrix"],
        right["gate_matrix"],
    )
    return {
        "num_qubits": num_qubits,
        "magic_count": int(left["magic_count"]) + int(right["magic_count"]),
        "magic_depth": int(_surface_code_maxplus_depth(magic_matrix)),
        "magic_matrix": magic_matrix,
        "gate_count": int(left["gate_count"]) + int(right["gate_count"]),
        "gate_depth": int(_surface_code_maxplus_depth(gate_matrix)),
        "gate_matrix": gate_matrix,
    }


def _surface_code_decompose_only_metrics_from_summary(
    summary: Mapping[str, Any],
    *,
    ham_name: str,
    pf_label: PFLabel,
    source: str,
    target_error: float,
    step_time: float,
    rotation_precision: float,
    compile_info_path: str | Path,
    generator: str,
) -> Dict[str, Any]:
    """IR summary から decompose-only step metrics を構築する。"""
    gate_count = int(summary["gate_count"])
    gate_depth = int(summary["gate_depth"])
    magic_count = int(summary["magic_count"])
    magic_depth = int(summary["magic_depth"])
    runtime_wo = max(gate_count, gate_depth)
    metrics = {
        "magic_state_consumption_count": magic_count,
        "magic_state_consumption_depth": magic_depth,
        "t_count": magic_count,
        "t_depth": magic_depth,
        "runtime": runtime_wo,
        "runtime_without_topology": runtime_wo,
        "qubit_volume": 0,
        "gate_count": gate_count,
        "gate_depth": gate_depth,
        "measurement_feedback_count": magic_count,
        "measurement_feedback_depth": magic_depth,
        "magic_factory_count": 0,
        "chip_cell_count": 0,
        "target_error": float(target_error),
        "step_time": float(step_time),
        "rotation_precision": float(rotation_precision),
        "cache_key": _surface_code_cache_key(float(target_error)),
        "generator": str(generator),
        "auto_generated": True,
        "source": str(source),
        "compile_mode": SURFACE_CODE_COMPILE_MODE,
        "compile_info_json": str(Path(compile_info_path).expanduser().resolve()),
    }
    return normalize_surface_code_step_metrics(
        metrics,
        context=f"{ham_name}_Operator_{pf_label}.decompose_only_surface_code_step",
    )


def _surface_code_decompose_only_step_metrics(
    opt_ir_path: str | Path,
    *,
    ham_name: str,
    pf_label: PFLabel,
    source: str,
    target_error: float,
    step_time: float,
    rotation_precision: float,
) -> Dict[str, Any]:
    """surface_code の parse+opt のみを用いた高速 step metrics。"""
    summary = _surface_code_ir_summary_from_opt_json(opt_ir_path, circuit_name="main")
    return _surface_code_decompose_only_metrics_from_summary(
        summary,
        ham_name=ham_name,
        pf_label=pf_label,
        source=source,
        target_error=target_error,
        step_time=step_time,
        rotation_precision=rotation_precision,
        compile_info_path=opt_ir_path,
        generator="decompose_only_ir",
    )


def _surface_code_rotation_precision(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
    use_original: bool = False,
    step_time: float | None = None,
) -> float:
    """surface_code parse に渡す回転合成精度を返す。"""
    mode = str(SURFACE_CODE_ROTATION_PRECISION_MODE)
    if mode == "fixed":
        precision = float(SURFACE_CODE_FIXED_ROTATION_PRECISION)
        if (not np.isfinite(precision)) or precision <= 0:
            raise ValueError(
                f"Invalid SURFACE_CODE_FIXED_ROTATION_PRECISION={precision!r}"
            )
        return precision

    if mode not in {"task_budget", "layer_linear_floor"}:
        raise ValueError(
            "SURFACE_CODE_ROTATION_PRECISION_MODE must be either "
            "'fixed', 'task_budget', or 'layer_linear_floor'."
        )

    coeff, expo = _load_compare_alpha_and_exponent(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    step_t = (
        float(step_time)
        if step_time is not None
        else _surface_code_step_time(
            ham_name,
            pf_label,
            source=source,
            target_error=float(target_error),
            use_original=use_original,
        )
    )
    qpe_factor = _qpe_iteration_factor(
        float(coeff),
        float(expo),
        float(target_error),
    )
    denominator = None
    if mode == "task_budget":
        denominator = _surface_code_rotation_count_for_precision(
            ham_name,
            pf_label,
            source=source,
            use_original=use_original,
        ) * float(qpe_factor)
    elif mode == "layer_linear_floor":
        denominator = _surface_code_rotation_layer_depth_for_precision(
            ham_name,
            pf_label,
            source=source,
            use_original=use_original,
        ) * float(qpe_factor)

    precision = (
        step_t
        * float(SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION)
        * float(target_error)
    ) / float(denominator)
    if mode == "layer_linear_floor":
        precision = max(
            float(SURFACE_CODE_ROTATION_PRECISION_FLOOR),
            float(precision),
        )
    if (not np.isfinite(precision)) or precision <= 0:
        raise ValueError(
            "Failed to compute a positive surface-code rotation precision for "
            f"{ham_name}_Operator_{pf_label}: {precision!r}"
        )
    return float(precision)


def _surface_code_log(
    message: str,
    *,
    source: str,
    ham_name: str | None = None,
    pf_label: PFLabel | None = None,
    target_error: float | None = None,
) -> None:
    """surface_code 見積もりの進捗ログを標準出力へ出す。"""
    prefix = f"[surface-code][{source}]"
    if ham_name is not None and pf_label is not None:
        prefix += f" {ham_name}_Operator_{pf_label}"
    if target_error is not None:
        prefix += f" {_surface_code_cache_key(float(target_error))}"
    print(f"{prefix}: {message}", flush=True)


def _surface_code_payload_cache_entries(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """payload 内の surface_code_step cache 一覧を返す。"""
    entries: List[Dict[str, Any]] = []
    raw_cache = payload.get("surface_code_step_cache")
    if isinstance(raw_cache, Sequence):
        for idx, item in enumerate(raw_cache):
            if not isinstance(item, Mapping):
                continue
            entries.append(
                normalize_surface_code_step_metrics(
                    item,
                    context=f"surface_code_step_cache[{idx}]",
                )
            )
    raw_latest = payload.get("surface_code_step")
    if isinstance(raw_latest, Mapping):
        latest = normalize_surface_code_step_metrics(
            raw_latest,
            context="surface_code_step",
        )
        if not any(
            entry.get("cache_key") == latest.get("cache_key")
            and entry.get("target_error") == latest.get("target_error")
            for entry in entries
        ):
            entries.append(latest)
    return entries


def _match_surface_code_step_metrics(
    metrics: Mapping[str, Any],
    *,
    target_error: float | None,
) -> bool:
    """metrics が指定 target_error に一致するかを判定する。"""
    if metrics.get("compile_mode") != SURFACE_CODE_COMPILE_MODE:
        return False
    if target_error is None:
        return True
    if metrics.get("cache_key") != _surface_code_cache_key(float(target_error)):
        return False
    stored = metrics.get("target_error")
    if stored is None:
        return False
    return math.isclose(float(stored), float(target_error), rel_tol=1e-12, abs_tol=0.0)


def _find_surface_code_step_metrics_in_payload(
    payload: Mapping[str, Any],
    *,
    target_error: float | None,
) -> Dict[str, Any] | None:
    """payload から target_error に対応する surface_code_step を探す。"""
    entries = _surface_code_payload_cache_entries(payload)
    for entry in entries:
        if _match_surface_code_step_metrics(entry, target_error=target_error):
            return dict(entry)
    raw_latest = payload.get("surface_code_step")
    if isinstance(raw_latest, Mapping) and target_error is None:
        latest = normalize_surface_code_step_metrics(
            raw_latest,
            context="surface_code_step",
        )
        if _match_surface_code_step_metrics(latest, target_error=None):
            return latest
    return None


def _surface_code_runtime_root(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
) -> Path:
    """surface_code 自動生成用の runtime/cached ファイル保存先を返す。"""
    safe_ham = re.sub(r"[^A-Za-z0-9_.-]+", "_", ham_name)
    safe_pf = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pf_label))
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", _surface_code_cache_key(target_error))
    return (
        SURFACE_CODE_CACHE_DIR
        / source
        / SURFACE_CODE_COMPILE_MODE
        / f"{safe_ham}__{safe_pf}"
        / safe_target
    )


def _surface_code_runtime_compile_info_path(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
) -> Path:
    """runtime cache 内の compile_info.json の想定パスを返す。"""
    return (
        _surface_code_runtime_root(
            ham_name,
            pf_label,
            source=source,
            target_error=target_error,
        )
        / "compile_info.json"
    )


def _extend_env_path_list(
    env: Dict[str, str],
    key: str,
    entries: Sequence[Path | str],
) -> None:
    """環境変数 PATH 系にディレクトリを前置する。"""
    normalized = [str(Path(entry)) for entry in entries if str(entry)]
    if not normalized:
        return
    current = env.get(key, "")
    parts = normalized + ([current] if current else [])
    env[key] = os.pathsep.join(parts)


def _prepare_surface_code_runtime_env(
    runtime_root: Path,
    *,
    library_dirs: Sequence[Path | str] = (),
    rotation_precision: float | None = None,
) -> Dict[str, str]:
    """qiskit/qcsf 実行時の writable env を用意する。"""
    tmp_dir = runtime_root / "tmp"
    mpl_dir = runtime_root / "mplconfig"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    _extend_env_path_list(env, "LD_LIBRARY_PATH", library_dirs)
    _extend_env_path_list(env, "DYLD_LIBRARY_PATH", library_dirs)
    gridsynth_path = Path(SURFACE_CODE_GRIDSYNTH_PATH).expanduser().resolve()
    if gridsynth_path.exists():
        env["GRIDSYNTH_PATH"] = str(gridsynth_path)
    if rotation_precision is not None:
        env["QSVT_OPENQASM_ROTATION_PRECISION"] = f"{float(rotation_precision):.17g}"
    if (
        SURFACE_CODE_COMPILE_MODE not in {"proxy", "decompose_only"}
        and SURFACE_CODE_COMPILE_SKIP_REDUNDANT_IR_PREPROCESS
    ):
        env["QSVT_COMPILE_SKIP_IR_PREPROCESS"] = "1"
    if (
        SURFACE_CODE_COMPILE_MODE not in {"proxy", "decompose_only"}
        and SURFACE_CODE_COMPILE_SKIP_OUTPUT
    ):
        env["QSVT_COMPILE_SKIP_OUTPUT"] = "1"
    return env


def _surface_code_opt_passes() -> List[str]:
    """現在の compile mode に応じた qcsf opt pass 列を返す。"""
    if SURFACE_CODE_COMPILE_MODE == "decompose_only":
        return [
            "ir::decompose_inst",
            "ir::ignore_global_phase",
            "ir::delete_consecutive_same_pauli",
            "ir::delete_opt_hint",
        ]
    return [
        "ir::recursive_inliner",
        "ir::static_condition_pruning",
        "ir::decompose_inst",
        "ir::ignore_global_phase",
        "ir::delete_consecutive_same_pauli",
        "ir::delete_opt_hint",
    ]


def _parse_molecule_type_from_ham_name(ham_name: str) -> int:
    """ham_name から H-chain 長を抽出する。"""
    match = re.match(r"H(?P<chain>\d+)_", str(ham_name))
    if match is None:
        raise ValueError(f"Could not parse molecule_type from ham_name={ham_name!r}")
    return int(match.group("chain"))


def _parse_distance_from_ham_name(ham_name: str) -> float:
    """ham_name から H-chain 距離を抽出する。"""
    match = re.search(r"_distance_(?P<dist>\d+)", str(ham_name))
    if match is None:
        return float(DEFAULT_DISTANCE)
    return float(int(match.group("dist"))) / 100.0


def _surface_code_integrals(
    molecule_type: int,
    *,
    distance: float,
    basis: str,
) -> tuple[float, Any, Any]:
    """surface_code 用 circuit 生成に必要な分子積分を返す。"""
    import pyscf
    from pyscf import gto, scf
    from .chemistry_hamiltonian import geo

    geometry, multiplicity, charge = geo(molecule_type, distance)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.spin = multiplicity - 1
    mol.charge = charge
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    constant = float(mf.energy_nuc())
    mo_coeff = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body = mo_coeff.T @ h_core @ mo_coeff
    eri_mo = pyscf.ao2mo.kernel(mf.mol, mo_coeff)
    eri_mo = pyscf.ao2mo.restore(1, eri_mo, mo_coeff.shape[0])
    two_body = np.asarray(eri_mo.transpose(0, 2, 3, 1), order="C")
    return constant, one_body, two_body


def _symmetrize_two_body_for_surface_code(two_body: Any) -> Any:
    """DF circuit 再構築用に two-body tensor を対称化する。"""
    t = np.asarray(two_body)
    parts = [
        t,
        np.transpose(t, (1, 0, 2, 3)),
        np.transpose(t, (0, 1, 3, 2)),
        np.transpose(t, (1, 0, 3, 2)),
        np.transpose(t, (2, 3, 0, 1)),
        np.transpose(t, (3, 2, 0, 1)),
        np.transpose(t, (2, 3, 1, 0)),
        np.transpose(t, (3, 2, 1, 0)),
    ]
    sym = sum(parts) / len(parts)
    return np.real_if_close(sym, tol=1e-8)


def _surface_code_artifact_path(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> Path:
    """surface_code step metrics を保存する artifact パスを返す。"""
    artifact_name = f"{ham_name}_Operator_{pf_label}"
    return surface_code_step_dir(source, use_original=use_original) / artifact_name


def _attach_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    metrics: Mapping[str, Any],
    *,
    source: str,
    use_original: bool = False,
) -> Dict[str, Any]:
    """artifact に surface_code step metrics を保存する。"""
    artifact_name = f"{ham_name}_Operator_{pf_label}"
    payload = _load_surface_code_artifact_payload(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    metrics_with_mode = dict(metrics)
    metrics_with_mode.setdefault("compile_mode", SURFACE_CODE_COMPILE_MODE)
    normalized = normalize_surface_code_step_metrics(
        metrics_with_mode,
        context=f"{artifact_name}.surface_code_step",
    )
    cache_entries = _surface_code_payload_cache_entries(payload)
    filtered_entries = []
    for entry in cache_entries:
        if (
            normalized.get("target_error") is not None
            and entry.get("target_error") is not None
            and math.isclose(
                float(entry["target_error"]),
                float(normalized["target_error"]),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            continue
        filtered_entries.append(entry)
    filtered_entries.append(normalized)
    payload["surface_code_step"] = normalized
    payload["surface_code_step_cache"] = filtered_entries
    path = _surface_code_artifact_path(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    _surface_code_log(
        "saved artifact cache",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=normalized.get("target_error"),
    )
    return normalized


def attach_df_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """DF artifact に surface_code step metrics を保存する。"""
    return _attach_surface_code_step_metrics(
        ham_name,
        pf_label,
        metrics,
        source="df",
    )


def attach_df_surface_code_step_metrics_from_compile_info_json(
    ham_name: str,
    pf_label: PFLabel,
    compile_info_path: str | Path,
) -> Dict[str, Any]:
    """compile_info.json を読み、DF artifact へ surface_code step metrics を保存する。"""
    metrics = surface_code_step_metrics_from_compile_info_json(compile_info_path)
    return attach_df_surface_code_step_metrics(ham_name, pf_label, metrics)


def attach_grouped_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    metrics: Mapping[str, Any],
    *,
    use_original: bool = False,
) -> Dict[str, Any]:
    """Grouped artifact に surface_code step metrics を保存する。"""
    return _attach_surface_code_step_metrics(
        ham_name,
        pf_label,
        metrics,
        source="gr",
        use_original=use_original,
    )


def attach_grouped_surface_code_step_metrics_from_compile_info_json(
    ham_name: str,
    pf_label: PFLabel,
    compile_info_path: str | Path,
    *,
    use_original: bool = False,
) -> Dict[str, Any]:
    """compile_info.json を読み、grouped artifact へ surface_code step metrics を保存する。"""
    metrics = surface_code_step_metrics_from_compile_info_json(compile_info_path)
    return attach_grouped_surface_code_step_metrics(
        ham_name,
        pf_label,
        metrics,
        use_original=use_original,
    )


def _extract_surface_code_payload_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """surface_code 関連キーだけを payload から抜き出す。"""
    extracted: Dict[str, Any] = {}
    if "surface_code_step" in payload:
        extracted["surface_code_step"] = payload["surface_code_step"]
    if "surface_code_step_cache" in payload:
        extracted["surface_code_step_cache"] = payload["surface_code_step_cache"]
    return extracted


def _load_surface_code_legacy_payload(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> Dict[str, Any]:
    """旧 trotter_expo_coeff artifact に埋め込まれた surface_code cache を読む。"""
    try:
        if source == "df":
            payload = _load_df_artifact_payload(ham_name, pf_label)
        elif source == "gr":
            payload = _load_grouped_artifact_payload(
                ham_name,
                pf_label,
                use_original=use_original,
            )
        else:
            raise ValueError(f"Unsupported source: {source}")
    except Exception:
        return {}
    return _extract_surface_code_payload_fields(payload)


def _load_surface_code_artifact_payload(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> Dict[str, Any]:
    """surface_code step metrics の保存先 payload を読む。"""
    path = _surface_code_artifact_path(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    if path.exists():
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid surface_code artifact payload: {path}")
        return dict(payload)
    return _load_surface_code_legacy_payload(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )


def _surface_code_step_time(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
    use_original: bool = False,
) -> float:
    """target_error に対応する 1-step 時間幅 t を返す。"""
    coeff, expo = _load_compare_alpha_and_exponent(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    return float((target_error / float(coeff) * (float(expo) + 1.0)) ** (1.0 / float(expo)))


def _build_df_surface_code_step_circuit(
    ham_name: str,
    pf_label: PFLabel,
    *,
    step_time: float,
) -> Any:
    """DF 1-step の Qiskit 回路を再構築する。"""
    from openfermion.chem.molecular_data import spinorb_from_spatial

    from .df_trotter.circuit import build_df_trotter_circuit
    from .df_trotter.decompose import df_decompose_from_integrals
    from .df_trotter.model import Block
    from .df_trotter.ops import build_df_blocks, build_one_body_gaussian_block

    molecule_type = _parse_molecule_type_from_ham_name(ham_name)
    distance = _parse_distance_from_ham_name(ham_name)
    constant, one_body, two_body = _surface_code_integrals(
        molecule_type,
        distance=distance,
        basis=DEFAULT_BASIS,
    )

    rank_fraction = get_df_rank_fraction_for_molecule(int(molecule_type))
    rank = None
    tol = None
    if rank_fraction is not None:
        n_spatial = int(np.asarray(one_body).shape[0])
        full_rank = int(n_spatial**2)
        if float(rank_fraction) >= 1.0:
            rank = full_rank
            tol = 0.0
        else:
            rank = int(round(full_rank * float(rank_fraction)))
            rank = max(1, min(rank, full_rank))

    two_body = _symmetrize_two_body_for_surface_code(two_body)
    one_body_spin, _two_body_spin = spinorb_from_spatial(one_body, two_body * 0.5)
    raw_model = df_decompose_from_integrals(
        one_body,
        two_body,
        constant=constant,
        rank=rank,
        tol=tol,
    )
    model = raw_model.hermitize()
    h_eff = one_body_spin + model.one_body_correction

    df_blocks = build_df_blocks(model)
    blocks: List[Block] = []
    one_body_block = build_one_body_gaussian_block(h_eff)
    blocks.append(Block.from_one_body_gaussian(one_body_block))
    blocks.extend(Block.from_df(block) for block in df_blocks)

    qc = build_df_trotter_circuit(
        blocks,
        time=float(step_time),
        num_qubits=model.N,
        pf_label=normalize_pf_label(pf_label),
        energy_shift=0.0,
    )
    qc.global_phase = 0.0
    return qc


def _build_grouped_surface_code_step_circuit(
    ham_name: str,
    pf_label: PFLabel,
    *,
    step_time: float,
) -> Any:
    """Grouped 1-step の Qiskit 回路を再構築する。"""
    from qiskit import QuantumCircuit

    from openfermion import InteractionOperator
    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial

    from .chemistry_hamiltonian import min_hamiltonian_grouper
    from .qiskit_time_evolution_grouping import w_trotter_grouper
    from .qiskit_time_evolution_pyscf import _build_grouped_jw_list

    molecule_type = _parse_molecule_type_from_ham_name(ham_name)
    distance = _parse_distance_from_ham_name(ham_name)
    constant, one_body, two_body = _surface_code_integrals(
        molecule_type,
        distance=distance,
        basis=DEFAULT_BASIS,
    )

    if molecule_type in (2, 3):
        h1s, h2s = spinorb_from_spatial(one_body, two_body * 0.5)
        jw_hamiltonian = jordan_wigner(
            get_fermion_operator(InteractionOperator(constant, h1s, h2s))
        )
        num_qubits = int(h1s.shape[0])
        grouped_ops, _grouped_name = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
        commuting_cliques = [[op] for op in grouped_ops]
    else:
        commuting_cliques = _build_grouped_jw_list(constant, one_body, two_body)
        num_qubits = int(2 * np.asarray(one_body).shape[0])

    qc = QuantumCircuit(int(num_qubits))
    w_trotter_grouper(
        qc,
        commuting_cliques,
        float(step_time),
        int(num_qubits),
        normalize_pf_label(pf_label),
    )
    qc.global_phase = 0.0
    return qc


def _surface_code_basis_circuit_from_circuit(
    qc: Any,
    *,
    runtime_root: Path,
) -> Any:
    """Qiskit 回路を surface_code 用 basis gate 回路へ分解する。"""
    env = _prepare_surface_code_runtime_env(runtime_root)
    os.environ.update(
        {
            "TMPDIR": env["TMPDIR"],
            "TMP": env["TMP"],
            "TEMP": env["TEMP"],
            "MPLCONFIGDIR": env["MPLCONFIGDIR"],
        }
    )
    import tempfile

    tempfile.tempdir = env["TMPDIR"]

    from .qiskit_time_evolution_utils import _decompose_to_basis

    qc_basis = _decompose_to_basis(
        qc,
        basis_gates=SURFACE_CODE_QASM_BASIS_GATES,
        decompose_reps=int(SURFACE_CODE_QASM_DECOMPOSE_REPS),
        optimization_level=0,
    )
    qc_basis.global_phase = 0.0
    return qc_basis


def _surface_code_qasm_text_from_basis_circuit(qc_basis: Any) -> str:
    """basis gate 回路を OpenQASM2 文字列へ変換する。"""
    try:
        from qiskit import qasm2

        return str(qasm2.dumps(qc_basis))
    except Exception:
        if hasattr(qc_basis, "qasm"):
            return str(qc_basis.qasm())
        raise RuntimeError("Failed to export circuit to OpenQASM2.")


def _surface_code_qasm_text_from_circuit(
    qc: Any,
    *,
    runtime_root: Path,
) -> str:
    """Qiskit 回路を OpenQASM2 文字列へ変換する。"""
    qc_basis = _surface_code_basis_circuit_from_circuit(qc, runtime_root=runtime_root)
    return _surface_code_qasm_text_from_basis_circuit(qc_basis)


def _surface_code_chunk_qasm_paths(
    qc_basis: Any,
    *,
    runtime_root: Path,
    max_ops: int,
) -> List[Dict[str, Any]]:
    """basis gate 回路を連続チャンクへ分割し、各 chunk の QASM を保存する。"""
    if int(max_ops) <= 0:
        raise ValueError(f"max_ops must be positive: {max_ops}")

    from qiskit import QuantumCircuit

    qubit_ids = {id(bit): idx for idx, bit in enumerate(qc_basis.qubits)}
    clbit_ids = {id(bit): idx for idx, bit in enumerate(getattr(qc_basis, "clbits", []))}
    num_qubits = len(qc_basis.qubits)
    num_clbits = len(getattr(qc_basis, "clbits", []))
    chunks_root = runtime_root / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    chunk_specs: List[Dict[str, Any]] = []
    chunk_index = 0
    current_count = 0
    current = QuantumCircuit(num_qubits, num_clbits)
    current.global_phase = 0.0

    def _flush_chunk(circuit: Any, index: int, op_count: int) -> None:
        chunk_dir = chunks_root / f"chunk_{index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        qasm_path = chunk_dir / "step.qasm"
        ir_path = chunk_dir / "step_ir.json"
        opt_path = chunk_dir / "step_opt.json"
        opt_yaml = chunk_dir / "opt.yaml"
        qasm_path.write_text(
            _surface_code_qasm_text_from_basis_circuit(circuit),
            encoding="utf-8",
        )
        opt_yaml.write_text(
            "\n".join(
                [
                    f"input: {ir_path}",
                    "circuit: main",
                    f"output: {opt_path}",
                    "pass:",
                    *[f"- {name}" for name in _surface_code_opt_passes()],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        chunk_specs.append(
            {
                "chunk_index": int(index),
                "chunk_dir": chunk_dir,
                "qasm_path": qasm_path,
                "ir_path": ir_path,
                "opt_path": opt_path,
                "opt_yaml": opt_yaml,
                "op_count": int(op_count),
            }
        )

    for inst in qc_basis.data:
        op = inst.operation
        qargs = [current.qubits[qubit_ids[id(q)]] for q in inst.qubits]
        cargs = [current.clbits[clbit_ids[id(c)]] for c in inst.clbits]
        try:
            op_copy = op.copy()
        except Exception:
            op_copy = op
        current.append(op_copy, qargs, cargs)
        current_count += 1
        if current_count >= int(max_ops):
            _flush_chunk(current, chunk_index, current_count)
            chunk_index += 1
            current = QuantumCircuit(num_qubits, num_clbits)
            current.global_phase = 0.0
            current_count = 0

    if current_count > 0:
        _flush_chunk(current, chunk_index, current_count)

    return chunk_specs


def _run_surface_code_command(
    cmd: Sequence[str],
    *,
    runtime_root: Path,
    rotation_precision: float | None = None,
) -> None:
    """surface_code CLI を実行し、失敗時は詳細付きで例外化する。"""
    binary_path = Path(cmd[0]).expanduser().resolve()
    env = _prepare_surface_code_runtime_env(
        runtime_root,
        library_dirs=(binary_path.parent,),
        rotation_precision=rotation_precision,
    )
    completed = subprocess.run(
        list(cmd),
        cwd=str(runtime_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    details = "\n".join(part for part in (stdout, stderr) if part)
    if (
        "error while loading shared libraries" in details
        or "GLIBC_" in details
        or "GLIBCXX_" in details
    ):
        raise RuntimeError(
            "surface_code qcsf binary is not runnable on this machine. "
            f"Binary: {binary_path}. "
            f"Added {binary_path.parent} to LD_LIBRARY_PATH automatically, "
            "but the remaining loader error indicates that either "
            "`libqsvt.so` cannot be loaded or the binary/library was built "
            "against a newer system ABI (for example GLIBC/GLIBCXX) than this "
            "environment provides. Point `SURFACE_CODE_QCSF_PATH` to a "
            "compatible local build, or generate `compile_info.json` outside "
            "this environment and attach it with "
            "`attach_*_surface_code_step_metrics_from_compile_info_json(...)`."
            + (f"\n{details}" if details else "")
        )
    raise RuntimeError(
        f"surface_code command failed (code={completed.returncode}): {' '.join(cmd)}"
        + (f"\n{details}" if details else "")
    )


def _run_surface_code_command_logged(
    cmd: Sequence[str],
    *,
    runtime_root: Path,
    source: str,
    ham_name: str,
    pf_label: PFLabel,
    target_error: float,
    stage_name: str,
    rotation_precision: float | None = None,
) -> None:
    """surface_code CLI を実行し、開始/完了と経過時間をログ出力する。"""
    _surface_code_log(
        f"running {stage_name}",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    t0 = time.perf_counter()
    _run_surface_code_command(
        cmd,
        runtime_root=runtime_root,
        rotation_precision=rotation_precision,
    )
    elapsed = time.perf_counter() - t0
    _surface_code_log(
        f"finished {stage_name} ({elapsed:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )


def _ensure_surface_code_binary_usable(
    qcsf_path: Path,
    *,
    runtime_root: Path,
    rotation_precision: float | None = None,
) -> None:
    """qcsf 実行バイナリがこの環境で起動可能か事前確認する。"""
    _run_surface_code_command(
        [str(qcsf_path), "--help"],
        runtime_root=runtime_root,
        rotation_precision=rotation_precision,
    )


def _surface_code_process_decompose_chunk(
    *,
    qcsf_path: str,
    runtime_root: str,
    qasm_path: str,
    ir_path: str,
    opt_path: str,
    opt_yaml: str,
    rotation_precision: float,
) -> Dict[str, Any]:
    """decompose-only 用 chunk を parse+opt して IR summary を返す。"""
    runtime_root_path = Path(runtime_root).expanduser().resolve()
    _run_surface_code_command(
        [
            str(qcsf_path),
            "parse",
            "--input",
            str(qasm_path),
            "--output",
            str(ir_path),
            "--format",
            "OpenQASM2",
            "--verbose",
        ],
        runtime_root=runtime_root_path,
        rotation_precision=float(rotation_precision),
    )
    _run_surface_code_command(
        [
            str(qcsf_path),
            "opt",
            "--pipeline",
            str(opt_yaml),
            "--verbose",
        ],
        runtime_root=runtime_root_path,
        rotation_precision=float(rotation_precision),
    )
    return _surface_code_ir_summary_from_opt_json(opt_path, circuit_name="main")


def _surface_code_chunked_decompose_only_step_metrics(
    qc_basis: Any,
    *,
    runtime_root: Path,
    qcsf_path: Path,
    ham_name: str,
    pf_label: PFLabel,
    source: str,
    target_error: float,
    step_time: float,
    rotation_precision: float,
) -> Dict[str, Any]:
    """巨大回路を chunk に分割して decompose-only 指標を集計する。"""
    max_ops = int(SURFACE_CODE_DECOMPOSE_CHUNK_MAX_OPS)
    chunk_specs = _surface_code_chunk_qasm_paths(
        qc_basis,
        runtime_root=runtime_root,
        max_ops=max_ops,
    )
    if not chunk_specs:
        raise ValueError("No chunk generated for decompose-only surface-code run.")

    workers = max(1, min(int(SURFACE_CODE_DECOMPOSE_MAX_WORKERS), len(chunk_specs)))
    _surface_code_log(
        f"chunked decompose_only: {len(qc_basis.data)} basis ops -> "
        f"{len(chunk_specs)} chunks (max_ops={max_ops}, workers={workers})",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )

    started = time.perf_counter()
    summaries_by_index: Dict[int, Dict[str, Any]] = {}
    if workers == 1 or len(chunk_specs) == 1:
        for spec in chunk_specs:
            summaries_by_index[int(spec["chunk_index"])] = _surface_code_process_decompose_chunk(
                qcsf_path=str(qcsf_path),
                runtime_root=str(spec["chunk_dir"]),
                qasm_path=str(spec["qasm_path"]),
                ir_path=str(spec["ir_path"]),
                opt_path=str(spec["opt_path"]),
                opt_yaml=str(spec["opt_yaml"]),
                rotation_precision=float(rotation_precision),
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _surface_code_process_decompose_chunk,
                    qcsf_path=str(qcsf_path),
                    runtime_root=str(spec["chunk_dir"]),
                    qasm_path=str(spec["qasm_path"]),
                    ir_path=str(spec["ir_path"]),
                    opt_path=str(spec["opt_path"]),
                    opt_yaml=str(spec["opt_yaml"]),
                    rotation_precision=float(rotation_precision),
                ): int(spec["chunk_index"])
                for spec in chunk_specs
            }
            for future in as_completed(future_map):
                chunk_index = future_map[future]
                summaries_by_index[chunk_index] = future.result()

    ordered_indices = sorted(summaries_by_index.keys())
    total_summary = summaries_by_index[ordered_indices[0]]
    for chunk_index in ordered_indices[1:]:
        total_summary = _surface_code_compose_ir_summaries(
            total_summary,
            summaries_by_index[chunk_index],
        )

    summary_path = runtime_root / "step_opt_chunked_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_chunks": len(chunk_specs),
                "max_ops": max_ops,
                "workers": workers,
                "summary": total_summary,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    _surface_code_log(
        f"finished chunked decompose_only ({time.perf_counter() - started:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    return _surface_code_decompose_only_metrics_from_summary(
        total_summary,
        ham_name=ham_name,
        pf_label=pf_label,
        source=source,
        target_error=target_error,
        step_time=step_time,
        rotation_precision=rotation_precision,
        compile_info_path=summary_path,
        generator="decompose_only_ir_chunked",
    )


def _generate_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
    use_original: bool = False,
) -> Dict[str, Any]:
    """Qiskit -> OpenQASM2 -> qcsf の流れで step metrics を生成する。"""
    _surface_code_log(
        "auto-generate start",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    step_time = _surface_code_step_time(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
        use_original=use_original,
    )
    _surface_code_log(
        f"computed step_time={step_time:.6e}",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    rotation_precision = _surface_code_rotation_precision(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
        use_original=use_original,
        step_time=float(step_time),
    )
    _surface_code_log(
        "rotation_precision="
        f"{rotation_precision:.6e} mode={SURFACE_CODE_ROTATION_PRECISION_MODE}",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    if SURFACE_CODE_COMPILE_MODE == "proxy":
        _surface_code_log(
            "using proxy surface-code estimator",
            source=source,
            ham_name=ham_name,
            pf_label=pf_label,
            target_error=target_error,
        )
        return _surface_code_proxy_step_metrics(
            ham_name,
            pf_label,
            source=source,
            target_error=float(target_error),
            step_time=float(step_time),
            rotation_precision=float(rotation_precision),
            use_original=use_original,
        )
    qcsf_path = Path(SURFACE_CODE_QCSF_PATH).expanduser().resolve()
    if not qcsf_path.exists():
        raise FileNotFoundError(f"surface_code binary not found: {qcsf_path}")
    runtime_root = _surface_code_runtime_root(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    _surface_code_log(
        "checking qcsf binary",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    t0 = time.perf_counter()
    _ensure_surface_code_binary_usable(
        qcsf_path,
        runtime_root=runtime_root,
        rotation_precision=rotation_precision,
    )
    _surface_code_log(
        f"qcsf binary ok ({time.perf_counter() - t0:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )

    _surface_code_log(
        f"building {source} step circuit",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    t0 = time.perf_counter()
    if source == "df":
        qc = _build_df_surface_code_step_circuit(
            ham_name,
            pf_label,
            step_time=float(step_time),
        )
    elif source == "gr":
        qc = _build_grouped_surface_code_step_circuit(
            ham_name,
            pf_label,
            step_time=float(step_time),
        )
    else:
        raise ValueError(f"Unsupported source: {source}")
    _surface_code_log(
        f"built {source} step circuit ({time.perf_counter() - t0:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )

    _surface_code_log(
        "building basis-gate circuit",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    t0 = time.perf_counter()
    qc_basis = _surface_code_basis_circuit_from_circuit(qc, runtime_root=runtime_root)
    _surface_code_log(
        f"built basis-gate circuit ({time.perf_counter() - t0:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    basis_op_count = len(qc_basis.data)
    if (
        SURFACE_CODE_COMPILE_MODE == "decompose_only"
        and basis_op_count > int(SURFACE_CODE_DECOMPOSE_CHUNK_MAX_OPS)
    ):
        return _surface_code_chunked_decompose_only_step_metrics(
            qc_basis,
            runtime_root=runtime_root,
            qcsf_path=qcsf_path,
            ham_name=ham_name,
            pf_label=pf_label,
            source=source,
            target_error=float(target_error),
            step_time=float(step_time),
            rotation_precision=float(rotation_precision),
        )

    _surface_code_log(
        "exporting OpenQASM2",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    t0 = time.perf_counter()
    qasm_text = _surface_code_qasm_text_from_basis_circuit(qc_basis)
    _surface_code_log(
        f"exported OpenQASM2 ({time.perf_counter() - t0:.1f}s)",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    qasm_path = runtime_root / "step.qasm"
    ir_path = runtime_root / "step_ir.json"
    opt_path = runtime_root / "step_opt.json"
    asm_path = runtime_root / "step.asm"
    compile_info_path = runtime_root / "compile_info.json"
    opt_yaml = runtime_root / "opt.yaml"

    qasm_path.write_text(qasm_text, encoding="utf-8")
    opt_yaml.write_text(
        "\n".join(
            [
                f"input: {ir_path}",
                "circuit: main",
                f"output: {opt_path}",
                "pass:",
                *[f"- {name}" for name in _surface_code_opt_passes()],
                "",
            ]
        ),
        encoding="utf-8",
    )

    _run_surface_code_command_logged(
        [
            str(qcsf_path),
            "parse",
            "--input",
            str(qasm_path),
            "--output",
            str(ir_path),
            "--format",
            "OpenQASM2",
            "--verbose",
        ],
        runtime_root=runtime_root,
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
        stage_name="qcsf parse",
        rotation_precision=rotation_precision,
    )
    _run_surface_code_command_logged(
        [
            str(qcsf_path),
            "opt",
            "--pipeline",
            str(opt_yaml),
            "--verbose",
        ],
        runtime_root=runtime_root,
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
        stage_name="qcsf opt",
        rotation_precision=rotation_precision,
    )
    if SURFACE_CODE_COMPILE_MODE == "decompose_only":
        _surface_code_log(
            "using decomposed IR estimator",
            source=source,
            ham_name=ham_name,
            pf_label=pf_label,
            target_error=target_error,
        )
        return _surface_code_decompose_only_step_metrics(
            opt_path,
            ham_name=ham_name,
            pf_label=pf_label,
            source=source,
            target_error=float(target_error),
            step_time=float(step_time),
            rotation_precision=float(rotation_precision),
        )

    topology_path = Path(SURFACE_CODE_TOPOLOGY_PATH).expanduser().resolve()
    if not topology_path.exists():
        raise FileNotFoundError(f"surface_code topology file not found: {topology_path}")
    compile_yaml = runtime_root / "compile.yaml"
    compile_yaml.write_text(
        "\n".join(
            [
                "source: IR",
                f"input: {opt_path}",
                "circuit: main",
                "target: FTQC",
                f"output: {asm_path}",
                f"ftqc_topology: {topology_path}",
                f"ftqc_machine_type: {SURFACE_CODE_MACHINE_TYPE}",
                f"ftqc_magic_generation_period: {int(SURFACE_CODE_MAGIC_GENERATION_PERIOD)}",
                f"ftqc_maximum_magic_state_stock: {int(SURFACE_CODE_MAX_MAGIC_STATE_STOCK)}",
                f"ftqc_entanglement_generation_period: {int(SURFACE_CODE_ENTANGLEMENT_GENERATION_PERIOD)}",
                f"ftqc_maximum_entangled_state_stock: {int(SURFACE_CODE_MAX_ENTANGLED_STATE_STOCK)}",
                f"ftqc_reaction_time: {int(SURFACE_CODE_REACTION_TIME)}",
                f"ftqc_dump_compile_info_to_json: {compile_info_path}",
                "ftqc_pass:",
                "  - ftqc::init_compile_info",
                "  - ftqc::calc_info_without_topology",
                "  - ftqc::dump_compile_info",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _run_surface_code_command_logged(
        [
            str(qcsf_path),
            "compile",
            "--pipeline",
            str(compile_yaml),
            "--verbose",
        ],
        runtime_root=runtime_root,
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
        stage_name="qcsf compile",
        rotation_precision=rotation_precision,
    )

    metrics = surface_code_step_metrics_from_compile_info_json(compile_info_path)
    metrics["target_error"] = float(target_error)
    metrics["step_time"] = float(step_time)
    metrics["cache_key"] = _surface_code_cache_key(float(target_error))
    metrics["generator"] = "auto_qcsf"
    metrics["auto_generated"] = True
    metrics["source"] = str(source)
    metrics["compile_mode"] = SURFACE_CODE_COMPILE_MODE
    metrics["rotation_precision"] = float(rotation_precision)
    normalized = normalize_surface_code_step_metrics(
        metrics,
        context=f"{ham_name}_Operator_{pf_label}.generated_surface_code_step",
    )
    _surface_code_log(
        "auto-generate done",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    return normalized


def _auto_populate_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float,
    use_original: bool = False,
) -> Dict[str, Any]:
    """surface_code_step を自動生成して artifact に保存する。"""
    _surface_code_log(
        "artifact cache miss -> generating",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    metrics = _generate_surface_code_step_metrics(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
        use_original=use_original,
    )
    return _attach_surface_code_step_metrics(
        ham_name,
        pf_label,
        metrics,
        source=source,
        use_original=use_original,
    )


def _load_surface_code_step_metrics_from_runtime_cache(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float | None,
    use_original: bool = False,
    attach_to_artifact: bool = True,
) -> Dict[str, Any] | None:
    """runtime cache の compile_info.json を artifact cache へ取り込む。"""
    if target_error is None:
        return None

    runtime_root = _surface_code_runtime_root(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
    )
    compile_info_path = _surface_code_runtime_compile_info_path(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
    )
    if not compile_info_path.exists():
        if runtime_root.exists():
            _surface_code_log(
                "runtime cache exists but compile_info.json is missing",
                source=source,
                ham_name=ham_name,
                pf_label=pf_label,
                target_error=target_error,
            )
        return None

    _surface_code_log(
        "runtime cache hit -> importing compile_info.json",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    metrics = surface_code_step_metrics_from_compile_info_json(compile_info_path)
    metrics["target_error"] = float(target_error)
    metrics["step_time"] = _surface_code_step_time(
        ham_name,
        pf_label,
        source=source,
        target_error=float(target_error),
        use_original=use_original,
    )
    metrics["cache_key"] = _surface_code_cache_key(float(target_error))
    metrics["generator"] = "runtime_cache"
    metrics["auto_generated"] = True
    metrics["source"] = str(source)
    metrics["compile_mode"] = SURFACE_CODE_COMPILE_MODE
    metrics.setdefault(
        "rotation_precision",
        _surface_code_rotation_precision(
            ham_name,
            pf_label,
            source=source,
            target_error=float(target_error),
            use_original=use_original,
            step_time=float(metrics["step_time"]),
        ),
    )
    normalized = normalize_surface_code_step_metrics(
        metrics,
        context=f"{ham_name}_Operator_{pf_label}.runtime_surface_code_step",
    )
    if not attach_to_artifact:
        return normalized
    return _attach_surface_code_step_metrics(
        ham_name,
        pf_label,
        normalized,
        source=source,
        use_original=use_original,
    )


def _load_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float | None = None,
    auto_generate: bool = SURFACE_CODE_AUTO_POPULATE,
    use_original: bool = False,
) -> Dict[str, Any]:
    """artifact から保存済み surface_code step metrics を読む。"""
    artifact_name = f"{ham_name}_Operator_{pf_label}"
    _surface_code_log(
        "loading step metrics",
        source=source,
        ham_name=ham_name,
        pf_label=pf_label,
        target_error=target_error,
    )
    payload = _load_surface_code_artifact_payload(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    cached = _find_surface_code_step_metrics_in_payload(
        payload,
        target_error=target_error,
    )
    if cached is not None:
        _surface_code_log(
            "artifact cache hit",
            source=source,
            ham_name=ham_name,
            pf_label=pf_label,
            target_error=target_error,
        )
        return cached
    runtime_cached = _load_surface_code_step_metrics_from_runtime_cache(
        ham_name,
        pf_label,
        source=source,
        target_error=target_error,
        use_original=use_original,
        attach_to_artifact=True,
    )
    if runtime_cached is not None:
        return runtime_cached
    if auto_generate:
        return _auto_populate_surface_code_step_metrics(
            ham_name,
            pf_label,
            source=source,
            target_error=float(target_error or TARGET_ERROR),
            use_original=use_original,
        )
    raise ValueError(
        f"{source} artifact missing surface_code_step: {artifact_name}"
    )


def _load_df_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    target_error: float | None = None,
    auto_generate: bool = SURFACE_CODE_AUTO_POPULATE,
) -> Dict[str, Any]:
    """DF artifact から保存済み surface_code step metrics を読む。"""
    return _load_surface_code_step_metrics(
        ham_name,
        pf_label,
        source="df",
        target_error=target_error,
        auto_generate=auto_generate,
    )


def _load_grouped_surface_code_step_metrics(
    ham_name: str,
    pf_label: PFLabel,
    *,
    target_error: float | None = None,
    auto_generate: bool = SURFACE_CODE_AUTO_POPULATE,
    use_original: bool = False,
) -> Dict[str, Any]:
    """Grouped artifact から保存済み surface_code step metrics を読む。"""
    return _load_surface_code_step_metrics(
        ham_name,
        pf_label,
        source="gr",
        target_error=target_error,
        auto_generate=auto_generate,
        use_original=use_original,
    )


def backfill_surface_code_step_cache_from_runtime_cache(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    source: str = "gr",
    target_error: float = TARGET_ERROR,
    use_original: bool = False,
    error_on_missing: bool = False,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """既存 runtime cache から artifact の surface_code_step_cache を埋める。"""
    if Hchain < 3:
        raise ValueError(
            "Hchain must be >= 3 for surface_code step-cache backfill."
        )
    _surface_code_log(
        f"backfill start Hchain={Hchain}, pf={list(n_w_list)}",
        source=source,
        target_error=target_error,
    )
    chain_list = [i for i in range(3, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in chain_list]
    num_qubits = [2 * i for i in chain_list]
    results: Dict[int, Dict[PFLabel, Dict[str, Any]]] = {}
    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"
        per_pf: Dict[PFLabel, Dict[str, Any]] = {}
        for pf_label in n_w_list:
            try:
                per_pf[pf_label] = _load_surface_code_step_metrics(
                    ham_name,
                    pf_label,
                    source=source,
                    target_error=float(target_error),
                    auto_generate=False,
                    use_original=use_original,
                )
            except Exception as exc:
                if error_on_missing:
                    raise
                print(
                    "[surface-code][backfill skip "
                    f"{source}] {ham_name}_Operator_{pf_label}: {exc}"
                )
        if per_pf:
            results[qubits] = per_pf
    _surface_code_log(
        f"backfill done: {sum(len(v) for v in results.values())} entries",
        source=source,
        target_error=target_error,
    )
    return results


def backfill_grouped_surface_code_step_cache_from_runtime_cache(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    target_error: float = TARGET_ERROR,
    use_original: bool = False,
    error_on_missing: bool = False,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """Grouped 用 runtime cache から artifact cache を埋める。"""
    return backfill_surface_code_step_cache_from_runtime_cache(
        Hchain,
        n_w_list,
        source="gr",
        target_error=target_error,
        use_original=use_original,
        error_on_missing=error_on_missing,
    )


def backfill_df_surface_code_step_cache_from_runtime_cache(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    target_error: float = TARGET_ERROR,
    error_on_missing: bool = False,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """DF 用 runtime cache から artifact cache を埋める。"""
    return backfill_surface_code_step_cache_from_runtime_cache(
        Hchain,
        n_w_list,
        source="df",
        target_error=target_error,
        error_on_missing=error_on_missing,
    )


def _surface_code_cycle_failure_rate(
    code_distance: int,
    *,
    p_phys: float,
    p_th: float,
    a_eff: float,
) -> float:
    """有効 logical cycle failure rate を近似式で計算する。"""
    if code_distance <= 0 or code_distance % 2 == 0:
        raise ValueError(
            f"code_distance must be a positive odd integer: {code_distance}"
        )
    if p_phys <= 0:
        raise ValueError("p_phys must be positive.")
    if p_th <= 0:
        raise ValueError("p_th must be positive.")
    if a_eff <= 0:
        raise ValueError("a_eff must be positive.")
    exponent = (code_distance + 1) / 2.0
    return float(a_eff * ((p_phys / p_th) ** exponent))


def _surface_code_failure_probability(lambda_value: float) -> float:
    """Poisson 近似で failure probability を計算する。"""
    if lambda_value <= 0:
        return 0.0
    if lambda_value > 700:
        return 1.0
    return float(1.0 - math.exp(-lambda_value))


def estimate_df_surface_code_task_resources(
    ham_name: str,
    pf_label: PFLabel,
    *,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    step_metrics: Mapping[str, Any] | None = None,
    use_original: bool = False,
) -> Dict[str, Any]:
    """DF 1-step の surface-code metrics から QPE タスク全体資源を見積もる。"""
    return _estimate_surface_code_task_resources(
        ham_name,
        pf_label,
        source="df",
        target_error=target_error,
        p_th=p_th,
        a_eff_values=a_eff_values,
        p_phys_values=p_phys_values,
        delta_fail_values=delta_fail_values,
        code_distances=code_distances,
        runtime_key=runtime_key,
        step_metrics=step_metrics,
        use_original=use_original,
    )


def _estimate_surface_code_task_resources(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    step_metrics: Mapping[str, Any] | None = None,
    use_original: bool = False,
) -> Dict[str, Any]:
    """1-step の surface-code metrics から QPE タスク全体資源を見積もる。"""
    if target_error <= 0:
        raise ValueError("target_error must be positive.")
    if runtime_key not in {"runtime", "runtime_without_topology"}:
        raise ValueError(
            "runtime_key must be either 'runtime' or 'runtime_without_topology'."
        )

    coeff, expo = _load_compare_alpha_and_exponent(
        ham_name,
        pf_label,
        source=source,
        use_original=use_original,
    )
    qpe_factor = _qpe_iteration_factor(
        float(coeff),
        float(expo),
        float(target_error),
    )
    step = normalize_surface_code_step_metrics(
        step_metrics
        if step_metrics is not None
        else _load_surface_code_step_metrics(
            ham_name,
            pf_label,
            source=source,
            target_error=float(target_error),
            use_original=use_original,
        ),
        context=f"{ham_name}_Operator_{pf_label}.surface_code_step",
    )

    a_eff_list = _normalize_positive_scalar_list(
        a_eff_values if a_eff_values is not None else SURFACE_CODE_A_EFF_CASES,
        field="a_eff",
    )
    p_phys_list = _normalize_positive_scalar_list(
        p_phys_values if p_phys_values is not None else SURFACE_CODE_P_PHYS_CASES,
        field="p_phys",
    )
    delta_fail_list = _normalize_positive_scalar_list(
        delta_fail_values
        if delta_fail_values is not None
        else SURFACE_CODE_DELTA_FAIL_CASES,
        field="delta_fail",
    )
    code_distance_list = _normalize_code_distance_candidates(code_distances)

    step_qubit_volume = float(step["qubit_volume"])
    total_magic_state_count = qpe_factor * float(step["magic_state_consumption_count"])
    total_magic_state_depth = qpe_factor * float(step["magic_state_consumption_depth"])
    total_runtime = qpe_factor * float(step[runtime_key])
    total_runtime_with_topology = qpe_factor * float(step["runtime"])
    total_runtime_without_topology = qpe_factor * float(
        step["runtime_without_topology"]
    )
    total_qubit_volume = qpe_factor * step_qubit_volume
    total_t_count = (
        qpe_factor * float(step["t_count"]) if "t_count" in step else None
    )
    total_t_depth = (
        qpe_factor * float(step["t_depth"]) if "t_depth" in step else None
    )

    scenarios: List[Dict[str, Any]] = []
    for a_eff in a_eff_list:
        for p_phys in p_phys_list:
            for delta_fail in delta_fail_list:
                selected: Dict[str, Any] | None = None
                last_candidate: Dict[str, Any] | None = None
                for code_distance in code_distance_list:
                    p_cycle = _surface_code_cycle_failure_rate(
                        code_distance,
                        p_phys=p_phys,
                        p_th=float(p_th),
                        a_eff=a_eff,
                    )
                    lambda_step = step_qubit_volume * p_cycle
                    lambda_task = total_qubit_volume * p_cycle
                    p_fail_step = _surface_code_failure_probability(lambda_step)
                    p_fail_task = _surface_code_failure_probability(lambda_task)
                    candidate = {
                        "code_distance": int(code_distance),
                        "p_cycle_logical": float(p_cycle),
                        "lambda_step": float(lambda_step),
                        "lambda_task": float(lambda_task),
                        "p_fail_step": float(p_fail_step),
                        "p_fail_task": float(p_fail_task),
                    }
                    last_candidate = candidate
                    if p_fail_task <= delta_fail:
                        selected = candidate
                        break

                scenario: Dict[str, Any] = {
                    "a_eff": float(a_eff),
                    "p_phys": float(p_phys),
                    "delta_fail": float(delta_fail),
                    "meets_target": selected is not None,
                    "d_min": int(selected["code_distance"]) if selected is not None else None,
                }
                if selected is not None:
                    scenario.update(selected)
                elif last_candidate is not None:
                    scenario.update(
                        {
                            "code_distance_max_checked": int(
                                last_candidate["code_distance"]
                            ),
                            "p_cycle_logical_at_max_checked": float(
                                last_candidate["p_cycle_logical"]
                            ),
                            "lambda_step_at_max_checked": float(
                                last_candidate["lambda_step"]
                            ),
                            "lambda_task_at_max_checked": float(
                                last_candidate["lambda_task"]
                            ),
                            "p_fail_step_at_max_checked": float(
                                last_candidate["p_fail_step"]
                            ),
                            "p_fail_task_at_max_checked": float(
                                last_candidate["p_fail_task"]
                            ),
                        }
                    )
                scenarios.append(scenario)

    totals: Dict[str, Any] = {
        "total_magic_state_count": float(total_magic_state_count),
        "total_magic_state_depth": float(total_magic_state_depth),
        "total_runtime": float(total_runtime),
        "total_runtime_with_topology": float(total_runtime_with_topology),
        "total_runtime_without_topology": float(total_runtime_without_topology),
        "runtime_key": runtime_key,
        "total_qubit_volume": float(total_qubit_volume),
    }
    if total_t_count is not None:
        totals["total_t_count"] = float(total_t_count)
    if total_t_depth is not None:
        totals["total_t_depth"] = float(total_t_depth)

    return {
        "ham_name": ham_name,
        "pf_label": str(pf_label),
        "source": source,
        "target_error": float(target_error),
        "coeff": float(coeff),
        "expo": float(expo),
        # existing qpe_factor is used as an effective total block-count proxy.
        "effective_block_count": float(qpe_factor),
        "step_metrics": step,
        "totals": totals,
        "failure_model": {
            "p_th": float(p_th),
            "code_distances": code_distance_list,
            "a_eff_values": [float(v) for v in a_eff_list],
            "p_phys_values": [float(v) for v in p_phys_list],
            "delta_fail_values": [float(v) for v in delta_fail_list],
        },
        "scenarios": scenarios,
    }


def estimate_grouped_surface_code_task_resources(
    ham_name: str,
    pf_label: PFLabel,
    *,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    step_metrics: Mapping[str, Any] | None = None,
    use_original: bool = False,
) -> Dict[str, Any]:
    """Grouped 1-step の surface-code metrics から QPE タスク全体資源を見積もる。"""
    return _estimate_surface_code_task_resources(
        ham_name,
        pf_label,
        source="gr",
        target_error=target_error,
        p_th=p_th,
        a_eff_values=a_eff_values,
        p_phys_values=p_phys_values,
        delta_fail_values=delta_fail_values,
        code_distances=code_distances,
        runtime_key=runtime_key,
        step_metrics=step_metrics,
        use_original=use_original,
    )


def surface_code_task_resource_sweep_df(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    use_original: bool = False,
    error_on_missing: bool = True,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """DF artifact 群に対して surface-code タスク資源をまとめて見積もる。"""
    return _surface_code_task_resource_sweep(
        Hchain,
        n_w_list,
        source="df",
        target_error=target_error,
        p_th=p_th,
        a_eff_values=a_eff_values,
        p_phys_values=p_phys_values,
        delta_fail_values=delta_fail_values,
        code_distances=code_distances,
        runtime_key=runtime_key,
        use_original=use_original,
        error_on_missing=error_on_missing,
    )


def _surface_code_task_resource_sweep(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    source: str,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    use_original: bool = False,
    error_on_missing: bool = True,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """artifact 群に対して surface-code タスク資源をまとめて見積もる。"""
    if Hchain < 3:
        raise ValueError(
            "Hchain must be >= 3 for surface_code task-resource sweep."
        )

    results: Dict[int, Dict[PFLabel, Dict[str, Any]]] = {}
    errors: List[str] = []
    chain_list = [i for i in range(3, Hchain + 1)]
    mol_labels = [f"H{i}" for i in chain_list]
    num_qubits = [2 * i for i in chain_list]

    for qubits, mol in zip(num_qubits, mol_labels):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        per_pf: Dict[PFLabel, Dict[str, Any]] = {}
        for pf_label in n_w_list:
            if pf_label == "10th(Morales)" and qubits == 30:
                continue
            try:
                if source == "df":
                    per_pf[pf_label] = estimate_df_surface_code_task_resources(
                        ham_name,
                        pf_label,
                        target_error=target_error,
                        p_th=p_th,
                        a_eff_values=a_eff_values,
                        p_phys_values=p_phys_values,
                        delta_fail_values=delta_fail_values,
                        code_distances=code_distances,
                        runtime_key=runtime_key,
                        use_original=use_original,
                    )
                elif source == "gr":
                    per_pf[pf_label] = estimate_grouped_surface_code_task_resources(
                        ham_name,
                        pf_label,
                        target_error=target_error,
                        p_th=p_th,
                        a_eff_values=a_eff_values,
                        p_phys_values=p_phys_values,
                        delta_fail_values=delta_fail_values,
                        code_distances=code_distances,
                        runtime_key=runtime_key,
                        use_original=use_original,
                    )
                else:
                    raise ValueError(f"Unsupported source: {source}")
            except Exception as exc:
                if error_on_missing:
                    raise
                errors.append(f"{ham_name}_Operator_{pf_label}: {exc}")
                continue
        if per_pf:
            results[int(qubits)] = per_pf

    if not results and errors:
        preview = "; ".join(errors[:4])
        suffix = "" if len(errors) <= 4 else f"; ... ({len(errors)} entries)"
        raise RuntimeError(
            "surface_code task-resource sweep produced no valid data. "
            f"First errors: {preview}{suffix}"
        )
    return results


def surface_code_task_resource_sweep_grouped(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    use_original: bool = False,
    error_on_missing: bool = True,
) -> Dict[int, Dict[PFLabel, Dict[str, Any]]]:
    """Grouped artifact 群に対して surface-code タスク資源をまとめて見積もる。"""
    return _surface_code_task_resource_sweep(
        Hchain,
        n_w_list,
        source="gr",
        target_error=target_error,
        p_th=p_th,
        a_eff_values=a_eff_values,
        p_phys_values=p_phys_values,
        delta_fail_values=delta_fail_values,
        code_distances=code_distances,
        runtime_key=runtime_key,
        use_original=use_original,
        error_on_missing=error_on_missing,
    )


def _surface_code_metric_label(metric: str) -> str:
    """surface-code 指標名から軸ラベルを返す。"""
    labels = {
        "total_magic_state_count": "Total magic-state count",
        "total_magic_state_depth": "Total magic-state depth",
        "total_t_count": "Total T-count",
        "total_t_depth": "Total T-depth",
        "total_runtime": "Total runtime",
        "total_runtime_with_topology": "Total runtime (with topology)",
        "total_runtime_without_topology": "Total runtime (without topology)",
        "total_qubit_volume": "Total qubit volume",
        "d_min": "Minimum code distance",
        "p_fail_task": "Task failure probability",
        "p_fail_step": "Step failure probability",
        "lambda_task": "Task failure-rate proxy",
        "lambda_step": "Step failure-rate proxy",
        "p_cycle_logical": "Logical cycle failure rate",
    }
    return labels.get(metric, metric.replace("_", " "))


def _match_surface_code_scenario(
    scenario: Mapping[str, Any],
    *,
    a_eff: float | None,
    p_phys: float | None,
    delta_fail: float | None,
) -> bool:
    """scenario が指定パラメータに一致するかを判定する。"""
    if a_eff is not None and not math.isclose(float(scenario["a_eff"]), float(a_eff)):
        return False
    if p_phys is not None and not math.isclose(float(scenario["p_phys"]), float(p_phys)):
        return False
    if delta_fail is not None and not math.isclose(
        float(scenario["delta_fail"]),
        float(delta_fail),
    ):
        return False
    return True


def _pick_surface_code_scenario(
    result: Mapping[str, Any],
    *,
    a_eff: float | None,
    p_phys: float | None,
    delta_fail: float | None,
) -> Mapping[str, Any]:
    """result 内の scenario から指定条件に一致する 1 件を選ぶ。"""
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, Sequence):
        raise ValueError("surface_code result missing scenarios.")
    matched = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, Mapping)
        and _match_surface_code_scenario(
            scenario,
            a_eff=a_eff,
            p_phys=p_phys,
            delta_fail=delta_fail,
        )
    ]
    if len(matched) == 1:
        return matched[0]
    if len(matched) == 0:
        raise ValueError(
            "No surface_code scenario matched "
            f"(a_eff={a_eff}, p_phys={p_phys}, delta_fail={delta_fail})."
        )
    raise ValueError(
        "Multiple surface_code scenarios matched. "
        "Specify a_eff, p_phys, and delta_fail more narrowly."
    )


def _surface_code_result_metric_value(
    result: Mapping[str, Any],
    *,
    metric: str,
    a_eff: float | None = None,
    p_phys: float | None = None,
    delta_fail: float | None = None,
) -> float:
    """surface-code 推定結果から描画対象の値を取り出す。"""
    totals = result.get("totals")
    if isinstance(totals, Mapping) and metric in totals:
        return float(totals[metric])

    scenario = _pick_surface_code_scenario(
        result,
        a_eff=a_eff,
        p_phys=p_phys,
        delta_fail=delta_fail,
    )
    if metric == "d_min":
        d_min = scenario.get("d_min")
        if d_min is None:
            raise ValueError(
                "surface_code scenario did not meet target within checked code distances."
            )
        return float(d_min)

    value = scenario.get(metric)
    if value is None:
        raise ValueError(f"Unsupported surface_code metric: {metric}")
    return float(value)


def _surface_code_task_resource_series(
    sweep_results: Mapping[int, Mapping[PFLabel, Mapping[str, Any]]],
    n_w_list: Sequence[PFLabel],
    *,
    metric: str,
    a_eff: float | None = None,
    p_phys: float | None = None,
    delta_fail: float | None = None,
    error_on_missing: bool = False,
    source: str = "df",
) -> Dict[str, Dict[str, List[float]]]:
    """sweep 結果から PF 別プロット系列を構築する。"""
    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    missing_messages: List[str] = []
    for qubits in sorted(sweep_results.keys()):
        per_pf = sweep_results[qubits]
        for pf_label in n_w_list:
            result = per_pf.get(pf_label)
            if not isinstance(result, Mapping):
                continue
            try:
                value = _surface_code_result_metric_value(
                    result,
                    metric=metric,
                    a_eff=a_eff,
                    p_phys=p_phys,
                    delta_fail=delta_fail,
                )
            except Exception as exc:
                if error_on_missing:
                    raise
                msg = (
                    f"{result.get('ham_name', 'unknown')}_Operator_{pf_label}: {exc}"
                )
                missing_messages.append(msg)
                print(f"[surface-code][skip {source}] {msg}")
                continue
            if value <= 0:
                continue
            series[str(pf_label)]["x"].append(float(qubits))
            series[str(pf_label)]["y"].append(float(value))
    series_dict = {k: {"x": list(v["x"]), "y": list(v["y"])} for k, v in series.items()}
    if missing_messages:
        series_dict["_missing"] = {"messages": missing_messages}
    return series_dict


def surface_code_task_resource_extrapolation(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    source: str = "df",
    metric: str = "total_magic_state_count",
    target_error: float = TARGET_ERROR,
    p_th: float = SURFACE_CODE_P_TH,
    a_eff: float | None = None,
    p_phys: float | None = None,
    delta_fail: float | None = None,
    a_eff_values: Sequence[float] | None = None,
    p_phys_values: Sequence[float] | None = None,
    delta_fail_values: Sequence[float] | None = None,
    code_distances: Sequence[int] | None = None,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    use_original: bool = False,
    show_bands: bool = True,
    band_height: float = 0.06,
    band_alpha: float = 0.28,
    error_on_missing: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """複数分子・複数 PF の surface-code 資源を同時プロットする。"""
    if source == "df":
        sweep_results = surface_code_task_resource_sweep_df(
            Hchain,
            n_w_list,
            target_error=target_error,
            p_th=p_th,
            a_eff_values=a_eff_values,
            p_phys_values=p_phys_values,
            delta_fail_values=delta_fail_values,
            code_distances=code_distances,
            runtime_key=runtime_key,
            use_original=use_original,
            error_on_missing=error_on_missing,
        )
    elif source == "gr":
        sweep_results = surface_code_task_resource_sweep_grouped(
            Hchain,
            n_w_list,
            target_error=target_error,
            p_th=p_th,
            a_eff_values=a_eff_values,
            p_phys_values=p_phys_values,
            delta_fail_values=delta_fail_values,
            code_distances=code_distances,
            runtime_key=runtime_key,
            use_original=use_original,
            error_on_missing=error_on_missing,
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    series = _surface_code_task_resource_series(
        sweep_results,
        n_w_list,
        metric=metric,
        a_eff=a_eff,
        p_phys=p_phys,
        delta_fail=delta_fail,
        error_on_missing=error_on_missing,
        source=source,
    )
    missing_info = series.pop("_missing", None)
    if not series:
        details = ""
        if isinstance(missing_info, Mapping):
            messages = list(missing_info.get("messages", []))
            if messages:
                preview = "; ".join(str(msg) for msg in messages[:4])
                suffix = "" if len(messages) <= 4 else f"; ... ({len(messages)} entries)"
                details = f" Missing examples: {preview}{suffix}"
        raise FileNotFoundError(
            "No valid surface_code data found for "
            f"source={source}, metric={metric}. "
            "The most common cause is that the relevant artifact does not yet have "
            "`surface_code_step` attached. Populate it with "
            "`attach_df_surface_code_step_metrics_from_compile_info_json(...)` or "
            "`attach_grouped_surface_code_step_metrics_from_compile_info_json(...)`."
            f"{details}"
        )

    plt.figure(figsize=(8, 6), dpi=200)
    first_x = min(
        float(min(points["x"]))
        for points in series.values()
        if points["x"]
    )
    for pf_label in n_w_list:
        key = str(pf_label)
        points = series.get(key)
        if not points:
            continue
        x = np.asarray(points["x"], dtype=float)
        y = np.asarray(points["y"], dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        label = label_replace(key)
        for x_i, y_i in zip(x, y):
            plt.plot(
                x_i,
                y_i,
                ls="None",
                marker=MARKER_MAP.get(key, "o"),
                color=COLOR_MAP.get(key, None),
                label=label if math.isclose(float(x_i), first_x) else None,
            )

    ax = plt.gca()
    set_loglog_axes(ax)
    x_values = [x for points in series.values() for x in points["x"]]
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    if x_max <= x_min:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)

    _apply_loglog_fit_with_bands(
        ax,
        series,
        show_bands=show_bands,
        band_height=band_height,
        band_alpha=band_alpha,
    )

    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel(_surface_code_metric_label(metric), fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.show()
    return series


def _load_grouped_artifact_payload(
    ham_name: str,
    pf_label: PFLabel,
    *,
    use_original: bool = False,
) -> Dict[str, Any]:
    """Grouped 外挿用の保存バイナリを読み込む。"""
    target_path = f"{ham_name}_Operator_{pf_label}"
    data = load_data(target_path, gr=True, use_original=use_original)
    if isinstance(data, Mapping):
        return dict(data)

    # 互換: 古い形式で _ave のみある場合は coeff と既知次数 p を組み合わせる。
    target_path_ave = f"{ham_name}_Operator_{pf_label}_ave"
    coeff = load_data(target_path_ave, gr=True, use_original=use_original)
    return {"coeff": coeff, "expo": P_DIR[pf_label]}


def _artifact_positive_scalar(value: Any, *, field: str, context: str) -> float:
    """artifact 値(スカラー/長さ1配列)を正の float に正規化する。"""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Missing {field} in {context}")
    scalar = float(arr[0])
    if (not np.isfinite(scalar)) or scalar <= 0:
        raise ValueError(f"Invalid {field}={scalar} in {context}")
    return scalar


def _load_alpha_from_ave(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> float:
    """alpha(coeff) を *_ave 優先で読み込み、無ければ従来形式へフォールバックする。"""
    target_name = f"{ham_name}_Operator_{pf_label}_ave"
    try:
        if source == "gr":
            raw = load_data(target_name, gr=True, use_original=use_original)
        elif source == "df":
            raw = load_data(str(PICKLE_DIR_DF_PATH / target_name), gr=None)
        else:
            raise ValueError(f"Unsupported source: {source}")

        # 互換: dict 形式で保存されている場合は coeff を参照する。
        if isinstance(raw, Mapping):
            raw = raw.get("coeff")

        return _artifact_positive_scalar(
            raw,
            field="coeff(_ave)",
            context=f"{target_name}[{source}]",
        )
    except FileNotFoundError:
        # *_ave が無い場合は従来の保存形式を読む。
        legacy_name = f"{ham_name}_Operator_{pf_label}"
        if source == "gr":
            payload = _load_grouped_artifact_payload(
                ham_name,
                pf_label,
                use_original=use_original,
            )
        elif source == "df":
            payload = _load_df_artifact_payload(ham_name, pf_label)
        else:
            raise ValueError(f"Unsupported source: {source}")
        return _artifact_positive_scalar(
            payload.get("coeff"),
            field="coeff(legacy)",
            context=f"{legacy_name}[{source}]",
        )


def _load_compare_alpha_and_exponent(
    ham_name: str,
    pf_label: PFLabel,
    *,
    source: str,
    use_original: bool = False,
) -> tuple[float, float]:
    """Return (alpha, p) for gr/df comparison.

    Default:
      alpha from *_ave, p from fixed P_DIR[pf_label].
    Fallback when *_ave is missing:
      - gr: legacy coeff with fixed p (same p rule).
      - df: legacy coeff/expo (previous behavior).
    """
    fixed_p = float(P_DIR[pf_label])
    target_name = f"{ham_name}_Operator_{pf_label}_ave"
    try:
        if source == "gr":
            raw = load_data(target_name, gr=True, use_original=use_original)
        elif source == "df":
            raw = load_data(str(PICKLE_DIR_DF_PATH / target_name), gr=None)
        else:
            raise ValueError(f"Unsupported source: {source}")

        if isinstance(raw, Mapping):
            raw = raw.get("coeff")
        alpha = _artifact_positive_scalar(
            raw,
            field="coeff(_ave)",
            context=f"{target_name}[{source}]",
        )
        return alpha, fixed_p
    except FileNotFoundError:
        legacy_name = f"{ham_name}_Operator_{pf_label}"
        if source == "gr":
            payload = _load_grouped_artifact_payload(
                ham_name,
                pf_label,
                use_original=use_original,
            )
            alpha = _artifact_positive_scalar(
                payload.get("coeff"),
                field="coeff(legacy)",
                context=f"{legacy_name}[gr]",
            )
            return alpha, fixed_p
        if source == "df":
            payload = _load_df_artifact_payload(ham_name, pf_label)
            alpha = _artifact_positive_scalar(
                payload.get("coeff"),
                field="coeff(legacy)",
                context=f"{legacy_name}[df]",
            )
            expo = _artifact_positive_scalar(
                payload.get("expo"),
                field="expo(legacy)",
                context=f"{legacy_name}[df]",
            )
            return alpha, expo
        raise ValueError(f"Unsupported source: {source}")


def _qpe_iteration_factor(alpha: float, p: float, epsilon_e: float) -> float:
    """QPE 反復係数 β((1+p)/(pε_E))((α(1+p))/ε_E)^(1/p) を計算する。"""
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if p <= 0:
        raise ValueError("p must be positive.")
    if epsilon_e <= 0:
        raise ValueError("epsilon_e must be positive.")
    return float(
        BETA
        * ((1.0 + p) / (p * epsilon_e))
        * ((alpha * (1.0 + p) / epsilon_e) ** (1.0 / p))
    )


def qpe_iteration_factor_compare_gr_df(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    target_error: float = TARGET_ERROR,
    use_original: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """gr/df の *_ave alpha と P_DIR[pflabel] の p で QPE 反復係数を比較する。

    plot value:
      β((1+p)/(pε_E))((α(1+p))/ε_E)^(1/p)
    """
    if Hchain < 3:
        raise ValueError("Hchain must be >= 3 for gr/df comparison.")
    if target_error <= 0:
        raise ValueError("target_error must be positive.")

    chain_list = [i for i in range(3, Hchain + 1)]
    mol_labels = [f"H{i}" for i in chain_list]
    num_qubits = [2 * i for i in chain_list]

    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )

    for qubits, mol in zip(num_qubits, mol_labels):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:
                continue
            p_from_config = float(P_DIR[n_w])

            grouped_key = f"{n_w}|gr"
            grouped_name = f"{ham_name}_Operator_{n_w}"
            try:
                alpha_gr = _load_alpha_from_ave(
                    ham_name,
                    n_w,
                    source="gr",
                    use_original=use_original,
                )
                val_gr = _qpe_iteration_factor(
                    alpha_gr,
                    p_from_config,
                    float(target_error),
                )
                series[grouped_key]["x"].append(float(qubits))
                series[grouped_key]["y"].append(val_gr)
            except Exception as exc:
                print(f"[qpe-factor][skip gr] {grouped_name}_ave: {exc}")

            df_key = f"{n_w}|df"
            df_name = f"{ham_name}_Operator_{n_w}"
            try:
                alpha_df = _load_alpha_from_ave(
                    ham_name,
                    n_w,
                    source="df",
                    use_original=use_original,
                )
                val_df = _qpe_iteration_factor(
                    alpha_df,
                    p_from_config,
                    float(target_error),
                )
                series[df_key]["x"].append(float(qubits))
                series[df_key]["y"].append(val_df)
            except Exception as exc:
                print(f"[qpe-factor][skip df] {df_name}_ave: {exc}")

    has_data = any(len(d["x"]) > 0 for d in series.values())
    if not has_data:
        raise FileNotFoundError(
            "No valid grouped/df artifacts found for qpe_iteration_factor_compare_gr_df."
        )

    _fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    set_loglog_axes(ax)

    for key, data in series.items():
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        if x.size == 0:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        pf, source = key.split("|", 1)
        label = f"{label_replace(pf)} ({source})"
        if source == "gr":
            ax.plot(
                x,
                y,
                marker=MARKER_MAP.get(pf, "o"),
                linestyle="-",
                color=COLOR_MAP.get(pf, None),
                label=label,
            )
        else:
            ax.plot(
                x,
                y,
                marker="x",
                linestyle="--",
                color=COLOR_MAP.get(pf, None),
                label=label,
            )

    x_min = float(min(num_qubits))
    x_max = float(max(num_qubits))
    if x_max <= x_min:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel(
        r"$\beta\left(\frac{1+p}{p\varepsilon_E}\right)\left(\frac{\alpha(1+p)}{\varepsilon_E}\right)^{1/p}$",
        fontsize=15,
    )
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    handles_u, labels_u = unique_legend_entries(handles, labels)
    if handles_u:
        ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    return {k: {"x": list(v["x"]), "y": list(v["y"])} for k, v in series.items()}


def _pick_df_rz_layer_value(
    rz_layers: Mapping[str, Any],
    preferred_key: str | None = None,
) -> Tuple[str, float]:
    """DF artifact の rz_layers から利用するレイヤー値を選ぶ。

    Default priority prefers full reference RZ depth:
      total_ref_rz_depth > ref_rz_depth > (legacy/nonclifford keys...)
    """
    candidate_keys: List[str] = []
    if preferred_key:
        candidate_keys.append(str(preferred_key))
    candidate_keys.extend(
        [
            "total_ref_rz_depth",
            "ref_rz_depth",
            "u_ref_rz_depth",
            "d_ref_rz_depth",
            "total_nonclifford_z_coloring_depth",
            "total_nonclifford_z_depth",
            "total_nonclifford_rz_depth",
        ]
    )
    for key in candidate_keys:
        if key not in rz_layers:
            continue
        try:
            value = float(rz_layers[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return key, value
    raise ValueError(
        "No positive rz layer metric found in DF artifact. "
        f"available keys={list(rz_layers.keys())}"
    )


def t_depth_extrapolation_df(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    rz_layer: Optional[bool] = None,
    show_bands: bool = True,
    band_height: float = 0.06,
    band_alpha: float = 0.28,
    rz_layer_key: str | None = None,
) -> None:
    """DF の保存データ(trotter_expo_coeff_df)を使って T-depth / RZ レイヤー数を外挿する。

    rz_layer_key=None の既定では DF 側レイヤーは total_ref_rz_depth を優先する。
    """
    if Hchain < 3:
        raise ValueError("Hchain must be >= 3 for t_depth_extrapolation_df.")

    target_error = TARGET_ERROR
    total_dir: Dict[int, Dict[str, float]] = {}

    chain_list = [i for i in range(3, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in chain_list]
    num_qubits = [2 * i for i in chain_list]
    plt.figure(figsize=(8, 6), dpi=200)

    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        total_dir[qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:
                continue

            payload = _load_df_artifact_payload(ham_name, n_w)
            if "expo" not in payload or "coeff" not in payload:
                raise ValueError(
                    f"DF artifact missing expo/coeff: {ham_name}_Operator_{n_w}"
                )
            try:
                expo = float(payload["expo"])
                coeff = float(payload["coeff"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid expo/coeff in DF artifact: {ham_name}_Operator_{n_w}"
                ) from exc
            if expo <= 0 or coeff <= 0:
                raise ValueError(
                    f"Non-positive expo/coeff in DF artifact: {ham_name}_Operator_{n_w}"
                )

            rz_layers_raw = payload.get("rz_layers")
            if not isinstance(rz_layers_raw, Mapping):
                raise ValueError(
                    f"DF artifact missing rz_layers: {ham_name}_Operator_{n_w}"
                )
            _metric_key, rz_layer_value = _pick_df_rz_layer_value(
                rz_layers_raw, preferred_key=rz_layer_key
            )

            # DF版では DECOMPO_NUM の代わりに保存済み rz layer を使う。
            N_0 = rz_layer_value
            pf_layer_rz = rz_layer_value

            t = (target_error / coeff * (expo + 1)) ** (1 / expo)
            qpe_factor = _qpe_iteration_factor(
                float(coeff),
                float(expo),
                float(target_error),
            )

            eps_rot = (t * 0.01 * target_error) / (N_0 * qpe_factor)
            T_rot = 3 * np.log2(1 / eps_rot)
            D_T = pf_layer_rz * T_rot
            tot_dt = qpe_factor * D_T
            tot_rz_layer = qpe_factor * pf_layer_rz

            if rz_layer:
                total_dir[qubits][n_w] = tot_rz_layer
            else:
                total_dir[qubits][n_w] = tot_dt

    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            plt.plot(
                qubit,
                gate,
                ls="None",
                marker=MARKER_MAP[pf],
                color=COLOR_MAP[pf],
                label=lb if qubit == num_qubits[0] else None,
            )
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    set_loglog_axes(ax)
    x_min = float(min(num_qubits))
    x_max = float(max(num_qubits))
    if x_max <= x_min:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)

    _apply_loglog_fit_with_bands(
        ax,
        series,
        show_bands=show_bands,
        band_height=band_height,
        band_alpha=band_alpha,
    )

    ax.set_xlabel("Num qubits", fontsize=15)
    if rz_layer:
        ax.set_ylabel("Num RZ layer", fontsize=15)
    else:
        ax.set_ylabel("T-depth", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    plt.show()


def t_depth_extrapolation_compare_gr_df(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    rz_layer: Optional[bool] = None,
    rz_layer_key: str | None = None,
    use_original: bool = False,
) -> None:
    """Grouped と DF の t_depth~/rz_layer を同一プロットで比較する。

    Default:
      alpha は *_ave から、p は固定の P_DIR[pf_label] を使う。
      DF の rz layer は total_nonclifford_z_coloring_depth を優先して選ぶ。
    Fallback:
      *_ave が無い場合は従来読込に戻る（df は legacy expo を使用）。
    """
    if Hchain < 3:
        raise ValueError("Hchain must be >= 3 for gr/df comparison.")

    target_error = TARGET_ERROR
    chain_list = [i for i in range(3, Hchain + 1)]
    mol_labels = [f"H{i}" for i in chain_list]
    num_qubits = [2 * i for i in chain_list]

    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    effective_rz_layer_key = (
        rz_layer_key
        if rz_layer_key is not None
        else "total_nonclifford_z_coloring_depth"
    )

    for qubits, mol in zip(num_qubits, mol_labels):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:
                continue

            # grouped source
            grouped_key = f"{n_w}|gr"
            grouped_path = f"{ham_name}_Operator_{n_w}_ave"
            try:
                coeff_gr, expo_gr = _load_compare_alpha_and_exponent(
                    ham_name,
                    n_w,
                    source="gr",
                    use_original=use_original,
                )
                N_0_gr = float(DECOMPO_NUM[mol][n_w])
                pf_layer_rz_gr = float(PF_RZ_LAYER[mol][n_w])
                t_gr = (target_error / coeff_gr * (expo_gr + 1)) ** (1 / expo_gr)
                qpe_factor_gr = _qpe_iteration_factor(
                    float(coeff_gr),
                    float(expo_gr),
                    float(target_error),
                )
                eps_rot_gr = (t_gr * 0.01 * target_error) / (N_0_gr * qpe_factor_gr)
                T_rot_gr = 3 * np.log2(1 / eps_rot_gr)
                D_T_gr = pf_layer_rz_gr * T_rot_gr
                total_gr = (
                    qpe_factor_gr * pf_layer_rz_gr
                    if rz_layer
                    else qpe_factor_gr * D_T_gr
                )
                if total_gr > 0:
                    series[grouped_key]["x"].append(float(qubits))
                    series[grouped_key]["y"].append(float(total_gr))
            except Exception as exc:
                print(f"[compare][skip grouped] {grouped_path}: {exc}")

            # df source
            df_key = f"{n_w}|df"
            df_path = f"{ham_name}_Operator_{n_w}_ave"
            try:
                coeff_df, expo_df = _load_compare_alpha_and_exponent(
                    ham_name,
                    n_w,
                    source="df",
                    use_original=use_original,
                )
                payload = _load_df_artifact_payload(ham_name, n_w)
                rz_layers_raw = payload.get("rz_layers")
                if not isinstance(rz_layers_raw, Mapping):
                    raise ValueError("missing rz_layers")
                _metric_key, rz_layer_value = _pick_df_rz_layer_value(
                    rz_layers_raw, preferred_key=effective_rz_layer_key
                )
                N_0_df = rz_layer_value
                pf_layer_rz_df = rz_layer_value
                t_df = (target_error / coeff_df * (expo_df + 1)) ** (1 / expo_df)
                qpe_factor_df = _qpe_iteration_factor(
                    float(coeff_df),
                    float(expo_df),
                    float(target_error),
                )
                eps_rot_df = (t_df * 0.01 * target_error) / (N_0_df * qpe_factor_df)
                T_rot_df = 3 * np.log2(1 / eps_rot_df)
                D_T_df = pf_layer_rz_df * T_rot_df
                total_df = (
                    qpe_factor_df * pf_layer_rz_df
                    if rz_layer
                    else qpe_factor_df * D_T_df
                )
                if total_df > 0:
                    series[df_key]["x"].append(float(qubits))
                    series[df_key]["y"].append(float(total_df))
            except Exception as exc:
                print(f"[compare][skip df] {df_path}: {exc}")

    ax = plt.gca()
    set_loglog_axes(ax)

    for key, data in series.items():
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        if x.size == 0:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        pf, source = key.split("|", 1)
        label = f"{label_replace(pf)} ({source})"
        if source == "gr":
            ax.plot(
                x,
                y,
                marker=MARKER_MAP.get(pf, "o"),
                linestyle="-",
                color=COLOR_MAP.get(pf, None),
                label=label,
            )
        else:
            ax.plot(
                x,
                y,
                marker="x",
                linestyle="--",
                color=COLOR_MAP.get(pf, None),
                label=label,
            )

    x_min = float(min(num_qubits))
    x_max = float(max(num_qubits))
    if x_max <= x_min:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Num qubits", fontsize=15)
    if rz_layer:
        ax.set_ylabel("Num RZ layer", fontsize=15)
    else:
        ax.set_ylabel("T-depth", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    handles_u, labels_u = unique_legend_entries(handles, labels)
    if handles_u:
        ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.show()


def _conventional_t_depth_series(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    source: str,
    target_error: float = TARGET_ERROR,
    rz_layer_key: str | None = None,
    use_original: bool = False,
    error_on_missing: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """従来式の T-depth 系列を PF ごとに構築する。"""
    if Hchain < 3:
        raise ValueError("Hchain must be >= 3 for t-depth comparison.")
    if source not in {"gr", "df"}:
        raise ValueError(f"Unsupported source: {source}")

    _, mol_labels, num_qubits = _hchain_series(Hchain)
    effective_rz_layer_key = (
        rz_layer_key
        if rz_layer_key is not None
        else "total_nonclifford_z_coloring_depth"
    )
    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )

    for qubits, mol in zip(num_qubits, mol_labels):
        if qubits % 4 == 0:
            ham_name = mol + "_sto-3g_singlet_distance_100_charge_0_grouping"
        else:
            ham_name = mol + "_sto-3g_triplet_1+_distance_100_charge_1_grouping"

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:
                continue

            try:
                coeff, expo = _load_compare_alpha_and_exponent(
                    ham_name,
                    n_w,
                    source=source,
                    use_original=use_original,
                )
                if source == "gr":
                    N_0 = float(DECOMPO_NUM[mol][n_w])
                    pf_layer_rz = float(PF_RZ_LAYER[mol][n_w])
                else:
                    payload = _load_df_artifact_payload(ham_name, n_w)
                    rz_layers_raw = payload.get("rz_layers")
                    if not isinstance(rz_layers_raw, Mapping):
                        raise ValueError("missing rz_layers")
                    _metric_key, rz_layer_value = _pick_df_rz_layer_value(
                        rz_layers_raw, preferred_key=effective_rz_layer_key
                    )
                    N_0 = float(rz_layer_value)
                    pf_layer_rz = float(rz_layer_value)

                t = (target_error / coeff * (expo + 1)) ** (1 / expo)
                qpe_factor = _qpe_iteration_factor(
                    float(coeff),
                    float(expo),
                    float(target_error),
                )
                eps_rot = (t * 0.01 * target_error) / (N_0 * qpe_factor)
                T_rot = 3 * np.log2(1 / eps_rot)
                total_t_depth = qpe_factor * pf_layer_rz * T_rot
            except Exception as exc:
                if error_on_missing:
                    raise
                print(
                    f"[t-depth][skip {source} conventional] "
                    f"{ham_name}_Operator_{n_w}: {exc}"
                )
                continue

            if total_t_depth <= 0:
                continue
            series[str(n_w)]["x"].append(float(qubits))
            series[str(n_w)]["y"].append(float(total_t_depth))

    return {k: {"x": list(v["x"]), "y": list(v["y"])} for k, v in series.items()}


def t_depth_extrapolation_compare_conventional_decompose(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    *,
    source: str = "gr",
    target_error: float = TARGET_ERROR,
    rz_layer_key: str | None = None,
    use_original: bool = False,
    runtime_key: str = SURFACE_CODE_RUNTIME_METRIC,
    error_on_missing: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """従来の T-depth 外挿と decompose-only の T-depth を同一プロットで比較する。"""
    conventional = _conventional_t_depth_series(
        Hchain,
        n_w_list,
        source=source,
        target_error=target_error,
        rz_layer_key=rz_layer_key,
        use_original=use_original,
        error_on_missing=error_on_missing,
    )

    if source == "df":
        sweep_results = surface_code_task_resource_sweep_df(
            Hchain,
            n_w_list,
            target_error=target_error,
            runtime_key=runtime_key,
            use_original=use_original,
            error_on_missing=error_on_missing,
        )
    elif source == "gr":
        sweep_results = surface_code_task_resource_sweep_grouped(
            Hchain,
            n_w_list,
            target_error=target_error,
            runtime_key=runtime_key,
            use_original=use_original,
            error_on_missing=error_on_missing,
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    decompose = _surface_code_task_resource_series(
        sweep_results,
        n_w_list,
        metric="total_t_depth",
        error_on_missing=error_on_missing,
        source=source,
    )
    decompose.pop("_missing", None)

    if not conventional and not decompose:
        raise FileNotFoundError(
            "No valid T-depth data found for conventional or decompose-only series."
        )

    plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.gca()

    for pf_label in n_w_list:
        key = str(pf_label)
        label = label_replace(key)
        color = COLOR_MAP.get(key, None)

        conventional_points = conventional.get(key)
        if conventional_points and conventional_points["x"]:
            x = np.asarray(conventional_points["x"], dtype=float)
            y = np.asarray(conventional_points["y"], dtype=float)
            order = np.argsort(x)
            ax.plot(
                x[order],
                y[order],
                marker=MARKER_MAP.get(key, "o"),
                linestyle="-",
                color=color,
                label=f"{label} (conventional)",
            )

        decompose_points = decompose.get(key)
        if decompose_points and decompose_points["x"]:
            x = np.asarray(decompose_points["x"], dtype=float)
            y = np.asarray(decompose_points["y"], dtype=float)
            order = np.argsort(x)
            ax.plot(
                x[order],
                y[order],
                marker="x",
                linestyle="--",
                color=color,
                label=f"{label} (decompose)",
            )

    all_x = [
        value
        for points in list(conventional.values()) + list(decompose.values())
        for value in points["x"]
    ]
    if not all_x:
        raise FileNotFoundError("No valid x-values found for T-depth comparison.")

    set_loglog_axes(ax)
    x_min = float(min(all_x))
    x_max = float(max(all_x))
    if x_max <= x_min:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("T-depth", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    handles_u, labels_u = unique_legend_entries(handles, labels)
    if handles_u:
        ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.show()
    return {
        "conventional": conventional,
        "decompose": decompose,
    }


def t_depth_extrapolation_diff(
    Hchain: int,
    rz_layer: Optional[bool] = None,
    n_w_list: Sequence[PFLabel] = ("4th(new_2)", "8th(Morales)"),
    MIN_POS: float = 1e-18,
    X_MIN_CALC: float = 4,
    X_MAX_DISPLAY: float = 100.0,
    use_original: bool = False,
) -> None:
    """
    2つの PF を指定して T-depth を比較するプロット関数（双Y軸）。

      左Y: QPE 全体の T-depth（散布 + log–log フィット）
      右Y: フィット直線同士の絶対差 |ΔT|（ログ）

    - n_w_list, num_w_list は必ず 2 つずつ渡す
    - スケーリング直線/Δ の評価は x >= X_MIN_CALC のみ
    - x の右端は X_MAX_DISPLAY に固定

    依存: decompo_num, PF_RZ_layer, optimal_distance,
          jw_hamiltonian_maker, load_data, label_replace
    use_original=True で trotter_expo_coeff_gr_original を参照する。
    """

    target_error = TARGET_ERROR
    X_MAX_DISPLAY = 100.0


    # PF 表示名 → ラベル
    label_map = {pf: label_replace(pf) for pf in n_w_list}
    num_qubits = [i for i in range(4,(Hchain*2)+1,2)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]

    # ===== T-depth の計算 =====
    total_dir: Dict[int, Dict[str, float]] = {}  # {n_qubits: {pf_key: T-depth}}

    for qubits, mol in zip(num_qubits, Hchain_str):
        if qubits % 4 == 0:
            ham_name = mol + '_sto-3g_singlet_distance_100_charge_0_grouping'
        else:
            ham_name = mol + '_sto-3g_triplet_1+_distance_100_charge_1_grouping'

        total_dir[qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and qubits == 30:  # 10th は H15 で未評価
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            data = load_data(target_path, use_original=use_original)

            coeff = data
            expo = P_DIR[n_w]

            # N_0 PF による分解に含まれる RZ 数 T-depth 計算なら実質パウリ回転数
            N_0 = DECOMPO_NUM[mol][n_w]

            # L_Z RZ のレイヤー数
            pf_layer_rz = PF_RZ_LAYER[mol][n_w]

            t = (target_error / coeff * (expo + 1))**(1/expo)
            qpe_factor = _qpe_iteration_factor(
                float(coeff),
                float(expo),
                float(target_error),
            )

            # RZ の近似誤差は許容誤差の 1 パーセント
            eps_rot = (t * 0.01 * target_error) / (N_0 * qpe_factor)
            
            # RZ 近似誤差 T = 3log2(1/eps_rot)
            T_rot = 3 * np.log2(1/eps_rot)

            # PF のユニタリ１回分の T-depth
            D_T = pf_layer_rz * T_rot

            # QPE QC 全体での T-depth
            tot_dt = qpe_factor * D_T

            # QPE QC 全体での RZ レイヤー数
            tot_rz_layer = qpe_factor * pf_layer_rz

            if qubits == 30:
                print(f'T_rot:{T_rot} PF:{n_w}')

            if rz_layer:
                total_dir[qubits][n_w] =  tot_rz_layer
            else:
                total_dir[qubits][n_w] = tot_dt

    # ===== プロット（左Y: T-depth, 右Y: 差分）=====
    _fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    ax2 = ax.twinx()

    # 散布用データ（キーは pf
    series: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    for qubit, gdict in total_dir.items():
        for pf in n_w_list:
            if pf not in gdict:
                continue
            lb = label_map[pf]  # 表示用ラベル

            # 散布図
            ax.plot(
                qubit,
                gdict[pf],
                ls="None",
                marker=MARKER_MAP.get(pf, "o"),       # ← pf で引く
                color=COLOR_MAP.get(pf, None),       # ← pf で引く
                label=lb,
            )

            # フィット用データを pf キーで保持
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gdict[pf]))

    # 左軸スケール
    set_loglog_axes(ax)

    x_left_auto, _ = ax.get_xlim()
    ax.set_xlim(x_left_auto, X_MAX_DISPLAY)

    # ===== log–log フィット =====
    fit_params = {}  # キーは pf
    xfit_lo = max(X_MIN_CALC, x_left_auto)
    xfit_hi = X_MAX_DISPLAY

    for pf in n_w_list:
        if pf not in series:
            continue

        x = np.asarray(series[pf]["x"], float)
        y = np.asarray(series[pf]["y"], float)
        m = (x > 0) & (y > 0)
        if m.sum() < 2:
            continue

        fit = loglog_fit(x[m], y[m], mask_nonpositive=True)
        A = fit.coeff
        B = fit.slope
        fit_params[pf] = {"A": A, "B": B}

        if xfit_hi > xfit_lo:
            xx = np.logspace(np.log10(xfit_lo), np.log10(xfit_hi), 400)
            ax.plot(
                xx,
                A * xx ** B,
                "-",
                color=COLOR_MAP.get(pf),
                lw=1.0,
                alpha=0.95,
            )
            # デバッグ用: 100qubit 時の T-depth フィット値
            print(f"{label_map[pf]}_Tdepth_at_100qubits = {A * 100.0 ** B}")

    # ===== 右軸: フィット直線の差 |ΔT| =====
    # n_w_list の順に 2 個の pf があり、両方フィットできているときだけ描画
    pf_for_diff = [pf for pf in n_w_list if pf in fit_params]
    if len(pf_for_diff) == 2:
        pf_a, pf_b = pf_for_diff  # 表示順を n_w_list に揃える

        def yfit(pf: str, x: np.ndarray) -> np.ndarray:
            A = fit_params[pf]["A"]
            B = fit_params[pf]["B"]
            return A * x ** B

        if xfit_hi > xfit_lo:
            xx = np.logspace(np.log10(xfit_lo), np.log10(xfit_hi), 1200)
            diff = np.maximum(np.abs(yfit(pf_a, xx) - yfit(pf_b, xx)), MIN_POS)
            ax2.plot(
                xx,
                diff,
                "--",
                lw=2.0,
                alpha=0.9,
                color=COLOR_MAP.get(pf_a),
                label=f"|{label_map[pf_b]} - {label_map[pf_a]}|",
            )
            ax2.set_yscale("log")

    # ===== 目盛を左右で揃え、10^k 表記にする =====
    for a in (ax, ax2):
        a.yaxis.set_major_locator(LogLocator(base=10))
        a.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        a.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        a.yaxis.set_minor_formatter(plt.NullFormatter())

    ax2.set_ylim(ax.get_ylim())

    # ===== 凡例 =====
    h_u, l_u = unique_legend_entries(*ax.get_legend_handles_labels())

    hr, lr = ax2.get_legend_handles_labels()
    if hr:
        h_u += hr
        l_u += lr

    ax.legend(h_u, l_u, loc="upper left", framealpha=0.9)

    # 軸ラベル
    ax.set_xlabel("Num qubits", fontsize=15)
    if rz_layer:
        ax.set_ylabel("Num RZ layer", fontsize=15)
        ax2.set_ylabel("Difference in Num RZ layer", fontsize=15)
    else:
        ax.set_ylabel("T-depth", fontsize=15)
        ax2.set_ylabel("Difference in T-depth", fontsize=15)

    # グリッド
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="-",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.35,
    )

    plt.tight_layout()
    plt.show()


def best_product_formula_all(
    mol: str,
    ham_name: str,
    n_w_list: Sequence[PFLabel],
    use_original: bool = False,
) -> Tuple[Dict[str, List[float]], List[Dict[str, float]], Dict[str, Optional[float]]]:
    """精度掃引で最適な PF とコスト範囲を集計する（use_original=True で original を参照）。"""
    CA_list = [CA * (10 ** (-0.01 * i)) for i in range(-200, 300)]

    # PF ごとの結果テーブルを初期化
    result: Dict[str, List[float]] = {str(pair): [] for pair in n_w_list}
    CA_exp: Dict[str, Optional[float]] = {str(pair): None for pair in n_w_list}
    expo_dir: Dict[str, Optional[float]] = {str(pair): None for pair in n_w_list}
    coeff_dir: Dict[str, Optional[float]] = {str(pair): None for pair in n_w_list}
    cost_dir: Dict[str, Optional[float]] = {str(pair): None for pair in n_w_list}
    total_list: List[Dict[str, float]] = []

    for num_w in n_w_list:
        if num_w == "10th(Morales)" and mol == "H15":
            continue
        unit_expo = DECOMPO_NUM[mol][num_w]

        cost_dir[str(num_w)] = unit_expo

        target_path = f"{ham_name}_Operator_{num_w}_ave"
        # target_path = f"{ham_name}_Operator_{num_w}"

        try:
            # p(次数) 固定
            data = load_data(target_path, use_original=use_original)
            expo_dir[str(num_w)] = P_DIR[num_w]
            coeff_dir[str(num_w)] = data

            # p(次数) 固定なし
            # expo_dir[str(num_w)] = data['expo']
            # coeff_dir[str(num_w)] = data['coeff']

        except Exception as e:
            print(f"not found {target_path}")
            continue

    # 所望精度達成ゲート数計算
    for error_E in CA_list:
        min_total_expo = float("inf")
        best_trotter = None
        total: Dict[str, float] = {}
        for num_w in n_w_list:
            expo = expo_dir[str(num_w)]
            coeff = coeff_dir[str(num_w)]
            if expo is None or coeff is None:
                continue
            expo = float(expo)
            coeff = float(coeff)
            min_f = _compute_min_f(error_E, expo, coeff)

            # グルーピングあり
            unit_expo_val = cost_dir[str(num_w)]
            if unit_expo_val is None:
                continue

            total_expo = unit_expo_val * min_f
            # print(f'minf{min_f} cost{unit_expo_val} w{num_w}')
            total[str(num_w)] = total_expo
            if error_E == CA:
                CA_exp[str(num_w)] = total_expo

            if total_expo < min_total_expo:
                min_total_expo = total_expo
                best_trotter = str(num_w)
        total_list.append(total)
        error_fac = math.log10(error_E / CA)
        if best_trotter is None:
            continue
        result[best_trotter].append(error_fac)

    return result, total_list, CA_exp


def num_gate_plot_grouping(
    mol: str | int,
    n_w_list: Sequence[PFLabel] | None = None,
    *,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    use_original: bool = False,
) -> None:
    """係数データから総パウリ回転数をプロットする（グルーピング版）。"""
    # H-chain 表記を正規化
    if isinstance(mol, str):
        if not mol.startswith("H"):
            raise ValueError(f"Unsupported mol label: {mol}")
        mol_label = mol
        mol_type = int(mol[1:])
    else:
        mol_type = int(mol)
        mol_label = f"H{mol_type}"

    # 対象 PF が未指定なら分解テーブルから拾う
    if n_w_list is None:
        n_w_list = tuple(DECOMPO_NUM.get(mol_label, {}).keys())
    if not n_w_list:
        raise ValueError(f"No PF labels available for {mol_label}")

    # 係数データの保存名を組み立てる
    _, _, ham_name, _ = jw_hamiltonian_maker(mol_type, 1.0)
    ham_name = f"{ham_name}_grouping"

    # 誤差スイープに対するコストを集計
    CA_list = [CA * (10 ** (-0.01 * i)) for i in range(-200, 300)]
    result_gr, total_list_gr, _ = best_product_formula_all(
        mol_label, ham_name, n_w_list, use_original=use_original
    )

    # プロット
    plt.figure(figsize=(8, 6), dpi=200)
    for pf_key in n_w_list:
        pf_key = str(pf_key)
        x_vals = [err for err, d in zip(CA_list, total_list_gr) if pf_key in d]
        y_vals = [d[pf_key] for d in total_list_gr if pf_key in d]
        if not x_vals:
            continue
        label = label_replace(pf_key)
        color = COLOR_MAP.get(pf_key, "gray")
        plt.plot(x_vals, y_vals, label=label, color=color)

    # 最適 PF の精度帯を色付け
    for label, error_range in result_gr.items():
        if not error_range:
            continue
        min_error_expo = min(error_range)
        max_error_expo = max(error_range)
        min_error = CA * (10 ** (min_error_expo))
        max_error = CA * (10 ** (max_error_expo))
        label_name = label_replace(label)
        color = COLOR_MAP.get(label, "gray")
        plt.axvspan(
            min_error,
            max_error,
            color=color,
            alpha=0.3,
            label=f"{label_name}",
        )

    ax = plt.gca()
    ax.axvline(x=CA, color="r", linestyle="--", label="CA")
    set_loglog_axes(
        ax,
        xlabel="Target error [Hartree]",
        ylabel="Number of Pauli rotations",
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=12)
    plt.show()


def efficient_accuracy_range_plt_grouper(
    Hchain: int,
    n_w_list: Sequence[PFLabel],
    use_original: bool = False,
) -> None:
    """H-chain ごとの最適 PF の精度帯を描画する（use_original=True で original を参照）。"""
    xdic: Dict[str, int] = {}
    dic: Dict[str, Dict[str, List[float]]] = {str(n_w): {} for n_w in n_w_list}

    # H-chain ごとに最適 PF の精度範囲を集計
    for chain in range(2, Hchain + 1):
        mol = f"H{chain}"
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"

        xdic.setdefault(mol, n_qubits)
        result, _, _ = best_product_formula_all(
            mol, ham_name, n_w_list, use_original=use_original
        )

        for label, error_range in result.items():
            if error_range:
                min_error_expo = min(error_range)
                max_error_expo = max(error_range)
                accuracy_range = []
                accuracy_range.append(CA * (10 ** (min_error_expo)))
                accuracy_range.append(CA * (10 ** (max_error_expo)))
                dic[label].setdefault(mol, accuracy_range)
            else:
                continue
    """
    data = {
        'w2': { 'H2':[1, 5], 'H4':[2,6]},
        'w4': { 'H2':[6, 7], 'H4':[7,8]}
    }
    xdic = {'H2':4, 'H4':8}
    """

    def plot_with_horizontal_offset(
        data: Dict[str, Dict[str, List[float]]],
        xdic: Dict[str, int],
        offset: float = 0.2,
    ) -> None:

        plt.figure(figsize=(8, 6), dpi=200)

        # 凡例用のラベル管理
        color_labels: Set[str] = set()
        marker_labels: Set[str] = set()

        marker_map = {
            "H2": "o",
            "H3": "s",
            "H4": "^",
            "H5": "v",
            "H6": "<",
            "H7": ">",
            "H8": "D",
            "H9": "p",
            "H10": "h",
            "H11": "*",
            "H12": "x",
            "H13": "+",
            "H14": "P",
            "H15": "X",
        }

        x_offset_map: Dict[float, Dict[str, float]] = {}  # 横軸のオフセット管理

        def get_unique_x(x_base: float, shape: str, _label: str) -> float:
            # key = (shape, label)  # shape と label の組み合わせでユニークなキーを生成
            if x_base not in x_offset_map:
                x_offset_map[x_base] = {}
            if shape not in x_offset_map[x_base]:
                x_offset_map[x_base][shape] = len(x_offset_map[x_base]) * offset
            return x_base + x_offset_map[x_base][shape]

        for label, subsets in data.items():
            color = COLOR_MAP[label]  # グループ (w2, w4, w8) の色を取得
            for shape, y_range in subsets.items():
                x_base = xdic[shape]  # 基準横軸値を取得
                x_unique = get_unique_x(x_base, shape, label)  # 一意の横軸位置を取得
                plt.plot(
                    [x_unique, x_unique], y_range, marker=marker_map[shape], color=color
                )

                if label not in color_labels:
                    plt.plot([], [], color=color, label=label)
                    color_labels.add(label)

                if shape not in marker_labels:
                    plt.plot(
                        [],
                        [],
                        marker=marker_map[shape],
                        color="black",
                        linestyle="None",
                        label=shape,
                    )
                    marker_labels.add(shape)

        # グループ化された凡例の設定
        handles, labels = plt.gca().get_legend_handles_labels()
        color_handles = [h for h, l in zip(handles, labels) if l in data.keys()]
        color_labels_list = [l for l in labels if l in data.keys()]
        marker_handles = [h for h, l in zip(handles, labels) if l in xdic.keys()]
        marker_labels_list = [l for l in labels if l in xdic.keys()]

        ca_handle = plt.axhline(y=CA, color="r", linestyle="--", label="CA")

        # 凡例にCAを追加
        combined_handles = color_handles + marker_handles + [ca_handle]
        combined_labels_0 = color_labels_list + marker_labels_list + ["CA"]
        combined_labels_1 = [s.replace("_gr", "") for s in combined_labels_0]
        combined_labels = [s.replace("my1", "4(new)") for s in combined_labels_1]
        combined_labels_2 = [label_replace(lb) for lb in combined_labels]

        plt.yscale("log")
        plt.xlabel("Spin orbitals", fontsize=15)
        plt.ylabel("Target error [Hartree]", fontsize=15)
        plt.tick_params(labelsize=15)
        # plt.title("Accuracy range of each product formula for spin orbitals",fontsize=15)
        plt.legend(
            combined_handles,
            combined_labels_2,
            title=f"product \nformula",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            ncol=1,
            fontsize=13,
        )

        plt.show()

    plot_with_horizontal_offset(dic, xdic, offset=0.2)
