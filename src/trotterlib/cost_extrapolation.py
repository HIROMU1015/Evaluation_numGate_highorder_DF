from __future__ import annotations

import math
from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
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
    PFLabel,
    P_DIR,
    DECOMPO_NUM,
    PF_RZ_LAYER,
    PICKLE_DIR_DF_PATH,
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
    """DF artifact の rz_layers から利用するレイヤー値を選ぶ。"""
    candidate_keys: List[str] = []
    if preferred_key:
        candidate_keys.append(str(preferred_key))
    candidate_keys.extend(
        [
            "total_nonclifford_z_coloring_depth",
            "total_nonclifford_z_depth",
            "total_nonclifford_rz_depth",
            "ref_rz_depth",
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
    """DF の保存データ(trotter_expo_coeff_df)を使って T-depth / RZ レイヤー数を外挿する。"""
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
                    rz_layers_raw, preferred_key=rz_layer_key
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
