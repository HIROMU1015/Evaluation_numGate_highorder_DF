from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Patch  # type: ignore
from matplotlib.ticker import LogFormatterMathtext, LogLocator  # type: ignore

from Evaluation_numGate_highorder.src.trotterlib.config import (
    CA,
    BETA,
    PICKLE_DIR_GROUPED,
    COLOR_MAP,
    MARKER_MAP,
    P_DIR,
    DECOMPO_NUM,
    PF_RZ_LAYER,
)

from Evaluation_numGate_highorder.src.trotterlib.io_cache import load_data, label_replace
from Evaluation_numGate_highorder.src.trotterlib.chemistry_hamiltonian import jw_hamiltonian_maker
from Evaluation_numGate_highorder.src.trotterlib.product_formula import _get_w_list


def calculation_cost(clique_list, num_w, ham_name):
    """
    分解数計算用(decompo_num) clique_num_dir にそのクリークに含まれる項をインデックス付きで登録
    """

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


def exp_extrapolation(
    Hchain, n_w_list, show_bands=True, band_height=0.06, band_alpha=0.28
):
    
    Hchain_list = [i for i in range(2, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]

    eps = CA / 10

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR_GROUPED)
    total_dir = {}

    plt.figure(figsize=(8, 6), dpi=200)

    num_qubits = [i for i in range(4, (Hchain * 2) + 1, 2)]

    for chain, qubit, mol in zip(Hchain_list, num_qubits, Hchain_str):
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"

        total_dir[n_qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and n_qubits == 30:  # 10th は H15 で未評価
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = P_DIR[n_w]

            min_f = (
                BETA
                * (eps ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = DECOMPO_NUM[mol][n_w]
            total_expo = unit_expo * min_f

            total_dir[n_qubits][n_w] = total_expo

    series = defaultdict(lambda: {"x": [], "y": []})
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
    ax.set_xscale("log")
    ax.set_yscale("log")

    XMAX = 100
    xmin_current, xmax_current = ax.get_xlim()
    ax.set_xlim(xmin_current, max(xmax_current, XMAX))

    # ---- フィット ----
    fit_params = {}
    for lb, d in series.items():
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        m = (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        B, log10A = np.polyfit(np.log10(x), np.log10(y), 1)
        A = 10**log10A
        fit_params[lb] = {"A": A, "B": B}

        x_right = ax.get_xlim()[1]
        xfit = np.logspace(np.log10(x.min()), np.log10(x_right), 400)
        yfit = A * (xfit**B)
        ax.plot(xfit, yfit, "-", color=COLOR_MAP.get(lb), alpha=0.9, linewidth=1.5)

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

        winners_in_order = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            lb = labels_fit[imin[s]]
            if lb not in winners_in_order:
                winners_in_order.append(lb)
            ax.axvspan(
                xgrid[s],
                xgrid[e],
                ymin=0.0,
                ymax=band_height,  # ★ 高さを引数で
                color=COLOR_MAP.get(lb, "0.6"),
                alpha=band_alpha,  # ★ 透明度を引数で
                ec="none",
                zorder=0,
            )

        # 縦点線（任意）
        for i in switch_idx:
            ax.axvline(
                xgrid[i], linestyle=":", linewidth=1.0, color="k", alpha=0.5, zorder=1
            )

    # ---- 凡例 ----
    from matplotlib.patches import Patch

    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u, labels_u = [], []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)

    # 色帯も凡例に出すなら（show_bands True のときだけ）
    if show_bands and fit_params:
        # winners_in_order を上で作っているので、無ければ復元
        if "winners_in_order" not in locals():
            x_left, x_right = ax.get_xlim()
            xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)
            labels_fit = list(fit_params.keys())
            Y = np.vstack(
                [
                    fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"])
                    for lb in labels_fit
                ]
            )
            imin = np.argmin(Y, axis=0)
            switch_idx = np.where(np.diff(imin) != 0)[0] + 1
            bounds = np.r_[0, switch_idx, len(xgrid) - 1]
            winners_in_order = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                lb = labels_fit[imin[s]]
                if lb not in winners_in_order:
                    winners_in_order.append(lb)

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
        labels_u += [p.get_label() for p in proxies]

    ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    plt.show()


def exp_extrapolation_diff(
    Hchain,
    n_w_list=("4th(new_2)", "8th(Morales)"),
    MIN_POS=1e-18,
    X_MIN_CALC=4,
    X_MAX_DISPLAY=100,
):
    """
    単一図（左右Y軸）:
      左Y: 総パウリ回転数（散布 + log–log フィット）
      右Y: 2本のフィット直線の絶対差 |Δ|
    依存：decompo_num, optimal_distance, jw_hamiltonian_maker, load_data, label_replace,
         MARKER_MAP, COLOR_MAP, P_DIR
    """

    # 対象 H チェーン
    Hchain_list = [i for i in range(2, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]
    num_qubits = [i for i in range(4, (Hchain * 2) + 1, 2)]

    eps = CA / 10

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR_GROUPED)

    # 総回転数の算出
    total_dir = {}
    for chain, qubit, mol in zip(Hchain_list, num_qubits, Hchain_str):
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"
        total_dir[n_qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and n_qubits == 30:  # 10th は H15 で未評価
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = P_DIR[n_w]

            min_f = (
                BETA
                * (eps ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = DECOMPO_NUM[mol][n_w]
            total_expo = unit_expo * min_f

            total_dir[n_qubits][n_w] = total_expo

    # ---- プロット（単一図・双Y軸）----
    plt.figure(figsize=(8, 6), dpi=200)

    series = defaultdict(lambda: {"x": [], "y": []})
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
    ax.set_xscale("log")
    ax.set_yscale("log")

    # x の右端を固定
    x_left_auto, _ = ax.get_xlim()
    ax.set_xlim(x_left_auto, X_MAX_DISPLAY)

    # ---- フィット ----
    fit_params = {}
    xfit_lo = max(X_MIN_CALC, x_left_auto)
    xfit_hi = X_MAX_DISPLAY

    for lb, d in series.items():
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        m = (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        B, log10A = np.polyfit(np.log10(x), np.log10(y), 1)
        A = 10**log10A
        fit_params[lb] = {"A": A, "B": B}

        x_right = ax.get_xlim()[1]
        xfit = np.logspace(np.log10(x.min()), np.log10(x_right), 400)
        yfit = A * (xfit**B)
        ax.plot(xfit, yfit, "-", color=COLOR_MAP.get(lb), alpha=0.9, linewidth=1.5)

    # ---- 左軸：凡例 ----
    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u = []
    labels_u = []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)
    if handles_u:
        ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

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
        Hchain, n_w_list, rz_layer=None,show_bands=True, band_height=0.06, band_alpha=0.28
        ):

    eps = CA / 10

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR_GROUPED)
    total_dir = {}

    num_qubits = [i for i in range(4,(Hchain*2)+1,2)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]

    plt.figure(figsize=(8, 6), dpi=200)

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
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = P_DIR[n_w]

            # N_0 PF による分解に含まれる RZ 数 T-depth 計算なら実質パウリ回転数
            N_0 = DECOMPO_NUM[mol][n_w]

            # L_Z RZ のレイヤー数
            pf_layer_rz = PF_RZ_LAYER[mol][n_w]

            t = (eps / coeff * (expo + 1))**(1/expo)
            eps_qpe = eps * (expo / (expo + 1))
            M_qpe = BETA / (eps_qpe * t)

            # RZ の近似誤差は許容誤差の 1 パーセント
            eps_rot = (t * 0.01 * eps) / (N_0 * M_qpe)
            
            # RZ 近似誤差 T = 3log2(1/eps_rot)
            T_rot = 3 * np.log2(1/eps_rot)

            # PF のユニタリ１回分の T-depth
            D_T = pf_layer_rz * T_rot

            # QPE QC 全体での T-depth
            tot_dt = M_qpe * D_T

            # QPE QC 全体での RZ レイヤー数
            tot_rz_layer = M_qpe * pf_layer_rz

            if qubits == 30:
                print(f'T_rot:{T_rot} PF:{n_w}')

            if rz_layer:
                total_dir[qubits][n_w] =  tot_rz_layer
            else:
                total_dir[qubits][n_w] = tot_dt

    series = defaultdict(lambda: {"x": [], "y": []})
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
    ax.set_xscale("log")
    ax.set_yscale("log")

    XMAX = 100
    xmin_current, xmax_current = ax.get_xlim()
    ax.set_xlim(xmin_current, max(xmax_current, XMAX))

    # ---- フィット ----
    fit_params = {}
    for lb, d in series.items():
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        m = (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        B, log10A = np.polyfit(np.log10(x), np.log10(y), 1)
        A = 10**log10A
        fit_params[lb] = {"A": A, "B": B}

        x_right = ax.get_xlim()[1]
        xfit = np.logspace(np.log10(x.min()), np.log10(x_right), 400)
        yfit = A * (xfit ** B)
        ax.plot(xfit, yfit, '-', color=COLOR_MAP.get(lb), alpha=0.9, linewidth=1.5)

    # ---- 色帯の表示（トグル）----
    if show_bands and fit_params:
        # 表示範囲に合わせて評価用グリッドを作成
        x_left, x_right = ax.get_xlim()
        xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)

        labels_fit = list(fit_params.keys())
        # 各ラベルの y(x)
        Y = np.vstack([fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit])
        imin = np.argmin(Y, axis=0)   # 各 x で最小の系列のインデックス

        # 勝者切替の境界
        switch_idx = np.where(np.diff(imin) != 0)[0] + 1
        bounds = np.r_[0, switch_idx, len(xgrid)-1]

        winners_in_order = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            lb = labels_fit[imin[s]]
            if lb not in winners_in_order:
                winners_in_order.append(lb)
            ax.axvspan(xgrid[s], xgrid[e],
                       ymin=0.0, ymax=band_height,  # ★ 高さを引数で
                       color=COLOR_MAP.get(lb, "0.6"),
                       alpha=band_alpha,            # ★ 透明度を引数で
                       ec="none", zorder=0)

        # 縦点線（任意）
        for i in switch_idx:
            ax.axvline(xgrid[i], linestyle=":", linewidth=1.0, color="k", alpha=0.5, zorder=1)

    # ---- 凡例 ----
    from matplotlib.patches import Patch
    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u, labels_u = [], []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h); labels_u.append(lab); seen.add(lab)

    # 色帯も凡例に出すなら（show_bands True のときだけ）
    if show_bands and fit_params:
        # winners_in_order を上で作っているので、無ければ復元
        if 'winners_in_order' not in locals():
            x_left, x_right = ax.get_xlim()
            xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)
            labels_fit = list(fit_params.keys())
            Y = np.vstack([fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit])
            imin = np.argmin(Y, axis=0)
            switch_idx = np.where(np.diff(imin) != 0)[0] + 1
            bounds = np.r_[0, switch_idx, len(xgrid)-1]
            winners_in_order = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                lb = labels_fit[imin[s]]
                if lb not in winners_in_order:
                    winners_in_order.append(lb)

        proxies = [Patch(facecolor=COLOR_MAP[lb], alpha=0.6, edgecolor="none", label=f"{lb} (lowest)")
                   for lb in winners_in_order
                   if f"{lb} (lowest)" not in seen]
        handles_u += proxies
        labels_u += [p.get_label() for p in proxies]

    ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    if rz_layer:
        ax.set_ylabel("Num RZ layer", fontsize=15)
    else:
        ax.set_ylabel("T-depth", fontsize=15)
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)
    plt.show()


def t_depth_extrapolation_diff(
    Hchain,
    rz_layer=None,
    n_w_list=("4th(new_2)", "8th(Morales)"),
    MIN_POS=1e-18,
    X_MIN_CALC=4,
    X_MAX_DISPLAY=100,
):
    """
    2つの PF を指定して T-depth を比較するプロット関数（双Y軸）。

      左Y: QPE 全体の T-depth（散布 + log–log フィット）
      右Y: フィット直線同士の絶対差 |ΔT|（ログ）

    - n_w_list, num_w_list は必ず 2 つずつ渡す
    - スケーリング直線/Δ の評価は x >= X_MIN_CALC のみ
    - x の右端は X_MAX_DISPLAY に固定

    依存: decompo_num, PF_RZ_layer, optimal_distance,
          jw_hamiltonian_maker, load_data, label_replace
    """

    eps = CA / 10

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR_GROUPED)

    # PF 表示名 → ラベル
    label_map = {pf: label_replace(pf) for pf in n_w_list}
    labels = [label_map[pf] for pf in n_w_list]

    num_qubits = [i for i in range(4,(Hchain*2)+1,2)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]

    # ===== T-depth の計算 =====
    total_dir = {}  # {n_qubits: {pf_key: T-depth}}

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
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = P_DIR[n_w]

            # N_0 PF による分解に含まれる RZ 数 T-depth 計算なら実質パウリ回転数
            N_0 = DECOMPO_NUM[mol][n_w]

            # L_Z RZ のレイヤー数
            pf_layer_rz = PF_RZ_LAYER[mol][n_w]

            t = (eps / coeff * (expo + 1))**(1/expo)
            eps_qpe = eps * (expo / (expo + 1))
            M_qpe = BETA / (eps_qpe * t)

            # RZ の近似誤差は許容誤差の 1 パーセント
            eps_rot = (t * 0.01 * eps) / (N_0 * M_qpe)
            
            # RZ 近似誤差 T = 3log2(1/eps_rot)
            T_rot = 3 * np.log2(1/eps_rot)

            # PF のユニタリ１回分の T-depth
            D_T = pf_layer_rz * T_rot

            # QPE QC 全体での T-depth
            tot_dt = M_qpe * D_T

            # QPE QC 全体での RZ レイヤー数
            tot_rz_layer = M_qpe * pf_layer_rz

            if qubits == 30:
                print(f'T_rot:{T_rot} PF:{n_w}')

            if rz_layer:
                total_dir[qubits][n_w] =  tot_rz_layer
            else:
                total_dir[qubits][n_w] = tot_dt

    # ===== プロット（左Y: T-depth, 右Y: 差分）=====
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    ax2 = ax.twinx()

    # 散布用データ（キーは pf
    series = defaultdict(lambda: {"x": [], "y": []})
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
    ax.set_xscale("log")
    ax.set_yscale("log")

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

        B, log10A = np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)
        A = 10.0 ** log10A
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

        def yfit(pf, x):
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
    h, l = ax.get_legend_handles_labels()
    seen = set()
    h_u, l_u = [], []
    for hh, ll in zip(h, l):
        if ll not in seen:
            h_u.append(hh)
            l_u.append(ll)
            seen.add(ll)

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


def best_product_formula_all(mol, ham_name, n_w_list):
    CA_list = [CA * (10 ** (-0.01 * i)) for i in range(-200, 300)]

    result = {str(pair): [] for pair in n_w_list}
    CA_exp = {str(pair): [] for pair in n_w_list}
    expo_dir = {str(pair): None for pair in n_w_list}
    coeff_dir = {str(pair): None for pair in n_w_list}
    cost_dir = {str(pair): None for pair in n_w_list}
    total_list = []

    for num_w in n_w_list:
        unit_expo = DECOMPO_NUM[mol][num_w]

        cost_dir[str(num_w)] = unit_expo

        target_path = f"{ham_name}_Operator_{num_w}_ave"
        # target_path = f"{ham_name}_Operator_{num_w}"

        try:
            # p(次数) 固定
            data = load_data(target_path)
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
        total = {}
        for num_w in n_w_list:
            expo = expo_dir[str(num_w)]
            coeff = coeff_dir[str(num_w)]
            if expo == None or coeff == None:
                continue
            expo = float(expo)
            coeff = float(coeff)
            min_f = (
                BETA
                * (error_E ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = cost_dir[str(num_w)]

            total_expo = unit_expo * min_f
            # print(f'minf{min_f} cost{unit_expo} w{num_w}')
            total[str(num_w)] = total_expo
            if error_E == CA:
                CA_exp[str(num_w)] = total_expo

            if total_expo < min_total_expo:
                min_total_expo = total_expo
                best_trotter = str(num_w)
        total_list.append(total)
        error_fac = math.log10(error_E / CA)
        result[best_trotter].append(error_fac)

    return result, total_list, CA_exp


def efficient_accuracy_range_plt_grouper(Hchain, n_w_list):
    xdic = {}
    dic = {str(n_w): {} for n_w in n_w_list}

    for chain in range(2, Hchain + 1):
        mol = f"H{chain}"
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"

        xdic.setdefault(mol, n_qubits)
        result, _, _ = best_product_formula_all(mol, ham_name, n_w_list)

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

    def plot_with_horizontal_offset(data, xdic, offset=0.2):

        plt.figure(figsize=(8, 6), dpi=200)

        # 凡例用のラベル管理
        color_labels = set()
        marker_labels = set()

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

        x_offsets = {}  # 横軸のオフセット管理

        def get_unique_x(x_base, shape, label):
            # key = (shape, label)  # shape と label の組み合わせでユニークなキーを生成
            if x_base not in x_offsets:
                x_offsets[x_base] = {}
            if shape not in x_offsets[x_base]:
                x_offsets[x_base][shape] = len(x_offsets[x_base]) * offset
            return x_base + x_offsets[x_base][shape]

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
        color_labels = [l for l in labels if l in data.keys()]
        marker_handles = [h for h, l in zip(handles, labels) if l in xdic.keys()]
        marker_labels = [l for l in labels if l in xdic.keys()]

        ca_handle = plt.axhline(y=CA, color="r", linestyle="--", label="CA")

        # 凡例にCAを追加
        combined_handles = color_handles + marker_handles + [ca_handle]
        combined_labels_0 = color_labels + marker_labels + ["CA"]
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
