from __future__ import annotations

from multiprocessing import Pool
from typing import Any, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, save_npz

from .config import (
    CALCULATION_DIR,
    MATRIX_DIR,
    PFLabel,
    POOL_PROCESSES,
    ensure_artifact_dirs,
    pf_order,
)

from .chemistry_hamiltonian import (
    jw_hamiltonian_maker,
    ham_list_maker,
    ham_ground_energy,
    min_hamiltonian_grouper,
)
from .qiskit_time_evolution_grouping import tEvolution_vector_grouper
from .qiskit_time_evolution_pyscf import (
    make_fci_vector_from_pyscf_solver,
    make_fci_vector_from_pyscf_solver_grouper,
)
from .qiskit_time_evolution_ungrouped import tEvolution_vector
from .io_cache import save_data
from .matrix_pf_build import (
    folder_maker_multiprocessing_values,
    load_matrix_files,
)
from .matrix_multiply import multi_parallel_sparse_matrix_multiply_recursive
from .eig_error import error_cal_multi
from .analysis_utils import LogLogFitResult, loglog_average_coeff, loglog_fit, print_loglog_fit
from .plot_utils import set_loglog_axes


def trotter_error_plt(
    t_start: float,
    t_end: float,
    t_step: float,
    molecule_type: int,
    pf_label: PFLabel,
) -> Tuple[List[float], List[float]]:
    """対角化版の時間発展誤差を log–log でプロット（GRスタイル）。挙動不変。"""
    n_w = pf_order(pf_label)
    series_label = f"{pf_label}"
    # 時間刻みの候補を生成
    t_values = [round(t, 5) for t in np.arange(t_start, t_end, t_step)]

    # Hamiltonian と基底状態の準備
    jw_hamiltonian, _, ham_name, n_qubits = jw_hamiltonian_maker(molecule_type)
    energy, state_vec, _ = ham_ground_energy(jw_hamiltonian)

    # 出力先のディレクトリ設定
    ensure_artifact_dirs(include_pickle_dirs=False)
    num_w_dir = f"w{pf_label}"
    directory_path = CALCULATION_DIR / ham_name
    directory_path.mkdir(parents=True, exist_ok=True)

    # 未生成の t だけ行列生成をキック
    make_folder_t = []
    for t in t_values:
        ham_dir = MATRIX_DIR / f"{ham_name}_Operator_{num_w_dir}" / f"{t}"
        term_dir_path = directory_path / f"t_{t}_{num_w_dir}.npz"
        if not term_dir_path.exists() and not ham_dir.exists():
            make_folder_t.append(t)
    if len(make_folder_t) > 0:
        folder_maker_multiprocessing_values(
            make_folder_t, jw_hamiltonian, n_qubits, ham_name, pf_label
        )

    # 事前計算済みの行列を読み込み、無ければ生成して保存
    evolution_ops = []
    for t in t_values:
        term_path = directory_path / f"t_{t}_{num_w_dir}.npz"
        if not term_path.exists():
            folder_path, file_path = load_matrix_files(n_qubits, t, ham_name, pf_label)
            term = multi_parallel_sparse_matrix_multiply_recursive(
                folder_path, file_path, 32
            )
            evolution_ops.append(term)
            save_npz(term_path, term)
            print("save")
        else:
            data = load_npz(term_path)
            evolution_ops.append(data)

    # 多時刻の誤差を一括評価
    t_list, error_list = error_cal_multi(
        t_values, evolution_ops, state_vec, energy, num_eig=1
    )
    print("multiprocessing done")

    # log-log フィットと係数の集計
    ave_coeff = loglog_average_coeff(
        t_list, error_list, n_w, mask_nonpositive=False
    )

    # 回帰と r^2（log–log）
    fit = loglog_fit(t_list, error_list, mask_nonpositive=False, compute_r2=True)
    print_loglog_fit(fit, ave_coeff=ave_coeff)

    # プロット
    ax = plt.gca()
    set_loglog_axes(
        ax,
        title=f"{ham_name}_{series_label}",
        xlabel="time",
        ylabel="error",
    )
    ax.plot(t_list, error_list, marker="o", linestyle="-")
    plt.show()
    return t_list, error_list


def _collect_perturbation_errors(
    final_state_list: Sequence[Tuple[Any, ...]],
    energy: float,
    state_vec: np.ndarray,
) -> Tuple[List[float], List[float], List[int]]:
    """摂動論ベースの誤差と対応する時間列を集計する。"""
    error_list_perturb: List[float] = []
    times_out: List[float] = []
    counts: list[int] = []
    for item in final_state_list:
        t = item[0]
        statevector = item[1]
        time = -t
        statevector = statevector.data.reshape(-1, 1)
        phase_factor = np.exp(1j * energy * time)
        delta_state = (statevector - (phase_factor * state_vec)) / (1j * time)
        overlap = state_vec.conj().T @ delta_state
        overlap = overlap.real / np.cos(energy * time)
        error_list_perturb.extend(np.abs(overlap.real))
        times_out.append(time)
        rotation_count = int(item[2]) if len(item) > 2 else 0
        counts.append(rotation_count)
    return times_out, error_list_perturb, counts


def _prepare_grouped_qc_data(molecule_type: int):
    jw_hamiltonian, _, ham_name, num_qubits = jw_hamiltonian_maker(molecule_type)
    ham_terms = ham_list_maker(jw_hamiltonian)
    constant_term = next(iter(ham_terms[0].terms.values())).real

    if molecule_type in (2, 3):
        energy, state_vec, _ = ham_ground_energy(jw_hamiltonian)
        grouped_ops, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
        commuting_cliques = [[op] for op in grouped_ops]
    else:
        commuting_cliques, num_qubits, energy, state_vec = make_fci_vector_from_pyscf_solver_grouper(
            molecule_type
        )

    energy = energy - constant_term
    ham_name = f"{ham_name}_grouping"
    return commuting_cliques, num_qubits, energy, state_vec, ham_name


def _qc_gr_error_series(
    t_start: float,
    t_end: float,
    t_step: float,
    molecule_type: int,
    pf_label: PFLabel,
    *,
    prepped: tuple[list[Any], int, float, np.ndarray, str] | None = None,
) -> tuple[list[float], list[float], list[int], str, float, LogLogFitResult]:
    times = list(np.arange(t_start, t_end, t_step))
    neg_times = [-1.0 * t for t in times]
    if prepped is None:
        commuting_cliques, num_qubits, energy, state_vec, ham_name = _prepare_grouped_qc_data(
            molecule_type
        )
    else:
        commuting_cliques, num_qubits, energy, state_vec, ham_name = prepped  # type: ignore[assignment]
    task_args = [
        (commuting_cliques, t, num_qubits, state_vec, pf_label) for t in neg_times
    ]
    with Pool(processes=POOL_PROCESSES) as pool:
        final_state_list = pool.starmap(tEvolution_vector_grouper, task_args)

    times_out, error_list_perturb, counts = _collect_perturbation_errors(
        final_state_list,
        energy,
        state_vec,
    )

    avg_coeff = loglog_average_coeff(
        times_out, error_list_perturb, pf_order(pf_label), mask_nonpositive=False
    )
    fit = loglog_fit(
        times_out, error_list_perturb, mask_nonpositive=False, compute_r2=True
    )
    return times_out, error_list_perturb, counts, ham_name, avg_coeff, fit


def trotter_error_qc_gr_curve(
    t_start: float,
    t_end: float,
    t_step: float,
    molecule_type: int,
    pf_label: PFLabel,
 ) -> tuple[list[float], list[float], list[int], str, float, LogLogFitResult]:
    return _qc_gr_error_series(t_start, t_end, t_step, molecule_type, pf_label)


def plot_trotter_error_curve(
    times: Sequence[float],
    errors: Sequence[float],
    *,
    title: str,
    fit_result: LogLogFitResult | None = None,
) -> None:
    ax = plt.gca()
    set_loglog_axes(
        ax,
        title=title,
        xlabel="time",
        ylabel="error",
    )
    ax.plot(times, errors, marker="s", linestyle="-")
    if fit_result is not None:
        alpha = fit_result.coeff
        p = fit_result.slope
        fit_curve = [alpha * (t**p) for t in times]
        ax.plot(
            times,
            fit_curve,
            linestyle="--",
            label=f"fit: α={alpha:.2e}, p={p:.2f}",
        )
        ax.text(
            0.05,
            0.95,
            f"error exponent: {p:.2f}\nerror coefficient: {alpha:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    ax.legend()
    plt.show()


def trotter_error_plt_qc(
    t_start: float,
    t_end: float,
    t_step: float,
    molecule_type: int,
    pf_label: PFLabel,
) -> None:
    """非グルーピングQC版の時間発展誤差を log–log でプロット"""
    n_w = pf_order(pf_label)
    # 時間刻みを用意（符号を反転して計算）
    times = list(np.arange(t_start, t_end, t_step))
    neg_times = [-1.0 * t for t in times]

    # FCI ベースの基底状態・エネルギーを取得
    jw_hamiltonian, _, energy, state_vec, _ = make_fci_vector_from_pyscf_solver(
        molecule_type
    )
    _, _, _, num_qubits = jw_hamiltonian_maker(molecule_type)

    # 各 t の時間発展を並列に計算
    hamiltonian_terms = ham_list_maker(jw_hamiltonian)
    task_args = [
        (hamiltonian_terms, t, num_qubits, state_vec, pf_label) for t in neg_times
    ]
    with Pool(processes=32) as pool:
        final_state_list = pool.starmap(tEvolution_vector, task_args)

    # 摂動論ベースの誤差を集計
    times_out, error_list_perturb, _ = _collect_perturbation_errors(
        final_state_list,
        energy,
        state_vec,
    )

    # 指数は PF 次数テーブルを使用
    ave_coeff = loglog_average_coeff(
        times_out, error_list_perturb, n_w, mask_nonpositive=False
    )

    # 回帰と r^2（log–log）
    fit = loglog_fit(
        times_out, error_list_perturb, mask_nonpositive=False, compute_r2=True
    )
    print_loglog_fit(fit, ave_coeff=ave_coeff)

    # 対角化ベースの誤差も同図で比較
    _diag_times, error_list_ph = trotter_error_plt(
        t_start,
        t_end,
        t_step,
        molecule_type,
        pf_label,
    )

    ax = plt.gca()
    set_loglog_axes(
        ax,
        xlabel="Time",
        ylabel="Eigenvalue error [Hartree]",
        xlabel_kwargs={"fontsize": 15},
        ylabel_kwargs={"fontsize": 15},
    )
    ax.plot(
        times_out, error_list_ph, marker="s", linestyle="-", label="Diagonalization"
    )
    ax.legend()
    plt.show()

    # 摂動論ベースのプロット
    ax = plt.gca()
    set_loglog_axes(
        ax,
        xlabel="Time",
        ylabel="Eigenvalue error [Hartree]",
        xlabel_kwargs={"fontsize": 15},
        ylabel_kwargs={"fontsize": 15},
    )
    ax.plot(
        times_out,
        error_list_perturb,
        marker="s",
        linestyle="-",
        label="Perturbation",
        color="green",
    )
    ax.legend()
    plt.show()

    # 対角化との差分をプロット
    perturb_error = []
    for ph, per in zip(error_list_ph, error_list_perturb):
        perturb_error.append(abs(ph - per))

    ax = plt.gca()
    set_loglog_axes(
        ax,
        xlabel="Time",
        ylabel="Algorithm error [Hartree]",
        xlabel_kwargs={"fontsize": 15},
        ylabel_kwargs={"fontsize": 15},
    )
    ax.plot(times_out, perturb_error, marker="o", linestyle="-", color="red")
    plt.show()


def trotter_error_plt_qc_gr(
    t_start: float,
    t_end: float,
    t_step: float,
    molecule_type: int,
    pf_label: PFLabel,
    save_fit_params: bool,  # フィッティング結果保存
    save_avg_coeff: bool,  # fixed-p 保存
) -> None:
    """グルーピング版の時間発展誤差を log–log でプロット（挙動不変）。"""
    n_w = pf_order(pf_label)
    series_label = f"{pf_label}"
    # 時間刻みを用意（符号を反転して計算）
    times = list(np.arange(t_start, t_end, t_step))
    neg_times = [-1.0 * t for t in times]

    commuting_cliques, num_qubits, energy, state_vec, ham_name = _prepare_grouped_qc_data(
        molecule_type
    )
    print(f"energy_{energy}")
    times_out, error_list_perturb, _, ham_name, avg_coeff, fit = _qc_gr_error_series(
        t_start,
        t_end,
        t_step,
        molecule_type,
        pf_label,
        prepped=(commuting_cliques, num_qubits, energy, state_vec, ham_name),
    )
    print_loglog_fit(fit, ave_coeff=avg_coeff)

    # フィッティング結果の保存（必要時）
    if save_fit_params is True:
        data = {"expo": fit.slope, "coeff": fit.coeff}
        if pf_label is not None:
            target_path = f"{ham_name}_Operator_{pf_label}"
        else:
            target_path = f"{ham_name}_Operator_normal"
        save_data(target_path, data, gr=True)

    if save_avg_coeff is True:
        if pf_label is not None:
            target_path = f"{ham_name}_Operator_{pf_label}_ave"
        else:
            target_path = f"{ham_name}_Operator_normal_ave"
        save_data(target_path, avg_coeff, True)
    

    plot_trotter_error_curve(
        times_out,
        error_list_perturb,
        title=f"{ham_name}_{series_label}",
        fit_result=fit,
    )
