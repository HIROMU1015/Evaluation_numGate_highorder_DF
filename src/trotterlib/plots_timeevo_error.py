from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Optional, List, Tuple

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from scipy.sparse import load_npz, save_npz  # type: ignore

import qiskit_time_evolution as qte  # type: ignore

from config import POOL_PROCESSES, P_DIR

from chemistry_hamiltonian import (
    jw_hamiltonian_maker,
    ham_list_maker,
    ham_ground_energy,
    min_hamiltonian_grouper,
)
from io_cache import (
    save_data,
    load_data,
)
from matrix_pf_build import (
    folder_maker_multiprocessing_values,
    load_matrix_files,
)
from matrix_multiply import multi_parallel_sparse_matrix_multiply_recursive
from eig_error import error_cal_multi


def trotter_error_plt(
    s_time,
    e_time,
    dividing,
    mol_type,
    num_w,
):
    """対角化版の時間発展誤差を log–log でプロット（GRスタイル、ヘルパ未使用）。挙動不変。"""
    series_label = f"{num_w}"
    t_values = [round(t, 5) for t in np.arange(s_time, e_time, dividing)]

    hamiltonian, _, ham_name, n_qubits = jw_hamiltonian_maker(mol_type)
    E, ori_vec, _ = ham_ground_energy(hamiltonian)

    # 出力先
    parent_dir = os.path.join(os.getcwd(), "calculation")
    num_w_dir = "normal" if num_w is None else f"w{num_w}"
    directory_path = os.path.join(parent_dir, ham_name)
    os.makedirs(directory_path, exist_ok=True)

    make_folder_t = []
    for t in t_values:
        ham_dir = os.path.join(
            parent_dir, "matrix", f"{ham_name}_Operator_{num_w_dir}", f"{t}"
        )
        term_dir_path = os.path.join(directory_path, f"t_{t}_{num_w_dir}.npz")
        if not os.path.exists(term_dir_path) and not os.path.exists(ham_dir):
            make_folder_t.append(t)
    if len(make_folder_t) > 0:
        folder_maker_multiprocessing_values(
            make_folder_t, hamiltonian, n_qubits, ham_name, num_w
        )

    term_list = []
    for t in t_values:
        term_dir_path = os.path.join(directory_path, f"t_{t}_{num_w_dir}.npz")
        if not os.path.exists(term_dir_path):
            folder_path, file_path = load_matrix_files(n_qubits, t, ham_name, num_w)
            term = multi_parallel_sparse_matrix_multiply_recursive(
                folder_path, file_path, 32
            )
            term_list.append(term)
            save_npz(term_dir_path, term)
            print("save")
        else:
            data = load_npz(term_dir_path)
            term_list.append(data)

    t_list, error_list = error_cal_multi(t_values, term_list, ori_vec, E, num_eig=1)
    print("multiprocessing done")

    error_log = np.log10(error_list)
    time_log = np.log10(t_list)

    n_w = P_DIR[num_w]
    set_expo_error = error_log - n_w * time_log
    ave_coeff = 10 ** (np.mean(set_expo_error))

    # 回帰と r^2（log–log）
    linear_error = np.polyfit(time_log, error_log, 1)
    slope = linear_error[0]
    coeff = 10 ** linear_error[1]
    print("error exponent :" + str(slope))
    print("error coefficient :" + str(coeff))
    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)
    corr = np.corrcoef(x_flat, y_flat)[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)
    print(f"average_coeff:{ave_coeff}")

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title(f"{ham_name}_{series_label}")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.plot(t_list, error_list, marker="o", linestyle="-")
    plt.show()
    return t_list, error_list


def trotter_error_plt_qc(
    s_time,
    e_time,
    dividing,
    mol_type,
    num_w,
):
    """非グルーピングQC版の時間発展誤差を log–log でプロット"""
    t_values = list(np.arange(s_time, e_time, dividing))
    neg_t_values = [-1.0 * t for t in t_values]

    jw_hamiltonian, _, E, ori_vec, _ = qte.make_fci_vector_from_pyscf_solver(mol_type)
    _, _, _, num_qubits = jw_hamiltonian_maker(mol_type)

    clique_list = ham_list_maker(jw_hamiltonian)
    task_args = [(clique_list, t, num_qubits, ori_vec, num_w) for t in neg_t_values]
    with Pool(processes=32) as pool:
        final_state_list = pool.starmap(qte.tEvolution_vector, task_args)

    error_list_pertur = []
    t_values = []
    for t, vector in final_state_list:
        t *= -1
        vector = vector.data.reshape(-1, 1)
        tevolution = np.exp(1j * E * t)
        delta_psi = (vector - (tevolution * ori_vec)) / (1j * t)
        innerproduct = ori_vec.conj().T @ delta_psi
        innerproduct = innerproduct.real / np.cos(E * t)
        error_list_pertur.extend(abs((innerproduct.real)))
        t_values.append(t)

    error_log = np.log10(error_list_pertur)
    time_log = np.log10(t_values)

    # 指数は P_DIR を使用（ヘルパ不使用）
    n_w = P_DIR[num_w]
    set_expo_error = error_log - n_w * time_log
    ave_coeff = 10 ** (np.mean(set_expo_error))

    # 回帰と r^2（log–log）
    linear_error = np.polyfit(time_log, error_log, 1)
    slope = linear_error[0]
    coeff = 10 ** linear_error[1]
    print("error exponent :" + str(slope))
    print("error coefficient :" + str(coeff))
    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)
    corr = np.corrcoef(x_flat, y_flat)[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)
    print(f"average_coeff:{ave_coeff}")

    # 対角化ベースの誤差も同図で比較
    t_list, error_list_ph = trotter_error_plt(
        s_time,
        e_time,
        dividing,
        mol_type,
        num_w
    )

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Eigenvalue error [Hartree]", fontsize=15)
    plt.plot(
        t_values, error_list_ph, marker="s", linestyle="-", label="Diagonalization"
    )
    plt.legend()
    plt.show()

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Eigenvalue error [Hartree]", fontsize=15)
    plt.plot(
        t_values,
        error_list_pertur,
        marker="s",
        linestyle="-",
        label="Perturbation",
        color="green",
    )
    plt.legend()
    plt.show()

    pertuer_error = []
    for ph, per in zip(error_list_ph, error_list_pertur):
        pertuer_error.append(abs(ph - per))

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Algorithm error [Hartree]", fontsize=15)
    plt.plot(t_values, pertuer_error, marker="o", linestyle="-", color="red")
    plt.show()


def trotter_error_plt_qc_gr(
    s_time: float,
    e_time: float,
    dividing: float,
    mol_type: str,
    num_w: str,
    storage: bool,  # フィッティング結果保存
    avestrage: bool,  # fixed-p 保存
):
    """グルーピング版の時間発展誤差を log–log でプロット（挙動不変）。"""
    series_label = f"{num_w}"
    t_values = list(np.arange(s_time, e_time, dividing))
    neg_t_values = [-1.0 * t for t in t_values]

    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type)
    ham_list = ham_list_maker(jw)
    # 定数項（アイデンティティ項）の実数部を取得
    const = next(iter(ham_list[0].terms.values())).real

    if mol_type in (2, 3):
        jw_hamiltonian = jw
        E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
        clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    else:
        clique_list, num_qubits, E, ori_vec = (
            qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)
        )

    E = E - const
    print(f"energy_{E}")
    ham_name = f"{ham_name}_grouping"

    task_args = [(clique_list, t, num_qubits, ori_vec, num_w) for t in neg_t_values]
    with Pool(processes=POOL_PROCESSES) as pool:
        final_state_list = pool.starmap(qte.tEvolution_vector_grouper, task_args)

    error_list_pertur = []
    t_values = []
    for t, vector, _ in final_state_list:
        t *= -1
        vector = vector.data.reshape(-1, 1)
        tevolution = np.exp(1j * E * t)
        delta_psi = (vector - (tevolution * ori_vec)) / (1j * t)
        innerproduct = ori_vec.conj().T @ delta_psi
        innerproduct = innerproduct.real / np.cos(E * t)
        error_list_pertur.extend(abs(innerproduct.real))
        t_values.append(t)

    error_log = np.log10(error_list_pertur)
    time_log = np.log10(t_values)

    n_w = P_DIR[num_w]

    set_expo_error = error_log - n_w * time_log
    ave_coeff = np.mean(set_expo_error)
    ave_coeff = 10 ** (ave_coeff)

    linear_error = np.polyfit(time_log, error_log, 1)
    print("error exponent :" + str(linear_error[0]))
    # In Y = CX^a, logY = AlogX + B as 10^B = C
    coeff = 10 ** linear_error[1]
    print("error coefficient :" + str(coeff))
    print(f"average_coeff:{ave_coeff}")

    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)

    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr = corr_matrix[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)

    if storage is True:
        data = {"expo": linear_error[0], "coeff": coeff}
        if num_w is not None:
            target_path = f"{ham_name}_Operator_{num_w}"
        else:
            target_path = f"{ham_name}_Operator_normal"
        save_data(target_path, data, gr=True)

    if avestrage is True:
        if num_w is not None:
            target_path = f"{ham_name}_Operator_{num_w}_ave"
        else:
            target_path = f"{ham_name}_Operator_normal_ave"
        save_data(target_path, ave_coeff, True)
    

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title(f"{ham_name}_{series_label}")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.plot(t_values, error_list_pertur, marker="s", linestyle="-")
    plt.show()
