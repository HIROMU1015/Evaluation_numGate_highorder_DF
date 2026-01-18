from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any, List, Optional, Tuple

import numpy as np  # type: ignore
import scipy.sparse as sp  # type: ignore
from openfermion.linalg import get_sparse_operator  # type: ignore
from scipy.sparse import eye  # type: ignore
from scipy.sparse.linalg import expm  # type: ignore

from chemistry_hamiltonian import ham_list_maker
from product_formula import _get_w_list
from config import POOL_PROCESSES


def load_matrix_files(n_qubits, t, ham_name, num_w):
    """t に対応する疎行列ファイル群を読み込み順序で返す（挙動不変）。"""
    t = round(t, 10)
    sub = (
        f"{ham_name}_Operator_w{num_w}/{t}/{ham_name}_nostep_tOperator_w{num_w}"
        if num_w is not None
        else f"{ham_name}_Operator_normal/{t}/{ham_name}_nostep_tOperator"
    )
    directory_path = os.path.join(os.getcwd(), "matrix", sub)
    # "matrix_*.npz" を数値インデックス順にソート
    matrix_files = [
        name for name in os.listdir(directory_path) if name.startswith("matrix_")
    ]
    matrix_paths = sorted(
        matrix_files, key=lambda s: int(os.path.splitext(s)[0].split("_")[1])
    )
    return directory_path, matrix_paths


def eU_strage(jw_hamiltonian, t, n_qubits, w, idx, folder_path):
    """
    ハミルトニアンの項を疎行列の指数関数行列に変換し指定のフォルダに保存

    Args:
        jw_hamiltonian: JW変換されたハミルトニアン,
        t:  時間発展演算子のt
        n_qubits: ハミルトニアンのQビット数
        w: 積公式の係数
        idx: トロッター展開されたハミルトニアンの項のインデックス
        folder_path: 保存先のフォルダ

    Returns:
        idx: トロッター展開されたハミルトニアンの項のインデックス
    """

    def eU_exchanger(jw_hamiltonian, t, n_qubits, w):
        """各項の e^{i w t H_j} を構成（既存ロジックを保持）。"""
        dim = 2**n_qubits
        I = eye(dim, format="csc")

        def ham_to_cIsU(jw_hamiltonian, coefficient, t, n_qubits, w):
            # 係数で規格化した H を用い、cos・sin 展開で 1ステップ分の e^{iwtH} を構築
            for term in jw_hamiltonian:
                term_mat = get_sparse_operator(term, n_qubits)
                h_norm = term_mat / coefficient
                c = np.cos(w * t * coefficient)
                s = np.sin(w * t * coefficient)
                exp_op = complex(c, 0) * I + complex(0, s) * h_norm
            return exp_op

        for term, coefficient in jw_hamiltonian.terms.items():
            if term:
                eU = ham_to_cIsU(jw_hamiltonian, coefficient, t, n_qubits, w)
            else:
                eU = expm(1j * eye(2**n_qubits, format="csc") * coefficient * t * w)
        return eU

    eU = eU_exchanger(jw_hamiltonian, t, n_qubits, w)
    sp.save_npz(os.path.join(folder_path, f"matrix_{idx}.npz"), eU)
    idx += 1
    return idx

def S_2(ham_list, t, n_qubits, w, folder_path, idx):  # 左端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w / 2, idx, folder_path)
        print(f"idx {idx} ham {ham_list[i]}")
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w, idx, folder_path)
    # 終端まで
    for k in reversed(range(0, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w / 2, idx, folder_path)
    return idx


def S_2_trotter_left(ham_list, t, n_qubits, Max_w, nMax_w, folder_path, idx):  # 左端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, Max_w / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, Max_w, idx, folder_path)
    # 終端 - 1 個目まで
    for k in reversed(range(1, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, Max_w / 2, idx, folder_path)
    # 終端
    idx = eU_strage(ham_list[0], t, n_qubits, (Max_w + nMax_w) / 2, idx, folder_path)
    return idx


def S_2_trotter(ham_list, t, n_qubits, w_f, w_s, folder_path, idx):  # 左端、右端以外
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(1, J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w_f / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w_f, idx, folder_path)
    # 終端 - 1 個目まで
    for k in reversed(range(1, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w_f / 2, idx, folder_path)
    # 終端
    idx = eU_strage(ham_list[0], t, n_qubits, (w_f + w_s) / 2, idx, folder_path)
    return idx


def S_2_trotter_right(ham_list, t, n_qubits, w_i, folder_path, idx):  # 右端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(1, J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w_i / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w_i, idx, folder_path)
    # 終端 まで
    for k in reversed(range(0, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w_i / 2, idx, folder_path)
    return idx


def folder_maker_multiprocessing_values(
    t_values, jw_hamiltonian, n_qubits, ham_name, num_w
):
    """t ごとにフォルダと e^{iHt} 分解を並列生成。32 並列を固定（挙動不変）。"""
    workers = 32
    partition_size = (len(t_values) + workers - 1) // workers
    t_partitions = [
        t_values[i * partition_size : (i + 1) * partition_size] for i in range(workers)
    ]

    if num_w is not None:
        task_args = [
            (jw_hamiltonian, t, n_qubits, ham_name, num_w, i)
            for i, t in enumerate(t_partitions)
        ]
        with Pool(processes=workers) as pool:
            pool.starmap(w_trotter_folder_maker_multi, task_args)
    else:
        task_args = [
            (jw_hamiltonian, t, n_qubits, ham_name, i)
            for i, t in enumerate(t_partitions)
        ]
        with Pool(processes=workers) as pool:
            pool.starmap(normal_trotter_folder_maker_multi, task_args)
    print("done")


def normal_trotter_folder_maker_multi(
    jw_hamiltonian, t_list, n_qubits, ham_name, core_num
):
    """num_w=None の通常版フォルダ生成。"""
    base = os.path.join(os.getcwd(), "matrix", f"{ham_name}_Operator_normal")
    os.makedirs(base, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    for t in t_list:
        mt = round(t, 10)
        target_dir = os.path.join(base, str(mt))
        os.makedirs(target_dir, exist_ok=True)
        folder_path = os.path.join(target_dir, f"{ham_name}_nostep_tOperator")
        os.makedirs(folder_path, exist_ok=True)
        idx = 0
        for term in ham_list:
            idx = eU_strage(term, t, n_qubits, 1, idx, folder_path)


def w_trotter_folder_maker_multi(
    jw_hamiltonian, t_list, n_qubits, ham_name, num_w, core_num
):
    """重み付き（高次）PF のフォルダ生成。"""
    parent_dir = os.path.join(os.getcwd(), "matrix", f"{ham_name}_Operator_w{num_w}")
    os.makedirs(parent_dir, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    w_list = _get_w_list(num_w)
    m = len(w_list)
    for t in t_list:
        mt = round(t, 10)
        target_dir = os.path.join(parent_dir, str(mt))
        os.makedirs(target_dir, exist_ok=True)
        folder_path = os.path.join(target_dir, f"{ham_name}_nostep_tOperator_w{num_w}")
        os.makedirs(folder_path, exist_ok=True)
        idx = 0
        if m == 1:
            print(f"m{m}")
            idx = S_2(ham_list, t, n_qubits, w_list[0], folder_path, idx)
            print(f"idx{idx}")
            continue
        # w_m
        idx = S_2_trotter_left(
            ham_list, t, n_qubits, w_list[m - 1], w_list[m - 2], folder_path, idx
        )
        # w_{m-1} ~ w_1
        for i in reversed(range(1, m - 1)):
            idx = S_2_trotter(
                ham_list, t, n_qubits, w_list[i], w_list[i - 1], folder_path, idx
            )
        # w_0 ~ w_{m-1}
        for i in range(0, m - 1):
            idx = S_2_trotter(
                ham_list, t, n_qubits, w_list[i], w_list[i + 1], folder_path, idx
            )
        # w_m
        idx = S_2_trotter_right(ham_list, t, n_qubits, w_list[m - 1], folder_path, idx)
