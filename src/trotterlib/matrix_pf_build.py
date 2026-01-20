from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from openfermion.linalg import get_sparse_operator
from scipy.sparse import eye
from scipy.sparse.linalg import expm

from .chemistry_hamiltonian import ham_list_maker
from .product_formula import _get_w_list
from .config import MATRIX_DIR, PFLabel
from .pf_decomposition import iter_pf_steps


def _make_time_dirs(
    matrix_root: os.PathLike[str] | str,
    t: float,
    terms_dir_name: str,
) -> str:
    """時刻ごとの保存ディレクトリを作成して返す。"""
    t_rounded = round(t, 10)
    time_dir = os.path.join(os.fspath(matrix_root), str(t_rounded))
    os.makedirs(time_dir, exist_ok=True)
    terms_dir = os.path.join(time_dir, terms_dir_name)
    os.makedirs(terms_dir, exist_ok=True)
    return terms_dir


def load_matrix_files(
    _n_qubits: int, t: float, ham_name: str, num_w: PFLabel | None
) -> Tuple[str, List[str]]:
    """t に対応する疎行列ファイル群を読み込み順序で返す（挙動不変）。"""
    # 保存ディレクトリを構築
    t = round(t, 10)
    if num_w is not None:
        matrix_dir = (
            MATRIX_DIR
            / f"{ham_name}_Operator_w{num_w}"
            / f"{t}"
            / f"{ham_name}_nostep_tOperator_w{num_w}"
        )
    else:
        matrix_dir = (
            MATRIX_DIR
            / f"{ham_name}_Operator_normal"
            / f"{t}"
            / f"{ham_name}_nostep_tOperator"
        )
    # "matrix_*.npz" を数値インデックス順にソート
    matrix_files = [
        name for name in os.listdir(matrix_dir) if name.startswith("matrix_")
    ]
    matrix_paths = sorted(
        matrix_files, key=lambda s: int(os.path.splitext(s)[0].split("_")[1])
    )
    return os.fspath(matrix_dir), matrix_paths


def eU_strage(
    jw_hamiltonian: Any,
    t: float,
    n_qubits: int,
    w: float,
    idx: int,
    folder_path: Union[str, os.PathLike[str]],
) -> int:
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

    def eU_exchanger(
        jw_hamiltonian: Any, t: float, n_qubits: int, w: float
    ) -> sp.spmatrix:
        """各項の e^{i w t H_j} を構成（既存ロジックを保持）。"""
        # 各項の指数演算子を構築
        def ham_to_cIsU(
            jw_hamiltonian: Any,
            coefficient: float,
            t: float,
            n_qubits: int,
            w: float,
        ) -> sp.spmatrix:
            # 係数で規格化した H を用い、cos・sin 展開で 1ステップ分の e^{iwtH} を構築
            for term in jw_hamiltonian:
                term_mat = get_sparse_operator(term, n_qubits)
                h_norm = term_mat / coefficient
                c = np.cos(w * t * coefficient)
                s = np.sin(w * t * coefficient)
                exp_op = (
                    complex(c, 0) * eye(2**n_qubits, format="csc")
                    + complex(0, s) * h_norm
                )
            return exp_op

        for term, coefficient in jw_hamiltonian.terms.items():
            if term:
                eU = ham_to_cIsU(jw_hamiltonian, coefficient, t, n_qubits, w)
            else:
                eU = expm(1j * eye(2**n_qubits, format="csc") * coefficient * t * w)
        return eU

    # 指数演算子を保存
    eU = eU_exchanger(jw_hamiltonian, t, n_qubits, w)
    sp.save_npz(os.path.join(folder_path, f"matrix_{idx}.npz"), eU)
    idx += 1
    return idx


def _apply_pf_steps(
    ham_list: Sequence[Any],
    t: float,
    n_qubits: int,
    w_list: Sequence[float],
    folder_path: Union[str, os.PathLike[str]],
    idx: int,
    *,
    verbose: bool = False,
) -> int:
    """PF の手順に沿って指数演算子を生成・保存する。"""
    # PF の順序に従って指数演算子を生成
    for step_idx, (term_idx, weight) in enumerate(
        iter_pf_steps(len(ham_list), w_list)
    ):
        idx = eU_strage(ham_list[term_idx], t, n_qubits, weight, idx, folder_path)
        if verbose and step_idx < len(ham_list) - 1:
            print(f"idx {idx} ham {ham_list[term_idx]}")
    return idx


def folder_maker_multiprocessing_values(
    t_values: Sequence[float],
    jw_hamiltonian: Any,
    n_qubits: int,
    ham_name: str,
    num_w: PFLabel | None,
) -> None:
    """t ごとにフォルダと e^{iHt} 分解を並列生成。32 並列を固定（挙動不変）。"""
    # t のリストを分割して並列処理
    workers = 32
    partition_size = (len(t_values) + workers - 1) // workers
    t_partitions = [
        t_values[i * partition_size : (i + 1) * partition_size] for i in range(workers)
    ]

    if num_w is not None:
        # 高次 PF 用のフォルダ生成
        task_args_weighted = [
            (jw_hamiltonian, t, n_qubits, ham_name, num_w) for t in t_partitions
        ]
        with Pool(processes=workers) as pool:
            pool.starmap(w_trotter_folder_maker_multi, task_args_weighted)
    else:
        # 通常 PF 用のフォルダ生成
        task_args_normal = [(jw_hamiltonian, t, n_qubits, ham_name) for t in t_partitions]
        with Pool(processes=workers) as pool:
            pool.starmap(normal_trotter_folder_maker_multi, task_args_normal)
    print("done")


def normal_trotter_folder_maker_multi(
    jw_hamiltonian: Any,
    t_list: Sequence[float],
    n_qubits: int,
    ham_name: str,
) -> None:
    """num_w=None の通常版フォルダ生成。"""
    # 時刻ごとの保存ディレクトリを用意
    matrix_root = MATRIX_DIR / f"{ham_name}_Operator_normal"
    os.makedirs(matrix_root, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    for t in t_list:
        terms_dir = _make_time_dirs(
            matrix_root, t, f"{ham_name}_nostep_tOperator"
        )
        # 各項の指数演算子を保存
        idx = 0
        for term in ham_list:
            idx = eU_strage(term, t, n_qubits, 1, idx, terms_dir)


def w_trotter_folder_maker_multi(
    jw_hamiltonian: Any,
    t_list: Sequence[float],
    n_qubits: int,
    ham_name: str,
    num_w: PFLabel,
) -> None:
    """重み付き（高次）PF のフォルダ生成。"""
    # 時刻ごとの保存ディレクトリを用意
    matrix_root = MATRIX_DIR / f"{ham_name}_Operator_w{num_w}"
    os.makedirs(matrix_root, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    w_list = _get_w_list(num_w)
    m = len(w_list)
    for t in t_list:
        terms_dir = _make_time_dirs(
            matrix_root, t, f"{ham_name}_nostep_tOperator_w{num_w}"
        )
        # PF 分解に従って指数演算子を保存
        idx = 0
        if m == 1:
            print(f"m{m}")
            idx = _apply_pf_steps(
                ham_list, t, n_qubits, w_list, terms_dir, idx, verbose=True
            )
            print(f"idx{idx}")
            continue
        idx = _apply_pf_steps(ham_list, t, n_qubits, w_list, terms_dir, idx)
