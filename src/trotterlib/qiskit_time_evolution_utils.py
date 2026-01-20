from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .config import PFLabel
from .product_formula import _get_w_list as _pf_get_w_list


def free_var(name: str, scope: dict) -> None:
    """ローカル変数を解放して GC を促す（メモリ圧を下げるための補助）。"""
    # スコープから削除して GC を促進
    if name in scope:
        del scope[name]
        import gc

        gc.collect()


def apply_time_evolution(
    eigenvector: np.ndarray, time_evolution_circuit: QuantumCircuit
) -> Statevector:
    """|ψ⟩ を初期化せず、Statevector.evolve で時間発展させた最終状態を返す。"""
    # Statevector を直接 evolve させる
    sv = Statevector(eigenvector)
    final_sv = sv.evolve(time_evolution_circuit)
    return final_sv


def term_to_sparse_pauli(
    term: Tuple[Tuple[int, str], ...],
    n_qubits: int,
) -> SparsePauliOp:
    """OpenFermion term を Qiskit の SparsePauliOp に変換する。"""
    # 注意: 右端が q0（Qiskit）。既存挙動維持のためラベル反転は行わない。
    X = SparsePauliOp("X")
    Y = SparsePauliOp("Y")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")
    pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
    pauli_operators = [I] * n_qubits
    for index, pauli_op_name in term:
        pauli_operators[index] = pauli_dict[pauli_op_name]
    pauli_op = pauli_operators[0]
    for op in pauli_operators[1:]:
        pauli_op ^= op
    return pauli_op


def _get_w_list(num_w: PFLabel) -> List[float]:
    """積公式パラメータ w の系列を取得（分岐を関数化）。"""
    # product_formula 側の実装を使用
    return _pf_get_w_list(num_w)
