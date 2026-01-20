from typing import Any, Sequence, Tuple

from openfermion.ops import QubitOperator

from qiskit import QuantumCircuit
from qiskit.circuit.library import GlobalPhaseGate, PauliEvolutionGate
from qiskit.quantum_info import Statevector

from .config import PFLabel
from .qiskit_time_evolution_utils import (
    apply_time_evolution,
    _get_w_list,
    term_to_sparse_pauli,
)
from .pf_decomposition import iter_pf_steps


def add_term_to_circuit(
    hamiltonian: QubitOperator,
    num_qubits: int,
    time: float,
    weight: float,
    circuit: QuantumCircuit,
) -> None:
    """1項分の時間発展ゲートを回路に追加する。"""
    # ハミルトニアンの項からゲートを回路に追加
    for term, coeff in hamiltonian.terms.items():
        # 回転角度の計算
        rotation_angle = coeff.real * weight * time
        # PauliEvolutionGate を作成し、量子回路に追加
        if not term:
            # 恒等項は位相として追加
            circuit.append(GlobalPhaseGate(-1 * rotation_angle))
            return

        # パウリ演算子のテンソル積を構築
        pauli_tensor = term_to_sparse_pauli(tuple(term), num_qubits)
        evolution_gate = PauliEvolutionGate(
            pauli_tensor, time=rotation_angle, synthesis=None
        )
        circuit.append(evolution_gate, range(num_qubits))
    return


def w_trotter(
    circuit: QuantumCircuit,
    hamiltonian_terms: Sequence[QubitOperator],
    time: float,
    num_qubits: int,
    pf_label: PFLabel,
) -> None:
    """与えられた w シリーズで PF 分解を回路に追加し、累計の指数項数を返す。"""
    # PF 係数列に従って項を追加
    weights = _get_w_list(pf_label)
    for term_idx, weight in iter_pf_steps(len(hamiltonian_terms), weights):
        add_term_to_circuit(
            hamiltonian_terms[term_idx], num_qubits, time, weight, circuit
        )
    return


def tEvolution_vector(
    hamiltonian_terms: Sequence[QubitOperator],
    time: float,
    num_qubits: int,
    state_vec: Any,
    pf_label: PFLabel,
) -> Tuple[float, Statevector]:
    """非グルーピング版の時間発展回路を合成し、最終状態と指数項数を返す。"""
    # 回路を組み立てて時間発展
    evolution_circuit = QuantumCircuit(num_qubits)
    w_trotter(evolution_circuit, hamiltonian_terms, time, num_qubits, pf_label)
    final_statevector = apply_time_evolution(state_vec, evolution_circuit)
    return time, final_statevector
