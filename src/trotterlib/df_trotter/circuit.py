from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from ..config import PFLabel
from ..pf_decomposition import iter_pf_steps
from ..product_formula import _get_w_list
from .model import Block


def build_df_trotter_circuit(
    blocks: Sequence[Block],
    time: float,
    num_qubits: int,
    pf_label: PFLabel,
    energy_shift: float = 0.0,
) -> QuantumCircuit:
    """Build a product-formula circuit over mixed block types."""
    qc = QuantumCircuit(num_qubits)
    if energy_shift != 0.0:
        qc.global_phase += -float(energy_shift) * float(time)
    weights = _get_w_list(pf_label)
    for term_idx, weight in iter_pf_steps(len(blocks), weights):
        blocks[term_idx].apply(qc, weight * time)
    return qc


def simulate_statevector(qc: QuantumCircuit, psi0: np.ndarray) -> np.ndarray:
    """Evolve a statevector without explicit initialization gates."""
    global_phase = float(getattr(qc, "global_phase", 0.0) or 0.0)
    if global_phase != 0.0:
        qc = qc.copy()
        qc.global_phase = 0.0
    sv = Statevector(psi0)
    final_sv = sv.evolve(qc)
    vec = np.asarray(final_sv.data)
    if global_phase != 0.0:
        vec = np.exp(1j * global_phase) * vec
    return vec


def estimate_energy(statevector: np.ndarray, hamiltonian: Any) -> float:
    """Estimate <psi|H|psi> for common Hamiltonian representations."""
    vec = np.asarray(statevector).reshape(-1)
    if hasattr(hamiltonian, "terms"):
        from openfermion.linalg import get_sparse_operator

        ham_matrix = get_sparse_operator(hamiltonian)
        energy = np.vdot(vec, ham_matrix.dot(vec))
        return float(np.real_if_close(energy))
    if hasattr(hamiltonian, "to_matrix"):
        ham_matrix = hamiltonian.to_matrix()
        energy = np.vdot(vec, ham_matrix @ vec)
        return float(np.real_if_close(energy))

    ham_matrix = hamiltonian
    if hasattr(ham_matrix, "dot"):
        energy = np.vdot(vec, ham_matrix.dot(vec))
    else:
        energy = np.vdot(vec, ham_matrix @ vec)
    return float(np.real_if_close(energy))


def report_cost(qc: QuantumCircuit, *, basis_gates: Sequence[str] | None = None) -> dict:
    """Report gate counts, optionally after transpilation."""
    report = {"count_ops": qc.count_ops()}
    if basis_gates is not None:
        transpiled = transpile(qc, basis_gates=list(basis_gates), optimization_level=1)
        report["count_ops_transpiled"] = transpiled.count_ops()
    return report
