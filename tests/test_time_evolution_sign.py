import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from trotterlib.df_trotter.circuit import build_df_trotter_circuit
from trotterlib.df_trotter.ops import (
    apply_df_block,
    apply_one_body_gaussian_block,
)


def _expected_one_body_unitary(eps: np.ndarray, tau: float) -> np.ndarray:
    num_qubits = len(eps)
    dim = 2**num_qubits
    diag = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        occ = [(basis >> k) & 1 for k in range(num_qubits)]
        energy = sum(e * n for e, n in zip(eps, occ))
        diag[basis] = np.exp(-1j * tau * energy)
    return np.diag(diag)


def _expected_df_unitary(eta: np.ndarray, lam: float, tau: float) -> np.ndarray:
    num_qubits = len(eta)
    dim = 2**num_qubits
    diag = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        occ = [(basis >> k) & 1 for k in range(num_qubits)]
        total = sum(e * n for e, n in zip(eta, occ))
        diag[basis] = np.exp(-1j * tau * lam * (total**2))
    return np.diag(diag)


def test_one_body_gaussian_block_sign() -> None:
    eps = np.array([0.3, -0.2], dtype=float)
    tau = 0.4
    qc = QuantumCircuit(2)
    apply_one_body_gaussian_block(qc, [], eps, tau)
    actual = Operator(qc).data
    expected = _expected_one_body_unitary(eps, tau)
    assert np.allclose(actual, expected, atol=1e-10)


def test_df_block_sign() -> None:
    eta = np.array([0.5, -0.1], dtype=float)
    lam = 0.8
    tau = 0.25
    qc = QuantumCircuit(2)
    apply_df_block(qc, [], eta, lam, tau)
    actual = Operator(qc).data
    expected = _expected_df_unitary(eta, lam, tau)
    assert np.allclose(actual, expected, atol=1e-10)


def test_build_df_trotter_circuit_energy_shift() -> None:
    num_qubits = 2
    time = 0.3
    energy_shift = 1.25
    qc = build_df_trotter_circuit(
        [], time=time, num_qubits=num_qubits, pf_label="2nd", energy_shift=energy_shift
    )
    actual = Operator(qc).data
    expected = np.exp(-1j * energy_shift * time) * np.eye(2**num_qubits)
    assert np.allclose(actual, expected, atol=1e-10)
