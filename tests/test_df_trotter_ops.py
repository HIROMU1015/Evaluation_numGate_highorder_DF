import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RZGate
from qiskit.quantum_info import Operator

from trotterlib.df_trotter.ops import apply_D_squared, apply_df_block


def _expected_d_squared(eta: np.ndarray, lam: float, tau: float) -> np.ndarray:
    num_qubits = len(eta)
    dim = 2**num_qubits
    diag = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        occ = [(basis >> k) & 1 for k in range(num_qubits)]
        total = sum(e * n for e, n in zip(eta, occ))
        diag[basis] = np.exp(-1j * tau * lam * (total**2))
    return np.diag(diag)


def test_apply_d_squared_matches_matrix() -> None:
    eta = np.array([0.7, -0.3], dtype=float)
    lam = 0.5
    tau = 0.2
    qc = QuantumCircuit(2)
    apply_D_squared(qc, eta, lam, tau)
    actual = Operator(qc).data
    expected = _expected_d_squared(eta, lam, tau)
    assert np.allclose(actual, expected, atol=1e-10)


def test_apply_df_block_uses_u_then_u_dagger() -> None:
    eta = np.array([0.4], dtype=float)
    lam = 1.1
    tau = 0.3
    u_ops = [(RZGate(0.2), (0,)), (RXGate(0.5), (0,))]

    qc = QuantumCircuit(1)
    apply_df_block(qc, u_ops, eta, lam, tau)
    actual = Operator(qc).data

    u_circ = QuantumCircuit(1)
    u_circ.append(RZGate(0.2), [0])
    u_circ.append(RXGate(0.5), [0])
    U = Operator(u_circ).data

    alpha = tau * lam * eta[0] * eta[0]
    D = np.diag([1.0, np.exp(-1j * alpha)])
    expected = U @ D @ U.conj().T
    assert np.allclose(actual, expected, atol=1e-10)
