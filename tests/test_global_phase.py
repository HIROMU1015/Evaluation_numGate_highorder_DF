import numpy as np

from qiskit import QuantumCircuit

from trotterlib.df_trotter.circuit import simulate_statevector


def test_simulate_statevector_applies_global_phase() -> None:
    qc = QuantumCircuit(1)
    qc.global_phase = 0.123
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi = simulate_statevector(qc, psi0)
    expected = np.exp(1j * 0.123) * psi0
    assert np.allclose(psi, expected)
