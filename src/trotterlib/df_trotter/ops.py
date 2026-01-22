from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

from .decompose import diag_hermitian
from .model import DFBlock, OneBodyGaussianBlock

_MAX_DENSE_GAUSSIAN_QUBITS = 8


def _bit_reverse_permutation(num_qubits: int) -> np.ndarray:
    dim = 1 << num_qubits
    perm = np.zeros(dim, dtype=int)
    for i in range(dim):
        x = i
        r = 0
        for _ in range(num_qubits):
            r = (r << 1) | (x & 1)
            x >>= 1
        perm[i] = r
    return perm

def _as_real(value: Any, name: str, *, atol: float = 1e-12) -> float:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > atol:
            raise ValueError(f"{name} has non-negligible imaginary part.")
        return float(arr.real)
    return float(arr)


def _append_rzz(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
    if hasattr(qc, "rzz"):
        qc.rzz(theta, q0, q1)
        return
    try:
        from qiskit.circuit.library import RZZGate

        qc.append(RZZGate(theta), [q0, q1])
    except Exception:
        qc.cx(q0, q1)
        qc.rz(theta, q1)
        qc.cx(q0, q1)


def _import_fermionic_gaussian_decomposition() -> Any:
    try:
        from qiskit_nature.second_q.circuit.library import (
            fermionic_gaussian_decomposition_jw,
        )
    except ImportError:
        try:
            from qiskit_nature.circuit.library import (
                fermionic_gaussian_decomposition_jw,
            )
        except ImportError as exc:
            raise ImportError(
                "qiskit-nature is required for fermionic Gaussian decomposition."
            ) from exc
    return fermionic_gaussian_decomposition_jw


def _import_bogoliubov_transform() -> Any:
    try:
        from qiskit_nature.second_q.circuit.library import BogoliubovTransform
    except ImportError:
        try:
            from qiskit_nature.circuit.library import BogoliubovTransform
        except ImportError as exc:
            raise ImportError(
                "qiskit-nature is required for Bogoliubov transform decomposition."
            ) from exc
    return BogoliubovTransform


def _dense_gaussian_unitary_ops(U: np.ndarray) -> list[tuple[Any, Tuple[int, ...]]]:
    from openfermion import FermionOperator
    from openfermion.linalg import get_sparse_operator
    try:
        from scipy.linalg import expm as _scipy_expm
        from scipy.linalg import logm as _scipy_logm
    except Exception:
        _scipy_expm = None
        _scipy_logm = None

    try:
        from qiskit.circuit.library import UnitaryGate
    except Exception:
        from qiskit.extensions import UnitaryGate  # type: ignore

    num_qubits = U.shape[0]
    if num_qubits > _MAX_DENSE_GAUSSIAN_QUBITS:
        raise ImportError(
            "qiskit-nature is required for fermionic Gaussian decomposition "
            f"when num_qubits > {_MAX_DENSE_GAUSSIAN_QUBITS}."
        )

    if _scipy_logm is not None:
        anti_herm = _scipy_logm(U)
        anti_herm = 0.5 * (anti_herm - anti_herm.conj().T)
    else:
        evals, evecs = np.linalg.eig(U)
        angles = np.angle(evals)
        anti_herm = evecs @ np.diag(1j * angles) @ np.linalg.inv(evecs)
        anti_herm = 0.5 * (anti_herm - anti_herm.conj().T)
    op = FermionOperator()
    for p in range(num_qubits):
        for q in range(num_qubits):
            coeff = anti_herm[p, q]
            if abs(coeff) < 1e-14:
                continue
            op += FermionOperator(((p, 1), (q, 0)), coeff)

    mat = get_sparse_operator(op, n_qubits=num_qubits).toarray()
    if _scipy_expm is not None:
        unitary = _scipy_expm(mat)
    else:
        evals, evecs = np.linalg.eig(mat)
        unitary = evecs @ np.diag(np.exp(evals)) @ np.linalg.inv(evecs)
    perm = _bit_reverse_permutation(num_qubits)
    unitary = unitary[np.ix_(perm, perm)]
    gate = UnitaryGate(unitary)
    return [(gate, tuple(range(num_qubits)))]


def _qubit_index(qr: QuantumRegister, qubit: Any) -> int:
    if isinstance(qubit, int):
        return qubit
    if hasattr(qubit, "index"):
        return int(qubit.index)
    return int(qr.index(qubit))


def _normalize_u_ops(ops: Any, qr: QuantumRegister) -> list[tuple[Any, Tuple[int, ...]]]:
    if hasattr(ops, "data"):
        data = ops.data
    else:
        data = ops
    normalized = []
    for item in data:
        if hasattr(item, "operation") and hasattr(item, "qubits"):
            gate = item.operation
            qubits = item.qubits
        elif isinstance(item, tuple) and len(item) >= 2:
            gate = item[0]
            qubits = item[1]
        else:
            raise TypeError("Unsupported operation format from decomposition.")
        qubit_indices = tuple(_qubit_index(qr, q) for q in qubits)
        normalized.append((gate, qubit_indices))
    return normalized


def U_to_qiskit_ops_jw(U: np.ndarray) -> list[tuple[Any, Tuple[int, ...]]]:
    """Convert a fermionic Gaussian unitary into Qiskit ops under JW mapping."""
    U = np.asarray(U)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix.")
    num_qubits = U.shape[0]
    if num_qubits <= _MAX_DENSE_GAUSSIAN_QUBITS:
        try:
            return _dense_gaussian_unitary_ops(U)
        except Exception:
            pass

    try:
        BogoliubovTransform = _import_bogoliubov_transform()
    except ImportError:
        return _dense_gaussian_unitary_ops(U)

    circ = BogoliubovTransform(U)
    qr = circ.qregs[0] if circ.qregs else QuantumRegister(num_qubits)
    return _normalize_u_ops(circ, qr)


def apply_D_one_body(qc: QuantumCircuit, eps: np.ndarray, tau: float) -> float:
    """Apply exp(-i tau sum_k eps_k n_k) using RZ and global phase."""
    eps = np.asarray(eps)
    phase = 0.0
    for k, eps_k in enumerate(eps):
        angle = _as_real(-tau * eps_k, "rz_angle")
        if abs(angle) > 0.0:
            qc.rz(angle, k)
        phase += angle / 2.0
    qc.global_phase += _as_real(phase, "global_phase")
    return phase


def apply_D_squared(
    qc: QuantumCircuit, eta: np.ndarray, lam: float, tau: float
) -> float:
    """Apply exp(-i tau lam (sum_k eta_k n_k)^2) using RZ/RZZ and global phase."""
    eta = np.asarray(eta)
    lam = _as_real(lam, "lambda")
    tau = _as_real(tau, "tau")
    tau = -tau
    num_qubits = len(eta)
    rz_angles = np.zeros(num_qubits, dtype=float)
    phase = 0.0

    for k in range(num_qubits):
        alpha = _as_real(tau * lam * eta[k] * eta[k], "alpha")
        rz_angles[k] += alpha
        phase += alpha / 2.0

    for k in range(num_qubits):
        for j in range(k + 1, num_qubits):
            beta = _as_real(2.0 * tau * lam * eta[k] * eta[j], "beta")
            rz_angles[k] += beta / 2.0
            rz_angles[j] += beta / 2.0
            phase += beta / 4.0
            _append_rzz(qc, -beta / 2.0, k, j)

    for k, angle in enumerate(rz_angles):
        if abs(angle) > 0.0:
            qc.rz(angle, k)

    qc.global_phase += _as_real(phase, "global_phase")
    return phase


def _append_u_ops(
    qc: QuantumCircuit, u_ops: Iterable[tuple[Any, Tuple[int, ...]]], *, inverse: bool
) -> None:
    ops = list(u_ops)
    if inverse:
        ops = list(reversed(ops))
    for gate, qubits in ops:
        op = gate.inverse() if inverse else gate
        qc.append(op, list(qubits))


def apply_df_block(
    qc: QuantumCircuit,
    u_ops: Sequence[tuple[Any, Tuple[int, ...]]],
    eta: np.ndarray,
    lam: float,
    tau: float,
) -> None:
    _append_u_ops(qc, u_ops, inverse=True)
    apply_D_squared(qc, eta, lam, tau)
    _append_u_ops(qc, u_ops, inverse=False)


def apply_one_body_gaussian_block(
    qc: QuantumCircuit,
    u_ops: Sequence[tuple[Any, Tuple[int, ...]]],
    eps: np.ndarray,
    tau: float,
) -> None:
    _append_u_ops(qc, u_ops, inverse=True)
    apply_D_one_body(qc, eps, tau)
    _append_u_ops(qc, u_ops, inverse=False)


def apply_pauli_block(qc: QuantumCircuit, qubit_op: Any, tau: float) -> None:
    from ..qiskit_time_evolution_ungrouped import add_term_to_circuit

    add_term_to_circuit(qubit_op, qc.num_qubits, tau, 1.0, qc)


def build_df_blocks(
    model: Any, *, sort: str = "descending_abs"
) -> list[DFBlock]:
    blocks = []
    for lam, g_mat in zip(model.lambdas, model.G_list):
        U, eta = diag_hermitian(g_mat, sort=sort)
        u_ops = U_to_qiskit_ops_jw(U)
        blocks.append(DFBlock(U_ops=u_ops, eta=eta, lam=_as_real(lam, "lambda")))
    return blocks


def build_one_body_gaussian_block(
    one_body: np.ndarray, *, sort: str = "descending_abs"
) -> OneBodyGaussianBlock:
    U, eps = diag_hermitian(one_body, sort=sort)
    u_ops = U_to_qiskit_ops_jw(U)
    return OneBodyGaussianBlock(U_ops=u_ops, eps=eps)
