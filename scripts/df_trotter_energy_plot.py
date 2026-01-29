from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from openfermion import FermionOperator, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from openfermion.transforms import get_interaction_operator
from scipy.sparse.linalg import eigsh

from trotterlib.chemistry_hamiltonian import geo
from trotterlib.config import (
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    PFLabel,
    normalize_pf_label,
    pf_order,
)
from trotterlib.df_trotter.decompose import df_decompose_from_integrals, diag_hermitian
from trotterlib.df_trotter.model import Block, DFModel
from trotterlib.df_trotter.ops import (
    apply_df_block,
    apply_one_body_gaussian_block,
    U_to_qiskit_ops_jw,
    build_df_blocks,
    build_one_body_gaussian_block,
)
from trotterlib.df_trotter.two_body import (
    interaction_operator_from_chemist_integrals,
    two_body_tensor_from_df_model as _two_body_tensor_from_df_model,
)
from qiskit.quantum_info import Operator

from trotterlib.df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from trotterlib.analysis_utils import loglog_fit, print_loglog_fit
from trotterlib.eig_error import error_cal_multi
from trotterlib.plot_utils import set_loglog_axes
from trotterlib.qiskit_time_evolution_pyscf import (
    _run_scf_and_integrals,
    make_fci_vector_from_pyscf_solver,
)


def _one_body_fermion_op(coeff_mat: np.ndarray) -> FermionOperator:
    coeff_mat = np.asarray(coeff_mat)
    n = coeff_mat.shape[0]
    op = FermionOperator()
    for p in range(n):
        for q in range(n):
            coeff = coeff_mat[p, q]
            if abs(coeff) < 1e-14:
                continue
            op += FermionOperator(((p, 1), (q, 0)), coeff)
    return op


def _one_body_matrix(coeff_mat: np.ndarray) -> np.ndarray:
    n = coeff_mat.shape[0]
    op = _one_body_fermion_op(coeff_mat)
    return get_sparse_operator(op, n_qubits=n).toarray()


def _df_two_body_matrix(model) -> np.ndarray:
    n = model.N
    dim = 2**n
    acc = np.zeros((dim, dim), dtype=np.complex128)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        A = _one_body_matrix(g_mat)
        acc += lam * (A @ A)
    acc += _one_body_matrix(model.one_body_correction)
    acc += model.constant_correction * np.eye(dim, dtype=np.complex128)
    return acc


def _hamiltonian_matrix_from_df_model(
    constant: float, one_body_spin: np.ndarray, model: DFModel
) -> np.ndarray:
    dim = 2**model.N
    h1 = _one_body_matrix(one_body_spin)
    h2 = _df_two_body_matrix(model)
    return constant * np.eye(dim, dtype=np.complex128) + h1 + h2


def _hamiltonian_matrix_from_df_tensor(
    constant: float, one_body_spin: np.ndarray, model: DFModel
) -> np.ndarray:
    n = model.N
    two_body_chemist = np.zeros((n, n, n, n), dtype=np.complex128)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        two_body_chemist += lam * np.einsum("pq,rs->pqrs", g_mat, g_mat, optimize=True)
    op = interaction_operator_from_chemist_integrals(
        constant + model.constant_correction,
        one_body_spin + model.one_body_correction,
        two_body_chemist,
    )
    return get_sparse_operator(op, n_qubits=model.N).toarray()


def _hamiltonian_matrix(
    constant: float, one_body_spin: np.ndarray, two_body_spin: np.ndarray
) -> np.ndarray:
    op = InteractionOperator(constant, one_body_spin, two_body_spin)
    n = one_body_spin.shape[0]
    return get_sparse_operator(op, n_qubits=n).toarray()


def _effective_df_hamiltonian_sparse(
    constant: float, one_body_spin: np.ndarray, model: DFModel
):
    h_eff = one_body_spin + model.one_body_correction
    u_one, eps = diag_hermitian(h_eff, assume_hermitian=True)
    herm_one_body = u_one @ np.diag(eps) @ u_one.conj().T
    total_op = FermionOperator((), constant + model.constant_correction)
    total_op += _one_body_fermion_op(herm_one_body)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        u_g, eta = diag_hermitian(g_mat, assume_hermitian=True)
        g_herm = u_g @ np.diag(eta) @ u_g.conj().T
        a_op = _one_body_fermion_op(g_herm)
        total_op += lam * (a_op * a_op)
    interaction_op = get_interaction_operator(total_op)
    return get_sparse_operator(interaction_op, n_qubits=model.N)


def _ground_state_from_sparse(h_sparse):
    evals, evecs = eigsh(h_sparse, k=1, which="SA")
    idx = int(np.argmin(evals.real))
    return float(evals.real[idx]), evecs[:, idx]


def _reorder_sparse_matrix(mat, perm: np.ndarray):
    return mat[perm][:, perm]


def _unitary_from_circuit(qc) -> np.ndarray:
    num_qubits = qc.num_qubits
    dim = 2**num_qubits
    if dim <= 256:
        unitary = np.zeros((dim, dim), dtype=np.complex128)
        for idx in range(dim):
            basis = np.zeros(dim, dtype=np.complex128)
            basis[idx] = 1.0
            unitary[:, idx] = simulate_statevector(qc, basis)
        return unitary
    return Operator(qc).data


def _eigenphase_error(time: float, energy: float, unitary: np.ndarray) -> float:
    if time == 0.0:
        return 0.0
    evals = np.linalg.eigvals(unitary)
    phases = np.angle(evals)
    phases = np.where(phases > 0, phases - 2 * np.pi, phases)
    n_wrap = int((-energy * time) // (2 * np.pi)) + 1
    phases = phases - (2 * (n_wrap - 1) * np.pi)
    energies = np.array([ph.real / time for ph in phases], dtype=float)
    return float(np.min(np.abs(energies - energy)))


def _count_rz_gates(qc: QuantumCircuit) -> int:
    counts = qc.count_ops()
    rz = int(counts.get("rz", 0))
    rz += int(counts.get("RZ", 0))
    return rz


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


def _reorder_vector(vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec).reshape(-1)
    return arr[perm]


def _reorder_matrix(mat: np.ndarray, perm: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat)
    return arr[np.ix_(perm, perm)]


def _phase_energy_from_circuit(
    blocks: Sequence[Block],
    psi0: np.ndarray,
    time: float,
    *,
    num_qubits: int,
    pf_label: PFLabel,
    energy_shift: float,
) -> float:
    if time == 0.0:
        return 0.0
    qc = build_df_trotter_circuit(
        blocks,
        time=time,
        num_qubits=num_qubits,
        pf_label=pf_label,
        energy_shift=energy_shift,
    )
    psi_t = simulate_statevector(qc, psi0)
    s = np.vdot(np.asarray(psi0).reshape(-1), psi_t)
    return float(-np.angle(s) / time)


def _h_chain_integrals(
    molecule_type: int, *, distance: float | None, basis: str | None
) -> tuple[float, np.ndarray, np.ndarray]:
    if distance is None:
        distance = DEFAULT_DISTANCE
    if basis is None:
        basis = DEFAULT_BASIS
    geometry, multiplicity, charge = geo(molecule_type, distance)
    description = f"distance_{int(distance * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry, basis, multiplicity, charge, description=description
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    one_body, two_body = molecule.get_integrals()
    constant = float(molecule.nuclear_repulsion)
    return constant, one_body, two_body


def _h_chain_integrals_pyscf(
    molecule_type: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    _, _, constant, one_body, two_body = _run_scf_and_integrals(molecule_type)
    return float(constant), one_body, two_body


def _symmetrize_two_body(two_body: np.ndarray) -> np.ndarray:
    t = np.asarray(two_body)
    parts = [
        t,
        np.transpose(t, (1, 0, 2, 3)),
        np.transpose(t, (0, 1, 3, 2)),
        np.transpose(t, (1, 0, 3, 2)),
        np.transpose(t, (2, 3, 0, 1)),
        np.transpose(t, (3, 2, 0, 1)),
        np.transpose(t, (2, 3, 1, 0)),
        np.transpose(t, (3, 2, 1, 0)),
    ]
    sym = sum(parts) / len(parts)
    return np.real_if_close(sym, tol=1e-8)


def _perturbation_error(
    time: float, energy: float, psi0: np.ndarray, psi_t: np.ndarray
) -> float:
    if time == 0.0:
        return 0.0
    phase_factor = np.exp(-1j * energy * time)
    delta_state = psi_t - phase_factor * psi0
    denom = time * np.sin(energy * time)
    if abs(denom) < 1e-12:
        denom = energy * (time**2)
    if denom == 0.0:
        return 0.0
    delta_e = np.vdot(psi0, delta_state).real / denom
    return float(abs(delta_e))


def _df_model_diagnostics(model: DFModel) -> dict[str, float]:
    g_nonherm = [
        float(np.linalg.norm(g_mat - g_mat.conj().T)) for g_mat in model.G_list
    ]
    g_norms = [float(np.linalg.norm(g_mat)) for g_mat in model.G_list]
    one_body_nonherm = float(
        np.linalg.norm(model.one_body_correction - model.one_body_correction.conj().T)
    )
    lambdas = np.asarray(model.lambdas)
    abs_lam = np.abs(lambdas)
    max_imag = float(np.max(np.abs(np.imag(lambdas)))) if lambdas.size else 0.0
    return {
        "max_g_nonherm": max(g_nonherm) if g_nonherm else 0.0,
        "max_g_norm": max(g_norms) if g_norms else 0.0,
        "one_body_nonherm": one_body_nonherm,
        "lam_min": float(abs_lam.min()) if abs_lam.size else 0.0,
        "lam_max": float(abs_lam.max()) if abs_lam.size else 0.0,
        "lam_max_imag": max_imag,
    }


def _hamiltonian_hermiticity(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat - mat.conj().T))


def _run_sanity_checks(
    *,
    constant: float,
    one_body_spin: np.ndarray,
    two_body_spin: np.ndarray,
    model: DFModel,
    blocks: Sequence[Block],
    energy_ref: float | None,
    psi0: np.ndarray | None,
    reference: str,
    rank: int | None,
    tol: float | None,
    pf_label: PFLabel,
    debug_print: Callable[[str], None],
) -> None:
    warnings: list[str] = []
    perm = _bit_reverse_permutation(model.N)
    full_rank = False
    max_rank = None

    # Basic tensor sanity checks.
    one_body_norm = np.linalg.norm(one_body_spin)
    one_body_diff = np.linalg.norm(one_body_spin - one_body_spin.conj().T)
    if one_body_diff > 1e-8 * max(1.0, one_body_norm):
        warnings.append(f"one_body is not Hermitian (diff={one_body_diff:.3e}).")

    if two_body_spin.size:
        sym1 = np.linalg.norm(
            two_body_spin - np.transpose(two_body_spin, (1, 0, 3, 2))
        )
        sym2 = np.linalg.norm(
            two_body_spin - np.transpose(two_body_spin, (2, 3, 0, 1))
        )
        two_body_norm = np.linalg.norm(two_body_spin)
        if max(sym1, sym2) > 1e-8 * max(1.0, two_body_norm):
            warnings.append(
                f"two_body symmetry mismatch (sym1={sym1:.3e}, sym2={sym2:.3e})."
            )

    if model.N != one_body_spin.shape[0]:
        warnings.append(
            f"model.N={model.N} does not match one_body size={one_body_spin.shape[0]}."
        )
    if one_body_spin.size:
        n_spatial = int(one_body_spin.shape[0] // 2)
        max_rank = int(n_spatial**2)
        if rank is None:
            full_rank = tol == 0.0
        elif rank >= max_rank and (tol is None or tol == 0.0):
            full_rank = True

    # G_list hermiticity check (cheap, always on).
    g_norms = []
    g_nonherm = []
    for g_mat in model.G_list:
        g_norm = np.linalg.norm(g_mat)
        g_norms.append(g_norm)
        g_nonherm.append(np.linalg.norm(g_mat - g_mat.conj().T))
    max_nonherm = 0.0
    if g_nonherm:
        max_nonherm = float(np.max(g_nonherm))
        max_norm = float(np.max(g_norms)) if g_norms else 1.0
        if reference != "df" and max_nonherm > 1e-8 * max(1.0, max_norm):
            warnings.append(
                f"G_list not Hermitian (max ||G-G†||={max_nonherm:.3e})."
            )

    def _match_up_to_phase(u: np.ndarray, target: np.ndarray, tol: float = 1e-8) -> bool:
        if u.shape != target.shape:
            return False
        overlap = np.vdot(target.flatten(), u.flatten())
        if overlap == 0.0:
            return False
        phase = overlap / abs(overlap)
        return np.linalg.norm(u - phase * target) < tol

    if model.N >= 2:
        phases = np.array([0.3, -0.5], dtype=float)
        U_test = np.diag(np.exp(1j * phases))
        try:
            u_ops = U_to_qiskit_ops_jw(U_test)
            qc = QuantumCircuit(2)
            for gate, qubits in u_ops:
                qc.append(gate, list(qubits))
            u_ops_mat = Operator(qc).data

            # Expected Fock-space unitary for diagonal U, matching Qiskit basis order.
            expected = np.eye(4, dtype=np.complex128)
            for basis in range(4):
                occ0 = (basis >> 0) & 1
                occ1 = (basis >> 1) & 1
                phase = occ0 * phases[0] + occ1 * phases[1]
                expected[basis, basis] = np.exp(1j * phase)
            expected_dag = expected.conj().T

            if _match_up_to_phase(u_ops_mat, expected, tol=1e-8):
                pass
            elif _match_up_to_phase(u_ops_mat, expected_dag, tol=1e-8):
                warnings.append(
                    "U_to_qiskit_ops_jw appears to implement U† on Fock space."
                )
            else:
                warnings.append(
                    "U_to_qiskit_ops_jw unitary mismatch for diagonal U (neither U nor U†)."
                )
        except Exception:
            warnings.append("U_to_qiskit_ops_jw sanity check skipped.")

    # Check sign conventions on diagonal blocks.
    eps = np.array([0.3], dtype=float)
    tau = 0.2
    qc = QuantumCircuit(1)
    apply_one_body_gaussian_block(qc, [], eps, tau)
    expected = np.diag([1.0, np.exp(-1j * tau * eps[0])])
    if not np.allclose(Operator(qc).data, expected, atol=1e-10):
        warnings.append("one-body block sign mismatch (exp(-i) check).")

    eta = np.array([0.4], dtype=float)
    lam = 0.7
    qc = QuantumCircuit(1)
    apply_df_block(qc, [], eta, lam, tau)
    expected = np.diag([1.0, np.exp(-1j * tau * lam * (eta[0] ** 2))])
    if not np.allclose(Operator(qc).data, expected, atol=1e-10):
        warnings.append("DF block sign mismatch (exp(-i) check).")

    # Check energy_shift / global phase propagation on an empty circuit.
    shift = 0.9
    qc = build_df_trotter_circuit(
        [],
        time=tau,
        num_qubits=1,
        pf_label=pf_label,
        energy_shift=shift,
    )
    psi = simulate_statevector(qc, np.array([1.0, 0.0], dtype=np.complex128))
    expected = np.exp(-1j * shift * tau) * np.array([1.0, 0.0], dtype=np.complex128)
    if not np.allclose(psi, expected, atol=1e-10):
        warnings.append("energy_shift/global_phase mismatch in circuit evolution.")

    # Perturbation formula should be ~0 for exact diagonal evolution.
    psi0_check = np.array([0.0, 1.0], dtype=np.complex128)
    energy_check = float(eps[0])
    qc = QuantumCircuit(1)
    apply_one_body_gaussian_block(qc, [], eps, tau)
    psi_t = simulate_statevector(qc, psi0_check)
    perr = _perturbation_error(tau, energy_check, psi0_check, psi_t)
    if perr > 1e-10:
        warnings.append(
            f"perturbation formula not zero on exact evolution (err={perr:.3e})."
        )

    # DF reconstruction consistency (small systems only).
    if one_body_spin.size and model.N <= 8 and reference != "df":
        h_exact = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
        h_df = _hamiltonian_matrix_from_df_model(constant, one_body_spin, model)
        diff = np.linalg.norm(h_exact - h_df)
        rel = diff / max(1.0, np.linalg.norm(h_exact))
        if rank is None and tol is None:
            warnings.append(
                "rank=None with tol=None uses OpenFermion default truncation; "
                "set tol=0.0 for full-rank reconstruction."
            )
        if full_rank and rel > 1e-6:
            warnings.append(
                f"DF full-rank reconstruction mismatch (rel={rel:.3e})."
            )
        if not full_rank and rel > 1e-2:
            warnings.append(
                f"DF truncation error is large (rel={rel:.3e}); scaling may be dominated by truncation."
            )

        # Check hermiticity impact on A^2 reconstruction (small systems only).
        if max_nonherm > 0.0 and model.N <= 8:
            h_df_herm = constant * np.eye(2 ** model.N, dtype=np.complex128)
            h_df_herm += _one_body_matrix(one_body_spin)
            for lam, g_mat in zip(model.lambdas, model.G_list):
                g_herm = 0.5 * (g_mat + g_mat.conj().T)
                A = _one_body_matrix(g_herm)
                h_df_herm += lam * (A @ A)
            h_df_herm += _one_body_matrix(model.one_body_correction)
            h_df_herm += model.constant_correction * np.eye(
                2 ** model.N, dtype=np.complex128
            )
            diff_herm = np.linalg.norm(h_df - h_df_herm)
            warnings.append(
                f"H_df mismatch from Hermitian G (||Δ||_F={diff_herm:.3e})."
            )

    # Sparse reconstruction consistency (moderate systems).
    if one_body_spin.size and model.N <= 12 and reference != "df":
        try:
            h_exact_sparse = get_sparse_operator(
                InteractionOperator(constant, one_body_spin, two_body_spin),
                n_qubits=model.N,
            )
            two_body_df = _two_body_tensor_from_df_model(model)
            h_df_sparse = get_sparse_operator(
                InteractionOperator(
                    constant + model.constant_correction,
                    one_body_spin + model.one_body_correction,
                    two_body_df,
                ),
                n_qubits=model.N,
            )
            diff_sparse = h_exact_sparse - h_df_sparse
            diff = np.linalg.norm(diff_sparse.data)
            rel = diff / max(1.0, np.linalg.norm(h_exact_sparse.data))
            if full_rank and rel > 1e-6:
                warnings.append(
                    f"DF full-rank reconstruction mismatch (sparse rel={rel:.3e})."
                )
            if not full_rank and rel > 1e-2:
                warnings.append(
                    f"DF truncation error is large (sparse rel={rel:.3e})."
                )
        except Exception as exc:
            warnings.append(f"sparse reconstruction check skipped ({exc}).")

    # Phase consistency check on the first step.
    if psi0 is not None and energy_ref is not None and len(blocks) > 0:
        t_check = 1e-3
        psi0_vec = np.asarray(psi0).reshape(-1)
        qc = build_df_trotter_circuit(
            blocks,
            time=t_check,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )
        psi_t = simulate_statevector(qc, psi0)
        s = np.vdot(psi0_vec, psi_t)
        if t_check != 0.0:
            e_phase = -np.angle(s) / t_check
            dE_phase = e_phase - float(energy_ref)
            if reference == "df" and abs(dE_phase) > 1e-3:
                warnings.append(
                    f"phase mismatch vs DF reference (dE_phase={dE_phase:+.3e})."
                )
            if reference == "exact" and not full_rank:
                warnings.append(
                    "reference='exact' with DF truncation: phase mismatch expected; "
                    "use reference='df' to isolate Trotter error."
                )
        if model.N <= 6:
            h_df = _hamiltonian_matrix_from_df_model(constant, one_body_spin, model)
            h_df = _reorder_matrix(h_df, perm)
            evals, evecs = np.linalg.eigh(h_df)
            phases = np.exp(-1j * evals * t_check)
            u_exact = (evecs * phases) @ evecs.conj().T
            psi_exact = u_exact @ psi0_vec
            s_exact = np.vdot(psi0_vec, psi_exact)
            if t_check != 0.0:
                e_phase_exact = -np.angle(s_exact) / t_check
                dE_exact = e_phase_exact - float(energy_ref)
                if abs(dE_exact) > 1e-6:
                    warnings.append(
                        f"exact expm phase mismatch (dE_phase={dE_exact:+.3e})."
                    )
            overlap = np.vdot(psi_exact, psi_t)
            if abs(overlap) > 0.0:
                phase_align = overlap / abs(overlap)
                phase_shift = -np.angle(phase_align) / t_check
                residual = np.linalg.norm(psi_exact - phase_align.conj() * psi_t)
                if residual > 1e-6:
                    warnings.append(
                        "circuit state deviates from exact expm beyond global phase "
                        f"(residual={residual:.3e})."
                    )
                elif abs(phase_shift) > 1e-6:
                    warnings.append(
                        f"global phase offset vs expm (ΔE_phase={phase_shift:+.3e})."
                    )
            else:
                if np.linalg.norm(psi_exact - psi_t) > 1e-6:
                    warnings.append(
                        "circuit state deviates from exact expm at t=1e-3 "
                        "(PF/sign/global-phase mismatch)."
                    )

    if warnings:
        for msg in warnings:
            debug_print(f"sanity: {msg}")
    else:
        debug_print("sanity: all checks passed.")


def df_trotter_energy_error_curve(
    t_start: float,
    t_end: float,
    t_step: float,
    *,
    molecule_type: int = 2,
    pf_label: PFLabel = "2nd",
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    estimator: str = "perturbation",
    reference: str = "exact",
    debug: bool = True,
    debug_every: int = 1,
    debug_max: int | None = 30,
    debug_print: Callable[[str], None] = print,
    debug_compare_expectation: bool = True,
) -> tuple[list[float], list[float]]:
    pf_label = normalize_pf_label(pf_label)
    if t_step <= 0:
        raise ValueError("t_step must be positive.")
    if debug_every <= 0:
        raise ValueError("debug_every must be positive.")
    if estimator not in ("expectation", "perturbation", "eigenphase"):
        raise ValueError(
            "estimator must be 'perturbation', 'expectation', or 'eigenphase'."
        )
    if reference not in ("exact", "df"):
        raise ValueError("reference must be 'exact' or 'df'.")
    if estimator == "expectation" and reference not in ("exact", "df"):
        raise ValueError("reference must be 'exact' or 'df' for expectation estimator.")
    if reference == "exact" and rank is None and rank_fraction is None and tol is None:
        tol = 0.0
    times = [float(t) for t in np.arange(t_start, t_end, t_step) if t != 0.0]

    use_pyscf_integrals = reference == "exact" and distance is None and basis is None
    if use_pyscf_integrals:
        constant, one_body, two_body = _h_chain_integrals_pyscf(molecule_type)
    else:
        constant, one_body, two_body = _h_chain_integrals(
            molecule_type, distance=distance, basis=basis
        )
    two_body = _symmetrize_two_body(two_body)
    if rank_fraction is not None:
        if rank is not None:
            raise ValueError("rank and rank_fraction are mutually exclusive.")
        if rank_fraction <= 0:
            raise ValueError("rank_fraction must be positive.")
        n_spatial = int(one_body.shape[0])
        full_rank = int(n_spatial**2)
        if rank_fraction >= 1.0:
            rank = full_rank
            if tol is None:
                tol = 0.0
        else:
            rank = int(round(full_rank * rank_fraction))
            rank = max(1, min(rank, full_rank))
    one_body_spin, two_body_spin = spinorb_from_spatial(one_body, two_body * 0.5)

    raw_model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=rank, tol=tol
    )
    model = raw_model.hermitize()
    h_eff = one_body_spin + model.one_body_correction
    perm = _bit_reverse_permutation(model.N)

    blocks: list[Block] = []
    one_body_block = build_one_body_gaussian_block(h_eff)
    blocks.append(Block.from_one_body_gaussian(one_body_block))
    blocks.extend(Block.from_df(b) for b in build_df_blocks(model))

    energy_ref = None
    psi0 = None
    state_vec = None
    h_ref = None
    phase_energy_ref = None
    h_exact_open = None
    h_df_open = None
    use_fci = False
    if reference == "exact":
        if distance is None and basis is None and molecule_type != 6:
            _, n_qubits, energy_ref, state_vec, _ = make_fci_vector_from_pyscf_solver(
                molecule_type
            )
            if n_qubits != model.N:
                raise ValueError("FCI qubit count does not match DF model size.")
            psi0 = _reorder_vector(np.asarray(state_vec).reshape(-1), perm)
            use_fci = True
        else:
            h_exact_open = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
            evals, evecs = np.linalg.eigh(h_exact_open)
            idx = int(np.argmin(evals.real))
            energy_ref = float(evals.real[idx])
            psi0 = _reorder_vector(evecs[:, idx], perm)
            state_vec = psi0.reshape(-1, 1)
        if h_exact_open is None:
            h_exact_open = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
        h_ref = _reorder_matrix(h_exact_open, perm)

        if use_fci and psi0 is not None and energy_ref is not None:
            e0_check = np.vdot(psi0, h_ref @ psi0).real
            if abs(e0_check - float(energy_ref)) > 1e-6:
                if debug:
                    debug_print(
                        "sanity: FCI vector mismatch; falling back to diagonalization "
                        f"(ΔE={e0_check - float(energy_ref):+.3e})."
                    )
                evals, evecs = np.linalg.eigh(h_exact_open)
                idx = int(np.argmin(evals.real))
                energy_ref = float(evals.real[idx])
                psi0 = _reorder_vector(evecs[:, idx], perm)
                state_vec = psi0.reshape(-1, 1)
                h_ref = _reorder_matrix(h_exact_open, perm)

    if reference == "df":
        h_df_sparse = _effective_df_hamiltonian_sparse(
            constant, one_body_spin, model
        )
        energy_ref, psi0 = _ground_state_from_sparse(h_df_sparse)
        psi0 = _reorder_vector(np.asarray(psi0).reshape(-1), perm)
        state_vec = psi0.reshape(-1, 1)
        if estimator == "expectation":
            h_ref = _reorder_sparse_matrix(h_df_sparse, perm)

        if psi0 is not None and blocks:
            phase_energy = _phase_energy_from_circuit(
                blocks,
                psi0,
                1e-3,
                num_qubits=model.N,
                pf_label=pf_label,
                energy_shift=constant + model.constant_correction,
            )
            if debug:
                shift = phase_energy - float(energy_ref)
                if abs(shift) > 1e-9:
                    debug_print(
                        "df reference calibration: "
                        f"raw={float(energy_ref):+.6e} "
                        f"phase={phase_energy:+.6e} "
                        f"shift={shift:+.3e}"
                    )
                if abs(shift) > 1e-3:
                    debug_print(
                        "df calibration warning: large phase shift detected; "
                        "check global_phase propagation or constant energy shifts."
                    )
            phase_energy_ref = phase_energy

    if debug:
        raw_stats = _df_model_diagnostics(raw_model)
        herm_stats = _df_model_diagnostics(model)
        debug_print(
            "df model diagnostics: "
            f"raw max||G-G†||={raw_stats['max_g_nonherm']:.3e} "
            f"raw max||G||={raw_stats['max_g_norm']:.3e} "
            f"raw ||one_body-one_body†||={raw_stats['one_body_nonherm']:.3e} "
            f"raw |lambda|min={raw_stats['lam_min']:.3e} "
            f"raw |lambda|max={raw_stats['lam_max']:.3e} "
            f"raw max|Im(lambda)|={raw_stats['lam_max_imag']:.3e} "
            f"herm max||G-G†||={herm_stats['max_g_nonherm']:.3e} "
            f"herm ||one_body-one_body†||={herm_stats['one_body_nonherm']:.3e} "
            f"herm max|Im(lambda)|={herm_stats['lam_max_imag']:.3e}"
        )
        if reference == "df":
            debug_print(
                "df reference: "
                f"constant={constant:.6e} "
                f"constant_correction={model.constant_correction:.6e} "
                f"one_body_correction_norm={np.linalg.norm(model.one_body_correction):.3e} "
                f"num_lambda={len(model.lambdas)} "
                f"|lambda|min={herm_stats['lam_min']:.3e} |lambda|max={herm_stats['lam_max']:.3e} "
                f"energy_ref={float(energy_ref) if energy_ref is not None else float('nan'):.6e}"
            )
        else:
            msg = f"exact reference: energy_ref={float(energy_ref):.6e}"
            if h_ref is not None and psi0 is not None:
                e0 = np.vdot(psi0, h_ref @ psi0).real
                msg += f" <psi0|H|psi0>={float(e0):.6e}"
            debug_print(msg)
            if h_exact_open is not None and model.N <= 8:
                h_df_open = _hamiltonian_matrix_from_df_model(
                    constant, one_body_spin, model
                )
                diff = np.linalg.norm(h_exact_open - h_df_open)
                rel = diff / max(1.0, np.linalg.norm(h_exact_open))
                e_df = float(np.min(np.linalg.eigvalsh(h_df_open).real))
                debug_print(
                    "exact vs df (A^2): "
                    f"||ΔH||_F={diff:.3e} rel={rel:.3e} "
                    f"E0_df={e_df:.6e} ΔE0={e_df - float(energy_ref):+.3e}"
                )
                h_df_tensor = _hamiltonian_matrix_from_df_tensor(
                    constant, one_body_spin, model
                )
                diff_tensor = np.linalg.norm(h_exact_open - h_df_tensor)
                rel_tensor = diff_tensor / max(1.0, np.linalg.norm(h_exact_open))
                e_df_tensor = float(np.min(np.linalg.eigvalsh(h_df_tensor).real))
                debug_print(
                    "exact vs df (tensor): "
                    f"||ΔH||_F={diff_tensor:.3e} rel={rel_tensor:.3e} "
                    f"E0_df={e_df_tensor:.6e} ΔE0={e_df_tensor - float(energy_ref):+.3e}"
                )
        if model.N <= 8:
            h_df_raw = _hamiltonian_matrix_from_df_model(
                constant, one_body_spin, raw_model
            )
            h_df_herm = _hamiltonian_matrix_from_df_model(
                constant, one_body_spin, model
            )
            diff_h = np.linalg.norm(h_df_raw - h_df_herm)
            debug_print(
                "df hermitize check: "
                f"||H_raw-H_herm||_F={diff_h:.3e} "
                f"||H_raw-H_raw†||_F={_hamiltonian_hermiticity(h_df_raw):.3e} "
                f"||H_herm-H_herm†||_F={_hamiltonian_hermiticity(h_df_herm):.3e}"
            )
            h_df_tensor = _hamiltonian_matrix_from_df_tensor(
                constant, one_body_spin, model
            )
            diff_tensor = np.linalg.norm(h_df_herm - h_df_tensor)
            rel_tensor = diff_tensor / max(1.0, np.linalg.norm(h_df_herm))
            e_df_herm = float(np.min(np.linalg.eigvalsh(h_df_herm).real))
            e_df_tensor = float(np.min(np.linalg.eigvalsh(h_df_tensor).real))
            debug_print(
                "df reconstruction check: "
                f"||H_herm-H_tensor||_F={diff_tensor:.3e} rel={rel_tensor:.3e} "
                f"E0_herm={e_df_herm:.6e} ΔE0={e_df_tensor - e_df_herm:+.3e}"
            )
        _run_sanity_checks(
            constant=constant,
            one_body_spin=one_body_spin,
            two_body_spin=two_body_spin,
            model=model,
            blocks=blocks,
            energy_ref=float(energy_ref) if energy_ref is not None else None,
            psi0=psi0,
            reference=reference,
            rank=rank,
            tol=tol,
            pf_label=pf_label,
            debug_print=debug_print,
        )

    if debug and times:
        qc = build_df_trotter_circuit(
            blocks,
            time=float(times[0]),
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )
        debug_print(f"rz_count={_count_rz_gates(qc)}")

    if reference == "df" and phase_energy_ref is not None:
        energy_ref_eff = float(phase_energy_ref)
    else:
        energy_ref_eff = float(energy_ref) if energy_ref is not None else 0.0

    errors: list[float] = []
    if estimator == "perturbation":
        debug_count = 0
        for idx, time in enumerate(times):
            t_sim = float(time)
            qc = build_df_trotter_circuit(
                blocks,
                time=t_sim,
                num_qubits=model.N,
                pf_label=pf_label,
                energy_shift=constant + model.constant_correction,
            )
            psi_t = simulate_statevector(qc, psi0)  # type: ignore[arg-type]
            if reference == "df":
                psi0_vec = np.asarray(psi0).reshape(-1)
                s = np.vdot(psi0_vec, psi_t)
                if t_sim != 0.0:
                    e_phase = -np.angle(s) / t_sim
                else:
                    e_phase = 0.0
                phase_err = abs(e_phase - energy_ref_eff)
                power = max(1.0, pf_order(pf_label) / 2.0)
                errors.append(float(phase_err**power))
            else:
                errors.append(
                    _perturbation_error(t_sim, energy_ref_eff, psi0, psi_t)  # type: ignore[arg-type]
                )
            if debug:
                if (debug_max is None or debug_count < debug_max) and (
                    idx % debug_every == 0
                ):
                    psi0_vec = np.asarray(psi0).reshape(-1)
                    s = np.vdot(psi0_vec, psi_t)
                    amp = abs(s)
                    leak = 1.0 - amp**2
                    phase_err = np.angle(s) + energy_ref_eff * t_sim
                    if t_sim != 0.0:
                        e_phase = -np.angle(s) / t_sim
                        dE_phase = e_phase - energy_ref_eff
                    else:
                        e_phase = 0.0
                        dE_phase = 0.0
                    phase = np.exp(-1j * energy_ref_eff * t_sim)
                    delta_state = (psi_t - phase * psi0_vec) / (1j * t_sim)
                    z = np.vdot(psi0_vec, delta_state) * np.exp(1j * energy_ref_eff * t_sim)
                    cos_et = np.cos(energy_ref_eff * t_sim)
                    msg = (
                        f"t={t_sim:.3e} |<0|t>|={amp:.6f} leak={leak:.3e} "
                        f"phase_err={phase_err:+.3e} "
                        f"E_phase={e_phase:+.6e} dE_phase={dE_phase:+.3e} "
                        f"z=({z.real:+.3e}+i{z.imag:+.3e}) cosEt={cos_et:.6f}"
                    )
                    if (
                        debug_compare_expectation
                        and h_ref is not None
                        and model.N <= 14
                    ):
                        energy_est = np.vdot(psi_t, h_ref @ psi_t).real
                        msg += (
                            f" E_est={float(energy_est):.6e}"
                            f" dE={float(energy_est - float(energy_ref)):+.3e}"
                        )
                    debug_print(msg)
                    debug_count += 1
        return times, errors

    if estimator == "eigenphase":
        unitaries: list[np.ndarray] = []
        for time in times:
            t_sim = -float(time)
            qc = build_df_trotter_circuit(
                blocks,
                time=t_sim,
                num_qubits=model.N,
                pf_label=pf_label,
                energy_shift=constant + model.constant_correction,
            )
            unitaries.append(_unitary_from_circuit(qc))
        times_out, error_list = error_cal_multi(
            times, unitaries, state_vec, float(energy_ref), num_eig=1  # type: ignore[arg-type]
        )
        return times_out, error_list

    for time in times:
        t_sim = float(time)
        qc = build_df_trotter_circuit(
            blocks,
            time=t_sim,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )
        psi_t = simulate_statevector(qc, psi0)  # type: ignore[arg-type]
        energy_est = np.vdot(psi_t, h_ref @ psi_t).real  # type: ignore[operator]
        errors.append(float(abs(energy_est - float(energy_ref))))

    return times, errors


def df_trotter_energy_error_plot(
    t_start: float,
    t_end: float,
    t_step: float,
    *,
    molecule_type: int = 2,
    pf_label: PFLabel = "2nd",
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    estimator: str = "perturbation",
    reference: str = "exact",
    fit: bool = True,
) -> tuple[list[float], list[float]]:
    times, errors = df_trotter_energy_error_curve(
        t_start,
        t_end,
        t_step,
        molecule_type=molecule_type,
        pf_label=pf_label,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
        estimator=estimator,
        reference=reference,
    )

    ax = plt.gca()
    set_loglog_axes(
        ax,
        xlabel="time",
        ylabel="energy error [Hartree]",
        title=f"H{molecule_type}_DF_{pf_label}",
    )
    ax.plot(times, errors, marker="o", linestyle="-", label="error")
    if fit:
        try:
            fit_result = loglog_fit(times, errors, mask_nonpositive=True, compute_r2=True)
            alpha = fit_result.coeff
            p = fit_result.slope
            fit_curve = [alpha * (t**p) for t in times]
            ax.plot(
                times,
                fit_curve,
                linestyle="--",
                label=f"fit: α={alpha:.2e}, p={p:.2f}",
            )
            print_loglog_fit(fit_result)
        except ValueError as exc:
            print(f"log-log fit skipped: {exc}")
    ax.legend()
    plt.show()
    return times, errors


__all__ = ["df_trotter_energy_error_curve", "df_trotter_energy_error_plot"]
