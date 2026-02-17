from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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
    PICKLE_DIR_DF_PATH,
    get_df_rank_fraction_for_molecule,
    get_df_rank_selection_for_molecule,
    normalize_pf_label,
)
from trotterlib.df_trotter.decompose import df_decompose_from_integrals, diag_hermitian
from trotterlib.df_trotter.model import Block, DFModel
from trotterlib.df_trotter.ops import (
    apply_D_one_body,
    apply_D_squared,
    apply_df_block,
    apply_one_body_gaussian_block,
    U_to_qiskit_ops_jw,
    build_df_blocks,
    build_one_body_gaussian_block,
    build_df_blocks_givens,
    build_one_body_gaussian_block_givens,
)
from trotterlib.df_trotter.two_body import (
    interaction_operator_from_chemist_integrals,
    two_body_tensor_from_df_model as _two_body_tensor_from_df_model,
)
from qiskit.quantum_info import Operator, SparsePauliOp

from trotterlib.df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from trotterlib.analysis_utils import loglog_fit, print_loglog_fit
from trotterlib.eig_error import error_cal_multi
from trotterlib.plot_utils import set_loglog_axes
from trotterlib.qiskit_time_evolution_pyscf import (
    _run_scf_and_integrals,
    make_fci_vector_from_pyscf_solver,
)
from trotterlib.qiskit_time_evolution_utils import (
    rz_costs_from_circuit,
    rz_costs_from_u_ops,
    debug_trace_u_decomposition,
    nonclifford_rz_costs_from_circuit,
    d_nonclifford_costs_from_circuit,
    u_nonclifford_costs_from_u_ops,
)
from trotterlib.pf_decomposition import iter_pf_steps
from trotterlib.product_formula import _get_w_list


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


def _hermitize_df_model(model: DFModel) -> tuple[DFModel, dict[str, float]]:
    g_nonherm = [np.linalg.norm(g - g.conj().T) for g in model.G_list]
    max_g_nonherm = float(np.max(g_nonherm)) if g_nonherm else 0.0
    ob_nonherm = float(
        np.linalg.norm(model.one_body_correction - model.one_body_correction.conj().T)
    )
    g_list = [0.5 * (g + g.conj().T) for g in model.G_list]
    one_body = 0.5 * (
        model.one_body_correction + model.one_body_correction.conj().T
    )
    const = float(np.real_if_close(model.constant_correction))
    return (
        DFModel(
            lambdas=model.lambdas,
            G_list=g_list,
            one_body_correction=one_body,
            constant_correction=const,
            N=model.N,
        ),
        {"max_g_nonherm": max_g_nonherm, "one_body_nonherm": ob_nonherm},
    )


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


def _artifact_ham_name(
    molecule_type: int, *, distance: float | None, basis: str | None
) -> str:
    distance_value = DEFAULT_DISTANCE if distance is None else float(distance)
    basis_value = DEFAULT_BASIS if basis is None else basis
    geometry, multiplicity, charge = geo(molecule_type, distance_value)
    description = f"distance_{int(distance_value * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry,
        basis_value,
        multiplicity,
        charge,
        description=description,
    )
    return f"{Path(molecule.filename).stem}_grouping"


def _save_df_plot_artifact(file_name: str, data: dict[str, object]) -> None:
    PICKLE_DIR_DF_PATH.mkdir(parents=True, exist_ok=True)
    path = PICKLE_DIR_DF_PATH / file_name
    with path.open("wb") as f:
        pickle.dump(data, f)


def _collect_df_rz_layer_metrics(costs: dict[str, object]) -> dict[str, int]:
    def _to_int(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    metrics: dict[str, int | None] = {
        "ref_rz_depth": _to_int(costs.get("rz_depth")),
        "ref_depth_total": _to_int(costs.get("depth_total")),
    }

    d_only = costs.get("d_only_costs")
    if isinstance(d_only, dict):
        metrics["d_only_nonclifford_rz_depth"] = _to_int(
            d_only.get("nonclifford_rz_depth")
        )
        metrics["d_only_ref_rz_depth"] = _to_int(d_only.get("rz_depth"))

    totals = costs.get("nonclifford_total")
    if isinstance(totals, dict):
        metrics["total_nonclifford_rz_depth"] = _to_int(
            totals.get("total_nonclifford_rz_depth")
        )
        metrics["u_nonclifford_rz_depth"] = _to_int(
            totals.get("u_nonclifford_rz_depth")
        )
        metrics["d_nonclifford_rz_depth"] = _to_int(
            totals.get("d_nonclifford_rz_depth")
        )

    proxy_totals = costs.get("toffoli_proxy_total")
    if isinstance(proxy_totals, dict):
        metrics["total_nonclifford_z_depth"] = _to_int(
            proxy_totals.get("total_nonclifford_z_depth")
        )

    proxy_totals_coloring = costs.get("toffoli_proxy_total_coloring")
    if isinstance(proxy_totals_coloring, dict):
        metrics["total_nonclifford_z_coloring_depth"] = _to_int(
            proxy_totals_coloring.get("total_nonclifford_z_depth")
        )

    return {key: value for key, value in metrics.items() if value is not None}


def _select_rank_from_ccsd_target(
    molecule_type: int,
    *,
    target_error_ha: float,
    thresh_range: Sequence[float] | None,
    use_kernel: bool,
    no_triples: bool,
    record_in_config: bool = False,
) -> tuple[int, float, dict[str, Any]]:
    from trotterlib.ccsd import select_rank_fraction_for_molecule

    selection = select_rank_fraction_for_molecule(
        molecule_type=int(molecule_type),
        ccsd_target_error_ha=target_error_ha,
        thresh_range=thresh_range,
        use_kernel=use_kernel,
        no_triples=no_triples,
        record_in_config=record_in_config,
    )
    return (
        int(selection["selected_rank"]),
        float(selection["selected_rank_fraction"]),
        selection,
    )


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


def _df_block_generator_matrices_for_diagnostics(
    *,
    one_body_spin: np.ndarray,
    model: DFModel,
) -> tuple[list[str], list[np.ndarray]]:
    """Build dense generator matrices for each DF Trotter block (small-N diagnostics).

    This is used to diagnose cases where the Trotter error becomes ~0 because
    blocks (approximately) commute or the chosen reference state is (approximately)
    a common eigenstate of all blocks.

    Returns:
        (labels, mats) where mats are (2**N, 2**N) dense matrices in the computational basis.
    """
    h_eff = np.asarray(one_body_spin) + np.asarray(model.one_body_correction)
    labels: list[str] = ["one_body_gaussian"]
    mats: list[np.ndarray] = [_one_body_matrix(h_eff)]

    for idx, (lam, g_mat) in enumerate(zip(model.lambdas, model.G_list)):
        lam_r = float(np.real_if_close(lam))
        A = _one_body_matrix(np.asarray(g_mat))
        mats.append(lam_r * (A @ A))
        labels.append(f"df[{idx}]")

    return labels, mats


def _print_block_commutator_diagnostics(
    *,
    labels: Sequence[str],
    mats: Sequence[np.ndarray],
    psi0: np.ndarray | None,
    debug_print: Callable[[str], None],
    top_k: int = 8,
) -> None:
    """Print commutator/eigenstate diagnostics for block Hamiltonians (small-N)."""
    n = len(mats)
    if n <= 1:
        debug_print("commutator diagnostics: not enough blocks to compare.")
        return

    norms = np.array([np.linalg.norm(m) for m in mats], dtype=float)
    norms = np.where(norms > 0.0, norms, 1.0)

    pairs: list[tuple[float, float, int, int]] = []
    for i in range(n):
        Hi = mats[i]
        for j in range(i + 1, n):
            Hj = mats[j]
            comm = Hi @ Hj - Hj @ Hi
            comm_norm = float(np.linalg.norm(comm))
            denom = float(norms[i] * norms[j])
            ratio = comm_norm / denom if denom > 0.0 else comm_norm
            pairs.append((ratio, comm_norm, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    ratios = np.array([p[0] for p in pairs], dtype=float)
    debug_print(
        "commutator diagnostics: "
        f"num_blocks={n} num_pairs={len(pairs)} "
        f"max_ratio={float(np.max(ratios)):.3e} "
        f"median_ratio={float(np.median(ratios)):.3e}"
    )
    if float(np.max(ratios)) < 1e-12:
        debug_print(
            "commutator diagnostics: all block commutators are ~0; "
            "Trotter error can be near machine precision for any state."
        )

    for ratio, comm_norm, i, j in pairs[: max(1, top_k)]:
        debug_print(
            "commutator pair: "
            f"{labels[i]} vs {labels[j]} "
            f"||[.,.]||_F={comm_norm:.3e} "
            f"ratio={ratio:.3e}"
        )

    if psi0 is None:
        return

    psi = np.asarray(psi0).reshape(-1)
    psi_norm = float(np.linalg.norm(psi))
    if psi_norm == 0.0:
        return
    psi = psi / psi_norm

    eig_res: list[tuple[float, int, float]] = []
    for i, Hi in enumerate(mats):
        v = Hi @ psi
        e = float(np.vdot(psi, v).real)
        resid = float(np.linalg.norm(v - e * psi))
        denom = float(np.linalg.norm(v))
        rel = resid / denom if denom > 0.0 else resid
        eig_res.append((rel, i, e))
    eig_res.sort(key=lambda x: x[0], reverse=True)
    max_rel = eig_res[0][0]
    debug_print(
        "eigenstate diagnostics: "
        f"max_rel_resid={max_rel:.3e} "
        f"median_rel_resid={float(np.median([r for r,_,_ in eig_res])):.3e}"
    )
    if max_rel < 1e-10:
        debug_print(
            "eigenstate diagnostics: psi0 is ~a common eigenstate of all blocks; "
            "Trotter error on psi0 can be near machine precision even if blocks don't commute globally."
        )
    for rel, i, e in eig_res[: max(1, min(top_k, len(eig_res)))]:
        debug_print(
            "eigenstate block: "
            f"{labels[i]} rel_resid={rel:.3e} <H>={e:+.6e}"
        )

    comm_state: list[tuple[float, int, int]] = []
    for ratio, _comm_norm, i, j in pairs[: max(1, top_k)]:
        Hi = mats[i]
        Hj = mats[j]
        commpsi = (Hi @ (Hj @ psi)) - (Hj @ (Hi @ psi))
        commpsi_norm = float(np.linalg.norm(commpsi))
        denom = float(norms[i] * norms[j])
        rel = commpsi_norm / denom if denom > 0.0 else commpsi_norm
        comm_state.append((rel, i, j))
    comm_state.sort(key=lambda x: x[0], reverse=True)
    debug_print(
        "commutator-on-psi0: "
        f"max_rel={comm_state[0][0]:.3e} "
        f"median_rel={float(np.median([r for r,_,_ in comm_state])):.3e}"
    )
    for rel, i, j in comm_state[: max(1, min(top_k, len(comm_state)))]:
        debug_print(
            "commutator-on-psi0 pair: "
            f"{labels[i]} vs {labels[j]} rel={rel:.3e}"
        )





def _df_diag_pauli_terms_D_squared(
    eta: np.ndarray, lam: float
) -> tuple[float, np.ndarray, dict[tuple[int, int], float]]:
    """Return Pauli coefficients for lam * (eta·n)^2 in the computational basis.

    Using n_p = (I - Z_p)/2, we can write:
        (eta·n)^2 = c0 I + Σ_p c_p Z_p + Σ_{p<q} c_pq Z_p Z_q
    with
        S = Σ_p eta_p
        c0 = 1/4 (S^2 + Σ_p eta_p^2)
        c_p = -1/2 eta_p S
        c_pq = 1/2 eta_p eta_q
    Then multiply all coefficients by lam.
    """
    eta = np.asarray(eta, dtype=float).reshape(-1)
    S = float(np.sum(eta))
    s2 = float(np.sum(eta * eta))
    c0 = 0.25 * (S * S + s2) * float(lam)
    cz = (-0.5 * eta * S) * float(lam)
    czz: dict[tuple[int, int], float] = {}
    n = eta.size
    for p in range(n):
        for q in range(p + 1, n):
            coeff = 0.5 * float(lam) * float(eta[p]) * float(eta[q])
            if coeff != 0.0:
                czz[(p, q)] = coeff
    return c0, cz, czz


def _print_df_block_pauli_diagnostics(
    *,
    model: DFModel,
    blocks: Sequence[Block],
    debug_print: Callable[[str], None],
    abs_cutoff: float = 1e-12,
    top_k: int = 12,
    full_conjugated_if_n_leq: int = 6,
) -> None:
    """Print how each DF block looks in Pauli basis.

    - Always prints the *diagonal* Pauli terms for lam*(eta·n)^2 (I, Z, ZZ).
      This is the Hamiltonian in the rotated number basis *before* conjugation by U.

    - Optionally (for very small N), also prints the Pauli expansion of the *conjugated*
      block Hamiltonian:  H_block = U^† [lam*(eta·n)^2] U.
      This expansion is size O(4^N), so we only do it for N <= full_conjugated_if_n_leq.
    """
    N = int(model.N)
    df_blocks = [b for b in blocks if b.kind == "df"]
    if not df_blocks:
        return

    debug_print(
        f"df block pauli diagnostics: N={N} num_df_blocks={len(df_blocks)} "
        f"(abs_cutoff={abs_cutoff:g}, top_k={top_k}, full_conjugated_if_N<={full_conjugated_if_n_leq})"
    )

    for li, blk in enumerate(df_blocks):
        lam = float(blk.payload.lam)
        eta = np.asarray(blk.payload.eta, dtype=float).reshape(-1)
        c0, cz, czz = _df_diag_pauli_terms_D_squared(eta, lam)

        # Summaries (diagonal basis)
        nz_z = int(np.sum(np.abs(cz) > abs_cutoff))
        nz_zz = int(sum(1 for v in czz.values() if abs(v) > abs_cutoff))
        debug_print(f"df[{li}] diag basis (before U): lam={lam:+.6e}  nz(Z)={nz_z}  nz(ZZ)={nz_zz}  I={c0:+.6e}")

        # Print largest diagonal terms
        z_terms = [(abs(float(cz[p])), p, float(cz[p])) for p in range(N) if abs(float(cz[p])) > abs_cutoff]
        z_terms.sort(reverse=True)
        for _, p, coeff in z_terms[: max(0, min(top_k, len(z_terms)))]:
            debug_print(f"  Z[{p}]: {coeff:+.6e}")

        zz_terms = [(abs(v), p, q, v) for (p, q), v in czz.items() if abs(v) > abs_cutoff]
        zz_terms.sort(reverse=True)
        for _, p, q, coeff in zz_terms[: max(0, min(top_k, len(zz_terms)))]:
            debug_print(f"  Z[{p}]Z[{q}]: {coeff:+.6e}")

        # Full conjugated Pauli expansion (very small N only)
        if N <= int(full_conjugated_if_n_leq):
            try:
                qc_u = QuantumCircuit(N)
                for gate, qubits in blk.payload.U_ops:
                    qc_u.append(gate, list(qubits))
                U = Operator(qc_u).data

                # Build diagonal Hamiltonian matrix directly in computational basis
                dim = 1 << N
                diag = np.zeros(dim, dtype=float)
                # occupation pattern: qubit p corresponds to bit p in little-endian integer
                for x in range(dim):
                    dot = 0.0
                    for p in range(N):
                        if (x >> p) & 1:
                            dot += float(eta[p])
                    diag[x] = lam * (dot * dot)
                Hdiag = np.diag(diag)

                Hblk = U.conj().T @ Hdiag @ U
                paulis = SparsePauliOp.from_operator(Operator(Hblk))

                coeffs = np.asarray(paulis.coeffs)
                order = np.argsort(np.abs(coeffs))[::-1]
                debug_print(f"  conjugated H_block Pauli expansion: num_terms={len(coeffs)} (showing top {top_k})")
                for k in range(min(top_k, len(order))):
                    idx = int(order[k])
                    c = complex(coeffs[idx])
                    if abs(c) < abs_cutoff:
                        break
                    label = paulis.paulis[idx].to_label()
                    # Qiskit labels are big-endian; keep as-is but also show indices to avoid confusion.
                    debug_print(f"    {label}: {c.real:+.6e}{c.imag:+.1e}j")
            except Exception as exc:
                debug_print(f"  conjugated Pauli diagnostics skipped ({exc}).")


def _print_df_block_u_equivalence(
    *,
    model: DFModel,
    blocks: Sequence[Block],
    debug_print: Callable[[str], None],
    max_n: int = 8,
    tol: float = 1e-8,
) -> None:
    """Check whether DF blocks share the same U up to global phase."""
    N = int(model.N)
    if N > max_n:
        return
    df_blocks = [b for b in blocks if b.kind == "df"]
    if len(df_blocks) < 2:
        return

    def _unitary_from_ops(u_ops: Sequence[tuple[Any, Tuple[int, ...]]]) -> np.ndarray:
        qc_u = QuantumCircuit(N)
        for gate, qubits in u_ops:
            qc_u.append(gate, list(qubits))
        return Operator(qc_u).data

    def _global_phase_error(u1: np.ndarray, u2: np.ndarray) -> tuple[float, float]:
        v = u1.conj().T @ u2
        tr = np.trace(v)
        if abs(tr) > 1e-12:
            phase = np.angle(tr)
        else:
            phase = np.angle(v.flat[0])
        target = np.exp(1j * phase) * np.eye(v.shape[0], dtype=v.dtype)
        err = float(np.max(np.abs(v - target)))
        return err, float(phase)

    u_mats: list[np.ndarray] = []
    for blk in df_blocks:
        u_mats.append(_unitary_from_ops(blk.payload.U_ops))

    matches: list[str] = []
    for i in range(len(u_mats)):
        for j in range(i + 1, len(u_mats)):
            err, phase = _global_phase_error(u_mats[i], u_mats[j])
            if err <= tol:
                matches.append(
                    f"df[{i}] ~= df[{j}] (global phase={phase:+.3e}, err={err:.3e})"
                )

    debug_print(
        f"df block U equivalence: N={N} blocks={len(df_blocks)} tol={tol:.1e}"
    )
    if matches:
        for line in matches:
            debug_print(f"  {line}")
    else:
        debug_print("  no matching U up to global phase")


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

    # Block commutator diagnostics (small systems).
    # Useful when Trotter error becomes ~0 (e.g., all blocks commute or psi0 is a common eigenstate).
    if model.N <= 8:
        try:
            labels, mats = _df_block_generator_matrices_for_diagnostics(
                one_body_spin=one_body_spin,
                model=model,
            )
            _print_block_commutator_diagnostics(
                labels=labels,
                mats=mats,
                psi0=psi0,
                debug_print=debug_print,
            )
        except Exception as exc:
            debug_print(f"commutator diagnostics skipped ({exc}).")

    # Pauli diagnostics for DF blocks (diagonal basis always; full conjugated only for very small N).
    # This is useful to see which blocks are effectively diagonal/commuting on the reference state.
    try:
        if model.N <= 8:
            _print_df_block_pauli_diagnostics(
                model=model,
                blocks=blocks,
                debug_print=debug_print,
            )
            _print_df_block_u_equivalence(
                model=model,
                blocks=blocks,
                debug_print=debug_print,
            )
    except Exception as exc:
        debug_print(f"df block pauli/U diagnostics skipped ({exc}).")

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


def _run_debug_diagnostics(
    *,
    raw_model: DFModel,
    model: DFModel,
    constant: float,
    one_body_spin: np.ndarray,
    two_body_spin: np.ndarray,
    blocks: Sequence[Block],
    reference: str,
    energy_ref: float | None,
    psi0: np.ndarray | None,
    h_ref: np.ndarray | None,
    h_exact_open: np.ndarray | None,
    rank: int | None,
    tol: float | None,
    pf_label: PFLabel,
    phase_energy_ref: float | None,
    debug_print: Callable[[str], None],
) -> None:
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
        energy_ref_val = float(energy_ref) if energy_ref is not None else float("nan")
        debug_print(
            "df reference: "
            f"constant={constant:.6e} "
            f"constant_correction={model.constant_correction:.6e} "
            f"one_body_correction_norm={np.linalg.norm(model.one_body_correction):.3e} "
            f"num_lambda={len(model.lambdas)} "
            f"|lambda|min={herm_stats['lam_min']:.3e} |lambda|max={herm_stats['lam_max']:.3e} "
            f"energy_ref={energy_ref_val:.6e}"
        )
        if phase_energy_ref is not None and energy_ref is not None:
            shift = phase_energy_ref - float(energy_ref)
            if abs(shift) > 1e-9:
                debug_print(
                    "df reference calibration: "
                    f"raw={float(energy_ref):+.6e} "
                    f"phase={phase_energy_ref:+.6e} "
                    f"shift={shift:+.3e}"
                )
            if abs(shift) > 1e-3:
                debug_print(
                    "df calibration warning: large phase shift detected; "
                    "check global_phase propagation or constant energy shifts."
                )
    else:
        energy_ref_val = float(energy_ref) if energy_ref is not None else float("nan")
        msg = f"exact reference: energy_ref={energy_ref_val:.6e}"
        if h_ref is not None and psi0 is not None and energy_ref is not None:
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
                f"E0_df={e_df:.6e} ΔE0={e_df - energy_ref_val:+.3e}"
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
                f"E0_df={e_df_tensor:.6e} ΔE0={e_df_tensor - energy_ref_val:+.3e}"
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


def _debug_rz_count(
    *,
    blocks: Sequence[Block],
    times: Sequence[float],
    num_qubits: int,
    pf_label: PFLabel,
    energy_shift: float,
    debug_print: Callable[[str], None],
) -> None:
    if not times:
        return
    qc = build_df_trotter_circuit(
        blocks,
        time=float(times[0]),
        num_qubits=num_qubits,
        pf_label=pf_label,
        energy_shift=energy_shift,
    )
    debug_print(f"ref_rz_count={_count_rz_gates(qc)}")


def _debug_perturbation_step(
    *,
    t_sim: float,
    psi0: np.ndarray,
    psi_t: np.ndarray,
    energy_ref_eff: float,
    perr: float,
    h_ref: np.ndarray | None,
    energy_ref: float | None,
    num_qubits: int,
    debug_compare_expectation: bool,
) -> str:
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
        f"z=({z.real:+.3e}+i{z.imag:+.3e}) cosEt={cos_et:.6f} "
        f"pert_err={perr:.3e}"
    )
    if (
        debug_compare_expectation
        and h_ref is not None
        and energy_ref is not None
        and num_qubits <= 14
    ):
        energy_est = np.vdot(psi_t, h_ref @ psi_t).real
        msg += (
            f" E_est={float(energy_est):.6e}"
            f" dE={float(energy_est - float(energy_ref)):+.3e}"
        )
    return msg


def _apply_d_block(qc: QuantumCircuit, block: Block, tau: float) -> None:
    if block.kind == "one_body_gaussian":
        apply_D_one_body(qc, block.payload.eps, tau)
        return
    if block.kind == "df":
        apply_D_squared(qc, block.payload.eta, block.payload.lam, tau)
        return


def _build_d_only_cost_circuit(
    blocks: Sequence[Block],
    time: float,
    *,
    num_qubits: int,
    pf_label: PFLabel,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    weights = _get_w_list(pf_label)
    for term_idx, weight in iter_pf_steps(len(blocks), weights):
        _apply_d_block(qc, blocks[term_idx], weight * time)
    return qc


def _build_d_block_circuit(
    block: Block,
    tau: float,
    *,
    num_qubits: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    _apply_d_block(qc, block, tau)
    return qc


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
    return_costs: bool = False,
    cost_basis_gates: Sequence[str] | None = None,
    cost_decompose_reps: int = 8,
    cost_optimization_level: int = 0,
    trace_u_debug: bool = False,
    ccsd_target_error_ha: float | None = None,
    ccsd_thresh_range: Sequence[float] | None = None,
    ccsd_use_kernel: bool = False,
    ccsd_no_triples: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], dict[str, object]]:
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
    if rank is None and rank_fraction is None and ccsd_target_error_ha is None:
        config_rank_fraction = get_df_rank_fraction_for_molecule(int(molecule_type))
        if config_rank_fraction is not None:
            rank_fraction = float(config_rank_fraction)
            if debug:
                debug_print(
                    "rank_fraction from config: "
                    f"molecule_type={int(molecule_type)} "
                    f"rank_fraction={rank_fraction:.6f}"
                )
    if reference == "exact" and rank is None and rank_fraction is None and tol is None:
        tol = 0.0
    if ccsd_target_error_ha is not None:
        if rank is not None or rank_fraction is not None:
            raise ValueError(
                "ccsd_target_error_ha cannot be combined with rank or rank_fraction."
            )
        if ccsd_target_error_ha <= 0:
            raise ValueError("ccsd_target_error_ha must be positive.")
        if distance is not None or basis is not None:
            raise ValueError(
                "ccsd_target_error_ha currently requires distance=None and basis=None."
            )

    times = [float(t) for t in np.arange(t_start, t_end, t_step) if t != 0.0]

    ccsd_selection: dict[str, Any] | None = None
    if ccsd_target_error_ha is not None:
        rank, selected_fraction, ccsd_selection = _select_rank_from_ccsd_target(
            molecule_type,
            target_error_ha=float(ccsd_target_error_ha),
            thresh_range=ccsd_thresh_range,
            use_kernel=ccsd_use_kernel,
            no_triples=ccsd_no_triples,
            record_in_config=True,
        )
        _, _, constant, one_body, two_body = _run_scf_and_integrals(molecule_type)
        if debug:
            debug_print(
                "ccsd rank selection: "
                f"target={float(ccsd_target_error_ha):.6e}Ha "
                f"selected_rank={rank} "
                f"selected_fraction={selected_fraction:.6f} "
                f"abs_error={float(ccsd_selection['selected_abs_ccsd_error_ha']):.6e}Ha "
                f"threshold={float(ccsd_selection['selected_threshold']):.6e} "
                f"target_met={bool(ccsd_selection['target_met'])} "
                f"scan={int(ccsd_selection.get('thresholds_evaluated', 0))}/"
                f"{int(ccsd_selection.get('thresholds_total', 0))} "
                f"stopped_early={bool(ccsd_selection.get('stopped_early', False))}"
            )
    else:
        use_pyscf_integrals = reference == "exact" and distance is None and basis is None
        if use_pyscf_integrals:
            constant, one_body, two_body = _h_chain_integrals_pyscf(molecule_type)
        else:
            constant, one_body, two_body = _h_chain_integrals(
                molecule_type, distance=distance, basis=basis
            )

    if ccsd_target_error_ha is None and rank_fraction is not None:
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
    elif rank_fraction is not None:
        raise ValueError("rank_fraction cannot be set when ccsd_target_error_ha is used.")
    two_body = _symmetrize_two_body(two_body)
    one_body_spin, two_body_spin = spinorb_from_spatial(one_body, two_body * 0.5)

    raw_model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=rank, tol=tol
    )
    model = raw_model.hermitize()
    h_eff = one_body_spin + model.one_body_correction
    perm = _bit_reverse_permutation(model.N)

    df_blocks = build_df_blocks(model)
    blocks: list[Block] = []
    one_body_block = build_one_body_gaussian_block(h_eff)
    blocks.append(Block.from_one_body_gaussian(one_body_block))
    blocks.extend(Block.from_df(b) for b in df_blocks)

    rz_costs: dict[str, object] | None = None
    if ccsd_selection is not None:
        rz_costs = {"ccsd_rank_selection": ccsd_selection}
    df_blocks_cost: list[DFBlock] | None = None
    one_body_block_cost: OneBodyGaussianBlock | None = None
    if trace_u_debug or return_costs:
        one_body_block_cost = build_one_body_gaussian_block_givens(h_eff)
        df_blocks_cost = build_df_blocks_givens(model)

    if trace_u_debug:
        t_debug = float(times[0]) if times else float(t_step)
        debug_trace_u_decomposition(
            one_body_block_cost.U_ops if one_body_block_cost else one_body_block.U_ops,
            "U one_body",
            num_qubits=model.N,
            decompose_reps=cost_decompose_reps,
            basis_gates=cost_basis_gates,
            opt_level=cost_optimization_level,
            debug_print=debug_print,
            )
        for idx, df_block in enumerate(df_blocks_cost or df_blocks):
            debug_trace_u_decomposition(
                df_block.U_ops,
                f"U df[{idx}]",
                num_qubits=model.N,
                decompose_reps=cost_decompose_reps,
                basis_gates=cost_basis_gates,
                opt_level=cost_optimization_level,
                debug_print=debug_print,
            )
        d_one_body_qc = _build_d_block_circuit(
            blocks[0],
            t_debug,
            num_qubits=model.N,
        )
        debug_trace_u_decomposition(
            d_one_body_qc,
            "D one_body",
            decompose_reps=cost_decompose_reps,
            basis_gates=cost_basis_gates,
            opt_level=cost_optimization_level,
            debug_print=debug_print,
        )
        df_idx = 0
        for blk in blocks[1:]:
            if blk.kind != "df":
                continue
            d_blk_qc = _build_d_block_circuit(
                blk,
                t_debug,
                num_qubits=model.N,
            )
            debug_trace_u_decomposition(
                d_blk_qc,
                f"D df[{df_idx}]",
                decompose_reps=cost_decompose_reps,
                basis_gates=cost_basis_gates,
                opt_level=cost_optimization_level,
                debug_print=debug_print,
            )
            df_idx += 1

    if return_costs:
        t_cost = float(times[0]) if times else float(t_step)
        blocks_cost: list[Block] = []
        if one_body_block_cost is None:
            one_body_block_cost = build_one_body_gaussian_block_givens(h_eff)
        if df_blocks_cost is None:
            df_blocks_cost = build_df_blocks_givens(model)
        blocks_cost.append(Block.from_one_body_gaussian(one_body_block_cost))
        blocks_cost.extend(Block.from_df(b) for b in df_blocks_cost)
        qc_cost = build_df_trotter_circuit(
            blocks_cost,
            time=t_cost,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )
        if any(inst.operation.name.lower() == "unitary" for inst in qc_cost.data):
            raise RuntimeError(
                "UnitaryGate found in cost circuit; Givens expansion failed."
            )
        rz_costs = rz_costs_from_circuit(
            qc_cost,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
        if ccsd_selection is not None:
            rz_costs["ccsd_rank_selection"] = ccsd_selection
        d_only_qc = _build_d_only_cost_circuit(
            blocks,
            t_cost,
            num_qubits=model.N,
            pf_label=pf_label,
        )
        if any(inst.operation.name.lower() == "unitary" for inst in d_only_qc.data):
            raise RuntimeError("UnitaryGate found in D-only cost circuit.")
        d_only_cost = nonclifford_rz_costs_from_circuit(
            d_only_qc,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
        d_only_proxy_cost = d_nonclifford_costs_from_circuit(
            d_only_qc,
            debug=debug,
            debug_print=debug_print,
        )
        u_costs: list[dict[str, object]] = []
        ob_cost = u_nonclifford_costs_from_u_ops(
            one_body_block_cost.U_ops,
            model.N,
            debug=debug,
            debug_print=debug_print,
        )
        ob_cost["label"] = "one_body"
        u_costs.append(ob_cost)
        d_qc = QuantumCircuit(model.N)
        apply_D_one_body(d_qc, one_body_block_cost.eps, 1.0)
        d_cost = nonclifford_rz_costs_from_circuit(
            d_qc,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
        d_cost_proxy = d_nonclifford_costs_from_circuit(
            d_qc,
            debug=debug,
            debug_print=debug_print,
        )
        d_cost["label"] = "one_body_D"
        u_costs.append(d_cost)
        d_block_costs: list[dict[str, object]] = []
        d_block_proxy_costs: list[dict[str, object]] = []
        d_block_costs.append(d_cost)
        d_cost_proxy["label"] = "one_body_D_proxy"
        d_block_proxy_costs.append(d_cost_proxy)
        for idx, df_block in enumerate(df_blocks_cost):
            df_cost = u_nonclifford_costs_from_u_ops(
                df_block.U_ops,
                model.N,
                debug=debug,
                debug_print=debug_print,
            )
            df_cost["label"] = f"df[{idx}]"
            u_costs.append(df_cost)
            d_blk_qc = _build_d_block_circuit(
                blocks[idx + 1],
                t_cost,
                num_qubits=model.N,
            )
            d_blk_cost = nonclifford_rz_costs_from_circuit(
                d_blk_qc,
                basis_gates=cost_basis_gates,
                decompose_reps=cost_decompose_reps,
                optimization_level=cost_optimization_level,
            )
            d_blk_proxy_cost = d_nonclifford_costs_from_circuit(
                d_blk_qc,
                debug=debug,
                debug_print=debug_print,
            )
            d_blk_cost["label"] = f"df[{idx}]_D"
            d_block_costs.append(d_blk_cost)
            d_blk_proxy_cost["label"] = f"df[{idx}]_D_proxy"
            d_block_proxy_costs.append(d_blk_proxy_cost)
        rz_costs["u_costs"] = u_costs
        rz_costs["d_only_costs"] = d_only_cost
        rz_costs["d_block_costs"] = d_block_costs
        rz_costs["d_only_proxy_costs"] = d_only_proxy_cost
        rz_costs["d_block_proxy_costs"] = d_block_proxy_costs
        weights = _get_w_list(pf_label)
        u_total_count = 0
        u_total_depth = 0
        u_total_coloring_depth = 0
        for term_idx, _weight in iter_pf_steps(len(blocks), weights):
            blk = blocks[term_idx]
            if blk.kind == "one_body_gaussian":
                cost = u_costs[0]
            elif blk.kind == "df":
                df_i = term_idx - 1
                cost = u_costs[df_i + 1]
            else:
                continue
            u_total_count += 2 * int(cost.get("u_nonclifford_rz_count", 0))
            u_total_depth += 2 * int(cost.get("u_nonclifford_rz_depth", 0))
            u_total_coloring_depth += 2 * int(
                cost.get("u_nonclifford_z_coloring_depth", 0)
            )
        d_total_count = int(d_only_cost.get("nonclifford_rz_count", 0))
        d_total_depth = int(d_only_cost.get("nonclifford_rz_depth", 0))
        rz_costs["nonclifford_total"] = {
            "u_nonclifford_rz_count": u_total_count,
            "u_nonclifford_rz_depth": u_total_depth,
            "d_nonclifford_rz_count": d_total_count,
            "d_nonclifford_rz_depth": d_total_depth,
            "total_nonclifford_rz_count": u_total_count + d_total_count,
            "total_nonclifford_rz_depth": u_total_depth + d_total_depth,
        }
        d_proxy_count = int(d_only_proxy_cost.get("nonclifford_total", 0))
        d_proxy_depth = int(d_only_proxy_cost.get("combined_nonclifford_depth", 0))
        d_total_coloring_depth = 0
        for term_idx, _weight in iter_pf_steps(len(blocks), weights):
            blk = blocks[term_idx]
            if blk.kind == "one_body_gaussian":
                d_cost = d_block_proxy_costs[0]
            elif blk.kind == "df":
                df_i = term_idx - 1
                d_cost = d_block_proxy_costs[df_i + 1]
            else:
                continue
            d_total_coloring_depth += int(d_cost.get("coloring_nonclifford_depth", 0))
        rz_costs["toffoli_proxy_total"] = {
            "u_nonclifford_z_count": u_total_count,
            "u_nonclifford_z_depth": u_total_depth,
            "d_nonclifford_z_count": d_proxy_count,
            "d_nonclifford_z_depth": d_proxy_depth,
            "total_nonclifford_z_count": u_total_count + d_proxy_count,
            "total_nonclifford_z_depth": u_total_depth + d_proxy_depth,
        }
        rz_costs["toffoli_proxy_total_coloring"] = {
            "u_nonclifford_z_count": u_total_count,
            "u_nonclifford_z_depth": u_total_coloring_depth,
            "d_nonclifford_z_count": d_proxy_count,
            "d_nonclifford_z_depth": d_total_coloring_depth,
            "total_nonclifford_z_count": u_total_count + d_proxy_count,
            "total_nonclifford_z_depth": u_total_coloring_depth + d_total_coloring_depth,
        }

    energy_ref = None
    psi0 = None
    state_vec = None
    h_ref = None
    h_exact_open = None
    phase_energy_ref = None
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
            phase_energy_ref = phase_energy

    if debug:
        _run_debug_diagnostics(
            raw_model=raw_model,
            model=model,
            constant=constant,
            one_body_spin=one_body_spin,
            two_body_spin=two_body_spin,
            blocks=blocks,
            reference=reference,
            energy_ref=float(energy_ref) if energy_ref is not None else None,
            psi0=psi0,
            h_ref=h_ref,
            h_exact_open=h_exact_open,
            rank=rank,
            tol=tol,
            pf_label=pf_label,
            phase_energy_ref=phase_energy_ref,
            debug_print=debug_print,
        )

    if debug:
        _debug_rz_count(
            blocks=blocks,
            times=times,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
            debug_print=debug_print,
        )

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
            perr = _perturbation_error(
                t_sim, energy_ref_eff, psi0, psi_t  # type: ignore[arg-type]
            )
            errors.append(perr)
            if debug:
                if (debug_max is None or debug_count < debug_max) and (
                    idx % debug_every == 0
                ):
                    msg = _debug_perturbation_step(
                        t_sim=t_sim,
                        psi0=psi0,  # type: ignore[arg-type]
                        psi_t=psi_t,
                        energy_ref_eff=energy_ref_eff,
                        perr=perr,
                        h_ref=h_ref,
                        energy_ref=float(energy_ref) if energy_ref is not None else None,
                        num_qubits=model.N,
                        debug_compare_expectation=debug_compare_expectation,
                    )
                    debug_print(msg)
                    debug_count += 1
        if return_costs:
            return times, errors, rz_costs or {}
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
        if return_costs:
            return times_out, error_list, rz_costs or {}
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

    if return_costs:
        return times, errors, rz_costs or {}
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
    debug: bool = False,
    debug_every: int = 1,
    debug_max: int | None = 30,
    debug_print: Callable[[str], None] = print,
    debug_compare_expectation: bool = True,
    report_costs: bool = False,
    return_costs: bool = False,
    cost_basis_gates: Sequence[str] | None = None,
    cost_decompose_reps: int = 8,
    cost_optimization_level: int = 0,
    trace_u_debug: bool = False,
    ccsd_target_error_ha: float | None = None,
    ccsd_thresh_range: Sequence[float] | None = None,
    ccsd_use_kernel: bool = False,
    ccsd_no_triples: bool = False,
    save_fit_params: bool = False,
    save_rz_layers: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], dict[str, object]]:
    compute_costs_by_default = True
    result = df_trotter_energy_error_curve(
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
        debug=debug,
        debug_every=debug_every,
        debug_max=debug_max,
        debug_print=debug_print,
        debug_compare_expectation=debug_compare_expectation,
        return_costs=compute_costs_by_default or report_costs or return_costs,
        cost_basis_gates=cost_basis_gates,
        cost_decompose_reps=cost_decompose_reps,
        cost_optimization_level=cost_optimization_level,
        trace_u_debug=trace_u_debug,
        ccsd_target_error_ha=ccsd_target_error_ha,
        ccsd_thresh_range=ccsd_thresh_range,
        ccsd_use_kernel=ccsd_use_kernel,
        ccsd_no_triples=ccsd_no_triples,
    )
    times, errors, costs = result  # type: ignore[misc]

    fraction_for_legend: float | None = None
    rank_for_legend: tuple[int, int] | None = None
    if isinstance(costs, dict):
        ccsd_sel = costs.get("ccsd_rank_selection")
        if isinstance(ccsd_sel, dict):
            selected_rank = ccsd_sel.get("selected_rank")
            full_rank = ccsd_sel.get("full_rank")
            if selected_rank is not None and full_rank is not None:
                rank_for_legend = (int(selected_rank), int(full_rank))
            selected_fraction = ccsd_sel.get("selected_rank_fraction")
            if selected_fraction is not None:
                fraction_for_legend = float(selected_fraction)
    if fraction_for_legend is None and rank_fraction is not None:
        fraction_for_legend = float(rank_fraction)
    if rank_for_legend is None:
        config_selection = get_df_rank_selection_for_molecule(int(molecule_type))
        if isinstance(config_selection, dict):
            selected_rank_cfg = config_selection.get("selected_rank")
            full_rank_cfg = config_selection.get("full_rank")
            if selected_rank_cfg is not None and full_rank_cfg is not None:
                rank_for_legend = (int(selected_rank_cfg), int(full_rank_cfg))
    if fraction_for_legend is None:
        config_fraction = get_df_rank_fraction_for_molecule(int(molecule_type))
        if config_fraction is not None:
            fraction_for_legend = float(config_fraction)

    if rank_for_legend is None and fraction_for_legend is not None:
        full_rank_guess = max(1, int(molecule_type) ** 2)
        selected_rank_guess = int(round(full_rank_guess * fraction_for_legend))
        selected_rank_guess = max(1, min(selected_rank_guess, full_rank_guess))
        rank_for_legend = (selected_rank_guess, full_rank_guess)

    error_label = "error()"
    if rank_for_legend is not None:
        selected_rank_legend, full_rank_legend = rank_for_legend
        error_label = f"error(L={selected_rank_legend}/{full_rank_legend})"

    ax = plt.gca()
    set_loglog_axes(
        ax,
        xlabel="time",
        ylabel="energy error [Hartree]",
        title=f"H{molecule_type}_DF_{pf_label}",
    )
    ax.plot(times, errors, marker="o", linestyle="-", label=error_label)
    fit_result = None
    fit_error: str | None = None
    if fit or save_fit_params:
        try:
            fit_result = loglog_fit(times, errors, mask_nonpositive=True, compute_r2=True)
            if fit:
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
            fit_error = str(exc)
            if fit:
                print(f"log-log fit skipped: {exc}")
    ax.legend()
    plt.show()

    if save_fit_params or save_rz_layers:
        if pf_label is not None:
            artifact_name = (
                f"{_artifact_ham_name(molecule_type, distance=distance, basis=basis)}"
                f"_Operator_{pf_label}"
            )
        else:
            artifact_name = (
                f"{_artifact_ham_name(molecule_type, distance=distance, basis=basis)}"
                "_Operator_normal"
            )
        payload: dict[str, object] = {}
        if save_fit_params:
            if fit_result is not None:
                payload["expo"] = float(fit_result.slope)
                payload["coeff"] = float(fit_result.coeff)
                if fit_result.r2 is not None:
                    payload["r2"] = float(fit_result.r2)
            else:
                payload["fit_error"] = fit_error or "log-log fit unavailable"
        if save_rz_layers:
            payload["rz_layers"] = (
                _collect_df_rz_layer_metrics(costs) if isinstance(costs, dict) else {}
            )
        _save_df_plot_artifact(artifact_name, payload)

    if report_costs and costs is not None:
        ccsd_sel = costs.get("ccsd_rank_selection")
        if ccsd_sel:
            selected_fraction = float(ccsd_sel.get("selected_rank_fraction", 0.0))
            selected_rank = int(ccsd_sel.get("selected_rank", 0))
            full_rank = int(ccsd_sel.get("full_rank", 0))
            rank_ratio = (
                f"{selected_rank}/{full_rank}" if full_rank > 0 else f"{selected_rank}"
            )
            print(
                "ccsd_fraction: "
                f"rank_fraction={selected_fraction:.6f} "
                f"({selected_fraction * 100:.2f}%) "
                f"rank={rank_ratio}"
            )
            print(
                "ccsd_rank_selection: "
                f"target={float(ccsd_sel.get('target_error_ha', 0.0)):.6e}Ha "
                f"selected_rank={selected_rank} "
                f"selected_fraction={selected_fraction:.6f} "
                f"abs_error={float(ccsd_sel.get('selected_abs_ccsd_error_ha', 0.0)):.6e}Ha "
                f"target_met={bool(ccsd_sel.get('target_met', False))}"
            )
        rz_count = costs.get("rz_count")
        rz_depth = costs.get("rz_depth")
        depth_total = costs.get("depth_total")
        nonbasis = costs.get("nonbasis_ops", [])
        msg = (
            f"ref_rz_count={rz_count} ref_rz_depth={rz_depth} "
            f"ref_depth_total={depth_total}"
        )
        if nonbasis:
            msg += f" nonbasis_ops={nonbasis}"
        print(msg)
        u_costs = costs.get("u_costs", [])
        if u_costs:
            for entry in u_costs:
                label = entry.get("label", "U")
                xx_cnt = entry.get("xx_plus_yy_count", 0)
                xx_layers = entry.get("xx_plus_yy_layers", 0)
                mapped = entry.get("u_nonclifford_rz_count", 0)
                u_depth = entry.get("u_nonclifford_rz_depth", 0)
                u_col_depth = entry.get("u_nonclifford_z_coloring_depth", 0)
                line = (
                    f"{label}: xx_plus_yy_count={xx_cnt} xx_plus_yy_layers={xx_layers} "
                    f"mapped_nonClifford_rz_count={mapped} nonClifford_rz_depth={u_depth} "
                    f"proxy_coloring_depth={u_col_depth}"
                )
                print(line)
                rz_raw = entry.get("u_raw_rz_count")
                rz_raw_noncliff = entry.get("u_raw_rz_nonclifford_count")
                if rz_raw is not None or rz_raw_noncliff is not None:
                    print(
                        f"{label}: u_raw_rz_count={rz_raw} "
                        f"u_raw_rz_nonclifford_count={rz_raw_noncliff}"
                    )
        d_only = costs.get("d_only_costs")
        if d_only is not None:
            d_line = (
                "D-only ref: "
                f"nonClifford_rz_count={d_only.get('nonclifford_rz_count')} "
                f"nonClifford_rz_depth={d_only.get('nonclifford_rz_depth')} "
                f"ref_rz_count={d_only.get('rz_count')} "
                f"ref_rz_depth={d_only.get('rz_depth')}"
            )
            print(d_line)
        d_only_proxy = costs.get("d_only_proxy_costs")
        if d_only_proxy is not None:
            d_proxy_line = (
                "D-only proxy: "
                f"rz_total={d_only_proxy.get('rz_total')} "
                f"rz_nonclifford={d_only_proxy.get('rz_nonclifford')} "
                f"rzz_total={d_only_proxy.get('rzz_total')} "
                f"rzz_nonclifford={d_only_proxy.get('rzz_nonclifford')} "
                f"combined_depth={d_only_proxy.get('combined_nonclifford_depth')} "
                f"coloring_depth={d_only_proxy.get('coloring_nonclifford_depth')}"
            )
            print(d_proxy_line)
        d_block_costs = costs.get("d_block_costs", [])
        if d_block_costs:
            for entry in d_block_costs:
                label = entry.get("label", "D")
                rz_d = entry.get("nonclifford_rz_count")
                rz_d_depth = entry.get("nonclifford_rz_depth")
                line = (
                    f"{label}(ref): nonClifford_rz_count={rz_d} "
                    f"nonClifford_rz_depth={rz_d_depth}"
                )
                print(line)
        d_block_proxy_costs = costs.get("d_block_proxy_costs", [])
        if d_block_proxy_costs:
            for entry in d_block_proxy_costs:
                label = entry.get("label", "D_proxy")
                rz_d = entry.get("nonclifford_total")
                rz_d_depth = entry.get("combined_nonclifford_depth")
                rz_d_col_depth = entry.get("coloring_nonclifford_depth")
                line = (
                    f"{label}: nonClifford_z_count={rz_d} "
                    f"nonClifford_z_depth={rz_d_depth} "
                    f"proxy_coloring_depth={rz_d_col_depth}"
                )
                print(line)
        totals = costs.get("nonclifford_total")
        if totals:
            total_line = (
                "TOTAL(ref): "
                f"nonClifford_rz_count={totals.get('total_nonclifford_rz_count')} "
                f"nonClifford_rz_depth={totals.get('total_nonclifford_rz_depth')}"
            )
            print(total_line)
        proxy_totals = costs.get("toffoli_proxy_total")
        if proxy_totals:
            proxy_line = (
                "TOTAL(Toffoli proxy): "
                f"nonClifford_z_count={proxy_totals.get('total_nonclifford_z_count')} "
                f"nonClifford_z_depth={proxy_totals.get('total_nonclifford_z_depth')} "
                f"U={proxy_totals.get('u_nonclifford_z_count')} "
                f"D={proxy_totals.get('d_nonclifford_z_count')}"
            )
            print(proxy_line)
            if int(proxy_totals.get("u_nonclifford_z_count", 0)) == 0:
                print("sanity: U nonClifford proxy count is 0")
        proxy_col_totals = costs.get("toffoli_proxy_total_coloring")
        if proxy_col_totals:
            proxy_col_line = (
                "TOTAL(Toffoli proxy coloring): "
                f"nonClifford_z_count={proxy_col_totals.get('total_nonclifford_z_count')} "
                f"nonClifford_z_depth={proxy_col_totals.get('total_nonclifford_z_depth')} "
                f"U_depth={proxy_col_totals.get('u_nonclifford_z_depth')} "
                f"D_depth={proxy_col_totals.get('d_nonclifford_z_depth')}"
            )
            print(proxy_col_line)
    if return_costs:
        return times, errors, (costs or {})
    return times, errors


__all__ = ["df_trotter_energy_error_curve", "df_trotter_energy_error_plot"]
