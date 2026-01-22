from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from openfermion import FermionOperator, InteractionOperator
from openfermion.chem import MolecularData
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf

from trotterlib.chemistry_hamiltonian import geo
from trotterlib.config import DEFAULT_BASIS, DEFAULT_DISTANCE, PFLabel, pf_order
from trotterlib.df_trotter.decompose import df_decompose_from_integrals
from trotterlib.df_trotter.model import DFModel, Block
from trotterlib.df_trotter.ops import build_df_blocks, build_one_body_gaussian_block
from trotterlib.df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from trotterlib.analysis_utils import (
    loglog_average_coeff,
    loglog_fit,
    print_loglog_fit,
)
from trotterlib.plot_utils import set_loglog_axes
from trotterlib.qiskit_time_evolution_pyscf import (
    _run_scf_and_integrals,
    make_fci_vector_from_pyscf_solver,
)


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
    return np.asarray(vec).reshape(-1)[perm]


def _reorder_matrix(mat: np.ndarray, perm: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat)
    return arr[np.ix_(perm, perm)]


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
    return 0.25 * (
        t
        + np.transpose(t, (1, 0, 3, 2))
        + np.transpose(t, (2, 3, 0, 1))
        + np.transpose(t, (3, 2, 1, 0))
    )


def df_ground_energy_from_df(
    molecule_type: int,
    *,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
) -> float:
    """Return the DF Hamiltonian ground-state energy."""

    if rank_fraction is not None:
        if rank is not None:
            raise ValueError("rank and rank_fraction are mutually exclusive.")
        if rank_fraction <= 0:
            raise ValueError("rank_fraction must be positive.")

    use_pyscf_integrals = distance is None and basis is None
    if use_pyscf_integrals:
        constant, one_body, two_body = _h_chain_integrals_pyscf(molecule_type)
    else:
        constant, one_body, two_body = _h_chain_integrals(
            molecule_type, distance=distance, basis=basis
        )
    two_body = _symmetrize_two_body(two_body)
    if rank_fraction is not None:
        n_spatial = int(one_body.shape[0])
        full_rank = int(n_spatial**2)
        if rank_fraction >= 1.0:
            rank = full_rank
            if tol is None:
                tol = 0.0
        else:
            rank = int(round(full_rank * rank_fraction))
            rank = max(1, min(rank, full_rank))

    one_body_spin, _ = spinorb_from_spatial(one_body, two_body * 0.5)
    model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=rank, tol=tol
    )
    h_df_open = _hamiltonian_matrix_from_df_model(constant, one_body_spin, model)
    evals = np.linalg.eigvalsh(h_df_open)
    return float(np.min(evals.real))


def df_phase_match_times(
    molecule_type: int,
    n_values: Sequence[int],
    *,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    use_abs_energy: bool = True,
) -> tuple[float, list[float]]:
    """Return (E0, [t]) where E0 * t = n * pi/4 for each n in n_values."""

    energy = df_ground_energy_from_df(
        molecule_type,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
    )
    if abs(energy) < 1e-12:
        raise ValueError("Ground-state energy is too close to zero.")
    denom = abs(energy) if use_abs_energy else energy
    times = [float(n) * np.pi / (4.0 * denom) for n in n_values]
    return energy, times


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


def _df_two_body_matrix(model: DFModel) -> np.ndarray:
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


def _hamiltonian_matrix(
    constant: float, one_body_spin: np.ndarray, two_body_spin: np.ndarray
) -> np.ndarray:
    op = InteractionOperator(constant, one_body_spin, two_body_spin)
    n = one_body_spin.shape[0]
    return get_sparse_operator(op, n_qubits=n).toarray()


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


@dataclass(frozen=True)
class DFTrotterSetup:
    blocks: tuple[Block, ...]
    model: DFModel
    constant: float
    energy_shift: float
    pf_label: PFLabel
    num_qubits: int


def df_trotter_pauli_rotation_count(setup: DFTrotterSetup) -> int:
    qc = build_df_trotter_circuit(
        setup.blocks,
        time=1.0,
        num_qubits=setup.num_qubits,
        pf_label=setup.pf_label,
        energy_shift=setup.energy_shift,
    )
    counts = qc.count_ops()
    pauli_ops = ("rz", "RZ", "rzz", "RZZGate")
    return sum(int(counts.get(op, 0)) for op in pauli_ops)


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


def df_trotter_energy_error_curve_perturb(
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
    reference: str = "exact",
    calibrate_phase: bool = True,
    return_setup: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], DFTrotterSetup]:
    if t_step <= 0:
        raise ValueError("t_step must be positive.")
    if reference not in ("exact", "df"):
        raise ValueError("reference must be 'exact' or 'df'.")

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

    model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=rank, tol=tol
    )
    perm = _bit_reverse_permutation(model.N)
    h_eff = one_body_spin + model.one_body_correction

    blocks: list[Block] = []
    one_body_block = build_one_body_gaussian_block(h_eff)
    blocks.append(Block.from_one_body_gaussian(one_body_block))
    blocks.extend(Block.from_df(b) for b in build_df_blocks(model))

    setup = DFTrotterSetup(
        blocks=tuple(blocks),
        model=model,
        constant=constant,
        energy_shift=constant + model.constant_correction,
        pf_label=pf_label,
        num_qubits=model.N,
    )

    psi0: np.ndarray
    use_fci = False
    if reference == "exact" and distance is None and basis is None and molecule_type != 6:
        _, n_qubits, energy_ref, state_vec, _ = make_fci_vector_from_pyscf_solver(
            molecule_type
        )
        if n_qubits != model.N:
            raise ValueError("FCI qubit count does not match DF model size.")
        psi0 = _reorder_vector(np.asarray(state_vec).reshape(-1), perm)
        use_fci = True
    else:
        if reference == "exact":
            h_ref_open = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
        else:
            h_ref_open = _hamiltonian_matrix_from_df_model(
                constant, one_body_spin, model
            )
        evals, evecs = np.linalg.eigh(h_ref_open)
        idx = int(np.argmin(evals.real))
        energy_ref = float(evals.real[idx])
        psi0 = _reorder_vector(evecs[:, idx], perm)

    if reference == "exact" and use_fci:
        h_exact_open = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
        h_ref = _reorder_matrix(h_exact_open, perm)
        e0_check = np.vdot(psi0, h_ref @ psi0).real
        if abs(e0_check - float(energy_ref)) > 1e-6:
            evals, evecs = np.linalg.eigh(h_exact_open)
            idx = int(np.argmin(evals.real))
            energy_ref = float(evals.real[idx])
            psi0 = _reorder_vector(evecs[:, idx], perm)

    energy_ref_eff = energy_ref
    if calibrate_phase and reference == "df":
        energy_ref_eff = _phase_energy_from_circuit(
            blocks,
            psi0,
            1e-3,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )

    errors: list[float] = []
    for time in times:
        qc = build_df_trotter_circuit(
            blocks,
            time=time,
            num_qubits=model.N,
            pf_label=pf_label,
            energy_shift=constant + model.constant_correction,
        )
        psi_t = simulate_statevector(qc, psi0)
        errors.append(_perturbation_error(time, energy_ref_eff, psi0, psi_t))

    if return_setup:
        return times, errors, setup
    return times, errors


def df_trotter_energy_error_plot_perturb(
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
    reference: str = "exact",
    calibrate_phase: bool = True,
    fit: bool = True,
) -> tuple[list[float], list[float]]:
    times, errors = df_trotter_energy_error_curve_perturb(
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
        reference=reference,
        calibrate_phase=calibrate_phase,
    )

    plot_df_trotter_error_curve(times, errors, molecule_type=molecule_type, pf_label=pf_label, fit=fit)
    return times, errors


def plot_df_trotter_error_curve(
    times: Sequence[float],
    errors: Sequence[float],
    *,
    molecule_type: int,
    pf_label: PFLabel,
    fit: bool = True,
) -> None:
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
            ax.text(
                0.05,
                0.95,
                f"error exponent: {p:.2f}\nerror coefficient: {alpha:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        except ValueError as exc:
            print(f"log-log fit skipped: {exc}")
    ax.legend()
    plt.show()


def df_trotter_fixed_order_coeff(
    times: Sequence[float],
    errors: Sequence[float],
    pf_label: PFLabel,
    *,
    exponent: float | None = None,
    mask_nonpositive: bool = True,
) -> float:
    """Return the prefactor α in error ≈ α · t^p, where p matches the PF label."""

    if exponent is None:
        exponent = pf_order(pf_label)
    return loglog_average_coeff(
        times,
        errors,
        exponent,
        mask_nonpositive=mask_nonpositive,
    )


__all__ = [
    "df_trotter_energy_error_curve_perturb",
    "df_trotter_energy_error_plot_perturb",
    "df_trotter_fixed_order_coeff",
    "DFTrotterSetup",
    "df_trotter_pauli_rotation_count",
    "plot_df_trotter_error_curve",
    "df_ground_energy_from_df",
    "df_phase_match_times",
]
