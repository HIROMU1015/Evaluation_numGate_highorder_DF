from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from openfermion import FermionOperator, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator

from trotterlib.df_trotter.decompose import df_decompose_from_integrals
from trotterlib.chemistry_hamiltonian import geo
from trotterlib.config import DEFAULT_BASIS, DEFAULT_DISTANCE


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


def _hamiltonian_matrix(
    constant: float, one_body_spin: np.ndarray, two_body_spin: np.ndarray
) -> np.ndarray:
    op = InteractionOperator(constant, one_body_spin, two_body_spin)
    n = one_body_spin.shape[0]
    return get_sparse_operator(op, n_qubits=n).toarray()


def _h_chain_integrals(
    molecule_type: int, *, distance: float | None, basis: str | None
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf
    except Exception as exc:
        raise RuntimeError(
            "openfermionpyscf is required to build molecular integrals."
        ) from exc

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


def _estimate_full_rank(
    one_body: np.ndarray, two_body: np.ndarray, constant: float
) -> int:
    model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=None, tol=0.0
    )
    return int(len(model.lambdas))


def _normalize_ranks(ranks: Iterable[int] | int, full_rank: int) -> list[int]:
    if isinstance(ranks, int):
        ranks = [ranks]
    normalized = []
    for rank in ranks:
        value = max(1, min(int(rank), full_rank))
        normalized.append(value)
    return sorted(set(normalized))


def _ranks_from_fractions(
    full_rank: int, fractions: Sequence[float], *, include_full: bool
) -> list[int]:
    ranks = []
    for frac in fractions:
        if frac <= 0:
            continue
        ranks.append(int(round(full_rank * frac)))
    if include_full:
        ranks.append(full_rank)
    return _normalize_ranks(ranks, full_rank)


def _maybe_shift_positional_molecule_type(
    ranks: Iterable[int] | int | None,
    *,
    rank_fractions: Sequence[float] | None,
    num_points: int | None,
    molecule_type: int,
    distance: float | None,
    basis: str | None,
    tol: float | None,
) -> tuple[Iterable[int] | int | None, int]:
    if (
        ranks is not None
        and isinstance(ranks, int)
        and rank_fractions is None
        and num_points is None
        and molecule_type == 2
        and distance is None
        and basis is None
        and tol is None
    ):
        return None, int(ranks)
    return ranks, molecule_type


def df_rank_error_curve(
    ranks: Iterable[int] | None = None,
    *,
    rank_fractions: Sequence[float] | None = None,
    include_full: bool = True,
    num_points: int | None = None,
    molecule_type: int = 2,
    distance: float | None = None,
    basis: str | None = None,
    tol: float | None = None,
) -> list[dict[str, float]]:
    constant, one_body, two_body = _h_chain_integrals(
        molecule_type, distance=distance, basis=basis
    )
    one_body_spin, two_body_spin = spinorb_from_spatial(one_body, two_body * 0.5)
    h_exact = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)
    h1 = _one_body_matrix(one_body_spin)
    I = np.eye(h1.shape[0], dtype=np.complex128)

    ranks, molecule_type = _maybe_shift_positional_molecule_type(
        ranks,
        rank_fractions=rank_fractions,
        num_points=num_points,
        molecule_type=molecule_type,
        distance=distance,
        basis=basis,
        tol=tol,
    )
    if ranks is not None and rank_fractions is not None:
        raise ValueError("Specify either ranks or rank_fractions, not both.")

    full_rank = _estimate_full_rank(one_body, two_body, constant)
    if ranks is None:
        if rank_fractions is None:
            if num_points is not None:
                points = max(2, int(num_points))
                rank_fractions = np.linspace(1.0, 0.1, points).tolist()
            else:
                rank_fractions = [1.0, 0.75, 0.5, 0.25]
        ranks = _ranks_from_fractions(
            full_rank, list(rank_fractions), include_full=include_full
        )
    else:
        ranks = _normalize_ranks(ranks, full_rank)

    results: list[dict[str, float]] = []
    for rank in ranks:
        model = df_decompose_from_integrals(
            one_body, two_body, constant=constant, rank=int(rank), tol=tol
        )
        h_df = constant * I + h1 + _df_two_body_matrix(model)
        diff = h_exact - h_df
        abs_err = float(np.linalg.norm(diff))
        rel_err = abs_err / float(np.linalg.norm(h_exact))
        results.append(
            {
                "rank": float(rank),
                "full_rank": float(full_rank),
                "rank_frac": float(rank) / float(full_rank),
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
        )
    return results


def print_df_rank_error_curve(
    ranks: Iterable[int] | None = None,
    *,
    rank_fractions: Sequence[float] | None = None,
    include_full: bool = True,
    num_points: int | None = None,
    molecule_type: int = 2,
    distance: float | None = None,
    basis: str | None = None,
    tol: float | None = None,
) -> None:
    ranks, molecule_type = _maybe_shift_positional_molecule_type(
        ranks,
        rank_fractions=rank_fractions,
        num_points=num_points,
        molecule_type=molecule_type,
        distance=distance,
        basis=basis,
        tol=tol,
    )
    results = df_rank_error_curve(
        ranks,
        rank_fractions=rank_fractions,
        include_full=include_full,
        num_points=num_points,
        molecule_type=molecule_type,
        distance=distance,
        basis=basis,
        tol=tol,
    )
    full_rank = int(results[0]["full_rank"]) if results else 0
    print(f"full_rank={full_rank}")
    print("rank  rank_frac  abs_err        rel_err")
    for row in results:
        print(
            f"{int(row['rank']):4d}  {row['rank_frac']:.6f}  "
            f"{row['abs_err']:.6e}  {row['rel_err']:.6e}"
        )


__all__ = ["df_rank_error_curve", "print_df_rank_error_curve"]
