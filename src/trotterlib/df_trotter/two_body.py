from __future__ import annotations

import numpy as np

from openfermion import FermionOperator, InteractionOperator
from openfermion.transforms import get_interaction_operator, normal_ordered

from .model import DFModel


_CHEMIST_TO_PHYSICIST_PERMUTATION = (0, 2, 3, 1)


def two_body_tensor_from_df_model(model: DFModel) -> np.ndarray:
    """Reconstruct the two-body tensor in physicist ordering from a DF model."""
    n = model.N
    h2_chemist = np.zeros((n, n, n, n), dtype=np.complex128)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        h2_chemist += lam * np.einsum("pq,rs->pqrs", g_mat, g_mat, optimize=True)
    return np.transpose(h2_chemist, _CHEMIST_TO_PHYSICIST_PERMUTATION)


def interaction_operator_from_chemist_integrals(
    constant: float,
    one_body_spin: np.ndarray,
    two_body_chemist: np.ndarray,
) -> InteractionOperator:
    n = one_body_spin.shape[0]
    op = FermionOperator((), constant)
    for p in range(n):
        for q in range(n):
            coeff = one_body_spin[p, q]
            if abs(coeff) < 1e-14:
                continue
            op += FermionOperator(((p, 1), (q, 0)), coeff)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    coeff = two_body_chemist[p, q, r, s]
                    if abs(coeff) < 1e-14:
                        continue
                    op += FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)), coeff)
    op = normal_ordered(op)
    return get_interaction_operator(op)


def interaction_operator_from_df_model(
    constant: float,
    one_body_spin: np.ndarray,
    model: DFModel,
) -> InteractionOperator:
    n_spin = model.N
    n_spatial = n_spin // 2
    two_body_spatial = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=np.complex128)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        g_spatial = g_mat[::2, ::2]
        two_body_spatial += lam * np.einsum("pq,rs->pqrs", g_spatial, g_spatial, optimize=True)

    two_body_chemist = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.complex128)
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    coeff = two_body_spatial[p, q, r, s]
                    if abs(coeff) < 1e-14:
                        continue
                    two_body_chemist[2 * p, 2 * q, 2 * r, 2 * s] = coeff
                    two_body_chemist[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = coeff
                    two_body_chemist[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = coeff
                    two_body_chemist[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = coeff
    return interaction_operator_from_chemist_integrals(
        constant + model.constant_correction,
        one_body_spin + model.one_body_correction,
        two_body_chemist,
    )
