from __future__ import annotations

import numpy as np

from .model import DFModel


_CHEMIST_TO_PHYSICIST_PERMUTATION = (0, 2, 3, 1)


def two_body_tensor_from_df_model(model: DFModel) -> np.ndarray:
    """Reconstruct the two-body tensor in physicist ordering from a DF model."""
    n = model.N
    h2_chemist = np.zeros((n, n, n, n), dtype=np.complex128)
    for lam, g_mat in zip(model.lambdas, model.G_list):
        h2_chemist += lam * np.einsum("pq,rs->pqrs", g_mat, g_mat, optimize=True)
    return np.transpose(h2_chemist, _CHEMIST_TO_PHYSICIST_PERMUTATION)
