import numpy as np
import pytest

pytest.importorskip("openfermion")

from openfermion.circuits import get_chemist_two_body_coefficients

from trotterlib.df_trotter.model import DFModel
from trotterlib.df_trotter.two_body import two_body_tensor_from_df_model


def test_df_two_body_tensor_matches_openfermion_chemist_mapping() -> None:
    rng = np.random.default_rng(0)
    n_orb = 4
    g1 = rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb))
    g1 = 0.5 * (g1 + g1.conj().T)
    g2 = rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb))
    g2 = 0.5 * (g2 + g2.conj().T)
    lambdas = np.array([1.3, -0.7], dtype=float)

    h2_chemist = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.complex128)
    for lam, g_mat in zip(lambdas, [g1, g2]):
        h2_chemist += lam * np.einsum("pq,rs->pqrs", g_mat, g_mat, optimize=True)

    model = DFModel(
        lambdas=lambdas,
        G_list=[g1, g2],
        one_body_correction=np.zeros((n_orb, n_orb), dtype=np.complex128),
        constant_correction=0.0,
        N=n_orb,
    )
    two_body = two_body_tensor_from_df_model(model)
    _, chemist_back = get_chemist_two_body_coefficients(two_body, spin_basis=False)
    assert np.allclose(chemist_back, h2_chemist, atol=1e-10)
