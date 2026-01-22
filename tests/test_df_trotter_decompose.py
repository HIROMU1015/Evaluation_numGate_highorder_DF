import numpy as np
import pytest

from trotterlib.df_trotter.decompose import diag_hermitian, df_decompose_from_integrals


def test_diag_hermitian_sorts_and_is_unitary() -> None:
    mat = np.array([[1.0, 0.0], [0.0, -2.0]])
    U, evals = diag_hermitian(mat, sort="descending_abs")
    assert np.allclose(evals, np.array([-2.0, 1.0]))
    assert np.allclose(U.conj().T @ U, np.eye(2))

    _, evals_desc = diag_hermitian(mat, sort="descending")
    assert np.allclose(evals_desc, np.array([1.0, -2.0]))


def test_df_decompose_from_integrals_shapes() -> None:
    pytest.importorskip("openfermion")

    one_body = np.array([[0.5]])
    two_body = np.zeros((1, 1, 1, 1))
    two_body[0, 0, 0, 0] = 0.7

    model = df_decompose_from_integrals(one_body, two_body, constant=0.0, rank=1)
    assert model.N == 2
    assert model.one_body_correction.shape == (2, 2)
    assert len(model.lambdas) == len(model.G_list)
    for g_mat in model.G_list:
        assert g_mat.shape == (2, 2)
