import numpy as np
import pytest

pytest.importorskip("openfermion")

from openfermion import FermionOperator, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator

from trotterlib.df_trotter.decompose import df_decompose_from_integrals


def _random_symmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(n, n))
    return 0.5 * (mat + mat.T)


def _random_two_body(n: int, rng: np.random.Generator) -> np.ndarray:
    tensor = rng.normal(size=(n, n, n, n))
    tensor = 0.5 * (tensor + tensor.swapaxes(0, 1))
    tensor = 0.5 * (tensor + tensor.swapaxes(2, 3))
    tensor = 0.5 * (tensor + tensor.transpose(2, 3, 0, 1))
    return tensor


def _spin_orbital_integrals(
    one_body: np.ndarray, two_body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return spinorb_from_spatial(one_body, two_body * 0.5)


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


def _h2_integrals_from_openfermion() -> tuple[float, np.ndarray, np.ndarray]:
    pytest.importorskip("openfermionpyscf")
    pytest.importorskip("pyscf")
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf

    from trotterlib.chemistry_hamiltonian import geo
    from trotterlib.config import DEFAULT_BASIS, DEFAULT_DISTANCE

    geometry, multiplicity, charge = geo(2, DEFAULT_DISTANCE)
    description = f"distance_{int(DEFAULT_DISTANCE * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry, DEFAULT_BASIS, multiplicity, charge, description=description
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    one_body, two_body = molecule.get_integrals()
    constant = float(molecule.nuclear_repulsion)
    return constant, one_body, two_body


def _hamiltonian_matrix(
    constant: float, one_body_spin: np.ndarray, two_body_spin: np.ndarray
) -> np.ndarray:
    op = InteractionOperator(constant, one_body_spin, two_body_spin)
    n = one_body_spin.shape[0]
    return get_sparse_operator(op, n_qubits=n).toarray()


def test_df_rank_improves_two_body_reconstruction() -> None:
    rng = np.random.default_rng(0)
    n_orb = 2
    one_body = _random_symmetric(n_orb, rng)
    two_body = _random_two_body(n_orb, rng)
    _, two_body_spin = _spin_orbital_integrals(one_body, two_body)

    model_r1 = df_decompose_from_integrals(one_body, two_body, constant=0.0, rank=1)
    model_r2 = df_decompose_from_integrals(one_body, two_body, constant=0.0, rank=2)

    h2_exact = _hamiltonian_matrix(
        0.0,
        np.zeros((two_body_spin.shape[0], two_body_spin.shape[0])),
        two_body_spin,
    )
    h2_r1 = _df_two_body_matrix(model_r1)
    h2_r2 = _df_two_body_matrix(model_r2)
    err_r1 = np.linalg.norm(h2_exact - h2_r1)
    err_r2 = np.linalg.norm(h2_exact - h2_r2)

    assert err_r2 <= err_r1 + 1e-8, f"err_rank1={err_r1}, err_rank2={err_r2}"


def test_df_interaction_operator_consistency_rank_and_tol() -> None:
    rng = np.random.default_rng(1)
    n_orb = 2
    one_body = _random_symmetric(n_orb, rng)
    two_body = _random_two_body(n_orb, rng)
    constant = float(rng.normal())
    one_body_spin, two_body_spin = _spin_orbital_integrals(one_body, two_body)
    h_exact = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)

    model_r1 = df_decompose_from_integrals(one_body, two_body, constant=constant, rank=1)
    model_r2 = df_decompose_from_integrals(one_body, two_body, constant=constant, rank=2)

    h1 = _one_body_matrix(one_body_spin)
    I = np.eye(h1.shape[0], dtype=np.complex128)
    h_r1 = constant * I + h1 + _df_two_body_matrix(model_r1)
    h_r2 = constant * I + h1 + _df_two_body_matrix(model_r2)
    err_r1 = np.linalg.norm(h_exact - h_r1)
    err_r2 = np.linalg.norm(h_exact - h_r2)
    assert err_r2 <= err_r1 * (1 + 1e-6) + 1e-10, (
        f"err_rank1={err_r1}, err_rank2={err_r2}"
    )

    model_loose = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=None, tol=1e-2
    )
    model_tight = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=None, tol=1e-8
    )
    h_loose = constant * I + h1 + _df_two_body_matrix(model_loose)
    h_tight = constant * I + h1 + _df_two_body_matrix(model_tight)
    err_loose = np.linalg.norm(h_exact - h_loose)
    err_tight = np.linalg.norm(h_exact - h_tight)
    assert (
        err_tight <= err_loose * (1 + 1e-6) + 1e-10
    ), f"err_tol_loose={err_loose}, err_tol_tight={err_tight}"

    model_strict = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=None, tol=1e-12
    )
    h_strict = constant * I + h1 + _df_two_body_matrix(model_strict)
    rel = np.linalg.norm(h_exact - h_strict) / np.linalg.norm(h_exact)
    assert rel < 1e-6, f"rel_error={rel}"


def test_df_zero_two_body_has_small_corrections() -> None:
    rng = np.random.default_rng(2)
    n_orb = 2
    one_body = _random_symmetric(n_orb, rng)
    two_body = np.zeros((n_orb, n_orb, n_orb, n_orb))

    model = df_decompose_from_integrals(one_body, two_body, constant=0.0, tol=0.0)
    two_body_matrix = _df_two_body_matrix(model)
    two_body_norm = np.linalg.norm(two_body_matrix)
    correction_norm = np.linalg.norm(model.one_body_correction)
    constant_corr = abs(model.constant_correction)

    assert two_body_norm < 1e-8, f"two_body_norm={two_body_norm}"
    assert correction_norm < 1e-8, f"one_body_correction_norm={correction_norm}"
    assert constant_corr < 1e-8, f"constant_correction={constant_corr}"


def test_df_h2_openfermion_consistency() -> None:
    constant, one_body, two_body = _h2_integrals_from_openfermion()
    one_body_spin, two_body_spin = _spin_orbital_integrals(one_body, two_body)
    h_exact = _hamiltonian_matrix(constant, one_body_spin, two_body_spin)

    model = df_decompose_from_integrals(
        one_body, two_body, constant=constant, rank=None, tol=1e-12
    )
    h1 = _one_body_matrix(one_body_spin)
    I = np.eye(h1.shape[0], dtype=np.complex128)
    h_df = constant * I + h1 + _df_two_body_matrix(model)
    rel = np.linalg.norm(h_exact - h_df) / np.linalg.norm(h_exact)
    assert rel < 1e-6, f"rel_error={rel}"
