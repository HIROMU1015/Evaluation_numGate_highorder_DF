import sys
from pathlib import Path

import numpy as np
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import df_trotter_energy_plot as plot


def test_ground_state_from_sparse_physical_sector_restricts_basis(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())

    diag = np.arange(16, dtype=float)
    h_sparse = scipy.sparse.diags(diag, format="csc")

    energy, state, sector = plot._ground_state_from_sparse_physical_sector(
        h_sparse,
        molecule_type=2,
        distance=None,
        basis=None,
        n_qubits=4,
    )

    support = np.flatnonzero(np.abs(state) > 1e-12)
    select_indices = np.asarray(sector["select_indices"], dtype=int)
    expected_index = int(select_indices[np.argmin(diag[select_indices])])

    assert sector["nelec_alpha"] == 1
    assert sector["nelec_beta"] == 0
    assert support.tolist() == [expected_index]
    assert np.isclose(energy, diag[expected_index])


def test_df_trotter_energy_error_plot_sector_wraps_reference(monkeypatch):
    captured = {}

    def fake_plot(*args, **kwargs):
        captured["reference"] = kwargs.get("reference")
        captured["artifact_suffix"] = kwargs.get("artifact_suffix")
        return [0.1], [0.2]

    monkeypatch.setattr(plot, "df_trotter_energy_error_plot", fake_plot)

    result = plot.df_trotter_energy_error_plot_sector(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=2,
        pf_label="2nd",
        fit=False,
        debug=False,
        save_fit_params=False,
        save_rz_layers=False,
    )

    assert captured["reference"] == "df_sector"
    assert captured["artifact_suffix"] == "_df_sector"
    assert result == ([0.1], [0.2])


def test_ground_state_from_sparse_physical_sector_supports_davidson(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())

    diag = np.array([3.0, 2.0, 1.0, 0.0] * 4, dtype=float)
    h_sparse = scipy.sparse.diags(diag, format="csc")

    energy, state, sector = plot._ground_state_from_sparse_physical_sector(
        h_sparse,
        molecule_type=2,
        distance=None,
        basis=None,
        n_qubits=4,
        solver="davidson",
        davidson_max_subspace=10,
        davidson_max_iterations=80,
    )

    support = np.flatnonzero(np.abs(state) > 1e-9)
    select_indices = np.asarray(sector["select_indices"], dtype=int)
    expected_index = int(select_indices[np.argmin(diag[select_indices])])

    assert support.tolist() == [expected_index]
    assert np.isclose(energy, diag[expected_index], atol=1e-6)


def test_df_trotter_energy_error_curve_sector_passes_solver(monkeypatch):
    captured = {}

    def fake_curve(*args, **kwargs):
        captured["reference"] = kwargs.get("reference")
        captured["df_sector_solver"] = kwargs.get("df_sector_solver")
        return [0.1], [0.2]

    monkeypatch.setattr(plot, "df_trotter_energy_error_curve", fake_curve)

    result = plot.df_trotter_energy_error_curve_sector(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=2,
        pf_label="2nd",
        df_sector_solver="davidson",
    )

    assert captured["reference"] == "df_sector"
    assert captured["df_sector_solver"] == "davidson"
    assert result == ([0.1], [0.2])


def test_df_ground_state_physical_sector_uses_restricted_builder(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )
    monkeypatch.setattr(plot, "df_decompose_from_integrals", lambda *a, **k: FakeModel())
    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sparse",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("full sparse builder should not be used")
        ),
    )

    captured = {}

    def fake_sector_builder(
        constant,
        one_body_spin,
        model,
        *,
        nelec_alpha,
        nelec_beta,
        timings=None,
    ):
        captured["nelec_alpha"] = nelec_alpha
        captured["nelec_beta"] = nelec_beta
        if timings is not None:
            timings["build_restricted_sparse_s"] = 0.0
        return scipy.sparse.diags([5.0, 1.0], format="csc"), np.array([3, 6]), np.array(
            [True, False, False, False]
        )

    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sector_sparse",
        fake_sector_builder,
    )

    energy, state, info = plot.df_ground_state_physical_sector(molecule_type=2)

    assert np.isclose(energy, 1.0)
    assert np.flatnonzero(np.abs(state) > 1e-12).tolist() == [6]
    assert captured["nelec_alpha"] == 1
    assert captured["nelec_beta"] == 0
    assert info["restricted_dim"] == 2


def test_df_ground_state_physical_sector_returns_profile(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )
    monkeypatch.setattr(plot, "df_decompose_from_integrals", lambda *a, **k: FakeModel())
    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sector_sparse",
        lambda *a, **k: (
            scipy.sparse.diags([5.0, 1.0], format="csc"),
            np.array([3, 6]),
            np.array([True, False, False, False]),
        ),
    )

    energy, state, info = plot.df_ground_state_physical_sector(
        molecule_type=2,
        profile=True,
    )

    assert np.isclose(energy, 1.0)
    assert np.flatnonzero(np.abs(state) > 1e-12).tolist() == [6]
    assert "profile" in info
    assert "build_integrals_s" in info["profile"]
    assert "solve_ground_state_total_s" in info["profile"]
    assert "total_s" in info["profile"]


def test_matrix_free_df_sector_matches_restricted_sparse():
    one_body_spin = np.array(
        [
            [0.2, 0.0, 0.05, 0.0],
            [0.0, -0.1, 0.0, 0.04],
            [0.05, 0.0, 0.3, 0.0],
            [0.0, 0.04, 0.0, 0.15],
        ],
        dtype=np.complex128,
    )
    g_block = np.array(
        [
            [0.0, 0.0, 0.12, 0.0],
            [0.0, 0.0, 0.0, -0.08],
            [0.12, 0.0, 0.0, 0.0],
            [0.0, -0.08, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    model = plot.DFModel(
        lambdas=np.array([0.7], dtype=float),
        G_list=[g_block],
        one_body_correction=np.zeros((4, 4), dtype=np.complex128),
        constant_correction=0.05,
        N=4,
    )

    restricted_sparse, basis_indices, _ = plot._effective_df_hamiltonian_sector_sparse(
        0.1,
        one_body_spin,
        model,
        nelec_alpha=1,
        nelec_beta=1,
    )
    linear_operator, basis_indices_mf, _ = plot._build_df_sector_matrix_free_operator(
        0.1,
        one_body_spin,
        model,
        nelec_alpha=1,
        nelec_beta=1,
    )

    assert np.array_equal(basis_indices, basis_indices_mf)

    vec = np.array([1.0, -0.3, 0.2, 0.7], dtype=np.complex128)
    vec = vec / np.linalg.norm(vec)
    out_sparse = restricted_sparse @ vec
    out_mf = linear_operator @ vec
    assert np.allclose(out_sparse, out_mf, atol=1e-10)

    energy_sparse, state_sparse, _ = plot._ground_state_from_restricted_sparse(
        restricted_sparse,
        basis_indices=basis_indices,
        full_dim=16,
    )
    energy_mf, state_mf, _ = plot._ground_state_from_matrix_free_operator(
        linear_operator,
        basis_indices=basis_indices_mf,
        full_dim=16,
    )

    assert np.isclose(energy_sparse, energy_mf, atol=1e-10)
    overlap = abs(np.vdot(state_sparse, state_mf))
    assert np.isclose(overlap, 1.0, atol=1e-8)


def test_df_ground_state_physical_sector_matrix_free_compares_with_sparse(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )
    monkeypatch.setattr(plot, "df_decompose_from_integrals", lambda *a, **k: FakeModel())

    linop = plot.LinearOperator(
        (2, 2),
        matvec=lambda x: np.array([2.0 * x[0], 1.0 * x[1]], dtype=np.complex128),
        dtype=np.complex128,
    )

    monkeypatch.setattr(
        plot,
        "_build_df_sector_matrix_free_operator",
        lambda *a, **k: (linop, np.array([3, 6]), {"matrix_free": True, "restricted_dim": 2}),
    )
    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sector_sparse",
        lambda *a, **k: (
            scipy.sparse.diags([2.0, 1.0], format="csc"),
            np.array([3, 6]),
            np.array([True, False, False, False]),
        ),
    )

    energy, state, info = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
        compare_with_sparse_eigsh=True,
    )

    assert np.isclose(energy, 1.0)
    assert np.flatnonzero(np.abs(state) > 1e-12).tolist() == [6]
    assert "comparison" in info
    assert np.isclose(info["comparison"]["sparse_eigsh_energy"], 1.0)
    assert np.isclose(info["comparison"]["energy_delta"], 0.0)
    assert np.isclose(info["comparison"]["state_overlap_abs"], 1.0)


def test_df_ground_state_physical_sector_matrix_free_uses_linear_operator(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )
    monkeypatch.setattr(plot, "df_decompose_from_integrals", lambda *a, **k: FakeModel())
    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sector_sparse",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("restricted sparse builder should not be used")
        ),
    )

    def fake_build_matrix_free(constant, one_body_spin, model, *, nelec_alpha, nelec_beta, timings=None):
        linop = plot.LinearOperator(
            (2, 2),
            matvec=lambda x: np.array([2.0 * x[0], 1.0 * x[1]], dtype=np.complex128),
            dtype=np.complex128,
        )
        return linop, np.array([3, 6]), {"matrix_free": True, "restricted_dim": 2}

    monkeypatch.setattr(plot, "_build_df_sector_matrix_free_operator", fake_build_matrix_free)

    energy, state, info = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
    )

    assert np.isclose(energy, 1.0)
    assert np.flatnonzero(np.abs(state) > 1e-12).tolist() == [6]
    assert info["matrix_free"] is True


def test_df_ground_state_physical_sector_saves_and_loads_artifact(monkeypatch, tmp_path):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    monkeypatch.setattr(plot, "PICKLE_DIR_DF_GROUND_STATE_PATH", tmp_path)
    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )
    monkeypatch.setattr(plot, "df_decompose_from_integrals", lambda *a, **k: FakeModel())

    def fake_build_matrix_free(constant, one_body_spin, model, *, nelec_alpha, nelec_beta, timings=None):
        linop = plot.LinearOperator(
            (2, 2),
            matvec=lambda x: np.array([2.0 * x[0], 1.0 * x[1]], dtype=np.complex128),
            dtype=np.complex128,
        )
        return linop, np.array([3, 6]), {"matrix_free": True, "restricted_dim": 2}

    monkeypatch.setattr(plot, "_build_df_sector_matrix_free_operator", fake_build_matrix_free)

    energy_saved, state_saved, info_saved = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
        rank_fraction=0.5,
        save_artifact=True,
    )

    assert np.isclose(energy_saved, 1.0)
    assert info_saved["artifact_name"]
    artifact_path = tmp_path / info_saved["artifact_name"]
    assert artifact_path.exists()
    with artifact_path.open("rb") as f:
        artifact_data = plot.pickle.load(f)
    assert "state" not in artifact_data
    assert np.array_equal(artifact_data["basis_indices"], np.array([3, 6], dtype=np.uint32))
    assert np.allclose(artifact_data["restricted_state"], np.array([0.0, 1.0], dtype=np.complex128))

    monkeypatch.setattr(
        plot,
        "_build_df_sector_matrix_free_operator",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache should avoid recompute")),
    )

    energy_loaded, state_loaded, info_loaded = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
        rank_fraction=0.5,
        load_artifact=True,
    )

    assert np.isclose(energy_loaded, energy_saved)
    assert np.allclose(state_loaded, state_saved)
    assert info_loaded["artifact_name"] == info_saved["artifact_name"]
    assert info_loaded["loaded_from_artifact"] is True


def test_df_ground_state_physical_sector_uses_config_rank_fraction(monkeypatch):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    captured = {}

    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(plot, "get_df_rank_fraction_for_molecule", lambda molecule_type: 0.25)
    monkeypatch.setattr(
        plot,
        "_h_chain_integrals",
        lambda *args, **kwargs: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(plot, "_symmetrize_two_body", lambda two_body: two_body)
    monkeypatch.setattr(
        plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (np.zeros((4, 4), dtype=float), None),
    )

    def fake_decompose(one_body, two_body, *, constant, rank=None, tol=None):
        captured["rank"] = rank
        captured["tol"] = tol
        return FakeModel()

    monkeypatch.setattr(plot, "df_decompose_from_integrals", fake_decompose)
    monkeypatch.setattr(
        plot,
        "_effective_df_hamiltonian_sector_sparse",
        lambda *a, **k: (
            scipy.sparse.diags([5.0, 1.0], format="csc"),
            np.array([3, 6]),
            np.array([True, False, False, False]),
        ),
    )

    energy, state, info = plot.df_ground_state_physical_sector(molecule_type=2)

    assert np.isclose(energy, 1.0)
    assert captured["rank"] == 1
    assert np.isclose(info["rank_fraction"], 0.25)


def test_df_trotter_energy_error_curve_sector_passes_ground_state_cache(monkeypatch):
    captured = {}

    def fake_curve(*args, **kwargs):
        captured["reference"] = kwargs.get("reference")
        captured["df_sector_ground_state_cache"] = kwargs.get("df_sector_ground_state_cache")
        return [0.1], [0.2]

    monkeypatch.setattr(plot, "df_trotter_energy_error_curve", fake_curve)

    result = plot.df_trotter_energy_error_curve_sector(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=2,
        pf_label="2nd",
        df_sector_ground_state_cache=False,
    )

    assert captured["reference"] == "df_sector"
    assert captured["df_sector_ground_state_cache"] is False
    assert result == ([0.1], [0.2])


def test_rewrite_df_ground_state_artifacts_compact_rewrites_legacy_format(monkeypatch, tmp_path):
    monkeypatch.setattr(plot, "PICKLE_DIR_DF_GROUND_STATE_PATH", tmp_path)
    full_state = np.zeros(8, dtype=np.complex128)
    full_state[[1, 4]] = np.array([0.6, 0.8], dtype=np.complex128)
    legacy_path = tmp_path / "legacy_ground_state"
    with legacy_path.open("wb") as f:
        plot.pickle.dump(
            {
                "energy": -1.25,
                "state": full_state,
                "info": {"num_qubits": 3},
            },
            f,
        )

    results = plot.rewrite_df_ground_state_artifacts_compact(
        file_names=["legacy_ground_state"],
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert len(results) == 1
    result = results[0]
    assert result["rewritten"] is True
    assert result["already_compact"] is False

    with legacy_path.open("rb") as f:
        compact = plot.pickle.load(f)

    assert "state" not in compact
    assert np.array_equal(compact["basis_indices"], np.array([1, 4], dtype=np.uint32))
    assert np.allclose(compact["restricted_state"], np.array([0.6, 0.8], dtype=np.complex128))

    energy, state, info = plot._load_df_ground_state_artifact("legacy_ground_state")
    assert np.isclose(energy, -1.25)
    assert np.allclose(state, full_state)
    assert info["num_qubits"] == 3
