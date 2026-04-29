import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_df_trotter_energy_error_plot_sector_legend_uses_explicit_rank(monkeypatch):
    class FakeAx:
        def __init__(self):
            self.labels = []

        def plot(self, *_args, **kwargs):
            self.labels.append(kwargs.get("label"))

        def legend(self):
            return None

    ax = FakeAx()

    monkeypatch.setattr(
        plot,
        "df_trotter_energy_error_curve",
        lambda *args, **kwargs: ([0.1], [0.2]),
    )
    monkeypatch.setattr(
        plot,
        "get_df_rank_selection_for_molecule",
        lambda molecule_type: {"selected_rank": 11, "full_rank": 36},
    )
    monkeypatch.setattr(plot, "set_loglog_axes", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot.plt, "gca", lambda: ax)
    monkeypatch.setattr(plot.plt, "show", lambda: None)

    plot.df_trotter_energy_error_plot_sector(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=6,
        pf_label="2nd",
        rank=7,
        fit=False,
        debug=False,
        save_fit_params=False,
        save_rz_layers=False,
    )

    assert ax.labels[0] == "error(L=7/36)"


def test_df_trotter_energy_error_plot_sector_legend_uses_explicit_rank_fraction(monkeypatch):
    class FakeAx:
        def __init__(self):
            self.labels = []

        def plot(self, *_args, **kwargs):
            self.labels.append(kwargs.get("label"))

        def legend(self):
            return None

    ax = FakeAx()

    monkeypatch.setattr(
        plot,
        "df_trotter_energy_error_curve",
        lambda *args, **kwargs: ([0.1], [0.2]),
    )
    monkeypatch.setattr(
        plot,
        "get_df_rank_selection_for_molecule",
        lambda molecule_type: {"selected_rank": 11, "full_rank": 36},
    )
    monkeypatch.setattr(plot, "set_loglog_axes", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot.plt, "gca", lambda: ax)
    monkeypatch.setattr(plot.plt, "show", lambda: None)

    plot.df_trotter_energy_error_plot_sector(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=6,
        pf_label="2nd",
        rank_fraction=0.5,
        fit=False,
        debug=False,
        save_fit_params=False,
        save_rz_layers=False,
    )

    assert ax.labels[0] == "error(L=18/36)"


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


def test_resolve_plot_time_range_uses_ground_state_energy_for_default_start(monkeypatch):
    captured = {}

    def fake_ground_state(**kwargs):
        captured.update(kwargs)
        return -4.143732573029025, np.array([1.0 + 0.0j]), {"loaded_from_artifact": True}

    monkeypatch.setattr(plot, "get_df_rank_fraction_for_molecule", lambda molecule_type: 0.36)
    monkeypatch.setattr(plot, "df_ground_state_physical_sector", fake_ground_state)

    t_start, t_end, t_step = plot._resolve_plot_time_range(
        molecule_type=5,
        pf_label="2nd",
        t_start=None,
        t_end=None,
        t_step=None,
        rank=None,
        rank_fraction=None,
        tol=None,
        distance=None,
        basis=None,
        ground_state_solver_tol=1e-10,
        ground_state_solver_maxiter=123,
        load_ground_state_artifact=True,
        save_ground_state_artifact=True,
        debug=False,
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert np.isclose(t_start, 0.36)
    assert np.isclose(t_end, 0.37)
    assert np.isclose(t_step, 0.003)
    assert captured["molecule_type"] == 5
    assert captured["rank"] is None
    assert captured["rank_fraction"] is None
    assert captured["solver"] == "eigsh"
    assert captured["matrix_free"] is True
    assert captured["load_artifact"] is True
    assert captured["save_artifact"] is True
    assert captured["solver_maxiter"] == 123


def test_resolve_plot_time_range_uses_morales_phase_branch(monkeypatch):
    monkeypatch.setattr(plot, "get_df_rank_fraction_for_molecule", lambda molecule_type: 0.36)
    monkeypatch.setattr(
        plot,
        "df_ground_state_physical_sector",
        lambda **_kwargs: (-4.143732573029025, np.array([1.0 + 0.0j]), {}),
    )

    t_start, _t_end, _t_step = plot._resolve_plot_time_range(
        molecule_type=5,
        pf_label="10th(Morales)",
        t_start=None,
        t_end=None,
        t_step=None,
        debug=False,
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert np.isclose(t_start, 1.08)


def test_resolve_plot_time_range_falls_back_to_static_time_when_uncalibrated(monkeypatch):
    monkeypatch.setattr(
        plot,
        "df_ground_state_physical_sector",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("ground-state solver should not be used for static fallback")
        ),
    )

    t_start, t_end, t_step = plot._resolve_plot_time_range(
        molecule_type=2,
        pf_label="2nd",
        t_start=None,
        t_end=None,
        t_step=None,
        debug=False,
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert np.isclose(t_start, 0.73)
    assert np.isclose(t_end, 0.74)
    assert np.isclose(t_step, 0.003)


def test_auto_select_time_range_for_scaling_halves_until_fit_matches(monkeypatch):
    curve_calls: list[float] = []

    def fake_curve(
        t_start,
        t_end,
        t_step,
        **kwargs,
    ):
        curve_calls.append(float(t_start))
        if np.isclose(t_start, 0.25):
            return [0.25, 0.253, 0.256, 0.259], [0.4, 0.41, 0.42, 0.43]
        return [0.125, 0.128, 0.131, 0.134], [1.0e-4, 1.1e-4, 1.2e-4, 1.3e-4]

    fit_calls: list[float] = []

    def fake_loglog_fit(x, y, **kwargs):
        fit_calls.append(float(x[0]))
        if np.isclose(float(x[0]), 0.25):
            return SimpleNamespace(slope=0.4, intercept=0.0, coeff=1.0, r2=0.999)
        return SimpleNamespace(slope=4.05, intercept=0.0, coeff=1.0, r2=0.9999)

    monkeypatch.setattr(plot, "df_trotter_energy_error_curve", fake_curve)
    monkeypatch.setattr(plot, "loglog_fit", fake_loglog_fit)

    tuned = plot._auto_select_time_range_for_scaling(
        t_start=0.25,
        t_end=0.26,
        t_step=0.003,
        molecule_type=6,
        pf_label="4th",
        rank=None,
        rank_fraction=None,
        tol=None,
        distance=None,
        basis=None,
        estimator="perturbation",
        reference="df_sector",
        debug_compare_expectation=True,
        want_costs=False,
        cost_basis_gates=None,
        cost_decompose_reps=8,
        cost_optimization_level=0,
        trace_u_debug=False,
        ccsd_target_error_ha=None,
        ccsd_thresh_range=None,
        ccsd_use_kernel=False,
        ccsd_no_triples=False,
        df_sector_solver="eigsh",
        df_sector_solver_tol=1e-10,
        df_sector_solver_maxiter=None,
        df_sector_davidson_eps=1e-8,
        df_sector_davidson_max_subspace=80,
        df_sector_davidson_max_iterations=200,
        df_sector_matrix_free=True,
        df_sector_ground_state_cache=True,
        debug=False,
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert tuned is not None
    tuned_t_start, tuned_t_end, tuned_t_step, times, errors, costs, info = tuned
    assert np.isclose(tuned_t_start, 0.125)
    assert np.isclose(tuned_t_end, 0.135)
    assert np.isclose(tuned_t_step, 0.003)
    assert costs is None
    assert times == [0.125, 0.128, 0.131, 0.134]
    assert errors == [1.0e-4, 1.1e-4, 1.2e-4, 1.3e-4]
    assert curve_calls == [0.25, 0.125]
    assert fit_calls == [0.25, 0.125]
    assert np.isclose(float(info["selected_t_start"]), 0.125)


def test_auto_select_time_range_for_scaling_skips_second_order(monkeypatch):
    monkeypatch.setattr(
        plot,
        "df_trotter_energy_error_curve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("curve search should not run for second-order formulas")
        ),
    )

    tuned = plot._auto_select_time_range_for_scaling(
        t_start=0.36,
        t_end=0.37,
        t_step=0.003,
        molecule_type=5,
        pf_label="2nd",
        rank=None,
        rank_fraction=None,
        tol=None,
        distance=None,
        basis=None,
        estimator="perturbation",
        reference="df_sector",
        debug_compare_expectation=True,
        want_costs=False,
        cost_basis_gates=None,
        cost_decompose_reps=8,
        cost_optimization_level=0,
        trace_u_debug=False,
        ccsd_target_error_ha=None,
        ccsd_thresh_range=None,
        ccsd_use_kernel=False,
        ccsd_no_triples=False,
        df_sector_solver="eigsh",
        df_sector_solver_tol=1e-10,
        df_sector_solver_maxiter=None,
        df_sector_davidson_eps=1e-8,
        df_sector_davidson_max_subspace=80,
        df_sector_davidson_max_iterations=200,
        df_sector_matrix_free=True,
        df_sector_ground_state_cache=True,
        debug=False,
        debug_print=lambda *_args, **_kwargs: None,
    )

    assert tuned is None


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


def test_df_ground_state_physical_sector_appends_config_target_to_artifact_name(
    monkeypatch, tmp_path
):
    class FakeMol:
        nelec = (1, 0)

    class FakeModel:
        N = 4

        def hermitize(self):
            return self

    target_error_ha = 1.5936001019904e-4

    monkeypatch.setattr(plot, "PICKLE_DIR_DF_GROUND_STATE_PATH", tmp_path)
    monkeypatch.setattr(plot, "_build_pyscf_molecule", lambda *args, **kwargs: FakeMol())
    monkeypatch.setattr(plot, "get_df_rank_fraction_for_molecule", lambda molecule_type: 0.25)
    monkeypatch.setattr(
        plot,
        "get_df_rank_selection_for_molecule",
        lambda molecule_type: {
            "rank_fraction": 0.25,
            "selected_rank": 1,
            "full_rank": 4,
            "rank_ratio": "1/4",
            "target_error_ha": target_error_ha,
        },
    )
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

    energy, _state, info = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
        save_artifact=True,
    )

    expected_suffix = f"ccsd_target_{plot._artifact_target_error_token(target_error_ha)}ha"
    assert np.isclose(energy, 1.0)
    assert expected_suffix in info["artifact_name"]
    assert (tmp_path / info["artifact_name"]).exists()


def test_df_ground_state_physical_sector_ignores_mismatched_cache(monkeypatch):
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
        "_load_df_ground_state_artifact",
        lambda _name: (
            -9.0,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            {"hamiltonian_fingerprint": "stale"},
        ),
    )

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

    energy, state, info = plot.df_ground_state_physical_sector(
        molecule_type=2,
        matrix_free=True,
        load_artifact=True,
        save_artifact=False,
    )

    assert np.isclose(energy, 1.0)
    assert np.flatnonzero(np.abs(state) > 1e-12).tolist() == [6]
    assert info.get("loaded_from_artifact") is not True


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


def test_h_chain_integrals_uses_session_cache(monkeypatch):
    class FakeMolecule:
        def __init__(self, idx):
            self.nuclear_repulsion = 1.0
            self._idx = float(idx)

        def get_integrals(self):
            one_body = np.array([[self._idx]], dtype=float)
            two_body = np.array([[[[self._idx]]]], dtype=float)
            return one_body, two_body

    call_counter = {"count": 0}

    monkeypatch.setattr(plot, "geo", lambda *args, **kwargs: ([("H", (0.0, 0.0, 0.0))], 1, 1))
    monkeypatch.setattr(plot, "MolecularData", lambda *args, **kwargs: object())

    def fake_run_pyscf(_molecule, run_scf, run_fci):
        call_counter["count"] += 1
        return FakeMolecule(call_counter["count"])

    monkeypatch.setattr(plot, "run_pyscf", fake_run_pyscf)

    plot.clear_df_integral_session_cache()
    first = plot._h_chain_integrals(6, distance=None, basis=None)
    second = plot._h_chain_integrals(6, distance=None, basis=None)
    plot.clear_df_integral_session_cache()
    third = plot._h_chain_integrals(6, distance=None, basis=None)

    assert call_counter["count"] == 2
    assert np.allclose(first[1], second[1])
    assert np.allclose(first[2], second[2])
    assert not np.allclose(first[1], third[1])
    assert not np.allclose(first[2], third[2])


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
