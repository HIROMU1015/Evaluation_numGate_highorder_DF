import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import df_trotter_energy_plot as plot


def test_df_reference_skips_dense_diag_when_debug_false(monkeypatch):
    def fail_matrix_build(*args, **kwargs):
        raise AssertionError(
            "Dense many-body Hamiltonian construction should not run when debug=False."
        )

    monkeypatch.setattr(plot, "_hamiltonian_matrix", fail_matrix_build)
    monkeypatch.setattr(plot, "_hamiltonian_matrix_from_df_model", fail_matrix_build)
    monkeypatch.setattr(plot, "_hamiltonian_matrix_from_df_tensor", fail_matrix_build)

    times, errors = plot.df_trotter_energy_error_curve(
        t_start=0.25,
        t_end=0.253,
        t_step=0.003,
        molecule_type=2,
        pf_label="2nd",
        estimator="perturbation",
        reference="df",
        debug=False,
        return_costs=False,
    )

    assert len(times) == 2
    assert len(errors) == 2
    assert all(err >= 0.0 for err in errors)


@pytest.mark.slow
def test_df_fci_reference_skips_dense_diag_when_debug_false(monkeypatch):
    def fail_matrix_build(*args, **kwargs):
        raise AssertionError(
            "Dense many-body Hamiltonian construction should not run when debug=False."
        )

    monkeypatch.setattr(plot, "_hamiltonian_matrix", fail_matrix_build)
    monkeypatch.setattr(plot, "_hamiltonian_matrix_from_df_model", fail_matrix_build)
    monkeypatch.setattr(plot, "_hamiltonian_matrix_from_df_tensor", fail_matrix_build)

    times, errors = plot.df_trotter_energy_error_curve(
        t_start=0.25,
        t_end=0.253,
        t_step=0.003,
        molecule_type=2,
        pf_label="2nd",
        estimator="perturbation",
        reference="df_fci",
        debug=False,
        return_costs=False,
    )

    assert len(times) == 2
    assert len(errors) == 2
    assert all(err >= 0.0 for err in errors)


def test_df_fci_reference_falls_back_silently_when_debug_false(monkeypatch):
    sparse_vec = [1.0] + [0.0] * 15
    fci_vec = [0.0, 1.0] + [0.0] * 14

    def fake_sparse_ground_state(_h_sparse):
        return -1.23, sparse_vec

    def fake_df_fci(*args, **kwargs):
        return -0.99, fci_vec

    monkeypatch.setattr(plot, "_effective_df_hamiltonian_sparse", lambda *a, **k: object())
    monkeypatch.setattr(plot, "_ground_state_from_sparse", fake_sparse_ground_state)
    monkeypatch.setattr(plot, "_ground_state_from_df_integrals_fci", fake_df_fci)

    times, errors = plot.df_trotter_energy_error_curve(
        t_start=0.25,
        t_end=0.253,
        t_step=0.003,
        molecule_type=2,
        pf_label="2nd",
        estimator="perturbation",
        reference="df_fci",
        debug=False,
        return_costs=False,
    )

    assert len(times) == 2
    assert len(errors) == 2
    assert all(err >= 0.0 for err in errors)


def test_plot_does_not_request_costs_by_default(monkeypatch):
    captured = {}

    def fake_curve(t_start, t_end, t_step, **kwargs):
        captured["return_costs"] = kwargs.get("return_costs")
        return [0.1], [0.2]

    monkeypatch.setattr(plot, "df_trotter_energy_error_curve", fake_curve)

    result = plot.df_trotter_energy_error_plot(
        t_start=0.1,
        t_end=0.2,
        t_step=0.1,
        molecule_type=2,
        pf_label="2nd",
        estimator="perturbation",
        reference="df",
        fit=False,
        debug=False,
        save_fit_params=False,
        save_rz_layers=False,
    )

    assert captured["return_costs"] is False
    assert result == ([0.1], [0.2])
