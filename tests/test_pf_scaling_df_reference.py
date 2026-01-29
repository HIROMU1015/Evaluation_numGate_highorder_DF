import sys
from pathlib import Path

import numpy as np
import pytest

from trotterlib.analysis_utils import loglog_fit
from trotterlib.df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from trotterlib.df_trotter.decompose import df_decompose_from_integrals

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.df_trotter_energy_plot import (  # noqa: E402
    _h_chain_integrals,
    _symmetrize_two_body,
    df_trotter_energy_error_curve,
)


@pytest.mark.slow
def test_pf_scaling_df_reference_orders() -> None:
    settings = [
        ("2nd", 2.0),
        ("4th", 4.0),
        ("8th(Morales)", 8.0),
    ]
    for pf_label, expected in settings:
        times, errors = df_trotter_energy_error_curve(
            t_start=0.25,
            t_end=0.26,
            t_step=0.003,
            molecule_type=6,
            pf_label=pf_label,
            rank_fraction=1,
            estimator="perturbation",
            reference="df",
            debug=False,
        )
        fit = loglog_fit(times, errors, compute_r2=True)
        assert abs(fit.slope - expected) < 0.8


def test_df_model_hermitize_reduces_nonherm() -> None:
    constant, one_body, two_body = _h_chain_integrals(
        2, distance=None, basis=None
    )
    two_body = _symmetrize_two_body(two_body)
    model = df_decompose_from_integrals(one_body, two_body, constant=constant)
    herm_model = model.hermitize()

    max_g_nonherm = max(
        np.linalg.norm(g - g.conj().T) for g in herm_model.G_list
    )
    assert max_g_nonherm < 1e-10
    one_body_nonherm = np.linalg.norm(
        herm_model.one_body_correction - herm_model.one_body_correction.conj().T
    )
    assert one_body_nonherm < 1e-10


def test_global_phase_applied_in_simulation() -> None:
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    shift = 0.7
    time = 0.3
    qc = build_df_trotter_circuit(
        blocks=[],
        time=time,
        num_qubits=1,
        pf_label="2nd",
        energy_shift=shift,
    )
    psi_t = simulate_statevector(qc, psi0)
    expected = np.exp(-1j * shift * time) * psi0
    assert np.allclose(psi_t, expected, atol=1e-10)
