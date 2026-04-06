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
