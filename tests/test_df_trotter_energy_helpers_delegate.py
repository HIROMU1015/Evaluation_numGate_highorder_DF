import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import df_trotter_energy_helpers as helpers
from scripts import df_trotter_energy_plot as plot


def test_helper_curve_delegates_to_plot(monkeypatch):
    captured = {}
    sentinel = ([0.1], [0.2])

    def fake_curve(t_start, t_end, t_step, **kwargs):
        captured["t_start"] = t_start
        captured["t_end"] = t_end
        captured["t_step"] = t_step
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(plot, "df_trotter_energy_error_curve", fake_curve)

    result = helpers.df_trotter_energy_error_curve(
        0.1,
        0.2,
        0.01,
        molecule_type=4,
        pf_label="4th",
        reference="df_fci",
        return_costs=False,
    )

    assert result == sentinel
    assert captured["t_start"] == 0.1
    assert captured["t_end"] == 0.2
    assert captured["t_step"] == 0.01
    assert captured["kwargs"]["molecule_type"] == 4
    assert captured["kwargs"]["pf_label"] == "4th"
    assert captured["kwargs"]["reference"] == "df_fci"


def test_helper_plot_delegates_to_plot(monkeypatch):
    captured = {}
    sentinel = ([0.1], [0.2], {"rz_count": 3})

    def fake_plot(t_start, t_end, t_step, **kwargs):
        captured["t_start"] = t_start
        captured["t_end"] = t_end
        captured["t_step"] = t_step
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(plot, "df_trotter_energy_error_plot", fake_plot)

    result = helpers.df_trotter_energy_error_plot(
        0.1,
        0.2,
        0.01,
        molecule_type=4,
        pf_label="4th",
        reference="df_fci",
        save_fit_params=True,
        save_rz_layers=True,
    )

    assert result == sentinel
    assert captured["t_start"] == 0.1
    assert captured["t_end"] == 0.2
    assert captured["t_step"] == 0.01
    assert captured["kwargs"]["molecule_type"] == 4
    assert captured["kwargs"]["pf_label"] == "4th"
    assert captured["kwargs"]["reference"] == "df_fci"
    assert captured["kwargs"]["save_fit_params"] is True
    assert captured["kwargs"]["save_rz_layers"] is True
