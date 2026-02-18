from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts import df_trotter_energy_plot as df_plot  # noqa: E402


def test_df_trotter_ud_rz_layer_counts_splits_u_and_d(monkeypatch) -> None:
    monkeypatch.setattr(
        df_plot,
        "_h_chain_integrals",
        lambda molecule_type, *, distance, basis: (
            0.0,
            np.zeros((1, 1), dtype=float),
            np.zeros((1, 1, 1, 1), dtype=float),
        ),
    )
    monkeypatch.setattr(
        df_plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )

    class _FakeModel:
        one_body_correction = np.zeros((2, 2), dtype=float)
        constant_correction = 0.0
        N = 2

        def hermitize(self) -> "_FakeModel":
            return self

    monkeypatch.setattr(
        df_plot,
        "df_decompose_from_integrals",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        df_plot,
        "_compute_df_rz_costs",
        lambda **kwargs: {
            "rz_total_ud": {
                "u_rz_count": 12,
                "u_rz_depth": 7,
                "d_rz_count": 9,
                "d_rz_depth": 5,
                "total_rz_count": 21,
                "total_rz_depth": 12,
            },
            "nonclifford_total": {
                "u_nonclifford_rz_count": 4,
                "u_nonclifford_rz_depth": 3,
                "d_nonclifford_rz_count": 6,
                "d_nonclifford_rz_depth": 4,
                "total_nonclifford_rz_count": 10,
                "total_nonclifford_rz_depth": 7,
            },
            "u_costs": [
                {
                    "label": "one_body",
                    "u_ref_rz_count": 3,
                    "u_ref_rz_depth": 2,
                    "u_nonclifford_rz_count": 1,
                    "u_nonclifford_rz_depth": 1,
                }
            ],
            "d_block_costs": [
                {
                    "label": "one_body_D",
                    "rz_count": 5,
                    "rz_depth": 3,
                    "nonclifford_rz_count": 4,
                    "nonclifford_rz_depth": 2,
                }
            ],
        },
    )

    metrics = df_plot.df_trotter_ud_rz_layer_counts(
        molecule_type=2,
        pf_label="2nd",
        time=0.1,
        debug=False,
    )

    assert metrics["u_rz_depth"] == 7
    assert metrics["d_rz_depth"] == 5
    assert metrics["total_rz_depth"] == 12
    assert metrics["u_nonclifford_rz_depth"] == 3
    assert metrics["d_nonclifford_rz_depth"] == 4
    assert metrics["total_nonclifford_rz_depth"] == 7

    u_block_costs = metrics.get("u_block_costs")
    assert isinstance(u_block_costs, list)
    assert u_block_costs[0]["label"] == "one_body"
    assert u_block_costs[0]["rz_depth"] == 2

    d_block_costs = metrics.get("d_block_costs")
    assert isinstance(d_block_costs, list)
    assert d_block_costs[0]["label"] == "one_body_D"
    assert d_block_costs[0]["rz_depth"] == 3


def test_df_trotter_ud_rz_layer_counts_uses_config_rank_fraction(monkeypatch) -> None:
    monkeypatch.setattr(
        df_plot,
        "get_df_rank_fraction_for_molecule",
        lambda molecule_type: 0.25,
    )
    monkeypatch.setattr(
        df_plot,
        "_h_chain_integrals",
        lambda molecule_type, *, distance, basis: (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(
        df_plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )

    class _FakeModel:
        one_body_correction = np.zeros((2, 2), dtype=float)
        constant_correction = 0.0
        N = 2

        def hermitize(self) -> "_FakeModel":
            return self

    captured: dict[str, int | None] = {"rank": None}

    def _fake_df_decompose(*args, **kwargs):
        captured["rank"] = kwargs.get("rank")
        return _FakeModel()

    monkeypatch.setattr(df_plot, "df_decompose_from_integrals", _fake_df_decompose)
    monkeypatch.setattr(
        df_plot,
        "_compute_df_rz_costs",
        lambda **kwargs: {"rz_total_ud": {"total_rz_depth": 1}},
    )

    _ = df_plot.df_trotter_ud_rz_layer_counts(
        molecule_type=2,
        pf_label="2nd",
        time=0.1,
        rank_fraction=None,
        debug=False,
    )

    # one_body is (2, 2) -> full_rank=4, config rank_fraction=0.25 -> rank=1
    assert captured["rank"] == 1


def test_df_trotter_ud_rz_layer_counts_uses_config_time_when_none(monkeypatch) -> None:
    monkeypatch.setattr(
        df_plot,
        "get_default_df_time_for_molecule_pf",
        lambda molecule_type, pf_label: 0.37,
    )
    monkeypatch.setattr(
        df_plot,
        "_h_chain_integrals",
        lambda molecule_type, *, distance, basis: (
            0.0,
            np.zeros((1, 1), dtype=float),
            np.zeros((1, 1, 1, 1), dtype=float),
        ),
    )
    monkeypatch.setattr(
        df_plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )

    class _FakeModel:
        one_body_correction = np.zeros((2, 2), dtype=float)
        constant_correction = 0.0
        N = 2

        def hermitize(self) -> "_FakeModel":
            return self

    monkeypatch.setattr(
        df_plot,
        "df_decompose_from_integrals",
        lambda *args, **kwargs: _FakeModel(),
    )

    captured: dict[str, float | None] = {"t_cost": None}

    def _fake_compute_df_rz_costs(**kwargs):
        captured["t_cost"] = float(kwargs["t_cost"])
        return {"rz_total_ud": {"total_rz_depth": 1}}

    monkeypatch.setattr(df_plot, "_compute_df_rz_costs", _fake_compute_df_rz_costs)

    _ = df_plot.df_trotter_ud_rz_layer_counts(
        molecule_type=4,
        pf_label="2nd",
        time=None,
        debug=False,
    )

    assert captured["t_cost"] == 0.37


def test_df_trotter_energy_error_plot_uses_config_time_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        df_plot,
        "get_default_df_time_for_molecule_pf",
        lambda molecule_type, pf_label: 1.12,
    )

    captured: dict[str, float | None] = {"t_start": None, "t_end": None, "t_step": None}

    def _fake_curve(t_start, t_end, t_step, **kwargs):
        captured["t_start"] = float(t_start)
        captured["t_end"] = float(t_end)
        captured["t_step"] = float(t_step)
        return [float(t_start)], [1.0e-3], {}

    monkeypatch.setattr(df_plot, "df_trotter_energy_error_curve", _fake_curve)
    monkeypatch.setattr(df_plot.plt, "show", lambda: None)

    _ = df_plot.df_trotter_energy_error_plot(
        t_start=None,
        t_end=None,
        t_step=None,
        molecule_type=4,
        pf_label="2nd",
        fit=False,
        debug=False,
        save_fit_params=False,
    )

    assert captured["t_start"] == 1.12
    assert captured["t_step"] == 0.003
    assert captured["t_end"] == captured["t_start"] + 0.01


def test_df_trotter_energy_error_plot_saves_fixed_p_alpha_ave(monkeypatch) -> None:
    monkeypatch.setattr(
        df_plot,
        "df_trotter_energy_error_curve",
        lambda *args, **kwargs: ([1.0, 2.0, 4.0], [3.0, 12.0, 48.0], {}),
    )
    monkeypatch.setattr(
        df_plot,
        "_artifact_ham_name",
        lambda molecule_type, *, distance, basis: f"H{int(molecule_type)}_mock",
    )
    monkeypatch.setattr(df_plot.plt, "show", lambda: None)

    saved: dict[str, object] = {}

    def _fake_save(file_name: str, data: dict[str, object] | float) -> None:
        saved[file_name] = data

    monkeypatch.setattr(df_plot, "_save_df_plot_artifact", _fake_save)

    _ = df_plot.df_trotter_energy_error_plot(
        t_start=1.0,
        t_end=5.0,
        t_step=1.0,
        molecule_type=2,
        pf_label="2nd",
        fit=False,
        debug=False,
        save_fit_params=True,
        save_rz_layers=False,
    )

    artifact_name = "H2_mock_Operator_2nd"
    assert artifact_name in saved
    assert f"{artifact_name}_ave" in saved

    main_payload = saved[artifact_name]
    assert isinstance(main_payload, dict)
    assert main_payload.get("fixed_expo") == 2.0
    assert np.isclose(float(main_payload.get("avg_coeff", 0.0)), 3.0)

    ave_payload = saved[f"{artifact_name}_ave"]
    assert isinstance(ave_payload, float)
    assert np.isclose(float(ave_payload), 3.0)


def test_df_trotter_ud_rz_layer_counts_can_save_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(df_plot, "PICKLE_DIR_DF_RZ_LAYER_PATH", tmp_path)
    monkeypatch.setattr(
        df_plot,
        "_artifact_ham_name",
        lambda molecule_type, *, distance, basis: f"H{int(molecule_type)}_mock",
    )
    monkeypatch.setattr(
        df_plot,
        "_h_chain_integrals",
        lambda molecule_type, *, distance, basis: (
            0.0,
            np.zeros((1, 1), dtype=float),
            np.zeros((1, 1, 1, 1), dtype=float),
        ),
    )
    monkeypatch.setattr(
        df_plot,
        "spinorb_from_spatial",
        lambda one_body, two_body: (
            np.zeros((2, 2), dtype=float),
            np.zeros((2, 2, 2, 2), dtype=float),
        ),
    )

    class _FakeModel:
        one_body_correction = np.zeros((2, 2), dtype=float)
        constant_correction = 0.0
        N = 2

        def hermitize(self) -> "_FakeModel":
            return self

    monkeypatch.setattr(
        df_plot,
        "df_decompose_from_integrals",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        df_plot,
        "_compute_df_rz_costs",
        lambda **kwargs: {
            "rz_total_ud": {"u_rz_depth": 2, "d_rz_depth": 3, "total_rz_depth": 5}
        },
    )

    metrics = df_plot.df_trotter_ud_rz_layer_counts(
        molecule_type=2,
        pf_label="2nd",
        time=0.1,
        debug=False,
        save_rz_layers=True,
    )
    artifact_name = str(metrics["artifact_name"])
    saved_path = tmp_path / artifact_name
    assert saved_path.exists()
    with saved_path.open("rb") as f:
        saved = pickle.load(f)
    assert saved["u_rz_depth"] == 2
    assert saved["d_rz_depth"] == 3
    assert saved["num_qubits"] == 2


def test_plot_df_u_rz_depth_vs_num_qubits_reads_saved_layers(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(df_plot, "PICKLE_DIR_DF_RZ_LAYER_PATH", tmp_path)
    monkeypatch.setattr(
        df_plot,
        "_artifact_ham_name",
        lambda molecule_type, *, distance, basis: f"H{int(molecule_type)}_mock",
    )

    data_h2 = {
        "num_qubits": 4,
        "u_rz_depth": 10,
        "d_rz_depth": 8,
        "total_rz_depth": 20,
    }
    data_h3 = {
        "num_qubits": 6,
        "u_rz_depth": 30,
        "d_rz_depth": 24,
        "total_rz_depth": 60,
    }
    with (tmp_path / "H2_mock_Operator_2nd").open("wb") as f:
        pickle.dump(data_h2, f)
    with (tmp_path / "H3_mock_Operator_2nd").open("wb") as f:
        pickle.dump(data_h3, f)

    out = df_plot.plot_df_u_rz_depth_vs_num_qubits(
        molecule_types=[3, 2],
        pf_label="2nd",
        show=False,
        include_pf_rz_layer=True,
    )
    assert out["num_qubits"] == [4.0, 6.0]
    assert out["u_rz_depth"] == [10.0, 30.0]
    assert out["d_rz_depth"] == [8.0, 24.0]
    assert out["total_rz_depth"] == [20.0, 60.0]
    assert len(out["pf_rz_depth"]) == 2
