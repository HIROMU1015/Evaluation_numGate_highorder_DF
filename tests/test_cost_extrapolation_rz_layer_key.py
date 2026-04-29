from __future__ import annotations

import pytest

from trotterlib.cost_extrapolation import _pick_df_rz_layer_value
from trotterlib import cost_extrapolation as ce


def test_pick_df_rz_layer_value_prefers_total_ref_by_default() -> None:
    rz_layers = {
        "total_nonclifford_z_coloring_depth": 1700,
        "total_nonclifford_z_depth": 1710,
        "total_nonclifford_rz_depth": 1720,
        "total_ref_rz_depth": 23000,
    }

    key, value = _pick_df_rz_layer_value(rz_layers)
    assert key == "total_ref_rz_depth"
    assert value == 23000.0


def test_pick_df_rz_layer_value_respects_preferred_key() -> None:
    rz_layers = {
        "total_nonclifford_z_coloring_depth": 1700,
        "total_ref_rz_depth": 23000,
    }

    key, value = _pick_df_rz_layer_value(
        rz_layers,
        preferred_key="total_nonclifford_z_coloring_depth",
    )
    assert key == "total_nonclifford_z_coloring_depth"
    assert value == 1700.0


def test_pick_df_rz_layer_value_falls_back_when_total_ref_missing() -> None:
    rz_layers = {
        "total_nonclifford_z_coloring_depth": 1700,
        "total_nonclifford_z_depth": 1710,
        "total_nonclifford_rz_depth": 1720,
    }

    key, value = _pick_df_rz_layer_value(rz_layers)
    assert key == "total_nonclifford_z_coloring_depth"
    assert value == 1700.0


def test_normalize_compare_hchain_spec_accepts_range() -> None:
    assert ce._normalize_compare_hchain_spec(range(2, 8)) == [3, 4, 5, 6, 7]


def test_compare_gr_df_prefers_coloring_key_by_default(monkeypatch) -> None:
    captured: dict[str, str | None] = {"preferred_key": None}

    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (1.0, 2.0),
    )
    monkeypatch.setattr(ce, "DECOMPO_NUM", {"H3": {"2nd": 10.0}})
    monkeypatch.setattr(ce, "PF_RZ_LAYER", {"H3": {"2nd": 10.0}})
    monkeypatch.setattr(
        ce,
        "_load_df_compare_alpha_and_exponent_from_latest",
        lambda *args, **kwargs: (1.0, 2.0),
    )

    def _fake_pick(*_args, preferred_key=None, **_kwargs):
        captured["preferred_key"] = preferred_key
        return "total_nonclifford_z_coloring_depth", 1700.0

    monkeypatch.setattr(
        ce,
        "_load_df_compare_rz_layer_value_from_latest",
        _fake_pick,
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    ce.t_depth_extrapolation_compare_gr_df(
        Hchain=3,
        n_w_list=["2nd"],
        rz_layer=True,
    )

    assert captured["preferred_key"] == "total_nonclifford_z_coloring_depth"


def test_compare_gr_df_uses_explicit_rz_layer_key(monkeypatch) -> None:
    captured: dict[str, str | None] = {"preferred_key": None}

    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (1.0, 2.0),
    )
    monkeypatch.setattr(ce, "DECOMPO_NUM", {"H3": {"2nd": 10.0}})
    monkeypatch.setattr(ce, "PF_RZ_LAYER", {"H3": {"2nd": 10.0}})
    monkeypatch.setattr(
        ce,
        "_load_df_compare_alpha_and_exponent_from_latest",
        lambda *args, **kwargs: (1.0, 2.0),
    )

    def _fake_pick(*_args, preferred_key=None, **_kwargs):
        captured["preferred_key"] = preferred_key
        return "total_ref_rz_depth", 23000.0

    monkeypatch.setattr(
        ce,
        "_load_df_compare_rz_layer_value_from_latest",
        _fake_pick,
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    ce.t_depth_extrapolation_compare_gr_df(
        Hchain=3,
        n_w_list=["2nd"],
        rz_layer=True,
        rz_layer_key="total_ref_rz_depth",
    )

    assert captured["preferred_key"] == "total_ref_rz_depth"


def test_load_df_compare_alpha_and_exponent_from_latest_prefers_avg_coeff(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_df_gpu_latest_payload",
        lambda *args, **kwargs: {
            "fit": {
                "avg_coeff": 2.5,
                "coeff": 9.9,
                "exponent": 7.7,
            }
        },
    )

    alpha, expo = ce._load_df_compare_alpha_and_exponent_from_latest(
        "H3_sto-3g_triplet_1+_distance_100_charge_1_grouping",
        "4th",
    )

    assert alpha == 2.5
    assert expo == 4.0


def test_load_df_compare_rz_layer_value_from_latest_recomputes_when_missing(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ce,
        "_load_df_gpu_latest_payload",
        lambda *args, **kwargs: {"info": {"rank": 5, "t_start": 0.75}},
    )

    def _fake_recompute(ham_name, pf_label, *, payload=None):
        captured["ham_name"] = ham_name
        captured["pf_label"] = pf_label
        captured["payload"] = payload
        return {"total_nonclifford_z_coloring_depth": 1234}

    monkeypatch.setattr(
        ce,
        "_compute_df_gpu_latest_rz_layers_from_hamiltonian",
        _fake_recompute,
    )

    key, value = ce._load_df_compare_rz_layer_value_from_latest(
        "H13_sto-3g_triplet_1+_distance_100_charge_1_grouping",
        "8th(Morales)",
        preferred_key="total_nonclifford_z_coloring_depth",
    )

    assert key == "total_nonclifford_z_coloring_depth"
    assert value == 1234.0
    assert captured["pf_label"] == "8th(Morales)"
    assert isinstance(captured["payload"], dict)


def test_error_coefficient_compare_gr_df_uses_df_latest(monkeypatch) -> None:
    def _fake_load_compare_alpha_and_exponent(ham_name, pf_label, *, source, use_original=False):
        del ham_name, pf_label, use_original
        if source == "gr":
            return 10.0, 2.0
        return 1.0, 2.0

    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        _fake_load_compare_alpha_and_exponent,
    )
    monkeypatch.setattr(
        ce,
        "_load_df_compare_alpha_and_exponent_from_latest",
        lambda *args, **kwargs: (2.5, 4.0),
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    out = ce.error_coefficient_compare_gr_df(
        Hchain=3,
        n_w_list=["2nd"],
    )

    assert out["2nd|gr"]["x"] == [6.0]
    assert out["2nd|gr"]["y"] == [10.0]
    assert out["2nd|df"]["x"] == [6.0]
    assert out["2nd|df"]["y"] == [2.5]


def test_qpe_iteration_factor_compare_gr_df_uses_df_latest(monkeypatch) -> None:
    def _fake_load_alpha_from_ave(ham_name, pf_label, *, source, use_original=False):
        del ham_name, pf_label, use_original
        if source == "gr":
            return 10.0
        return 99.0

    monkeypatch.setattr(
        ce,
        "_load_alpha_from_ave",
        _fake_load_alpha_from_ave,
    )
    monkeypatch.setattr(
        ce,
        "_load_df_compare_alpha_and_exponent_from_latest",
        lambda *args, **kwargs: (2.5, 4.0),
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    out = ce.qpe_iteration_factor_compare_gr_df(
        Hchain=3,
        n_w_list=["2nd"],
    )

    expected_gr = ce._qpe_iteration_factor(10.0, float(ce.P_DIR["2nd"]), float(ce.TARGET_ERROR))
    expected_df = ce._qpe_iteration_factor(2.5, float(ce.P_DIR["2nd"]), float(ce.TARGET_ERROR))

    assert out["2nd|gr"]["x"] == [6.0]
    assert out["2nd|df"]["x"] == [6.0]
    assert out["2nd|gr"]["y"] == pytest.approx([expected_gr])
    assert out["2nd|df"]["y"] == pytest.approx([expected_df])


def test_compare_gr_df_extrapolated_returns_100qubit_difference(monkeypatch) -> None:
    monkeypatch.setattr(
        ce,
        "_compare_gr_df_series",
        lambda *args, **kwargs: {
            "2nd|gr": {"x": [6.0, 8.0, 10.0], "y": [60.0, 80.0, 100.0]},
            "2nd|df": {"x": [6.0, 8.0, 10.0], "y": [30.0, 40.0, 50.0]},
        },
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    result = ce.t_depth_extrapolation_compare_gr_df_extrapolated(
        Hchain=5,
        n_w_list=["2nd"],
        rz_layer=True,
        X_MIN_CALC=1.0,
        X_MAX_DISPLAY=100.0,
    )

    at_100 = result["at_100"]["2nd"]
    diff = result["differences_at_100"]["2nd"]

    assert at_100["gr"] == pytest.approx(1000.0)
    assert at_100["df"] == pytest.approx(500.0)
    assert diff["signed_gr_minus_df_at_100"] == pytest.approx(500.0)
    assert diff["abs_diff_at_100"] == pytest.approx(500.0)
