from __future__ import annotations

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
        "_load_df_artifact_payload",
        lambda *args, **kwargs: {"rz_layers": {"dummy": 1.0}},
    )

    def _fake_pick(_rz_layers, preferred_key=None):
        captured["preferred_key"] = preferred_key
        return "total_nonclifford_z_coloring_depth", 1700.0

    monkeypatch.setattr(ce, "_pick_df_rz_layer_value", _fake_pick)
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
        "_load_df_artifact_payload",
        lambda *args, **kwargs: {"rz_layers": {"dummy": 1.0}},
    )

    def _fake_pick(_rz_layers, preferred_key=None):
        captured["preferred_key"] = preferred_key
        return "total_ref_rz_depth", 23000.0

    monkeypatch.setattr(ce, "_pick_df_rz_layer_value", _fake_pick)
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    ce.t_depth_extrapolation_compare_gr_df(
        Hchain=3,
        n_w_list=["2nd"],
        rz_layer=True,
        rz_layer_key="total_ref_rz_depth",
    )

    assert captured["preferred_key"] == "total_ref_rz_depth"
