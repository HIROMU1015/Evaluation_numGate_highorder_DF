from __future__ import annotations

import numpy as np
from openfermion.ops import QubitOperator

from trotterlib import grouping_rz_layers as gr_layers


class _FakeMF:
    def __init__(self, n_orb: int) -> None:
        self.mo_coeff = np.zeros((n_orb, n_orb), dtype=float)


def test_estimate_rz_layers_from_grouping_supports_h_label(monkeypatch) -> None:
    monkeypatch.setattr(
        gr_layers,
        "_run_scf_and_integrals",
        lambda molecule_type: (
            None,
            _FakeMF(3),
            0.0,
            np.zeros((3, 3), dtype=float),
            np.zeros((3, 3, 3, 3), dtype=float),
        ),
    )
    monkeypatch.setattr(
        gr_layers,
        "_build_grouped_qubit_ops",
        lambda **kwargs: [
            QubitOperator("X0 Z1", 1.0) + QubitOperator("Z2", 0.5),
            QubitOperator("X0", 0.2) + QubitOperator("Y1", 0.3) + QubitOperator("", 0.7),
        ],
    )

    n_layers_list, layers_list, z_terms_list = gr_layers.estimate_rz_layers_from_grouping(
        "H2",
        bit_wise=False,
    )

    assert n_layers_list == [1, 1]
    assert len(layers_list) == 2
    assert len(z_terms_list) == 2
    assert set(z_terms_list[0].keys()) == {frozenset({0, 1}), frozenset({2})}
    assert set(z_terms_list[1].keys()) == {frozenset({0}), frozenset({1})}


def test_estimate_rz_layers_from_grouping_bitwise_large_uses_min(monkeypatch) -> None:
    monkeypatch.setattr(
        gr_layers,
        "_run_scf_and_integrals",
        lambda molecule_type: (
            None,
            _FakeMF(4),
            0.0,
            np.zeros((4, 4), dtype=float),
            np.zeros((4, 4, 4, 4), dtype=float),
        ),
    )
    monkeypatch.setattr(
        gr_layers,
        "_build_grouped_qubit_ops",
        lambda **kwargs: [
            QubitOperator("X0", 1.0) + QubitOperator("Y0 Z1", 1.0),
        ],
    )
    monkeypatch.setattr(
        gr_layers,
        "bitwise_optimize_z_terms",
        lambda z_terms, *, n_qubits, optimize_iters: (99, []),
    )

    n_layers_list, _layers_list, _z_terms_list, bitwise_t_depth_list = (
        gr_layers.estimate_rz_layers_from_grouping(
            4,
            bit_wise=True,
        )
    )

    assert n_layers_list == [2]
    assert bitwise_t_depth_list == [2]


def test_estimate_rz_layers_from_grouping_bitwise_small_no_min(monkeypatch) -> None:
    monkeypatch.setattr(
        gr_layers,
        "_run_scf_and_integrals",
        lambda molecule_type: (
            None,
            _FakeMF(3),
            0.0,
            np.zeros((3, 3), dtype=float),
            np.zeros((3, 3, 3, 3), dtype=float),
        ),
    )
    monkeypatch.setattr(
        gr_layers,
        "_build_grouped_qubit_ops",
        lambda **kwargs: [
            QubitOperator("X0", 1.0) + QubitOperator("Y0 Z1", 1.0),
        ],
    )
    monkeypatch.setattr(
        gr_layers,
        "bitwise_optimize_z_terms",
        lambda z_terms, *, n_qubits, optimize_iters: (99, []),
    )

    _n_layers_list, _layers_list, _z_terms_list, bitwise_t_depth_list = (
        gr_layers.estimate_rz_layers_from_grouping(
            "3",
            bit_wise=True,
        )
    )

    assert bitwise_t_depth_list == [99]

