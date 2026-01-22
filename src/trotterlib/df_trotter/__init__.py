from __future__ import annotations

from .circuit import (
    build_df_trotter_circuit,
    estimate_energy,
    report_cost,
    simulate_statevector,
)
from .decompose import df_decompose_from_integrals, diag_hermitian
from .model import Block, DFBlock, DFModel, OneBodyGaussianBlock
from .ops import (
    U_to_qiskit_ops_jw,
    apply_D_one_body,
    apply_D_squared,
    apply_df_block,
    apply_one_body_gaussian_block,
    apply_pauli_block,
    build_df_blocks,
    build_one_body_gaussian_block,
)


def make_integrals_and_fci(*args, **kwargs):  # type: ignore[no-untyped-def]
    from ..qiskit_time_evolution_pyscf import make_integrals_and_fci as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "Block",
    "DFBlock",
    "DFModel",
    "OneBodyGaussianBlock",
    "U_to_qiskit_ops_jw",
    "apply_D_one_body",
    "apply_D_squared",
    "apply_df_block",
    "apply_one_body_gaussian_block",
    "apply_pauli_block",
    "build_df_blocks",
    "build_one_body_gaussian_block",
    "build_df_trotter_circuit",
    "df_decompose_from_integrals",
    "diag_hermitian",
    "estimate_energy",
    "make_integrals_and_fci",
    "report_cost",
    "simulate_statevector",
]
