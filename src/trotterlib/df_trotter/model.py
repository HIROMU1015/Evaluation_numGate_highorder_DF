from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class DFModel:
    lambdas: np.ndarray
    G_list: list[np.ndarray]
    one_body_correction: np.ndarray
    constant_correction: float
    N: int


@dataclass(frozen=True)
class DFBlock:
    U_ops: Sequence[tuple[Any, Tuple[int, ...]]]
    eta: np.ndarray
    lam: float


@dataclass(frozen=True)
class OneBodyGaussianBlock:
    U_ops: Sequence[tuple[Any, Tuple[int, ...]]]
    eps: np.ndarray


@dataclass(frozen=True)
class Block:
    kind: str
    payload: Any

    def apply(self, qc: QuantumCircuit, tau: float) -> None:
        from .ops import (
            apply_df_block,
            apply_one_body_gaussian_block,
            apply_pauli_block,
        )

        if self.kind == "pauli":
            apply_pauli_block(qc, self.payload, tau)
            return
        if self.kind == "df":
            apply_df_block(
                qc, self.payload.U_ops, self.payload.eta, self.payload.lam, tau
            )
            return
        if self.kind == "one_body_gaussian":
            apply_one_body_gaussian_block(
                qc, self.payload.U_ops, self.payload.eps, tau
            )
            return
        raise ValueError(f"Unsupported block kind: {self.kind}")

    @classmethod
    def from_pauli(cls, qubit_op: Any) -> "Block":
        return cls(kind="pauli", payload=qubit_op)

    @classmethod
    def from_df(cls, df_block: DFBlock) -> "Block":
        return cls(kind="df", payload=df_block)

    @classmethod
    def from_one_body_gaussian(cls, block: OneBodyGaussianBlock) -> "Block":
        return cls(kind="one_body_gaussian", payload=block)
