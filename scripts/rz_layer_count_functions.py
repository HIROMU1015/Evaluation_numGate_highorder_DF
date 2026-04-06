from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from openfermion.chem import MolecularData
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from trotterlib.Almost_optimal_grouping import (
    Almost_optimal_grouper,
    make_spinorb_ham_upthendown_order,
)
from trotterlib.chemistry_hamiltonian import geo, min_hamiltonian_grouper
from trotterlib.config import (
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    PF_RZ_LAYER,
    PICKLE_DIR_DF_RZ_LAYER_PATH,
    PFLabel,
    get_default_df_time_for_molecule_pf,
    get_df_rank_fraction_for_molecule,
    normalize_pf_label,
)
from trotterlib.df_trotter.circuit import build_df_trotter_circuit
from trotterlib.df_trotter.decompose import df_decompose_from_integrals
from trotterlib.df_trotter.model import Block, DFModel
from trotterlib.df_trotter.ops import (
    apply_D_one_body,
    apply_D_squared,
    build_df_blocks_givens,
    build_one_body_gaussian_block_givens,
)
from trotterlib.pf_decomposition import iter_pf_steps
from trotterlib.plot_utils import set_loglog_axes
from trotterlib.product_formula import _get_w_list
from trotterlib.qiskit_time_evolution_pyscf import _run_scf_and_integrals

# Consolidated copy of DF/GR RZ-layer counting functions.

def _default_basis_gates() -> List[str]:
    return ["rz", "cx", "sx", "x"]


def _decompose_to_basis(
    qc: QuantumCircuit,
    *,
    basis_gates: Iterable[str] | None = None,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> QuantumCircuit:
    """分解＋transpileで基底ゲートに落とす（RZ計測用）。"""
    if basis_gates is None:
        basis_gates = _default_basis_gates()
    qc_work = qc
    if decompose_reps and decompose_reps > 0:
        qc_work = _decompose_nonbasis(
            qc_work, basis_gates=basis_gates, decompose_reps=decompose_reps
        )
    return transpile(
        qc_work,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
    )


def _nonbasis_ops(
    qc: QuantumCircuit, *, basis_gates: Iterable[str]
) -> List[str]:
    basis_set = {name.lower() for name in basis_gates}
    ignore = {"barrier", "measure", "reset", "delay"}
    extras = set()
    for inst, _qargs, _cargs in qc.data:
        name = inst.name.lower()
        if name in basis_set or name in ignore:
            continue
        extras.add(inst.name)
    return sorted(extras)


def _decompose_nonbasis(
    qc: QuantumCircuit,
    *,
    basis_gates: Iterable[str],
    decompose_reps: int,
) -> QuantumCircuit:
    qc_work = qc
    extras = _nonbasis_ops(qc_work, basis_gates=basis_gates)
    if not extras:
        return qc_work
    for _ in range(decompose_reps):
        qc_work = qc_work.decompose(gates_to_decompose=extras, reps=1)
        extras = _nonbasis_ops(qc_work, basis_gates=basis_gates)
        if not extras:
            break
    return qc_work


def _rz_depth_from_dag(dag) -> int:
    """DAG上でRZ重み付き最長路をDPで計算する。"""
    dp: dict[DAGOpNode, int] = {}
    max_depth = 0
    for node in dag.topological_op_nodes():
        max_pred = 0
        for pred in dag.predecessors(node):
            if isinstance(pred, DAGOpNode):
                max_pred = max(max_pred, dp.get(pred, 0))
        weight = 1 if node.op.name.lower() == "rz" else 0
        dp[node] = max_pred + weight
        if dp[node] > max_depth:
            max_depth = dp[node]
    return max_depth


def _weighted_depth_from_dag(dag, weight_fn: Callable[[DAGOpNode], int]) -> int:
    dp: dict[DAGOpNode, int] = {}
    max_depth = 0
    for node in dag.topological_op_nodes():
        max_pred = 0
        for pred in dag.predecessors(node):
            if isinstance(pred, DAGOpNode):
                max_pred = max(max_pred, dp.get(pred, 0))
        weight = int(weight_fn(node))
        dp[node] = max_pred + weight
        if dp[node] > max_depth:
            max_depth = dp[node]
    return max_depth


def _greedy_disjoint_layer_count(supports: Sequence[frozenset[int]]) -> int:
    """Greedy layer packing for supports (disjoint supports can share a layer)."""
    layers: list[set[int]] = []
    for supp in supports:
        placed = False
        for used in layers:
            if used.intersection(supp):
                continue
            used.update(supp)
            placed = True
            break
        if not placed:
            layers.append(set(supp))
    return len(layers)


def _support_from_qargs(qc: QuantumCircuit, qargs: Sequence[Any]) -> frozenset[int] | None:
    """Extract stable qubit indices as a support set via qc.find_bit(...).index."""
    if not qargs:
        return None
    idxs: list[int] = []
    for q in qargs:
        try:
            i = qc.find_bit(q).index
        except Exception:
            return None
        if i is None:
            return None
        idxs.append(int(i))
    return frozenset(idxs)


def _angle_to_float(angle: Any) -> float | None:
    try:
        val = float(angle)
    except Exception:
        try:
            val = float(np.asarray(angle))
        except Exception:
            return None
    if not np.isfinite(val):
        return None
    return val


def _is_clifford_angle(angle: Any, *, tol: float = 1e-10) -> bool:
    val = _angle_to_float(angle)
    if val is None:
        return False
    step = np.pi / 2.0
    ratio = val / step
    return abs(ratio - round(ratio)) <= tol


def _nonclifford_param_count(
    params: Sequence[Any], *, tol: float, default_if_empty: int = 1
) -> int:
    if not params:
        return default_if_empty
    count = 0
    for p in params:
        if not _is_clifford_angle(p, tol=tol):
            count += 1
    return count


def _is_nonclifford_rz_gate(op: Any, *, tol: float) -> bool:
    if op.name.lower() != "rz":
        return False
    angle = op.params[0] if op.params else None
    return not _is_clifford_angle(angle, tol=tol)


def _is_nonclifford_rzz_gate(op: Any, *, tol: float) -> bool:
    if op.name.lower() != "rzz":
        return False
    angle = op.params[0] if op.params else None
    return not _is_clifford_angle(angle, tol=tol)


def _is_nonclifford_z_proxy_gate(op: Any, *, tol: float) -> bool:
    name = op.name.lower()
    if name not in {"rz", "rzz"}:
        return False
    angle = op.params[0] if op.params else None
    return not _is_clifford_angle(angle, tol=tol)


def _nonclifford_rz_count_from_circuit(
    qc: QuantumCircuit, *, tol: float = 1e-10
) -> int:
    count = 0
    for inst, _qargs, _cargs in qc.data:
        if inst.name.lower() != "rz":
            continue
        angle = inst.params[0] if inst.params else None
        if not _is_clifford_angle(angle, tol=tol):
            count += 1
    return count


def _nonclifford_rzz_count_from_circuit(
    qc: QuantumCircuit, *, tol: float = 1e-10
) -> int:
    count = 0
    for inst, _qargs, _cargs in qc.data:
        if inst.name.lower() != "rzz":
            continue
        angle = inst.params[0] if inst.params else None
        if not _is_clifford_angle(angle, tol=tol):
            count += 1
    return count


def _nonclifford_rz_depth_from_dag(dag, *, tol: float = 1e-10) -> int:
    return _weighted_depth_from_dag(
        dag,
        lambda node: 1 if _is_nonclifford_rz_gate(node.op, tol=tol) else 0,
    )


def _nonclifford_rzz_depth_from_dag(dag, *, tol: float = 1e-10) -> int:
    return _weighted_depth_from_dag(
        dag,
        lambda node: 1 if _is_nonclifford_rzz_gate(node.op, tol=tol) else 0,
    )


def rz_costs_from_circuit(
    qc: QuantumCircuit,
    *,
    basis_gates: Iterable[str] | None = None,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> dict:
    """RZ数とRZレイヤー数を推定する（基底ゲートへ分解後に評価）。"""
    if basis_gates is None:
        basis_gates = _default_basis_gates()
    qc_basis = _decompose_to_basis(
        qc,
        basis_gates=basis_gates,
        decompose_reps=decompose_reps,
        optimization_level=optimization_level,
    )
    counts = qc_basis.count_ops()
    rz_count = int(counts.get("rz", 0)) + int(counts.get("RZ", 0))
    dag = circuit_to_dag(qc_basis)
    rz_depth = int(_rz_depth_from_dag(dag))
    nonbasis = _nonbasis_ops(qc_basis, basis_gates=basis_gates)
    return {
        "rz_count": rz_count,
        "rz_depth": rz_depth,
        "depth_total": int(qc_basis.depth()),
        "basis_gates": list(basis_gates),
        "nonbasis_ops": nonbasis,
    }


def nonclifford_rz_costs_from_circuit(
    qc: QuantumCircuit,
    *,
    basis_gates: Iterable[str] | None = None,
    decompose_reps: int = 8,
    optimization_level: int = 0,
    angle_tol: float = 1e-10,
) -> dict:
    """非Clifford RZの数/深さを推定する（基底ゲートへ分解後に評価）。"""
    if basis_gates is None:
        basis_gates = _default_basis_gates()
    qc_basis = _decompose_to_basis(
        qc,
        basis_gates=basis_gates,
        decompose_reps=decompose_reps,
        optimization_level=optimization_level,
    )
    counts = qc_basis.count_ops()
    rz_count = int(counts.get("rz", 0)) + int(counts.get("RZ", 0))
    noncliff_count = _nonclifford_rz_count_from_circuit(qc_basis, tol=angle_tol)
    dag = circuit_to_dag(qc_basis)
    rz_depth = int(_rz_depth_from_dag(dag))
    noncliff_depth = int(_nonclifford_rz_depth_from_dag(dag, tol=angle_tol))
    nonbasis = _nonbasis_ops(qc_basis, basis_gates=basis_gates)
    return {
        "rz_count": rz_count,
        "rz_depth": rz_depth,
        "nonclifford_rz_count": noncliff_count,
        "nonclifford_rz_depth": noncliff_depth,
        "depth_total": int(qc_basis.depth()),
        "basis_gates": list(basis_gates),
        "nonbasis_ops": nonbasis,
    }


def d_nonclifford_costs_from_circuit(
    qc: QuantumCircuit,
    *,
    angle_tol: float = 1e-10,
    debug: bool = False,
    debug_max: int = 10,
    debug_print: Callable[[str], None] = print,
) -> dict:
    rz_total = 0
    rzz_total = 0
    rz_noncliff = 0
    rzz_noncliff = 0
    rz_debug_count = 0
    rzz_debug_count = 0
    rz_debug_records: list[str] = []
    rzz_debug_records: list[str] = []
    support_errors: list[str] = []
    supports_noncliff: list[frozenset[int]] = []

    for inst, qargs, _cargs in qc.data:
        name = inst.name.lower()
        if name not in {"rz", "rzz"}:
            continue
        angle = inst.params[0] if inst.params else None
        val = _angle_to_float(angle)
        floatable = val is not None
        if floatable:
            step = np.pi / 2.0
            ratio = val / step
            theta_mod = abs(ratio - round(ratio))
            is_clifford = theta_mod <= angle_tol
        else:
            theta_mod = None
            is_clifford = False
        is_noncliff = not is_clifford
        if name == "rz":
            rz_total += 1
            if is_noncliff:
                rz_noncliff += 1
                supp = _support_from_qargs(qc, qargs[:1])
                if supp is None:
                    support_errors.append("failed to extract support for rz")
                else:
                    supports_noncliff.append(supp)
            if debug and rz_debug_count < debug_max:
                rz_debug_records.append(
                    "[d_rz] "
                    f"angle={val} floatable={floatable} theta_mod={theta_mod} "
                    f"clifford={not is_noncliff}"
                )
                rz_debug_count += 1
        else:
            rzz_total += 1
            if is_noncliff:
                rzz_noncliff += 1
                supp = _support_from_qargs(qc, qargs[:2])
                if supp is None:
                    support_errors.append("failed to extract support for rzz")
                else:
                    supports_noncliff.append(supp)
            if debug and rzz_debug_count < debug_max:
                rzz_debug_records.append(
                    "[d_rzz] "
                    f"angle={val} floatable={floatable} theta_mod={theta_mod} "
                    f"clifford={not is_noncliff}"
                )
                rzz_debug_count += 1

    dag = circuit_to_dag(qc)
    rz_depth = int(_nonclifford_rz_depth_from_dag(dag, tol=angle_tol))
    rzz_depth = int(_nonclifford_rzz_depth_from_dag(dag, tol=angle_tol))
    combined_depth = int(
        _weighted_depth_from_dag(
            dag,
            lambda node: 1
            if _is_nonclifford_z_proxy_gate(node.op, tol=angle_tol)
            else 0,
        )
    )
    noncliff_total = rz_noncliff + rzz_noncliff
    nonbasis_ops = _nonbasis_ops(qc, basis_gates=_default_basis_gates())
    coloring_depth = int(_greedy_disjoint_layer_count(supports_noncliff))

    if debug and (noncliff_total > 0 or nonbasis_ops):
        debug_print(
            f"[d_proxy] rz_total={rz_total} rz_nonclifford={rz_noncliff} "
            f"rzz_total={rzz_total} rzz_nonclifford={rzz_noncliff}"
        )
        debug_print(
            f"[d_proxy] depth(dag)={combined_depth} depth(coloring)={coloring_depth}"
        )
        for record in rz_debug_records:
            debug_print(record)
        for record in rzz_debug_records:
            debug_print(record)
        if support_errors:
            debug_print(f"[d_proxy][WARNING] support_errors(sample)={support_errors[:5]}")

    return {
        "rz_total": rz_total,
        "rzz_total": rzz_total,
        "rz_nonclifford": rz_noncliff,
        "rzz_nonclifford": rzz_noncliff,
        "nonclifford_total": noncliff_total,
        "rz_nonclifford_depth": rz_depth,
        "rzz_nonclifford_depth": rzz_depth,
        "combined_nonclifford_depth": combined_depth,
        "coloring_nonclifford_depth": coloring_depth,
        "nonbasis_ops": nonbasis_ops,
    }


def nonclifford_z_proxy_costs_from_circuit(
    qc: QuantumCircuit,
    *,
    angle_tol: float = 1e-10,
    debug: bool = False,
    debug_max: int = 10,
    debug_print: Callable[[str], None] = print,
) -> dict:
    """Raw circuit-based Toffoli proxy from non-Clifford RZ/RZZ only."""
    rz_total = 0
    rzz_total = 0
    rz_noncliff = 0
    rzz_noncliff = 0
    rz_debug_count = 0
    rzz_debug_count = 0
    rz_debug_records: list[str] = []
    rzz_debug_records: list[str] = []

    for inst, _qargs, _cargs in qc.data:
        name = inst.name.lower()
        if name not in {"rz", "rzz"}:
            continue
        angle = inst.params[0] if inst.params else None
        val = _angle_to_float(angle)
        floatable = val is not None
        if floatable:
            step = np.pi / 2.0
            ratio = val / step
            theta_mod = abs(ratio - round(ratio))
            is_clifford = theta_mod <= angle_tol
        else:
            theta_mod = None
            is_clifford = False
        is_noncliff = not is_clifford
        if name == "rz":
            rz_total += 1
            if is_noncliff:
                rz_noncliff += 1
            if rz_debug_count < debug_max:
                rz_debug_records.append(
                    "[d_rz] "
                    f"angle={val} floatable={floatable} theta_mod={theta_mod} "
                    f"clifford={not is_noncliff}"
                )
                rz_debug_count += 1
        else:
            rzz_total += 1
            if is_noncliff:
                rzz_noncliff += 1
            if rzz_debug_count < debug_max:
                rzz_debug_records.append(
                    "[d_rzz] "
                    f"angle={val} floatable={floatable} theta_mod={theta_mod} "
                    f"clifford={not is_noncliff}"
                )
                rzz_debug_count += 1

    dag = circuit_to_dag(qc)
    depth = int(
        _weighted_depth_from_dag(
            dag,
            lambda node: 1
            if _is_nonclifford_z_proxy_gate(node.op, tol=angle_tol)
            else 0,
        )
    )
    noncliff_total = rz_noncliff + rzz_noncliff

    if debug:
        debug_print(
            f"[d_proxy] rz_total={rz_total} rz_nonclifford={rz_noncliff} "
            f"rzz_total={rzz_total} rzz_nonclifford={rzz_noncliff}"
        )
        for record in rz_debug_records:
            debug_print(record)
        for record in rzz_debug_records:
            debug_print(record)

    return {
        "rz_total": rz_total,
        "rzz_total": rzz_total,
        "rz_nonclifford": rz_noncliff,
        "rzz_nonclifford": rzz_noncliff,
        "nonclifford_z_proxy_count": noncliff_total,
        "nonclifford_z_proxy_depth": depth,
    }


def u_nonclifford_costs_from_u_ops(
    u_ops: Sequence[tuple[Any, Tuple[int, ...]]],
    num_qubits: int,
    *,
    angle_tol: float = 1e-10,
    debug: bool = False,
    debug_max: int = 10,
    debug_print: Callable[[str], None] = print,
    fail_fast: bool = True,
) -> dict:
    """U回路の非Clifford RZコスト（xx_plus_yy換算含む）を算出する。"""
    qc = QuantumCircuit(num_qubits)
    for gate, qubits in u_ops:
        qc.append(gate, list(qubits))

    u_nonbasis_ops = _nonbasis_ops(qc, basis_gates=_default_basis_gates())
    rz_noncliff = _nonclifford_rz_count_from_circuit(qc, tol=angle_tol)
    rz_depth = int(
        _nonclifford_rz_depth_from_dag(circuit_to_dag(qc), tol=angle_tol)
    )

    supports_noncliff: list[frozenset[int]] = []
    support_errors: list[str] = []

    xx_count = 0
    xx_noncliff_count = 0
    xx_noncliff_gate_count = 0
    xx_edges: list[tuple[int, int]] = []
    xx_edge_errors: list[str] = []
    xx_debug_records: list[str] = []
    xx_debug_count = 0
    xx_param_clifford = 0
    xx_param_nonclifford = 0
    for inst, qargs, _cargs in qc.data:
        if inst.name.lower() != "xx_plus_yy":
            continue
        xx_count += 1
        params = list(inst.params) if inst.params else []
        param_infos: list[dict[str, Any]] = []
        if params:
            noncliff_params = 0
            for p in params:
                val = _angle_to_float(p)
                floatable = val is not None
                if floatable:
                    step = np.pi / 2.0
                    ratio = val / step
                    theta_mod = abs(ratio - round(ratio))
                    is_clifford = theta_mod <= angle_tol
                else:
                    theta_mod = None
                    is_clifford = False
                if not is_clifford:
                    noncliff_params += 1
                    xx_param_nonclifford += 1
                else:
                    xx_param_clifford += 1
                param_infos.append(
                    {
                        "value": val,
                        "floatable": floatable,
                        "theta_mod": theta_mod,
                        "is_clifford": is_clifford,
                    }
                )
        else:
            noncliff_params = 1
            xx_param_nonclifford += 1

        q_idx: tuple[int, int] | None = None
        edge_added = False
        edge_error = None
        if len(qargs) >= 2:
            try:
                q0 = qc.find_bit(qargs[0]).index
                q1 = qc.find_bit(qargs[1]).index
                if q0 is None or q1 is None:
                    raise ValueError("find_bit returned None")
                q_idx = (int(q0), int(q1))
            except Exception as exc:
                edge_error = (
                    "qubit index extraction failed "
                    f"(inst={inst.name} class={inst.__class__.__name__}): {exc}"
                )
        else:
            edge_error = f"insufficient qargs for xx_plus_yy (len={len(qargs)})"
        if edge_error:
            xx_edge_errors.append(edge_error)

        if noncliff_params > 0:
            xx_noncliff_count += noncliff_params
            xx_noncliff_gate_count += 1
            if q_idx is not None:
                edge = (min(q_idx[0], q_idx[1]), max(q_idx[0], q_idx[1]))
                xx_edges.append(edge)
                edge_added = True
                supports_noncliff.append(frozenset(edge))
            else:
                support_errors.append("failed to extract support for xx_plus_yy")

        if debug and xx_debug_count < debug_max:
            param_types = [type(p).__name__ for p in params]
            floatable_list = [info["floatable"] for info in param_infos]
            theta_mod_list = [info["theta_mod"] for info in param_infos]
            clifford_list = [info["is_clifford"] for info in param_infos]
            value_list = [info["value"] for info in param_infos]
            record = (
                "[xx_plus_yy] idx="
                f"{xx_debug_count} name={inst.name} class={inst.__class__.__name__} "
                f"params_repr={repr(inst.params)} param_types={param_types} "
                f"qidx={q_idx} floatable={floatable_list} "
                f"theta_mod={theta_mod_list} clifford={clifford_list} "
                f"param_values={value_list} noncliff_params={noncliff_params} "
                f"edge_added={edge_added}"
            )
            if not params:
                record += " params_missing=True"
            if edge_error:
                record += f" edge_error={edge_error}"
            xx_debug_records.append(record)
            xx_debug_count += 1

    rz_total = 0
    rz_noncliff_detail = 0
    rz_debug_count = 0
    rz_debug_records: list[str] = []
    for inst, qargs, _cargs in qc.data:
        if inst.name.lower() != "rz":
            continue
        rz_total += 1
        angle = inst.params[0] if inst.params else None
        val = _angle_to_float(angle)
        floatable = val is not None
        if floatable:
            step = np.pi / 2.0
            ratio = val / step
            theta_mod = abs(ratio - round(ratio))
            is_clifford = theta_mod <= angle_tol
        else:
            theta_mod = None
            is_clifford = False
        if not is_clifford:
            rz_noncliff_detail += 1
            supp = _support_from_qargs(qc, qargs[:1])
            if supp is None:
                support_errors.append("failed to extract support for rz")
            else:
                supports_noncliff.append(supp)
        if debug and rz_debug_count < debug_max:
            rz_debug_records.append(
                "[u_rz] "
                f"angle={val} floatable={floatable} theta_mod={theta_mod} "
                f"clifford={is_clifford}"
            )
            rz_debug_count += 1

    layers: list[set[int]] = []
    for q0, q1 in xx_edges:
        placed = False
        for used in layers:
            if q0 in used or q1 in used:
                continue
            used.update({q0, q1})
            placed = True
            break
        if not placed:
            layers.append({q0, q1})
    xx_layers_edge = len(layers)

    xx_layers_dag = None
    dag_exc: Exception | None = None
    try:
        dag = circuit_to_dag(qc)
        xx_layers_dag = int(
            _weighted_depth_from_dag(
                dag,
                lambda node: 1
                if (
                    node.op.name.lower() == "xx_plus_yy"
                    and _nonclifford_param_count(
                        list(node.op.params) if node.op.params else [],
                        tol=angle_tol,
                        default_if_empty=1,
                    )
                    > 0
                )
                else 0,
            )
        )
    except Exception as exc:
        dag_exc = exc

    xx_layers = xx_layers_dag if xx_layers_dag is not None else xx_layers_edge
    coloring_depth = int(_greedy_disjoint_layer_count(supports_noncliff))

    if debug and xx_count > 0 and xx_noncliff_count == 0:
        debug_print(
            "[xx_plus_yy][INFO] all params classified Clifford; "
            "nonClifford contribution is 0"
        )

    should_log_details = debug and (
        xx_noncliff_count > 0 or rz_noncliff_detail > 0 or u_nonbasis_ops
    )
    if should_log_details:
        debug_print(
            f"[u_proxy] depth(dag)={rz_depth + xx_layers} depth(coloring)={coloring_depth}"
        )
        debug_print(
            "[xx_plus_yy] param_clifford="
            f"{xx_param_clifford} param_nonclifford={xx_param_nonclifford}"
        )
        for record in xx_debug_records:
            debug_print(record)
        sample_edges = xx_edges[: min(5, len(xx_edges))]
        debug_print(
            "[xx_plus_yy] edges_count="
            f"{len(xx_edges)} edges_sample={sample_edges} "
            f"edge_errors_count={len(xx_edge_errors)}"
        )
        if xx_edge_errors:
            debug_print(
                f"[xx_plus_yy] edge_errors(sample)={xx_edge_errors[:5]}"
            )
        if dag_exc is not None:
            debug_print(f"[xx_plus_yy][WARNING] dag_weighted failed: {dag_exc}")
        debug_print(
            "[xx_plus_yy] layers edge_coloring="
            f"{xx_layers_edge} dag_weighted={xx_layers_dag}"
        )
        if (
            xx_layers_dag is not None
            and xx_layers_edge != xx_layers_dag
        ):
            debug_print(
                "[xx_plus_yy][WARNING] layer mismatch: "
                f"edge_coloring={xx_layers_edge} dag_weighted={xx_layers_dag}"
            )
        debug_print(
            f"[u_rz] total={rz_total} nonclifford={rz_noncliff_detail}"
        )
        for record in rz_debug_records:
            debug_print(record)
        if support_errors:
            debug_print(f"[u_proxy][WARNING] support_errors(sample)={support_errors[:5]}")

    u_noncliff_count = rz_noncliff + xx_noncliff_count

    if fail_fast and xx_count > 0 and xx_noncliff_count > 0 and xx_layers == 0:
        debug_print(
            "[xx_plus_yy][FAIL-FAST] nonClifford count>0 but layers==0"
        )
        debug_print(
            "[xx_plus_yy][FAIL-FAST] "
            f"xx_count={xx_count} xx_noncliff_gate_count={xx_noncliff_gate_count} "
            f"xx_noncliff_param_count={xx_noncliff_count} rz_noncliff_count={rz_noncliff} "
            f"edges_count={len(xx_edges)} xx_layers_edge={xx_layers_edge} "
            f"xx_layers_dag={xx_layers_dag}"
        )
        if xx_edge_errors:
            debug_print(
                f"[xx_plus_yy][FAIL-FAST] edge_errors(sample)={xx_edge_errors[:5]}"
            )
        for record in xx_debug_records:
            debug_print(record)
        raise RuntimeError(
            "xx_plus_yy counting bug: nonClifford count>0 but layers==0"
        )

    return {
        "rz_nonclifford_count": rz_noncliff,
        "rz_nonclifford_depth": rz_depth,
        "xx_plus_yy_count": xx_count,
        "xx_plus_yy_layers": xx_layers,
        "xx_plus_yy_nonclifford_count": xx_noncliff_count,
        "u_nonclifford_rz_count": u_noncliff_count,
        "u_nonclifford_rz_depth": rz_depth + xx_layers,
        "u_nonclifford_z_coloring_depth": coloring_depth,
        "u_raw_rz_count": rz_total,
        "u_raw_rz_nonclifford_count": rz_noncliff_detail,
    }


def rz_costs_from_u_ops(
    u_ops: Sequence[tuple[Any, Tuple[int, ...]]],
    num_qubits: int,
    *,
    basis_gates: Iterable[str] | None = None,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> dict:
    """U_ops だけの回路を作って RZ コストを推定する。"""
    qc = QuantumCircuit(num_qubits)
    for gate, qubits in u_ops:
        qc.append(gate, list(qubits))
    return rz_costs_from_circuit(
        qc,
        basis_gates=basis_gates,
        decompose_reps=decompose_reps,
        optimization_level=optimization_level,
    )


def debug_summarize_circuit(
    qc: QuantumCircuit,
    tag: str,
    *,
    show_top_ops: bool = True,
    max_ops: int = 10,
    debug_print: Callable[[str], None] = print,
) -> None:
    """回路の概要を簡易表示する（デバッグ用）。"""
    counts = qc.count_ops()
    summary = (
        f"[{tag}] num_qubits={qc.num_qubits} depth={qc.depth()} size={qc.size()}"
    )
    debug_print(summary)

    if show_top_ops:
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max_ops]
        debug_print(f"[{tag}] count_ops(top): {top}")

    op_names = [inst.operation.name for inst in qc.data[:max_ops]]
    uniq_names = sorted({inst.operation.name for inst in qc.data})
    debug_print(f"[{tag}] ops(first {max_ops}): {op_names}")
    debug_print(f"[{tag}] ops(unique {min(len(uniq_names), max_ops)}): {uniq_names[:max_ops]}")

    has_unitary = any(inst.operation.name.lower() == "unitary" for inst in qc.data)
    debug_print(f"[{tag}] has_unitary={has_unitary}")


def debug_trace_u_decomposition(
    u_qc_or_ops: QuantumCircuit | Sequence[tuple[Any, Tuple[int, ...]]],
    tag: str,
    *,
    num_qubits: int | None = None,
    decompose_reps: int = 10,
    basis_gates: Iterable[str] | None = None,
    opt_level: int = 0,
    warn_rz: int = 2000,
    warn_cx: int = 2000,
    debug_print: Callable[[str], None] = print,
) -> None:
    """U の分解状況を raw/decompose/transpile の3段階でログ出力する。"""
    if isinstance(u_qc_or_ops, QuantumCircuit):
        qc_raw = u_qc_or_ops
    else:
        if num_qubits is None:
            raise ValueError("num_qubits is required when u_qc_or_ops is ops.")
        qc_raw = QuantumCircuit(num_qubits)
        for gate, qubits in u_qc_or_ops:
            qc_raw.append(gate, list(qubits))

    debug_summarize_circuit(
        qc_raw,
        f"{tag}/raw",
        debug_print=debug_print,
    )

    basis_list = list(basis_gates or _default_basis_gates())
    qc_dec = (
        _decompose_nonbasis(
            qc_raw,
            basis_gates=basis_list,
            decompose_reps=decompose_reps,
        )
        if decompose_reps > 0
        else qc_raw
    )
    debug_summarize_circuit(
        qc_dec,
        f"{tag}/decompose",
        debug_print=debug_print,
    )

    qc_tr = transpile(
        qc_dec,
        basis_gates=basis_list,
        optimization_level=opt_level,
    )
    debug_summarize_circuit(
        qc_tr,
        f"{tag}/transpile",
        debug_print=debug_print,
    )

    has_unitary = any(inst.operation.name.lower() == "unitary" for inst in qc_tr.data)
    if has_unitary:
        debug_print(
            f"[{tag}] WARNING: UnitaryGate remains after decompose/transpile."
        )

    counts = qc_tr.count_ops()
    rz_count = int(counts.get("rz", 0)) + int(counts.get("RZ", 0))
    cx_count = int(counts.get("cx", 0)) + int(counts.get("CX", 0))
    if rz_count > warn_rz or cx_count > warn_cx:
        debug_print(
            f"[{tag}] WARNING: rz/cx extremely large (rz={rz_count}, cx={cx_count}); "
            "unitary synthesis path may be used."
        )
def _h_chain_integrals(
    molecule_type: int, *, distance: float | None, basis: str | None
) -> tuple[float, np.ndarray, np.ndarray]:
    if distance is None:
        distance = DEFAULT_DISTANCE
    if basis is None:
        basis = DEFAULT_BASIS
    geometry, multiplicity, charge = geo(molecule_type, distance)
    description = f"distance_{int(distance * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry, basis, multiplicity, charge, description=description
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    one_body, two_body = molecule.get_integrals()
    constant = float(molecule.nuclear_repulsion)
    return constant, one_body, two_body


def _h_chain_integrals_pyscf(
    molecule_type: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    _, _, constant, one_body, two_body = _run_scf_and_integrals(molecule_type)
    return float(constant), one_body, two_body


def _artifact_ham_name(
    molecule_type: int, *, distance: float | None, basis: str | None
) -> str:
    distance_value = DEFAULT_DISTANCE if distance is None else float(distance)
    basis_value = DEFAULT_BASIS if basis is None else basis
    geometry, multiplicity, charge = geo(molecule_type, distance_value)
    description = f"distance_{int(distance_value * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry,
        basis_value,
        multiplicity,
        charge,
        description=description,
    )
    return f"{Path(molecule.filename).stem}_grouping"


def _save_df_plot_artifact(file_name: str, data: dict[str, object]) -> None:
    PICKLE_DIR_DF_PATH.mkdir(parents=True, exist_ok=True)
    path = PICKLE_DIR_DF_PATH / file_name
    with path.open("wb") as f:
        pickle.dump(data, f)


def _save_df_rz_layer_artifact(file_name: str, data: dict[str, object]) -> None:
    PICKLE_DIR_DF_RZ_LAYER_PATH.mkdir(parents=True, exist_ok=True)
    path = PICKLE_DIR_DF_RZ_LAYER_PATH / file_name
    with path.open("wb") as f:
        pickle.dump(data, f)


def _load_df_rz_layer_artifact(file_name: str) -> dict[str, object] | None:
    path = PICKLE_DIR_DF_RZ_LAYER_PATH / file_name
    if not path.exists():
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data
    return None


def _collect_df_rz_layer_metrics(costs: dict[str, object]) -> dict[str, int]:
    def _to_int(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    metrics: dict[str, int | None] = {
        "ref_rz_depth": _to_int(costs.get("rz_depth")),
        "ref_depth_total": _to_int(costs.get("depth_total")),
    }

    d_only = costs.get("d_only_costs")
    if isinstance(d_only, dict):
        metrics["d_only_nonclifford_rz_depth"] = _to_int(
            d_only.get("nonclifford_rz_depth")
        )
        metrics["d_only_ref_rz_depth"] = _to_int(d_only.get("rz_depth"))

    totals = costs.get("nonclifford_total")
    if isinstance(totals, dict):
        metrics["total_nonclifford_rz_depth"] = _to_int(
            totals.get("total_nonclifford_rz_depth")
        )
        metrics["u_nonclifford_rz_depth"] = _to_int(
            totals.get("u_nonclifford_rz_depth")
        )
        metrics["d_nonclifford_rz_depth"] = _to_int(
            totals.get("d_nonclifford_rz_depth")
        )

    totals_rz = costs.get("rz_total_ud")
    if isinstance(totals_rz, dict):
        metrics["total_ref_rz_depth"] = _to_int(totals_rz.get("total_rz_depth"))
        metrics["u_ref_rz_depth"] = _to_int(totals_rz.get("u_rz_depth"))
        metrics["d_ref_rz_depth"] = _to_int(totals_rz.get("d_rz_depth"))

    proxy_totals = costs.get("toffoli_proxy_total")
    if isinstance(proxy_totals, dict):
        metrics["total_nonclifford_z_depth"] = _to_int(
            proxy_totals.get("total_nonclifford_z_depth")
        )

    proxy_totals_coloring = costs.get("toffoli_proxy_total_coloring")
    if isinstance(proxy_totals_coloring, dict):
        metrics["total_nonclifford_z_coloring_depth"] = _to_int(
            proxy_totals_coloring.get("total_nonclifford_z_depth")
        )

    return {key: value for key, value in metrics.items() if value is not None}


def _collect_df_ud_rz_layer_metrics(costs: dict[str, object]) -> dict[str, object]:
    def _to_int(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    metrics: dict[str, object] = {}
    totals_rz = costs.get("rz_total_ud")
    if isinstance(totals_rz, dict):
        for key in (
            "u_rz_count",
            "u_rz_depth",
            "d_rz_count",
            "d_rz_depth",
            "total_rz_count",
            "total_rz_depth",
        ):
            value = _to_int(totals_rz.get(key))
            if value is not None:
                metrics[key] = value

    totals = costs.get("nonclifford_total")
    if isinstance(totals, dict):
        for key in (
            "u_nonclifford_rz_count",
            "u_nonclifford_rz_depth",
            "d_nonclifford_rz_count",
            "d_nonclifford_rz_depth",
            "total_nonclifford_rz_count",
            "total_nonclifford_rz_depth",
        ):
            value = _to_int(totals.get(key))
            if value is not None:
                metrics[key] = value

    proxy_totals_coloring = costs.get("toffoli_proxy_total_coloring")
    if isinstance(proxy_totals_coloring, dict):
        proxy_key_to_metric_key = {
            "u_nonclifford_z_depth": "u_nonclifford_z_coloring_depth",
            "d_nonclifford_z_depth": "d_nonclifford_z_coloring_depth",
            "total_nonclifford_z_depth": "total_nonclifford_z_coloring_depth",
        }
        for proxy_key, metric_key in proxy_key_to_metric_key.items():
            value = _to_int(proxy_totals_coloring.get(proxy_key))
            if value is not None:
                metrics[metric_key] = value

    u_blocks: list[dict[str, object]] = []
    for entry in costs.get("u_costs", []):
        if not isinstance(entry, dict):
            continue
        block: dict[str, object] = {"label": str(entry.get("label", "U"))}
        rz_count = _to_int(entry.get("u_ref_rz_count"))
        rz_depth = _to_int(entry.get("u_ref_rz_depth"))
        count = _to_int(entry.get("u_nonclifford_rz_count"))
        depth = _to_int(entry.get("u_nonclifford_rz_depth"))
        if rz_count is not None:
            block["rz_count"] = rz_count
        if rz_depth is not None:
            block["rz_depth"] = rz_depth
        if count is not None:
            block["nonclifford_rz_count"] = count
        if depth is not None:
            block["nonclifford_rz_depth"] = depth
        if len(block) > 1:
            u_blocks.append(block)
    if u_blocks:
        metrics["u_block_costs"] = u_blocks

    d_blocks: list[dict[str, object]] = []
    for entry in costs.get("d_block_costs", []):
        if not isinstance(entry, dict):
            continue
        block = {"label": str(entry.get("label", "D"))}
        rz_count = _to_int(entry.get("rz_count"))
        rz_depth = _to_int(entry.get("rz_depth"))
        count = _to_int(entry.get("nonclifford_rz_count"))
        depth = _to_int(entry.get("nonclifford_rz_depth"))
        if rz_count is not None:
            block["rz_count"] = rz_count
        if rz_depth is not None:
            block["rz_depth"] = rz_depth
        if count is not None:
            block["nonclifford_rz_count"] = count
        if depth is not None:
            block["nonclifford_rz_depth"] = depth
        if len(block) > 1:
            d_blocks.append(block)
    if d_blocks:
        metrics["d_block_costs"] = d_blocks

    return metrics


def _default_df_time_or_raise(
    *,
    molecule_type: int,
    pf_label: PFLabel,
) -> float:
    t_default = get_default_df_time_for_molecule_pf(
        int(molecule_type), pf_label
    )
    if t_default is None:
        raise ValueError(
            "Default DF time is not configured for "
            f"molecule_type={int(molecule_type)}."
        )
    return float(t_default)


def _resolve_plot_time_range(
    *,
    molecule_type: int,
    pf_label: PFLabel,
    t_start: float | None,
    t_end: float | None,
    t_step: float | None,
) -> tuple[float, float, float]:
    if t_start is None:
        t_start = _default_df_time_or_raise(
            molecule_type=int(molecule_type),
            pf_label=pf_label,
        )
    if t_step is None:
        t_step = float(DEFAULT_DF_TIME_STEP)
    if t_end is None:
        t_end = float(t_start + DEFAULT_DF_TIME_WINDOW)
    return float(t_start), float(t_end), float(t_step)


def _select_rank_from_ccsd_target(
    molecule_type: int,
    *,
    target_error_ha: float,
    thresh_range: Sequence[float] | None,
    use_kernel: bool,
    no_triples: bool,
    record_in_config: bool = False,
) -> tuple[int, float, dict[str, Any]]:
    from trotterlib.ccsd import select_rank_fraction_for_molecule

    selection = select_rank_fraction_for_molecule(
        molecule_type=int(molecule_type),
        ccsd_target_error_ha=target_error_ha,
        thresh_range=thresh_range,
        use_kernel=use_kernel,
        no_triples=no_triples,
        record_in_config=record_in_config,
    )
    return (
        int(selection["selected_rank"]),
        float(selection["selected_rank_fraction"]),
        selection,
    )


def _symmetrize_two_body(two_body: np.ndarray) -> np.ndarray:
    t = np.asarray(two_body)
    parts = [
        t,
        np.transpose(t, (1, 0, 2, 3)),
        np.transpose(t, (0, 1, 3, 2)),
        np.transpose(t, (1, 0, 3, 2)),
        np.transpose(t, (2, 3, 0, 1)),
        np.transpose(t, (3, 2, 0, 1)),
        np.transpose(t, (2, 3, 1, 0)),
        np.transpose(t, (3, 2, 1, 0)),
    ]
    sym = sum(parts) / len(parts)
    return np.real_if_close(sym, tol=1e-8)


def _perturbation_error(
    time: float, energy: float, psi0: np.ndarray, psi_t: np.ndarray
) -> float:
    if time == 0.0:
        return 0.0
    phase_factor = np.exp(-1j * energy * time)
    delta_state = psi_t - phase_factor * psi0
    denom = time * np.sin(energy * time)
    if abs(denom) < 1e-12:
        denom = energy * (time**2)
    if denom == 0.0:
        return 0.0
    delta_e = np.vdot(psi0, delta_state).real / denom
    return float(abs(delta_e))


def _df_model_diagnostics(model: DFModel) -> dict[str, float]:
    g_nonherm = [
        float(np.linalg.norm(g_mat - g_mat.conj().T)) for g_mat in model.G_list
    ]
    g_norms = [float(np.linalg.norm(g_mat)) for g_mat in model.G_list]
def _apply_d_block(qc: QuantumCircuit, block: Block, tau: float) -> None:
    if block.kind == "one_body_gaussian":
        apply_D_one_body(qc, block.payload.eps, tau)
        return
    if block.kind == "df":
        apply_D_squared(qc, block.payload.eta, block.payload.lam, tau)
        return


def _build_d_only_cost_circuit(
    blocks: Sequence[Block],
    time: float,
    *,
    num_qubits: int,
    pf_label: PFLabel,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    weights = _get_w_list(pf_label)
    for term_idx, weight in iter_pf_steps(len(blocks), weights):
        _apply_d_block(qc, blocks[term_idx], weight * time)
    return qc


def _build_d_block_circuit(
    block: Block,
    tau: float,
    *,
    num_qubits: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    _apply_d_block(qc, block, tau)
    return qc


def _compute_df_rz_costs(
    *,
    model: DFModel,
    h_eff: np.ndarray,
    pf_label: PFLabel,
    t_cost: float,
    energy_shift: float,
    debug: bool,
    debug_print: Callable[[str], None],
    cost_basis_gates: Sequence[str] | None,
    cost_decompose_reps: int,
    cost_optimization_level: int,
    trace_u_debug: bool = False,
    ccsd_selection: dict[str, Any] | None = None,
) -> dict[str, object]:
    rz_costs: dict[str, object] = {}
    if ccsd_selection is not None:
        rz_costs["ccsd_rank_selection"] = ccsd_selection

    one_body_block_cost = build_one_body_gaussian_block_givens(h_eff)
    df_blocks_cost = build_df_blocks_givens(model)
    blocks_cost: list[Block] = []
    blocks_cost.append(Block.from_one_body_gaussian(one_body_block_cost))
    blocks_cost.extend(Block.from_df(b) for b in df_blocks_cost)

    if trace_u_debug:
        debug_trace_u_decomposition(
            one_body_block_cost.U_ops,
            "U one_body",
            num_qubits=model.N,
            decompose_reps=cost_decompose_reps,
            basis_gates=cost_basis_gates,
            opt_level=cost_optimization_level,
            debug_print=debug_print,
        )
        for idx, df_block in enumerate(df_blocks_cost):
            debug_trace_u_decomposition(
                df_block.U_ops,
                f"U df[{idx}]",
                num_qubits=model.N,
                decompose_reps=cost_decompose_reps,
                basis_gates=cost_basis_gates,
                opt_level=cost_optimization_level,
                debug_print=debug_print,
            )
        d_one_body_qc = _build_d_block_circuit(
            blocks_cost[0],
            t_cost,
            num_qubits=model.N,
        )
        debug_trace_u_decomposition(
            d_one_body_qc,
            "D one_body",
            decompose_reps=cost_decompose_reps,
            basis_gates=cost_basis_gates,
            opt_level=cost_optimization_level,
            debug_print=debug_print,
        )
        for idx, blk in enumerate(blocks_cost[1:]):
            d_blk_qc = _build_d_block_circuit(
                blk,
                t_cost,
                num_qubits=model.N,
            )
            debug_trace_u_decomposition(
                d_blk_qc,
                f"D df[{idx}]",
                decompose_reps=cost_decompose_reps,
                basis_gates=cost_basis_gates,
                opt_level=cost_optimization_level,
                debug_print=debug_print,
            )

    qc_cost = build_df_trotter_circuit(
        blocks_cost,
        time=t_cost,
        num_qubits=model.N,
        pf_label=pf_label,
        energy_shift=energy_shift,
    )
    if any(inst.operation.name.lower() == "unitary" for inst in qc_cost.data):
        raise RuntimeError(
            "UnitaryGate found in cost circuit; Givens expansion failed."
        )
    rz_costs.update(
        rz_costs_from_circuit(
            qc_cost,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
    )

    d_only_qc = _build_d_only_cost_circuit(
        blocks_cost,
        t_cost,
        num_qubits=model.N,
        pf_label=pf_label,
    )
    if any(inst.operation.name.lower() == "unitary" for inst in d_only_qc.data):
        raise RuntimeError("UnitaryGate found in D-only cost circuit.")
    d_only_cost = nonclifford_rz_costs_from_circuit(
        d_only_qc,
        basis_gates=cost_basis_gates,
        decompose_reps=cost_decompose_reps,
        optimization_level=cost_optimization_level,
    )
    d_only_proxy_cost = d_nonclifford_costs_from_circuit(
        d_only_qc,
        debug=debug,
        debug_print=debug_print,
    )

    u_costs: list[dict[str, object]] = []
    ob_cost = u_nonclifford_costs_from_u_ops(
        one_body_block_cost.U_ops,
        model.N,
        debug=debug,
        debug_print=debug_print,
    )
    ob_ref_cost = rz_costs_from_u_ops(
        one_body_block_cost.U_ops,
        model.N,
        basis_gates=cost_basis_gates,
        decompose_reps=cost_decompose_reps,
        optimization_level=cost_optimization_level,
    )
    ob_cost["u_ref_rz_count"] = int(ob_ref_cost.get("rz_count", 0))
    ob_cost["u_ref_rz_depth"] = int(ob_ref_cost.get("rz_depth", 0))
    ob_cost["label"] = "one_body"
    u_costs.append(ob_cost)

    d_qc = QuantumCircuit(model.N)
    apply_D_one_body(d_qc, one_body_block_cost.eps, 1.0)
    d_cost = nonclifford_rz_costs_from_circuit(
        d_qc,
        basis_gates=cost_basis_gates,
        decompose_reps=cost_decompose_reps,
        optimization_level=cost_optimization_level,
    )
    d_cost_proxy = d_nonclifford_costs_from_circuit(
        d_qc,
        debug=debug,
        debug_print=debug_print,
    )

    d_block_costs: list[dict[str, object]] = []
    d_block_proxy_costs: list[dict[str, object]] = []
    d_cost["label"] = "one_body_D"
    d_block_costs.append(d_cost)
    d_cost_proxy["label"] = "one_body_D_proxy"
    d_block_proxy_costs.append(d_cost_proxy)

    for idx, df_block in enumerate(df_blocks_cost):
        df_cost = u_nonclifford_costs_from_u_ops(
            df_block.U_ops,
            model.N,
            debug=debug,
            debug_print=debug_print,
        )
        df_ref_cost = rz_costs_from_u_ops(
            df_block.U_ops,
            model.N,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
        df_cost["u_ref_rz_count"] = int(df_ref_cost.get("rz_count", 0))
        df_cost["u_ref_rz_depth"] = int(df_ref_cost.get("rz_depth", 0))
        df_cost["label"] = f"df[{idx}]"
        u_costs.append(df_cost)

        d_blk_qc = _build_d_block_circuit(
            blocks_cost[idx + 1],
            t_cost,
            num_qubits=model.N,
        )
        d_blk_cost = nonclifford_rz_costs_from_circuit(
            d_blk_qc,
            basis_gates=cost_basis_gates,
            decompose_reps=cost_decompose_reps,
            optimization_level=cost_optimization_level,
        )
        d_blk_proxy_cost = d_nonclifford_costs_from_circuit(
            d_blk_qc,
            debug=debug,
            debug_print=debug_print,
        )
        d_blk_cost["label"] = f"df[{idx}]_D"
        d_block_costs.append(d_blk_cost)
        d_blk_proxy_cost["label"] = f"df[{idx}]_D_proxy"
        d_block_proxy_costs.append(d_blk_proxy_cost)

    rz_costs["u_costs"] = u_costs
    rz_costs["d_only_costs"] = d_only_cost
    rz_costs["d_block_costs"] = d_block_costs
    rz_costs["d_only_proxy_costs"] = d_only_proxy_cost
    rz_costs["d_block_proxy_costs"] = d_block_proxy_costs

    weights = _get_w_list(pf_label)
    u_total_count = 0
    u_total_depth = 0
    u_total_coloring_depth = 0
    u_total_ref_count = 0
    u_total_ref_depth = 0
    for term_idx, _weight in iter_pf_steps(len(blocks_cost), weights):
        blk = blocks_cost[term_idx]
        if blk.kind == "one_body_gaussian":
            cost = u_costs[0]
        elif blk.kind == "df":
            df_i = term_idx - 1
            cost = u_costs[df_i + 1]
        else:
            continue
        u_total_count += 2 * int(cost.get("u_nonclifford_rz_count", 0))
        u_total_depth += 2 * int(cost.get("u_nonclifford_rz_depth", 0))
        u_total_coloring_depth += 2 * int(
            cost.get("u_nonclifford_z_coloring_depth", 0)
        )
        u_total_ref_count += 2 * int(cost.get("u_ref_rz_count", 0))
        u_total_ref_depth += 2 * int(cost.get("u_ref_rz_depth", 0))

    d_total_count = int(d_only_cost.get("nonclifford_rz_count", 0))
    d_total_depth = int(d_only_cost.get("nonclifford_rz_depth", 0))
    d_total_ref_count = int(d_only_cost.get("rz_count", 0))
    d_total_ref_depth = int(d_only_cost.get("rz_depth", 0))
    rz_costs["nonclifford_total"] = {
        "u_nonclifford_rz_count": u_total_count,
        "u_nonclifford_rz_depth": u_total_depth,
        "d_nonclifford_rz_count": d_total_count,
        "d_nonclifford_rz_depth": d_total_depth,
        "total_nonclifford_rz_count": u_total_count + d_total_count,
        "total_nonclifford_rz_depth": u_total_depth + d_total_depth,
    }
    rz_costs["rz_total_ud"] = {
        "u_rz_count": u_total_ref_count,
        "u_rz_depth": u_total_ref_depth,
        "d_rz_count": d_total_ref_count,
        "d_rz_depth": d_total_ref_depth,
        "total_rz_count": u_total_ref_count + d_total_ref_count,
        "total_rz_depth": u_total_ref_depth + d_total_ref_depth,
    }

    d_proxy_count = int(d_only_proxy_cost.get("nonclifford_total", 0))
    d_proxy_depth = int(d_only_proxy_cost.get("combined_nonclifford_depth", 0))
    d_total_coloring_depth = 0
    for term_idx, _weight in iter_pf_steps(len(blocks_cost), weights):
        blk = blocks_cost[term_idx]
        if blk.kind == "one_body_gaussian":
            d_cost_term = d_block_proxy_costs[0]
        elif blk.kind == "df":
            df_i = term_idx - 1
            d_cost_term = d_block_proxy_costs[df_i + 1]
        else:
            continue
        d_total_coloring_depth += int(
            d_cost_term.get("coloring_nonclifford_depth", 0)
        )
    rz_costs["toffoli_proxy_total"] = {
        "u_nonclifford_z_count": u_total_count,
        "u_nonclifford_z_depth": u_total_depth,
        "d_nonclifford_z_count": d_proxy_count,
        "d_nonclifford_z_depth": d_proxy_depth,
        "total_nonclifford_z_count": u_total_count + d_proxy_count,
        "total_nonclifford_z_depth": u_total_depth + d_proxy_depth,
    }
    rz_costs["toffoli_proxy_total_coloring"] = {
        "u_nonclifford_z_count": u_total_count,
        "u_nonclifford_z_depth": u_total_coloring_depth,
        "d_nonclifford_z_count": d_proxy_count,
        "d_nonclifford_z_depth": d_total_coloring_depth,
        "total_nonclifford_z_count": u_total_count + d_proxy_count,
        "total_nonclifford_z_depth": u_total_coloring_depth + d_total_coloring_depth,
    }

    return rz_costs

def df_trotter_ud_rz_layer_counts(
    *,
    molecule_type: int = 2,
    pf_label: PFLabel = "2nd",
    time: float | None = None,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
    cost_basis_gates: Sequence[str] | None = None,
    cost_decompose_reps: int = 8,
    cost_optimization_level: int = 0,
    trace_u_debug: bool = False,
    ccsd_target_error_ha: float | None = None,
    ccsd_thresh_range: Sequence[float] | None = None,
    ccsd_use_kernel: bool = False,
    ccsd_no_triples: bool = False,
    save_rz_layers: bool = False,
) -> dict[str, object]:
    """Count U/D RZ-layer costs without running statevector evolution."""
    pf_label = normalize_pf_label(pf_label)
    if time is None:
        time = _default_df_time_or_raise(
            molecule_type=int(molecule_type),
            pf_label=pf_label,
        )
        if debug:
            debug_print(
                "time from config: "
                f"molecule_type={int(molecule_type)} "
                f"pf_label={pf_label} "
                f"time={float(time):.6f}"
            )
    if time <= 0:
        raise ValueError("time must be positive.")

    if rank is None and rank_fraction is None and ccsd_target_error_ha is None:
        config_rank_fraction = get_df_rank_fraction_for_molecule(int(molecule_type))
        if config_rank_fraction is not None:
            rank_fraction = float(config_rank_fraction)
            if debug:
                debug_print(
                    "rank_fraction from config: "
                    f"molecule_type={int(molecule_type)} "
                    f"rank_fraction={rank_fraction:.6f}"
                )

    if ccsd_target_error_ha is not None:
        if rank is not None or rank_fraction is not None:
            raise ValueError(
                "ccsd_target_error_ha cannot be combined with rank or rank_fraction."
            )
        if ccsd_target_error_ha <= 0:
            raise ValueError("ccsd_target_error_ha must be positive.")
        if distance is not None or basis is not None:
            raise ValueError(
                "ccsd_target_error_ha currently requires distance=None and basis=None."
            )

    ccsd_selection: dict[str, Any] | None = None
    if ccsd_target_error_ha is not None:
        rank, selected_fraction, ccsd_selection = _select_rank_from_ccsd_target(
            molecule_type,
            target_error_ha=float(ccsd_target_error_ha),
            thresh_range=ccsd_thresh_range,
            use_kernel=ccsd_use_kernel,
            no_triples=ccsd_no_triples,
            record_in_config=True,
        )
        _, _, constant, one_body, two_body = _run_scf_and_integrals(molecule_type)
        if debug:
            debug_print(
                "ccsd rank selection: "
                f"target={float(ccsd_target_error_ha):.6e}Ha "
                f"selected_rank={rank} "
                f"selected_fraction={selected_fraction:.6f} "
                f"abs_error={float(ccsd_selection['selected_abs_ccsd_error_ha']):.6e}Ha "
                f"threshold={float(ccsd_selection['selected_threshold']):.6e} "
                f"target_met={bool(ccsd_selection['target_met'])} "
                f"scan={int(ccsd_selection.get('thresholds_evaluated', 0))}/"
                f"{int(ccsd_selection.get('thresholds_total', 0))} "
                f"stopped_early={bool(ccsd_selection.get('stopped_early', False))}"
            )
    else:
        constant, one_body, two_body = _h_chain_integrals(
            molecule_type,
            distance=distance,
            basis=basis,
        )

    if ccsd_target_error_ha is None and rank_fraction is not None:
        if rank is not None:
            raise ValueError("rank and rank_fraction are mutually exclusive.")
        if rank_fraction <= 0:
            raise ValueError("rank_fraction must be positive.")
        n_spatial = int(one_body.shape[0])
        full_rank = int(n_spatial**2)
        if rank_fraction >= 1.0:
            rank = full_rank
            if tol is None:
                tol = 0.0
        else:
            rank = int(round(full_rank * rank_fraction))
            rank = max(1, min(rank, full_rank))
    elif rank_fraction is not None:
        raise ValueError("rank_fraction cannot be set when ccsd_target_error_ha is used.")

    two_body = _symmetrize_two_body(two_body)
    one_body_spin, _ = spinorb_from_spatial(one_body, two_body * 0.5)
    raw_model = df_decompose_from_integrals(
        one_body,
        two_body,
        constant=constant,
        rank=rank,
        tol=tol,
    )
    model = raw_model.hermitize()
    h_eff = one_body_spin + model.one_body_correction

    costs = _compute_df_rz_costs(
        model=model,
        h_eff=h_eff,
        pf_label=pf_label,
        t_cost=float(time),
        energy_shift=constant + model.constant_correction,
        debug=debug,
        debug_print=debug_print,
        cost_basis_gates=cost_basis_gates,
        cost_decompose_reps=cost_decompose_reps,
        cost_optimization_level=cost_optimization_level,
        trace_u_debug=trace_u_debug,
        ccsd_selection=ccsd_selection,
    )
    metrics = _collect_df_ud_rz_layer_metrics(costs)
    metrics["molecule_type"] = int(molecule_type)
    metrics["num_qubits"] = int(model.N)
    metrics["pf_label"] = str(pf_label)
    metrics["time"] = float(time)
    if ccsd_selection is not None:
        metrics["ccsd_rank_selection"] = ccsd_selection
    if save_rz_layers:
        artifact_name = (
            f"{_artifact_ham_name(molecule_type, distance=distance, basis=basis)}"
            f"_Operator_{pf_label}"
        )
        _save_df_rz_layer_artifact(artifact_name, metrics)
        metrics["artifact_name"] = artifact_name
    return metrics


def save_df_ud_rz_layer_sweep(
    molecule_types: Sequence[int],
    *,
    pf_label: PFLabel = "2nd",
    time: float | None = None,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
    cost_basis_gates: Sequence[str] | None = None,
    cost_decompose_reps: int = 8,
    cost_optimization_level: int = 0,
    trace_u_debug: bool = False,
    ccsd_target_error_ha: float | None = None,
    ccsd_thresh_range: Sequence[float] | None = None,
    ccsd_use_kernel: bool = False,
    ccsd_no_triples: bool = False,
) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for molecule_type in molecule_types:
        out[int(molecule_type)] = df_trotter_ud_rz_layer_counts(
            molecule_type=int(molecule_type),
            pf_label=pf_label,
            time=time,
            rank=rank,
            rank_fraction=rank_fraction,
            tol=tol,
            distance=distance,
            basis=basis,
            debug=debug,
            debug_print=debug_print,
            cost_basis_gates=cost_basis_gates,
            cost_decompose_reps=cost_decompose_reps,
            cost_optimization_level=cost_optimization_level,
            trace_u_debug=trace_u_debug,
            ccsd_target_error_ha=ccsd_target_error_ha,
            ccsd_thresh_range=ccsd_thresh_range,
            ccsd_use_kernel=ccsd_use_kernel,
            ccsd_no_triples=ccsd_no_triples,
            save_rz_layers=True,
        )
    return out


def plot_df_u_rz_depth_vs_num_qubits(
    molecule_types: Sequence[int],
    *,
    pf_label: PFLabel = "2nd",
    distance: float | None = None,
    basis: str | None = None,
    include_pf_rz_layer: bool = True,
    use_nonclifford_rz_depth: bool = False,
    use_nonclifford_z_coloring_depth: bool = False,
    show_d_rz_depth: bool = True,
    show_total_rz_depth: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
    debug_print: Callable[[str], None] = print,
) -> dict[str, list[float]]:
    pf_label = normalize_pf_label(pf_label)
    xs: list[float] = []
    ys_u: list[float] = []
    ys_d: list[float] = []
    ys_total: list[float] = []
    ys_pf: list[float] = []
    xs_pf: list[float] = []
    missing: list[int] = []

    if use_nonclifford_rz_depth and use_nonclifford_z_coloring_depth:
        raise ValueError(
            "use_nonclifford_rz_depth and use_nonclifford_z_coloring_depth "
            "cannot both be True."
        )

    if use_nonclifford_z_coloring_depth:
        u_key = "u_nonclifford_z_coloring_depth"
        d_key = "d_nonclifford_z_coloring_depth"
        total_key = "total_nonclifford_z_coloring_depth"
        label_u = "DF u_nonclifford_z_coloring_depth"
        label_d = "DF d_nonclifford_z_coloring_depth"
        label_total = "DF total_nonclifford_z_coloring_depth"
    elif use_nonclifford_rz_depth:
        u_key = "u_nonclifford_rz_depth"
        d_key = "d_nonclifford_rz_depth"
        total_key = "total_nonclifford_rz_depth"
        label_u = "DF u_nonclifford_rz_depth"
        label_d = "DF d_nonclifford_rz_depth"
        label_total = "DF total_nonclifford_rz_depth"
    else:
        u_key = "u_rz_depth"
        d_key = "d_rz_depth"
        total_key = "total_rz_depth"
        label_u = "DF u_rz_depth"
        label_d = "DF d_rz_depth"
        label_total = "DF total_rz_depth"

    for molecule_type in sorted({int(m) for m in molecule_types}):
        artifact_name = (
            f"{_artifact_ham_name(molecule_type, distance=distance, basis=basis)}"
            f"_Operator_{pf_label}"
        )
        data = _load_df_rz_layer_artifact(artifact_name)
        if data is None:
            missing.append(int(molecule_type))
            continue
        x = float(data.get("num_qubits", int(molecule_type) * 2))
        u_depth = data.get(u_key)
        d_depth = data.get(d_key)
        total_depth = data.get(total_key)
        if u_depth is None or d_depth is None or total_depth is None:
            missing.append(int(molecule_type))
            continue
        xs.append(x)
        ys_u.append(float(u_depth))
        ys_d.append(float(d_depth))
        ys_total.append(float(total_depth))

        if include_pf_rz_layer:
            pf_depth = PF_RZ_LAYER.get(f"H{int(molecule_type)}", {}).get(pf_label)
            if pf_depth is not None:
                xs_pf.append(x)
                ys_pf.append(float(pf_depth))

    if missing:
        raise FileNotFoundError(
            "Missing df_rz_layer artifacts or required keys for molecule_type="
            f"{sorted(set(missing))}. "
            "Run save_df_ud_rz_layer_sweep(...) or "
            "df_trotter_ud_rz_layer_counts(..., save_rz_layers=True) first."
        )

    order = np.argsort(np.asarray(xs))
    xs = [xs[i] for i in order]
    ys_u = [ys_u[i] for i in order]
    ys_d = [ys_d[i] for i in order]
    ys_total = [ys_total[i] for i in order]

    fig, ax = plt.subplots()
    ax.plot(xs, ys_u, marker="o", linestyle="-", label=label_u)
    if show_d_rz_depth:
        ax.plot(xs, ys_d, marker="d", linestyle="-", label=label_d)
    if show_total_rz_depth:
        ax.plot(xs, ys_total, marker="s", linestyle="--", label=label_total)
    if include_pf_rz_layer and xs_pf:
        order_pf = np.argsort(np.asarray(xs_pf))
        xs_pf = [xs_pf[i] for i in order_pf]
        ys_pf = [ys_pf[i] for i in order_pf]
        ax.plot(
            xs_pf,
            ys_pf,
            marker="^",
            linestyle="-.",
            label="GR_RZ_LAYER",
        )

    set_loglog_axes(
        ax,
        title=f"DF RZ depth vs qubits ({pf_label})",
    )
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Num RZ layer", fontsize=15)
    ax.yaxis.grid(True, which="both", alpha=0.3)
    ax.xaxis.grid(False, which="both")
    ax.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        debug_print(f"saved plot: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "num_qubits": xs,
        "u_rz_depth": ys_u,
        "d_rz_depth": ys_d,
        "total_rz_depth": ys_total,
        "pf_rz_depth": ys_pf,
    }
def _parse_molecule_type(mol_type: int | str) -> int:
    if isinstance(mol_type, int):
        return int(mol_type)
    text = str(mol_type).strip()
    if text.lower().startswith("h"):
        text = text[1:]
    return int(text)


def _build_grouped_qubit_ops(
    molecule_type: int,
    constant: float,
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    n_orb: int,
) -> List[QubitOperator]:
    if n_orb <= 3:
        interaction = make_spinorb_ham_upthendown_order(
            constant,
            one_body_integrals,
            two_body_integrals,
            validation=True,
        )
        ham_ferm = get_fermion_operator(interaction)
        ham_qubit: QubitOperator = jordan_wigner(ham_ferm)
        grouped_ops, _ = min_hamiltonian_grouper(
            ham_qubit,
            ham_name=f"H{int(molecule_type)}",
        )
        return grouped_ops

    grouper = Almost_optimal_grouper(
        constant,
        one_body_integrals,
        two_body_integrals,
        fermion_qubit_mapping=jordan_wigner,
        validation=False,
    )
    return [jordan_wigner(sum(group_terms)) for group_terms in grouper.group_term_list]


def extract_z_like_terms_from_qubit_group(
    q_group: QubitOperator,
    coeff_tol: float = 0.0,
) -> Dict[FrozenSet[int], complex]:
    """QubitOperator グループから Z 相当の support->coeff を抽出する。"""
    z_terms: Dict[FrozenSet[int], complex] = {}
    for term, coeff in q_group.terms.items():
        if abs(coeff) <= coeff_tol:
            continue
        support = frozenset(int(q) for q, _p in term)
        if not support:
            continue
        z_terms[support] = z_terms.get(support, 0.0 + 0.0j) + complex(coeff)

    return {
        supp: coeff
        for supp, coeff in z_terms.items()
        if abs(coeff) > coeff_tol
    }


def greedy_layering(
    supports: Sequence[FrozenSet[int]],
) -> List[List[FrozenSet[int]]]:
    """Disjoint support を同一レイヤーに詰める greedy 彩色。"""
    layers: List[List[FrozenSet[int]]] = []
    used_sets: List[set[int]] = []
    for supp in supports:
        placed = False
        for idx, used in enumerate(used_sets):
            if used.intersection(supp):
                continue
            layers[idx].append(supp)
            used.update(supp)
            placed = True
            break
        if not placed:
            layers.append([supp])
            used_sets.append(set(supp))
    return layers


def _supports_to_bitmasks(
    supports: Sequence[FrozenSet[int]],
    n_qubits: int,
) -> List[int]:
    out: List[int] = []
    for supp in supports:
        mask = 0
        for q in supp:
            if q < 0 or q >= n_qubits:
                raise ValueError(f"support index out of range: q={q}, n_qubits={n_qubits}")
            mask |= 1 << int(q)
        out.append(mask)
    return out


def _build_basis_and_coeffs(term_masks: Sequence[int]) -> Tuple[List[int], List[int]]:
    if not term_masks:
        return [], []

    basis: List[int] = []
    for val in term_masks:
        if val == 0:
            continue
        current = int(val)
        for b_val in basis:
            msb_b = b_val.bit_length() - 1
            if ((current >> msb_b) & 1) == 1:
                current ^= b_val
        if current != 0:
            basis.append(current)
            basis.sort(key=lambda x: x.bit_length(), reverse=True)

    r = len(basis)
    if r == 0:
        return [], [0] * len(term_masks)

    basis_rref = basis[:]
    coeff_map = [1 << i for i in range(r)]

    for i in range(r):
        msb = basis_rref[i].bit_length() - 1
        for j in range(i + 1, r):
            if ((basis_rref[j] >> msb) & 1) == 1:
                basis_rref[j] ^= basis_rref[i]
                coeff_map[j] ^= coeff_map[i]

    basis_msb = [(b, b.bit_length() - 1, coeff_map[i]) for i, b in enumerate(basis_rref)]
    basis_msb.sort(key=lambda x: x[1], reverse=True)

    coeffs: List[int] = []
    for val in term_masks:
        c_val = 0
        curr = int(val)
        for b, msb, c_mask in basis_msb:
            if ((curr >> msb) & 1) == 1:
                curr ^= b
                c_val ^= c_mask
        coeffs.append(c_val)

    return basis, coeffs


def _estimate_t_depth_greedy(coeffs: Sequence[int]) -> int:
    active = [int(c) for c in coeffs if int(c) != 0]
    n = len(active)
    if n == 0:
        return 0

    degrees = [0] * n
    for i in range(n):
        ci = active[i]
        d = 0
        for j in range(i + 1, n):
            if ci & active[j]:
                d += 1
                degrees[j] += 1
        degrees[i] += d

    sorted_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    colors: Dict[int, int] = {}
    for idx in sorted_indices:
        c_val = active[idx]
        used_colors = set()
        for other_idx, color in colors.items():
            if c_val & active[other_idx]:
                used_colors.add(color)
        color = 0
        while color in used_colors:
            color += 1
        colors[idx] = color
    return max(colors.values()) + 1 if colors else 0


def _optimize_coeffs(
    coeffs: Sequence[int],
    r: int,
    n_iter: int = 2000,
) -> Tuple[List[int], int]:
    best_coeffs = [int(c) for c in coeffs]
    best_cost = _estimate_t_depth_greedy(best_coeffs)

    if r <= 1:
        return best_coeffs, best_cost

    rng = random.Random(0)
    current_coeffs = best_coeffs[:]
    for _ in range(int(max(0, n_iter))):
        a = rng.randint(0, r - 1)
        b = rng.randint(0, r - 2)
        if b >= a:
            b += 1

        mask_a = 1 << a
        mask_b = 1 << b
        next_coeffs = [(c ^ mask_b) if (c & mask_a) else c for c in current_coeffs]
        new_cost = _estimate_t_depth_greedy(next_coeffs)
        if new_cost < best_cost:
            best_cost = new_cost
            best_coeffs = next_coeffs
            current_coeffs = next_coeffs

    return best_coeffs, best_cost


def bitwise_optimize_z_terms(
    z_terms: Mapping[FrozenSet[int], complex],
    *,
    n_qubits: int,
    optimize_iters: int = 2000,
) -> Tuple[int, List[int]]:
    """Z support 辞書から Bitwise 近似 T-depth を推定する。"""
    supports = [supp for supp, coeff in z_terms.items() if abs(coeff) > 0]
    term_masks = _supports_to_bitmasks(supports, n_qubits=n_qubits)
    basis, coeffs = _build_basis_and_coeffs(term_masks)
    opt_coeffs, cost = _optimize_coeffs(coeffs, len(basis), n_iter=optimize_iters)
    return int(cost), opt_coeffs


def estimate_rz_layers_from_grouping(
    mol_type: int | str,
    bit_wise: bool = False,
    coeff_tol: float = 0.0,
    bitwise_iters: int = 2000,
):
    """グルーピング済みハミルトニアンから RZ レイヤー数を推定する。

    Returns:
      bit_wise=False:
        n_layers_list, layers_list, z_terms_list
      bit_wise=True:
        n_layers_list, layers_list, z_terms_list, bitwise_T_depth_list
    """
    molecule_type = _parse_molecule_type(mol_type)
    _mol, mf, constant, one_body_integrals, two_body_integrals = _run_scf_and_integrals(
        molecule_type
    )
    n_orb = int(mf.mo_coeff.shape[0])
    n_qubits = 2 * n_orb

    grouped_qubit_ops = _build_grouped_qubit_ops(
        molecule_type=molecule_type,
        constant=float(constant),
        one_body_integrals=np.asarray(one_body_integrals),
        two_body_integrals=np.asarray(two_body_integrals),
        n_orb=n_orb,
    )

    n_layers_list: List[int] = []
    layers_list: List[List[List[FrozenSet[int]]]] = []
    z_terms_list: List[Dict[FrozenSet[int], complex]] = []
    bitwise_t_depth_list: List[int] = []

    for q_group in grouped_qubit_ops:
        z_terms_g = extract_z_like_terms_from_qubit_group(
            q_group,
            coeff_tol=coeff_tol,
        )
        supports_g = list(z_terms_g.keys())
        layers_all = greedy_layering(supports_g)
        layers_nonzero = [
            layer
            for layer in layers_all
            if any(abs(z_terms_g[supp]) > coeff_tol for supp in layer)
        ]
        n_layers_g = len(layers_nonzero)

        n_layers_list.append(n_layers_g)
        layers_list.append(layers_nonzero)
        z_terms_list.append(z_terms_g)

        if bit_wise:
            td_g, _ = bitwise_optimize_z_terms(
                z_terms_g,
                n_qubits=n_qubits,
                optimize_iters=bitwise_iters,
            )
            # 既存の notebook 実装に合わせ、大きい系では greedy 深さとの最小値を採用。
            if n_orb > 3:
                td_g = min(n_layers_g, td_g)
            bitwise_t_depth_list.append(int(td_g))

    if bit_wise:
        return n_layers_list, layers_list, z_terms_list, bitwise_t_depth_list
    return n_layers_list, layers_list, z_terms_list

DF_LAYER_COUNT_FUNCTIONS: dict[str, Callable[..., object]] = {
    "df_trotter_ud_rz_layer_counts": df_trotter_ud_rz_layer_counts,
    "save_df_ud_rz_layer_sweep": save_df_ud_rz_layer_sweep,
    "plot_df_u_rz_depth_vs_num_qubits": plot_df_u_rz_depth_vs_num_qubits,
    "rz_costs_from_circuit": rz_costs_from_circuit,
    "rz_costs_from_u_ops": rz_costs_from_u_ops,
    "nonclifford_rz_costs_from_circuit": nonclifford_rz_costs_from_circuit,
    "d_nonclifford_costs_from_circuit": d_nonclifford_costs_from_circuit,
    "u_nonclifford_costs_from_u_ops": u_nonclifford_costs_from_u_ops,
}

GR_LAYER_COUNT_FUNCTIONS: dict[str, Callable[..., object]] = {
    "estimate_rz_layers_from_grouping": estimate_rz_layers_from_grouping,
    "extract_z_like_terms_from_qubit_group": extract_z_like_terms_from_qubit_group,
    "greedy_layering": greedy_layering,
    "bitwise_optimize_z_terms": bitwise_optimize_z_terms,
}

ALL_LAYER_COUNT_FUNCTIONS: dict[str, Callable[..., object]] = {
    **DF_LAYER_COUNT_FUNCTIONS,
    **GR_LAYER_COUNT_FUNCTIONS,
}

__all__ = [
    "df_trotter_ud_rz_layer_counts",
    "save_df_ud_rz_layer_sweep",
    "plot_df_u_rz_depth_vs_num_qubits",
    "rz_costs_from_circuit",
    "rz_costs_from_u_ops",
    "nonclifford_rz_costs_from_circuit",
    "d_nonclifford_costs_from_circuit",
    "u_nonclifford_costs_from_u_ops",
    "estimate_rz_layers_from_grouping",
    "extract_z_like_terms_from_qubit_group",
    "greedy_layering",
    "bitwise_optimize_z_terms",
    "DF_LAYER_COUNT_FUNCTIONS",
    "GR_LAYER_COUNT_FUNCTIONS",
    "ALL_LAYER_COUNT_FUNCTIONS",
]
