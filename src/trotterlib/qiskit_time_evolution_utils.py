from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import SparsePauliOp, Statevector

from .config import PFLabel
from .product_formula import _get_w_list as _pf_get_w_list


def free_var(name: str, scope: dict) -> None:
    """ローカル変数を解放して GC を促す（メモリ圧を下げるための補助）。"""
    # スコープから削除して GC を促進
    if name in scope:
        del scope[name]
        import gc

        gc.collect()


def apply_time_evolution(
    eigenvector: np.ndarray, time_evolution_circuit: QuantumCircuit
) -> Statevector:
    """|ψ⟩ を初期化せず、Statevector.evolve で時間発展させた最終状態を返す。"""
    # Statevector を直接 evolve させる
    sv = Statevector(eigenvector)
    final_sv = sv.evolve(time_evolution_circuit)
    return final_sv


def term_to_sparse_pauli(
    term: Tuple[Tuple[int, str], ...],
    n_qubits: int,
) -> SparsePauliOp:
    """OpenFermion term を Qiskit の SparsePauliOp に変換する。"""
    # 注意: 右端が q0（Qiskit）。既存挙動維持のためラベル反転は行わない。
    X = SparsePauliOp("X")
    Y = SparsePauliOp("Y")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")
    pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
    pauli_operators = [I] * n_qubits
    for index, pauli_op_name in term:
        pauli_operators[index] = pauli_dict[pauli_op_name]
    pauli_op = pauli_operators[0]
    for op in pauli_operators[1:]:
        pauli_op ^= op
    return pauli_op


def _get_w_list(num_w: PFLabel) -> List[float]:
    """積公式パラメータ w の系列を取得（分岐を関数化）。"""
    # product_formula 側の実装を使用
    return _pf_get_w_list(num_w)


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
