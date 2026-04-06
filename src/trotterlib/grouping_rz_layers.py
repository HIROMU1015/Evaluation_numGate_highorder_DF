from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

import numpy as np
from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner

from .Almost_optimal_grouping import Almost_optimal_grouper, make_spinorb_ham_upthendown_order
from .chemistry_hamiltonian import min_hamiltonian_grouper
from .qiskit_time_evolution_pyscf import _run_scf_and_integrals


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

