from __future__ import annotations

import os
from typing import List, Optional, Tuple, Any

import numpy as np  # type: ignore
from openfermion import count_qubits  # type: ignore
from openfermion.chem import MolecularData  # type: ignore
from openfermion.linalg import get_sparse_operator  # type: ignore
from openfermion.ops import QubitOperator  # type: ignore
from openfermion.transforms import get_fermion_operator, jordan_wigner  # type: ignore
from openfermionpyscf import run_pyscf  # type: ignore
from scipy.sparse.linalg import eigsh  # type: ignore

from config import DEFAULT_BASIS, DEFAULT_DISTANCE


def call_geometry(Hchain, distance):
    # multiplicity と charge の規則
    if Hchain % 2 == 0:
        multiplicity = 1
        charge = 0
    else:
        multiplicity = 3
        charge = +1

    shift = (Hchain - 1) / 2.0

    geometry = [("H", (0.0, 0.0, distance * (i - shift))) for i in range(Hchain)]

    return geometry, multiplicity, charge


def geo(mol_type: str, distance: Optional[float] = None):
    if distance is None:
        distance = DEFAULT_DISTANCE
    return call_geometry(mol_type, distance)

def ham_list_maker(hamiltonian) -> List[QubitOperator]:
    """OpenFermion ハミルトニアンを項ごとのリストに変換する。"""
    return [term for term in hamiltonian]


def ham_ground_energy(jw_hamiltonian):
    """JW ハミルトニアンの基底エネルギー・固有ベクトル・最大固有値を返す。"""
    sum_matrix = get_sparse_operator(jw_hamiltonian)
    vals, vecs = eigsh(sum_matrix, k=1, return_eigenvectors=True, which="SA")
    max_eig = eigsh(sum_matrix, k=1, return_eigenvectors=False, which="LA")
    return vals[0], vecs, max_eig[0]


def jw_hamiltonian_maker(mol_type: str, distance: Optional[float] = None):
    """
    JW 変換ハミルトニアンを構築して返す。

    Returns:
        jw_hamiltonian, HF_energy, ham_name, num_qubits
    """
    basis = DEFAULT_BASIS
    if distance is None:
        distance = DEFAULT_DISTANCE
    geometry, multiplicity, charge = geo(mol_type, distance)
    name_distance = int(distance * 100)
    description = f"distance_{name_distance}_charge_{charge}"  # 保存先のファイル名
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    hf_energy = molecule.hf_energy
    jw_hamiltonian = jordan_wigner(
        get_fermion_operator(molecule.get_molecular_hamiltonian())
    )
    num_qubits = count_qubits(jw_hamiltonian)
    file_path = molecule.filename
    ham_name = os.path.splitext(os.path.basename(file_path))[0]
    print(ham_name)
    return jw_hamiltonian, hf_energy, ham_name, num_qubits


def min_hamiltonian_grouper(hamiltonian: QubitOperator, ham_name: str):
    """可換な項でグルーピングし、グループごとの QubitOperator のリストを返す。"""

    def are_commuting(op1: QubitOperator, op2: QubitOperator) -> bool:
        if len(op1.terms) != 1 or len(op2.terms) != 1:
            raise ValueError("Only single-term QubitOperators are supported.")
        term1 = list(op1.terms.keys())[0]
        term2 = list(op2.terms.keys())[0]
        n_anticommute = 0
        qubits = set(index for index, _ in term1).union(index for index, _ in term2)
        for q in qubits:
            op1_pauli = dict(term1).get(q, "I")
            op2_pauli = dict(term2).get(q, "I")
            if op1_pauli == "I" or op2_pauli == "I":
                continue
            if op1_pauli != op2_pauli:
                n_anticommute += 1
        return n_anticommute % 2 == 0

    def group_commuting_terms(qubit_hamiltonian: QubitOperator) -> List[QubitOperator]:
        terms = [
            QubitOperator(term, coeff)
            for term, coeff in qubit_hamiltonian.terms.items()
        ]
        groups: List[List[QubitOperator]] = []
        for term in terms:
            for group in groups:
                if all(are_commuting(term, other) for other in group):
                    group.append(term)
                    break
            else:
                groups.append([term])
        return [sum(group, QubitOperator()) for group in groups]

    grouped_ops = group_commuting_terms(hamiltonian)
    return grouped_ops, f"{ham_name}_grouping"
