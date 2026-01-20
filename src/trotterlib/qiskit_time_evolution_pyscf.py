from functools import reduce
from typing import List, Tuple

import numpy as np

import pyscf
from pyscf import fci, gto, scf
from pyscf.fci import cistring

from openfermion import InteractionOperator
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.chem.molecular_data import spinorb_from_spatial

from .Almost_optimal_grouping import Almost_optimal_grouper

from .chemistry_hamiltonian import geo

DEFAULT_BASIS = "sto-3g"  # PySCF 基底関数


def make_fci_vector_from_pyscf_solver_grouper(
    molecule_type: int,
) -> Tuple[List[List[QubitOperator]], int, float, np.ndarray]:
    """PySCF FCI から |ψ₀⟩ を構築し、グルーピング済み JW 演算子群とともに返す。"""
    # 分子情報を構築して SCF を実行
    geometry, multiplicity, molcharge = geo(molecule_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = DEFAULT_BASIS
    mol.spin = multiplicity - 1
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    # 積分（1体・2体）を構築
    constant = mf.energy_nuc()
    mo_coeff = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body_integrals = reduce(np.dot, (mo_coeff.T, h_core, mo_coeff))

    eri_mo = pyscf.ao2mo.kernel(mf.mol, mo_coeff)
    eri_mo = pyscf.ao2mo.restore(1, eri_mo, mo_coeff.shape[0])
    two_body_integrals = np.asarray(eri_mo.transpose(0, 2, 3, 1), order="C")

    # 近似グルーピングで JW 演算子を作成
    almost_optimal_grouper = Almost_optimal_grouper(
        constant, one_body_integrals, two_body_integrals, fermion_qubit_mapping=jordan_wigner, validation=True
    )
    grouping_term_list = almost_optimal_grouper.group_term_list
    # 定数項をグループ先頭へ（OpenFermion FermionOperator の空文字は恒等項）
    grouping_term_list[0].insert(0, FermionOperator("", almost_optimal_grouper._const_fermion))
    grouped_jw_list: List[List[QubitOperator]] = [
        jordan_wigner(sum(group_term)) for group_term in grouping_term_list
    ]

    # FCI を解いて基底状態を構築
    fci_solver = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = fci_solver.kernel()
    n_qubits = fci_solver.norb * 2
    n_orbitals = fci_solver.norb
    nelec_alpha, nelec_beta = fci_solver.nelec
    fci_vector = np.zeros(2**n_qubits, dtype=np.complex128)

    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta = cistring.make_strings(range(n_orbitals), nelec_beta)

    # 注意: 右が q0 となる Qiskit のビット順に合わせるにはビット反転が必要だが、
    # 既存コードは **反転しない** 前提で構築しているため、その挙動を維持する。
    # CI 係数をビット列へ展開
    for i, a_str in enumerate(ci_strings_alpha):
        alpha_index = list(format(a_str, f"0{n_qubits // 2}b"))[::-1]
        for j, b_str in enumerate(ci_strings_beta):
            beta_index = list(format(b_str, f"0{n_qubits // 2}b"))[::-1]
            bitstring = "".join(alpha_index) + "".join(beta_index)
            sign = 1
            N = len(alpha_index)
            for k in range(N):
                if alpha_index[k] == "1":
                    for l in range(N):
                        if beta_index[l] == "1":
                            sign *= -1
            index = int(bitstring, 2)
            fci_vector[index] = sign * ci_matrix[i][j]

    state_vec = fci_vector.reshape(-1, 1)
    return grouped_jw_list, n_qubits, energy, state_vec


def make_fci_vector_from_pyscf_solver(
    molecule_type: int,
) -> Tuple[QubitOperator, int, float, np.ndarray, np.ndarray]:
    """PySCFからFCIベクトルとJWハミルトニアンを生成（元コードと同一の並びと位相）。"""
    # --- Geometry / SCF ---
    # 分子情報を構築して SCF を実行
    geometry, multiplicity, molcharge = geo(molecule_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.spin = multiplicity - 1
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    # --- Molecular integrals ---
    # 積分（1体・2体）を構築
    constant = mf.energy_nuc()
    mo_coeff = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body = reduce(np.dot, (mo_coeff.T, h_core, mo_coeff))
    eri_mo = pyscf.ao2mo.kernel(mf.mol, mo_coeff)
    eri_mo = pyscf.ao2mo.restore(1, eri_mo, mo_coeff.shape[0])
    two_body = np.asarray(eri_mo.transpose(0, 2, 3, 1), order="C")

    # spin-orbital Hamiltonian → Fermion → JW
    # フェルミオン演算子を JW に変換
    h1s, h2s = spinorb_from_spatial(one_body, two_body * 0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(constant, h1s, h2s))
    jw_hamiltonian = jordan_wigner(ham_fermion)

    # --- FCI solve ---
    # FCI を解いて基底状態を構築
    fci_solver = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = fci_solver.kernel()
    num_orbitals = fci_solver.norb
    n_qubits = num_orbitals * 2
    nelec_alpha, nelec_beta = fci_solver.nelec

    # --- CI state on qubits: interleave [beta_k, alpha_k] for k=0.. ---
    state_vector = np.zeros(2 ** n_qubits, dtype=np.complex128)
    ci_strings_alpha = cistring.make_strings(range(num_orbitals), nelec_alpha)
    ci_strings_beta = cistring.make_strings(range(num_orbitals), nelec_beta)

    # CI 係数をビット列へ展開
    for i, a_str in enumerate(ci_strings_alpha):
        a_bits = format(a_str, f"0{n_qubits // 2}b")[::-1]  # LSBが軌道0
        for j, b_str in enumerate(ci_strings_beta):
            b_bits = format(b_str, f"0{n_qubits // 2}b")[::-1]

            # 交互に [β_k, α_k]
            interleaved = []
            for bit_a, bit_b in zip(a_bits, b_bits):
                interleaved.append(bit_b)
                interleaved.append(bit_a)
            bitstring = "".join(interleaved)

            # Jordan–Wigner 位相補正（元コードと同じ規則）
            sign = 1
            N = len(a_bits)
            for k in range(N):
                if a_bits[k] == "1":
                    # 反転後の添字で k より大きい位置が元の「左側」
                    for l in range(k + 1, N):
                        if b_bits[l] == "1":
                            sign *= -1

            index = int(bitstring, 2)
            state_vector[index] = sign * ci_matrix[i][j]

    state_vec = state_vector.reshape(-1, 1)
    return jw_hamiltonian, n_qubits, energy, state_vec, ci_matrix
