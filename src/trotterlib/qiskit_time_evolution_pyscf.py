from functools import reduce
from typing import Dict, List, Tuple

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


def _run_scf_and_integrals(
    molecule_type: int,
) -> Tuple[gto.Mole, scf.hf.RHF, float, np.ndarray, np.ndarray]:
    """Run SCF and return (mol, mf, constant, one_body, two_body) integrals."""
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
    constant = mf.energy_nuc()
    mo_coeff = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body_integrals = reduce(np.dot, (mo_coeff.T, h_core, mo_coeff))

    eri_mo = pyscf.ao2mo.kernel(mf.mol, mo_coeff)
    eri_mo = pyscf.ao2mo.restore(1, eri_mo, mo_coeff.shape[0])
    two_body_integrals = np.asarray(eri_mo.transpose(0, 2, 3, 1), order="C")
    return mol, mf, constant, one_body_integrals, two_body_integrals


def _solve_fci(
    mol: gto.Mole, mo_coeff: np.ndarray
) -> Tuple[float, np.ndarray, int, int, int, int]:
    """Solve FCI and return energy, ci_matrix, n_qubits, n_orbitals, nelec_alpha, nelec_beta."""
    fci_solver = fci.FCI(mol, mo_coeff)
    energy, ci_matrix = fci_solver.kernel()
    n_orbitals = fci_solver.norb
    n_qubits = n_orbitals * 2
    nelec_alpha, nelec_beta = fci_solver.nelec
    return energy, ci_matrix, n_qubits, n_orbitals, nelec_alpha, nelec_beta


def _build_fci_vector(
    ci_matrix: np.ndarray,
    n_qubits: int,
    n_orbitals: int,
    nelec_alpha: int,
    nelec_beta: int,
    *,
    mode: str,
) -> np.ndarray:
    """Build FCI statevector with the existing bit ordering."""
    fci_vector = np.zeros(2**n_qubits, dtype=np.complex128)
    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta = cistring.make_strings(range(n_orbitals), nelec_beta)

    if mode == "grouper":
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
        return fci_vector

    if mode == "interleaved":
        for i, a_str in enumerate(ci_strings_alpha):
            a_bits = format(a_str, f"0{n_qubits // 2}b")[::-1]
            for j, b_str in enumerate(ci_strings_beta):
                b_bits = format(b_str, f"0{n_qubits // 2}b")[::-1]

                interleaved = []
                for bit_a, bit_b in zip(a_bits, b_bits):
                    interleaved.append(bit_b)
                    interleaved.append(bit_a)
                bitstring = "".join(interleaved)

                sign = 1
                N = len(a_bits)
                for k in range(N):
                    if a_bits[k] == "1":
                        for l in range(k + 1, N):
                            if b_bits[l] == "1":
                                sign *= -1

                index = int(bitstring, 2)
                fci_vector[index] = sign * ci_matrix[i][j]
        return fci_vector

    raise ValueError(f"Unsupported FCI vector mode: {mode}")


def _build_grouped_jw_list(
    constant: float,
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
) -> List[List[QubitOperator]]:
    """Build grouped JW operators using the existing grouper."""
    almost_optimal_grouper = Almost_optimal_grouper(
        constant, one_body_integrals, two_body_integrals, fermion_qubit_mapping=jordan_wigner, validation=True
    )
    grouping_term_list = almost_optimal_grouper.group_term_list
    grouping_term_list[0].insert(0, FermionOperator("", almost_optimal_grouper._const_fermion))
    grouped_jw_list: List[List[QubitOperator]] = [
        jordan_wigner(sum(group_term)) for group_term in grouping_term_list
    ]
    return grouped_jw_list


def make_fci_vector_from_pyscf_solver_grouper(
    molecule_type: int,
) -> Tuple[List[List[QubitOperator]], int, float, np.ndarray]:
    """PySCF FCI から |ψ₀⟩ を構築し、グルーピング済み JW 演算子群とともに返す。"""
    mol, mf, constant, one_body_integrals, two_body_integrals = _run_scf_and_integrals(
        molecule_type
    )
    grouped_jw_list = _build_grouped_jw_list(
        constant, one_body_integrals, two_body_integrals
    )

    energy, ci_matrix, n_qubits, n_orbitals, nelec_alpha, nelec_beta = _solve_fci(
        mol, mf.mo_coeff
    )
    fci_vector = _build_fci_vector(
        ci_matrix,
        n_qubits,
        n_orbitals,
        nelec_alpha,
        nelec_beta,
        mode="grouper",
    )
    state_vec = fci_vector.reshape(-1, 1)
    return grouped_jw_list, n_qubits, energy, state_vec


def make_fci_vector_from_pyscf_solver(
    molecule_type: int,
) -> Tuple[QubitOperator, int, float, np.ndarray, np.ndarray]:
    """PySCFからFCIベクトルとJWハミルトニアンを生成（元コードと同一の並びと位相）。"""
    # --- Geometry / SCF ---
    # 分子情報を構築して SCF を実行
    mol, mf, constant, one_body, two_body = _run_scf_and_integrals(molecule_type)

    # spin-orbital Hamiltonian → Fermion → JW
    # フェルミオン演算子を JW に変換
    h1s, h2s = spinorb_from_spatial(one_body, two_body * 0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(constant, h1s, h2s))
    jw_hamiltonian = jordan_wigner(ham_fermion)

    # --- FCI solve ---
    # FCI を解いて基底状態を構築
    energy, ci_matrix, n_qubits, n_orbitals, nelec_alpha, nelec_beta = _solve_fci(
        mol, mf.mo_coeff
    )
    state_vector = _build_fci_vector(
        ci_matrix,
        n_qubits,
        n_orbitals,
        nelec_alpha,
        nelec_beta,
        mode="interleaved",
    )
    state_vec = state_vector.reshape(-1, 1)
    return jw_hamiltonian, n_qubits, energy, state_vec, ci_matrix


def make_integrals_and_fci(molecule_type: int) -> Dict[str, object]:
    """Return grouped JW list, FCI vector, and integrals without breaking existing I/O."""
    mol, mf, constant, one_body_integrals, two_body_integrals = _run_scf_and_integrals(
        molecule_type
    )
    grouped_jw_list = _build_grouped_jw_list(
        constant, one_body_integrals, two_body_integrals
    )
    energy, ci_matrix, n_qubits, n_orbitals, nelec_alpha, nelec_beta = _solve_fci(
        mol, mf.mo_coeff
    )
    fci_vector = _build_fci_vector(
        ci_matrix,
        n_qubits,
        n_orbitals,
        nelec_alpha,
        nelec_beta,
        mode="grouper",
    )
    state_vec = fci_vector.reshape(-1, 1)
    return {
        "grouping_jw_list": grouped_jw_list,
        "grouped_jw_list": grouped_jw_list,
        "n_qubits": n_qubits,
        "E_fci": energy,
        "psi0_vector": state_vec,
        "constant": constant,
        "one_body_integrals": one_body_integrals,
        "two_body_integrals": two_body_integrals,
    }


def _count_non_identity_terms(op: QubitOperator) -> int:
    """Return number of non-identity terms in a QubitOperator."""
    return sum(1 for term in op.terms if term)


def grouped_jw_list_stats(molecule_type: int) -> Dict[str, float | int]:
    """Report grouping stats: group count and term-count reduction ratio."""
    _, _, constant, one_body_integrals, two_body_integrals = _run_scf_and_integrals(
        molecule_type
    )
    grouped_jw_list = _build_grouped_jw_list(
        constant, one_body_integrals, two_body_integrals
    )

    h1s, h2s = spinorb_from_spatial(one_body_integrals, two_body_integrals * 0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(constant, h1s, h2s))
    jw_hamiltonian = jordan_wigner(ham_fermion)

    original_terms = _count_non_identity_terms(jw_hamiltonian)
    group_count = len(grouped_jw_list)
    grouped_terms = sum(_count_non_identity_terms(op) for op in grouped_jw_list)
    reduction_ratio = (group_count / original_terms) if original_terms else 0.0
    avg_terms_per_group = (original_terms / group_count) if group_count else 0.0

    print(f"original terms: {original_terms}")
    print(f"group count: {group_count}")
    print(f"grouped terms (sum): {grouped_terms}")
    print(f"reduction ratio (groups/original terms): {reduction_ratio:.3f}")
    print(f"avg terms per group: {avg_terms_per_group:.2f}")

    return {
        "original_terms": original_terms,
        "group_count": group_count,
        "grouped_terms": grouped_terms,
        "reduction_ratio": reduction_ratio,
        "avg_terms_per_group": avg_terms_per_group,
    }
