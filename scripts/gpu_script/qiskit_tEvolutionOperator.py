import numpy as np
from numpy import pi
from numpy.linalg import eig
from functools import reduce

from scipy.sparse import csr_matrix, identity, eye, load_npz
from scipy.sparse.linalg import eigs, expm,norm

from openfermion import InteractionOperator
from openfermion.ops import QubitOperator, FermionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from qulacs import QuantumCircuit
from qulacs.gate import RX, RY, RZ, CNOT, PauliRotation, merge, to_matrix_gate
from qulacs.circuit import QuantumCircuitOptimizer

import qiskit
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import PauliEvolutionGate, GlobalPhaseGate
from qiskit.quantum_info import Statevector
try:
    from qiskit_aer import AerSimulator
    AER_IMPORT_ERROR = None
except Exception as exc:
    AerSimulator = None
    AER_IMPORT_ERROR = exc

import pyscf
from pyscf.fci import cistring
from pyscf.ci import gcisd
from pyscf import gto, scf, fci, mcscf

import time
import os
import gc
import pickle
from datetime import datetime

import Trotter_sim as tsag
from Almost_optimal_grouping import Almost_optimal_grouper

def free_var(name, scope):
    if name in scope:
        del scope[name]
        import gc; gc.collect()

def evolve_with_statevector(eigenvector, time_evolution_circuit: QuantumCircuit):
    state = eigenvector if isinstance(eigenvector, Statevector) else Statevector(eigenvector)
    return state.evolve(time_evolution_circuit)

def make_spinorb_ham(constant, h1, h2):
    """
    Build a spin-orbital InteractionOperator from spatial integrals.

    Args:
        constant (float): nuclear repulsion energy
        h1 (ndarray): one-body spatial integrals [n_orb, n_orb]
        h2 (ndarray): two-body spatial integrals [n_orb, n_orb, n_orb, n_orb]
    Returns:
        InteractionOperator on spin-orbital basis
    """

    n_orb = h1.shape[0]
    n_so = 2 * n_orb
    # initialize spin-orbital integrals
    so_h1 = np.zeros((n_so, n_so))
    so_h2 = np.zeros((n_so, n_so, n_so, n_so))
    # fill one-body: h_pσ,qτ = h1[p,q] δσ,τ
    for p in range(n_orb):
        for q in range(n_orb):
            for sigma in [0,1]:  # 0: alpha, 1: beta
                i = 2*p + sigma
                j = 2*q + sigma
                so_h1[i,j] = h1[p,q]
    # fill two-body: <pσ qτ|rσ sτ> = h2[p,r,q,s] δστ δστ
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    for sigma in [0,1]:
                        for tau in [0,1]:
                            i = 2*p + sigma
                            j = 2*q + tau
                            k = 2*r + sigma
                            l = 2*s + tau
                            # chemists' notation (pr|qs)
                            so_h2[i,j,k,l] = h2[p,r,q,s]
    return InteractionOperator(constant, so_h1, so_h2)

def make_hamiltonian_pyscf(mol_type):
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.spin = multiplicity - 1          # 電子数は 4 → α = 2, β = 2
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    ## run SCF
    scf_solver = pyscf.scf.RHF(mol)
    scf_solver.verbose = 0
    scf_solver.kernel()

    n_qubits = 2 * scf_solver.mo_coeff.shape[0]
    print("# of qubits:", n_qubits)

    ## prepare Hamiltonian by making electron integrals. 
    constant = scf_solver.energy_nuc()
    mo = scf_solver.mo_coeff
    h_core = scf_solver.get_hcore()
    one_body_integrals = reduce(np.dot, (mo.T, h_core, mo))

    eri = pyscf.ao2mo.kernel(scf_solver.mol, mo)
    eri = pyscf.ao2mo.restore(1, eri, mo.shape[0])
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')

    ham_fermion = get_fermion_operator(make_spinorb_ham(constant, one_body_integrals, two_body_integrals))
    jw_hamiltonian = jordan_wigner(ham_fermion)
    return jw_hamiltonian, n_qubits


def make_fci_vector_from_pyscf_solver(mol_type):
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.spin = multiplicity - 1          # 電子数は 4 → α = 2, β = 2
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    n_qubits = 2 * mf.mo_coeff.shape[0]

    ## prepare Hamiltonian by making electron integrals. 
    constant = mf.energy_nuc()
    # print(f'cons{constant}')
    mo = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body_integrals = reduce(np.dot, (mo.T, h_core, mo))

    eri = pyscf.ao2mo.kernel(mf.mol, mo)
    eri = pyscf.ao2mo.restore(1, eri, mo.shape[0])
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')
    h1s, h2s = spinorb_from_spatial(one_body_integrals, two_body_integrals * 0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(constant, h1s, h2s))
    # ham_fermion = get_fermion_operator(make_spinorb_ham(constant, one_body_integrals, two_body_integrals))
    jw_hamiltonian = jordan_wigner(ham_fermion)

    pyscf_fci_solver = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = pyscf_fci_solver.kernel()
    n_qubits = pyscf_fci_solver.norb * 2
    n_orbitals = pyscf_fci_solver.norb
    n_electrons = sum(pyscf_fci_solver.nelec)
    fci_vector = np.zeros(2 ** n_qubits, dtype=np.complex128)
    ci_strings = cistring.make_strings(range(n_orbitals), n_electrons // 2)

    nelec_alpha, nelec_beta = pyscf_fci_solver.nelec  # 例: (3,3)

    # CI ビット列を正しく取得
    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta  = cistring.make_strings(range(n_orbitals), nelec_beta)

    alpha_list = cistring.gen_occslst(range(n_orbitals), nelec_alpha)
    beta_list  = cistring.gen_occslst(range(n_orbitals), nelec_beta)
    # print(f'ci_string {ci_strings}')
    if None:
        for i, alpha_ci_index in enumerate(ci_strings):
            for j, beta_ci_index in enumerate(ci_strings):
                alpha_index = format(alpha_ci_index, f"0{n_qubits//2}b")
                beta_index = format(beta_ci_index, f"0{n_qubits//2}b")
                qulacs_index = ""
                for bit_a, bit_b in zip(alpha_index, beta_index):
                    qulacs_index += bit_b + bit_a
                #process phase
                sign = 1
                for k, bit_a in enumerate(alpha_index):
                    # find beta electron left to the alpha spin
                    if bit_a == '1':
                        for l in range(k+1):
                            if beta_index[l] == '1':
                                sign *= -1
                qulacs_index = int(qulacs_index, 2)
                fci_vector[qulacs_index] = sign*pyscf_fci_solver.ci[i][j]


    if True:
        for i, alpha_ci_index in enumerate(ci_strings_alpha):
        #for i, alpha_ci_index in enumerate(alpha_list):
            for j, beta_ci_index in enumerate(ci_strings_beta):
            #for j, beta_ci_index in enumerate(beta_list):
                alpha_index = format(alpha_ci_index, f"0{n_qubits // 2}b")[::-1]
                beta_index = format(beta_ci_index, f"0{n_qubits // 2}b")[::-1]
                # alpha_index = format(alpha_ci_index, f"0{n_qubits // 2}b")
                # beta_index = format(beta_ci_index, f"0{n_qubits // 2}b")
                qulacs_index = ""
                for bit_a, bit_b in zip(alpha_index, beta_index):
                    qulacs_index += bit_b + bit_a
                #print(f'qulacs{qulacs_index}')

                # フェーズの補正（Jordan-Wigner 変換の影響）
                sign = 1
                N = len(alpha_index)
                if None:
                    for k, bit_a in enumerate(alpha_index):
                        if bit_a == '1':
                            for l in range(k+1):
                                if beta_index[l] == '1':
                                    sign *= -1
                    if sign == -1:
                        print(f'inverse a{alpha_index} b{beta_index}')
                if True:
                    for k in range(N):
                        if alpha_index[k] == '1':
                            # 反転後文字列では、元の「左側」に相当するのは
                            # インデックス k より**大きい**位置
                            for l in range(k+1, N):
                                if beta_index[l] == '1':
                                    sign *= -1

                index = int(qulacs_index, 2)
                # if abs(ci_matrix[i][j]) < 1e-10:
                #     continue
                # print(f'i{i} j{j}, alpha {alpha_index} beta {beta_index}, index {index}')
                #print(f'ind{abs(index - 2**n_qubits)-1}')

                try:
                    #print(sign * ci_matrix[i][j])
                    fci_vector[index] = sign * ci_matrix[i][j]
                except:
                    print(ci_matrix[i][j])


    id_fci = fci_vector.reshape(-1,1)
    #vector = id_fci[::-1]
    vector = id_fci
    return jw_hamiltonian, n_qubits, energy, vector, ci_matrix  # ← ndarray (1D) 形式で返す

def make_fci_vector_from_pyscf_solver_grouper(mol_type):
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.spin = multiplicity - 1          # 電子数は 4 → α = 2, β = 2
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    n_qubits = 2 * mf.mo_coeff.shape[0]

    ## prepare Hamiltonian by making electron integrals. 
    constant = mf.energy_nuc()
    mo = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body_integrals = reduce(np.dot, (mo.T, h_core, mo))

    eri = pyscf.ao2mo.kernel(mf.mol, mo)
    eri = pyscf.ao2mo.restore(1, eri, mo.shape[0])
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')

    almost_optimal_grouper = Almost_optimal_grouper(constant, one_body_integrals, two_body_integrals, fermion_qubit_mapping=jordan_wigner, validation=False)
    grouping_term_list = almost_optimal_grouper.group_term_list
    grouping_term_list[0].insert(0, FermionOperator('', almost_optimal_grouper._const_fermion))
    grouping_jw_list = [jordan_wigner(sum(group_term)) for group_term in grouping_term_list]

    pyscf_fci_solver = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = pyscf_fci_solver.kernel()
    n_qubits = pyscf_fci_solver.norb * 2
    n_orbitals = pyscf_fci_solver.norb
    n_electrons = sum(pyscf_fci_solver.nelec)
    fci_vector = np.zeros(2 ** n_qubits, dtype=np.complex128)
    ci_strings = cistring.make_strings(range(n_orbitals), n_electrons // 2)

    nelec_alpha, nelec_beta = pyscf_fci_solver.nelec  # 例: (3,3)

    # CI ビット列を正しく取得
    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta  = cistring.make_strings(range(n_orbitals), nelec_beta)

    # for i, alpha_ci_index in enumerate(ci_strings_alpha):
    #     for j, beta_ci_index in enumerate(ci_strings_beta):
    #         alpha_index = format(alpha_ci_index, f"0{n_qubits // 2}b")[::-1]
    #         beta_index = format(beta_ci_index, f"0{n_qubits // 2}b")[::-1]
    #         qulacs_index = ""
    #         for bit_a, bit_b in zip(alpha_index, beta_index):
    #             qulacs_index += bit_b + bit_a
    for i, a_str in enumerate(ci_strings_alpha):
        # a_bits: '01001...' を長さ n_orb のリストに
        alpha_index = list(format(a_str, f'0{n_qubits // 2}b'))[::-1]
        for j, b_str in enumerate(ci_strings_beta):
            beta_index = list(format(b_str, f'0{n_qubits // 2}b'))[::-1]

            # qubit 上の配置：先に α の n_orb ビット、あと β の n_orb ビット
            bitstring = ''.join(alpha_index) + ''.join(beta_index)
            
            # フェーズの補正（Jordan-Wigner 変換の影響）
            sign = 1
            N = len(alpha_index)
            for k in range(N):
                if alpha_index[k] == '1':
                    # 反転後文字列では、元の「左側」に相当するのは
                    # up-down の時はそれぞれ0番目から調べればよい
                    for l in range(N):
                        if beta_index[l] == '1':
                            sign *= -1

            # index = int(qulacs_index, 2)
            index = int(bitstring, 2)
            fci_vector[index] = sign * ci_matrix[i][j]

    id_fci = fci_vector.reshape(-1,1)
    vector = id_fci
    return grouping_jw_list, n_qubits, energy, vector  # ← ndarray (1D) 形式で返す

def make_fci_vector_from_pyscf_solver_grouper_ao(mol_type, n):
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
    basis = "cc-pVDZ"
    mol = gto.Mole()
    mol.atom = geometry
    #mol.basis = "sto-3g"
    mol.basis = basis
    mol.spin = multiplicity - 1          # 電子数は 4 → α = 2, β = 2
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    if multiplicity == 1:
        ham_name = f'{mol_type}_{basis}_singlet_distance_100_charge_0_grouping_ao{n}'
    else:
        ham_name = f'{mol_type}_{basis}_triplet_1+_distance_100_charge_1_grouping_ao{n}'
    mf = scf.RHF(mol)
    mf.kernel()
    print(f'nuc{ mf.energy_nuc()}')
    n_qubits = 2 * mf.mo_coeff.shape[0]

    occ  = mf.mo_occ
    print(occ)
    homo = max(i for i,o in enumerate(occ) if o>0)
    lumo = homo + 1

    # raw = range(homo - n , lumo + n + 1)
    # caslist = list(range(homo-n, lumo+n+1))
    below = homo - n//2 - n%2
    above = lumo + n//2 + 1
    raw = range(below, above)
    caslist = list(range(below,  above))
    n_mo = mf.mo_coeff.shape[1]     
    cas = [i for i in raw if 0 <= i < n_mo]
    print(cas, caslist)
    ncas    = len(caslist)
    nelecas = int(sum(occ[i] for i in cas))

    mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
    mc.kernel()
    ci_matrix = mc.ci
    energy = mc.e_tot
    # print("After kernel, mc.mo_coeff shape:", mc.mo_coeff.shape)
    # 活性空間のMO係数を取得（CASSCFで最適化された活性空間）
    mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]  # shape = (nao, ncas)
    # print("ncore:", mc.ncore, " ncas:", mc.ncas)
    # print("mo_cas shape:", mo_cas.shape)

    h1eff_full,_ = mc.get_h1eff(mc.mo_coeff)
    h2eff_full = mc.get_h2cas(mc.mo_coeff)
    # print(h2eff_full.shape)
    i0, i1 = mc.ncore, mc.ncore + mc.ncas
    h1eff = h1eff_full[i0:i1, i0:i1]
    # one-body integrals（ハートリー・フォックのハミルトニアン）
    hcore = mf.get_hcore()
    h1_active = reduce(np.dot, (mo_cas.T, hcore, mo_cas))  # shape = (ncas, ncas)
    # print(h1eff.shape, h1_active.shape)

    # two-body integrals（Coulomb積分）
    eri_ao2mo = pyscf.ao2mo.kernel(mol, mo_cas)  # MO変換済みの2電子積分
    eri_4index = pyscf.ao2mo.restore(1, eri_ao2mo, mc.ncas)  # shape = (ncas, ncas, ncas, ncas)
    eri_active = np.asarray(eri_4index.transpose(0, 3, 1, 2), order='C')       # (pq|rs) → chemist's notation
    eri_active1 = np.asarray(eri_4index.transpose(0, 2, 3, 1), order='C')
    # print(eri_active.shape)

    core_indices = list(range(mc.ncore))  # コア軌道のインデックス
    mo_core = mc.mo_coeff[:, core_indices]
    # print(f'mocore{core_indices}')
    # one-body core contribution
    e_core = 0.0
    hcore_mo = reduce(np.dot, (mo_core.T, hcore, mo_core))
    n_core = len(core_indices)
    for i in range(n_core):
        e_core += hcore_mo[i, i] * 2

    # two-body core contribution
    eri_core = pyscf.ao2mo.kernel(mol, mo_core)
    eri_core = pyscf.ao2mo.restore(1, eri_core, len(core_indices))
    eri_core = eri_core.transpose(0, 2, 3, 1)
    for i in range(len(core_indices)):
        for j in range(len(core_indices)):
            e_core += 2 * eri_core[i, j, j, i] - eri_core[i, j, i, j]


    constant = mf.energy_nuc() + e_core
    # print(constant)

    h1s, h2s = spinorb_from_spatial(h1eff_full, eri_active1*0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(0, h1s, h2s))
    E,_,_ = tsag.ham_ground_energy(jordan_wigner(ham_fermion))
    # print(E)

    almost_optimal_grouper = Almost_optimal_grouper(0, h1eff_full, eri_active1, fermion_qubit_mapping=jordan_wigner, validation=True)
    grouping_term_list = almost_optimal_grouper.group_term_list
    #del grouping_term_list[0][0]
    # print(grouping_term_list[0])
    # print([jordan_wigner(term) for term in grouping_term_list[0]])
    #grouping_term_list[0].insert(0, FermionOperator('', almost_optimal_grouper._const_fermion))
    grouping_jw_list = [jordan_wigner(sum(group_term)) for group_term in grouping_term_list]

    n_orbitals = mc.ncas
    n_qubits = 2 * n_orbitals
    fci_vector = np.zeros(2 ** n_qubits, dtype=np.complex128)
    nelec_alpha, nelec_beta = mc.nelecas  # 例: (3,3)

    # CI ビット列を正しく取得
    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta  = cistring.make_strings(range(n_orbitals), nelec_beta)

    # アルファ電子とβ電子の数が違う時に注意
    for i, a_str in enumerate(ci_strings_alpha):
        # a_bits: '01001...' を長さ n_orb のリストに
        alpha_index = list(format(a_str, f'0{n_qubits // 2}b'))[::-1]
        for j, b_str in enumerate(ci_strings_beta):
            beta_index = list(format(b_str, f'0{n_qubits // 2}b'))[::-1]

            # qubit 上の配置：先に α の n_orb ビット、あと β の n_orb ビット
            bitstring = ''.join(alpha_index) + ''.join(beta_index)
            
            # フェーズの補正（Jordan-Wigner 変換の影響）
            sign = 1
            N = len(alpha_index)
            for k in range(N):
                if alpha_index[k] == '1':
                    # 反転後文字列では、元の「左側」に相当するのは
                    # インデックス k より**大きい**位置
                    # up-down の時はそれぞれ0番目から調べればよい
                    for l in range(N):
                        if beta_index[l] == '1':
                            sign *= -1

            # index = int(qulacs_index, 2)
            index = int(bitstring, 2)
            # print(f'index{index}, i{i}, j{j}')
            fci_vector[index] = sign * ci_matrix[i][j]

    id_fci = fci_vector.reshape(-1,1)
    vector = id_fci
    return grouping_jw_list, n_qubits, E, vector, ham_name

def jw_hamiltonian_maker(n_atoms, multiplicity):
    basis = "sto-3g"  #基底関数
    multiplicity = 1  #基底状態のスピン多重度
    charge = 0        #分子の全電荷
    distance = 0.7 #オングストローム
    n_qubits = 2 * n_atoms
    geometry = [("H", (0, 0, i * distance)) for i in range(n_atoms)]  #xyz座標での原子間距離
    description = str(distance)  #保存先のファイル名
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule,run_scf=1,run_fci=0)
    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    return jw_hamiltonian

def eU_strage_grouper(clique, t, n_qubits, w, qc):
    """
    clique = [(coeff)[term]+()[]+...,(coeff)[term]+()[]+...,...]

    Args:
        clique: 積公式の項, 
        t:  時間発展演算子のt
        n_qubits: ハミルトニアンのQビット数
        w: 積公式の係数
        idx: トロッター展開されたハミルトニアンの項のインデックス
        folder_path: 保存先のフォルダ

    Returns:
        idx: トロッター展開されたハミルトニアンの項のインデックス
    """
    def ham_to_cIsU_clique(term_list, t, n_qubits, w, qc):
        for term in term_list:
            for op, coefficient in term.terms.items():
                add_term_to_circuit(qc, op, coefficient, n_qubits, t, w)
        

    def eU_exchanger(clique, t, n_qubits, w, qc):
        for terms in clique:
            term_list = [term for term in terms]
            ham_to_cIsU_clique(term_list, t, n_qubits, w, qc)

    eU_exchanger(clique, t, n_qubits, w, qc)

def get_eigenvalues_error(unitary, threshold, hosei, origin_vecs, t, originE):#Uは密行列
    unitary[np.abs(unitary) < threshold] = 0
    sparse_unitary = csr_matrix(unitary)
    sigma = np.exp(1j * hosei)
    eigenvalues = eigs(
        sparse_unitary, k=1, sigma=sigma, v0=origin_vecs ,return_eigenvectors=False
    )
    phases = np.angle(eigenvalues)
    normal_E = sorted(phases.real / t )[0]
    if normal_E > 0:
        normal_E = -1 * normal_E
    error = abs(normal_E - originE)
    print(f'error:{error}')
    return error


def add_term_to_circuit(hamiltonian, n_qubits, t, w, qc): #ハミルトニアンの項からゲートを回路に追加
    from qiskit.synthesis.evolution import SuzukiTrotter
    from qiskit.synthesis.evolution import MatrixExponential
    for term, coeff in hamiltonian.terms.items():
        X = SparsePauliOp("X")
        Y = SparsePauliOp("Y")
        Z = SparsePauliOp("Z")
        I = SparsePauliOp("I")
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        pauli_operators = [I] * n_qubits  
        
        #term の内容に基づき、適切なビットにパウリ演算子を配置
        for index, pauli_op_name in term:
            pauli_operators[index] = pauli_dict[pauli_op_name]
        
        #全体のパウリ演算子のテンソル積を生成
        pauli_op = pauli_operators[0]
        for op in pauli_operators[1:]:
            pauli_op ^= op

        #回転角度の計算
        angle = coeff.real * w * t
        # PauliEvolutionGate を作成し、量子回路に追加
        if not term :
            # qc.append(GlobalPhaseGate(-1*angle))
            return

        rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=None)
        #rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=SuzukiTrotter(order=4,reps=2))
        #rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=MatrixExponential())

        if None:
            unitary = Operator(rotation_gate).data
            matrix = get_sparse_operator(hamiltonian, n_qubits)
            eU = expm(-1j*matrix*w*t)
            diff = eU - unitary
            #print(np.vdot(eU.toarray(), unitary))
            if np.linalg.norm(diff) > 0.00000000000000001:
                print(f'diff_op {np.linalg.norm(diff)}, operator {rotation_gate}, ham {hamiltonian}')

        qc.append(rotation_gate, range(n_qubits))

def add_term_to_circuit_grouper(clique, t, n_qubits, w, qc): #ハミルトニアンの項からゲートを回路に追加
    # clique = [(coeff)[term]+()[]+...,(coeff)[term]+()[]+...,...] クリーク内の項は可換
    
    X = SparsePauliOp("X")
    Y = SparsePauliOp("Y")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")
    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    num_exp = 0
    for hamiltonian in clique:
        for term, coeff in hamiltonian.terms.items():
            pauli_operators = [I] * n_qubits  
            
            #term の内容に基づき、適切なビットにパウリ演算子を配置
            for index, pauli_op_name in term:
                pauli_operators[index] = pauli_dict[pauli_op_name]
            
            #全体のパウリ演算子のテンソル積を生成
            pauli_op = pauli_operators[0]
            for op in pauli_operators[1:]:
                pauli_op ^= op

            #回転角度の計算
            angle = coeff.real * w * t
            # PauliEvolutionGate を作成し、量子回路に追加
            if not term :
                ## set_statevector 使用時は飛ばす
                # qc.append(GlobalPhaseGate(-1*angle))
                # num_exp += 1
                continue
            
            rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=None)
            num_exp += 1
            qc.append(rotation_gate, range(n_qubits))
    return num_exp

def S_2_gr(clique_list, t, n_qubits, w, qc): # 左端
    """
    Args:
        ham_list:項ごとに格納したリスト , 
        t: 時間発展演算子の t , 
        n_qubits: ハミルトニアンのQビット数, 
        Max_w: S_2 の係数w, 
        nMax_w: となりの S_2 の係数, 
        folder_path: 保存フォルダのパス, 
        idx: トロッター展開した項のインデックス

    Returns:
        idx: トロッター展開した項のインデックス
    """

    J = len(clique_list)
    num_exp = 0
    # 折り返しの直前まで
    for i in range(J-1):
        add_exp = add_term_to_circuit_grouper(clique_list[i], t, n_qubits, w/2, qc)
        num_exp += add_exp

    # 折り返し
    add_exp = add_term_to_circuit_grouper(clique_list[J-1], t, n_qubits, w, qc)
    num_exp += add_exp

    # 終端まで
    for k in reversed(range(0,J-1)):
        add_exp = add_term_to_circuit_grouper(clique_list[k], t, n_qubits, w/2, qc)
        num_exp += add_exp
    return num_exp

def S_2(ham_list, t, n_qubits, w, qc, eU, speU): # 左端
    """
    Args:
        ham_list:項ごとに格納したリスト , 
        t: 時間発展演算子の t , 
        n_qubits: ハミルトニアンのQビット数, 
        Max_w: S_2 の係数w, 
        nMax_w: となりの S_2 の係数, 
        folder_path: 保存フォルダのパス, 
        idx: トロッター展開した項のインデックス

    Returns:
        idx: トロッター展開した項のインデックス
    """

    J = len(ham_list)
    idx = 0
    
    # 折り返しの直前まで
    for i in range(J-1):
        add_term_to_circuit(ham_list[i], n_qubits,t,  w/2, qc)
        #eU,speU = exp_check(eU,speU, ham_list[i], n_qubits,t,  w/2, qc,i,idx)
        #idx += 1

    # 折り返し
    add_term_to_circuit(ham_list[J-1], n_qubits, t, w, qc)
    #eU,speU = exp_check(eU,speU, ham_list[J-1], n_qubits, t, w, qc,J-1,idx)
    #idx += 1

    # 終端まで
    for k in reversed(range(0,J-1)):
        add_term_to_circuit(ham_list[k],n_qubits, t,  w/2, qc)
        #eU,speU = exp_check(eU,speU, ham_list[k],n_qubits, t,  w/2, qc,k,idx)
        #idx += 1
    #return qc

def S_2_trotter_left(A_list, n_qubits,t, Max_w, nMax_w, qc,eU,speU): #ham は項ごとのリスト
    J = len(A_list)
    idx = 0
    #折り返しの直前まで
    for i in range(J-1):
        add_term_to_circuit(A_list[i], n_qubits,t, Max_w / 2, qc)
        #eU,speU = exp_check(eU,speU, A_list[i], n_qubits,t, Max_w / 2, qc,i,idx)
        idx +=1

    #折り返し
    add_term_to_circuit(A_list[J-1], n_qubits,t, Max_w, qc)
    #eU,speU = exp_check(eU,speU, A_list[J-1], n_qubits,t, Max_w, qc,J-1,idx)
    idx +=1

    #終端 - 1 個目まで
    for k in reversed(range(1,J-1)):
        add_term_to_circuit(A_list[k], n_qubits, t, Max_w / 2, qc)
        #eU,speU = exp_check(eU,speU,A_list[k], n_qubits, t, Max_w / 2, qc,k,idx)
        idx +=1

    #終端
    add_term_to_circuit(A_list[0], n_qubits, t, (Max_w + nMax_w) / 2, qc)
    #eU,speU = exp_check(eU,speU,A_list[0], n_qubits, t, (Max_w + nMax_w) / 2, qc,0,idx)
    idx += 1
    #return eU,speU, idx


def exp_check(eU,speU , ham, n_qubits, t,w, qc, i,idx=None):
    if None:
        matrix = get_sparse_operator(ham, n_qubits)
        exp = expm(-1j*matrix*w*t )
        eU = exp @ eU 
        unitary = Operator(qc).data

        diff = eU - unitary
        if np.linalg.norm(diff) > 0.00000000000001:
            print(f'diff_{i} {np.linalg.norm(diff)}')
            print(f'num{i} ham {ham}')
            print(f'max diff {np.abs(diff).max()}')

        if not idx==None:
            spdata = load_npz(f'/home/abe/myproject/ham_dir/H4_sto-3g_singlet_distance_86_charge_0_Operator_w2/{-t}/H4_sto-3g_singlet_distance_86_charge_0_nostep_tOperator_w2/matrix_{idx}.npz')
            spdiff = exp - spdata
            if norm(spdiff) > 0.000000001:
                print(f'diff_idx_{idx} {norm(spdiff)} ham {ham} max {np.abs(spdiff).max()}')
            
            speU =  spdata @ speU 
            eUdiff = unitary - speU
            if np.linalg.norm(eUdiff) > 0.0000001:
                print(f'speUdiff_{idx} {np.linalg.norm(eUdiff)} max {np.abs(eUdiff).max()}')        

    return eU,speU


def S_2_trotter(A_list, n_qubits, t, w_f, w_s, qc,eU,speU,idx): #左端以外
    J = len(A_list)
    #折り返しの直前まで
    for i in range(1,J-1):
        add_term_to_circuit(A_list[i],n_qubits, t, w_f / 2, qc)
        #eU,speU = exp_check(eU,speU,A_list[i],n_qubits, t, w_f / 2, qc,i,idx)
        idx += 1

    #折り返し
    add_term_to_circuit(A_list[J-1], n_qubits, t, w_f, qc)
    #eU,speU = exp_check(eU,speU,A_list[J-1], n_qubits, t, w_f, qc,J-1,idx)
    idx += 1

    #終端 - 1 個目まで
    for k in reversed(range(1,J-1)):
        add_term_to_circuit(A_list[k], n_qubits, t, w_f / 2, qc)
        #eU,speU = exp_check(eU,speU,A_list[k], n_qubits, t, w_f / 2, qc,k,idx)
        idx += 1

    #終端
    add_term_to_circuit(A_list[0], n_qubits, t, (w_f + w_s)/2, qc)
    #eU,speU = exp_check(eU,speU,A_list[0], n_qubits, t, (w_f + w_s)/2, qc,0,idx)
    idx += 1
    #return eU,speU,idx


def S_2_trotter_right(A_list, n_qubits, t, w_i, qc,eU,speU,idx): #右端
    J = len(A_list)
    #折り返しの直前まで
    print(f'idx {idx}')
    for i in range(1,J-1):
        add_term_to_circuit(A_list[i], n_qubits,t, w_i / 2, qc)
        #eU,speU = exp_check(eU,speU,A_list[i], n_qubits,t, w_i / 2, qc,i,idx)
        idx += 1

    #折り返し
    print(f'idx {idx}')
    add_term_to_circuit(A_list[J-1], n_qubits, t, w_i, qc)
    #eU,speU = exp_check(eU,speU,A_list[J-1], n_qubits, t, w_i, qc,J-1,idx)
    idx += 1
    #終端 まで
    for k in reversed(range(0,J-1)):
        add_term_to_circuit(A_list[k], n_qubits, t, w_i / 2, qc)
        #eU,speU = exp_check(eU,speU,A_list[k], n_qubits, t, w_i / 2, qc,k,idx)
        idx += 1
    #return eU,speU,idx



def S_2_trotter_left_gr(clique_list, t, n_qubits, Max_w, nMax_w, qc): # 左端
    """
    Args:
        ham_list:項ごとに格納したリスト , 
        t: 時間発展演算子の t , 
        n_qubits: ハミルトニアンのQビット数, 
        Max_w: S_2 の係数w, 
        nMax_w: となりの S_2 の係数, 
        folder_path: 保存フォルダのパス, 
        idx: トロッター展開した項のインデックス

    Returns:
        idx: トロッター展開した項のインデックス
    """

    J = len(clique_list)
    num_exp = 0
    # 折り返しの直前まで
    for i in range(J-1):
        add_exp = add_term_to_circuit_grouper(clique_list[i], t, n_qubits, Max_w/2, qc)
        num_exp += add_exp

    # 折り返し
    add_exp = add_term_to_circuit_grouper(clique_list[J-1], t, n_qubits, Max_w, qc)
    num_exp += add_exp

    # 終端 - 1 個目まで
    for k in reversed(range(1,J-1)):
        add_exp = add_term_to_circuit_grouper(clique_list[k], t, n_qubits, Max_w/2, qc)
        num_exp += add_exp

    # 終端
    add_exp = add_term_to_circuit_grouper(clique_list[0], t, n_qubits, (Max_w + nMax_w) /2, qc)
    num_exp += add_exp
    return num_exp

def S_2_trotter_gr(clique_list, t, n_qubits, w_f, w_s, qc, num_exp): # 左端、右端以外
    """
    Args:
        ham_list:項ごとに格納したリスト , 
        t: 時間発展演算子の t , 
        n_qubits: ハミルトニアンのQビット数, 
        w_f: S_2 の係数w, 
        w_s: 右端側のとなりの S_2 の係数, 
        folder_path: 保存フォルダのパス, 
        idx: トロッター展開した項のインデックス

    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(clique_list)

    # 折り返しの直前まで
    for i in range(1,J-1):
        add_exp = add_term_to_circuit_grouper(clique_list[i], t, n_qubits, w_f /2, qc)
        num_exp += add_exp

    # 折り返し
    add_exp = add_term_to_circuit_grouper(clique_list[J-1], t, n_qubits, w_f, qc)
    num_exp += add_exp

    # 終端 - 1 個目まで
    for k in reversed(range(1,J-1)):
        add_exp = add_term_to_circuit_grouper(clique_list[k], t, n_qubits, w_f / 2, qc)
        num_exp += add_exp

    # 終端
    add_exp = add_term_to_circuit_grouper(clique_list[0], t, n_qubits, (w_f + w_s)/ 2, qc)
    num_exp += add_exp
    return num_exp

def S_2_trotter_right_gr(clique_list, t, n_qubits, w_i, qc, num_exp): #右端
    """
    Args:
        ham_list:項ごとに格納したリスト , 
        t: 時間発展演算子の t , 
        n_qubits: ハミルトニアンのQビット数, 
        w_i: S_2 の係数w, 
        folder_path: 保存フォルダのパス, 
        idx: トロッター展開した項のインデックス

    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(clique_list)

    # 折り返しの直前まで
    for i in range(1,J-1):
        add_exp = add_term_to_circuit_grouper(clique_list[i], t, n_qubits, w_i / 2, qc)
        num_exp += add_exp

    # 折り返し
    add_exp = add_term_to_circuit_grouper(clique_list[J-1], t, n_qubits, w_i, qc)
    num_exp += add_exp

    # 終端 まで
    for k in reversed(range(0,J-1)):
        add_exp = add_term_to_circuit_grouper(clique_list[k], t, n_qubits, w_i / 2, qc)
        num_exp += add_exp
    return num_exp

def w_trotter(qc, ham_list, t, n_qubits,num_w):
    eU = 0
    speU = 0
    idx = 0
    if num_w == 8:
        w_list = tsag.generate_w8_list()
    elif num_w == 10_15:
        w_list = tsag.generate_w10_15_list()
    elif num_w == 10_16:
        w_list = tsag.generate_w10_16_list()
    elif num_w == 3:
        w_list = tsag.generate_w3_list()
    elif num_w == 'yoshida':
        w_list = tsag.yoshida_list()
    elif num_w == 2:
        w_list = tsag.generate_w1_list()
    m = len(w_list)
    if m == 1:
        S_2(ham_list, t, n_qubits, w_list[0], qc,eU,speU)
        # print(qc)
        # count = 0
        # for instruction, qargs, cargs in qc.data:
        #     if isinstance(instruction, PauliEvolutionGate):
        #         count += 1
        #print(f'num_evolutionGate {count}, (4m+2)(J-1)-1 {(4*0+2)*(len(ham_list)-1)+1}')
        #U = Operator(qc)
        return 
    #w_m
    S_2_trotter_left(ham_list,  n_qubits, t, w_list[m-1], w_list[m-2], qc, eU,speU)

    #w_m-1 ~ w_1
    for i in reversed(range(1, m-1)):
        S_2_trotter(ham_list,  n_qubits, t, w_list[i], w_list[i-1], qc, eU,speU,idx)

    #w_0 ~ w_m-1
    for i in range(0,m-1):
        S_2_trotter(ham_list, n_qubits, t, w_list[i], w_list[i+1], qc, eU,speU,idx)

    #w_m

    S_2_trotter_right(ham_list, n_qubits,  t, w_list[m-1], qc, eU,speU,idx)
    # count = 0
    # for instruction, qargs, cargs in qc.data:
    #     if isinstance(instruction, PauliEvolutionGate):
    #         count += 1
    # print(f'num_evolutionGate {count}, (4m+2)(J-1)-1 {(4*(m-1)+2)*(len(ham_list)-1)+1}')
    #U = Operator(qc)
    #print(U.data)
    return 

def w_trotter_grouper(qc, clique_list, t, n_qubits,num_w):

    if num_w == 8:
        w_list = tsag.generate_w8_list()
    elif num_w == 10_15:
        w_list = tsag.generate_w10_15_list()
    elif num_w == 10_16:
        w_list = tsag.generate_w10_16_list()
    elif num_w == 3:
        w_list = tsag.generate_w3_list()
    elif num_w == 'yoshida':
        w_list = tsag.yoshida_list()
    elif num_w == 2:
        w_list = tsag.generate_w1_list()
    elif num_w == 'my1':
        w_list = tsag.generate_myw1_list()
    elif num_w == 'my4':
        w_list = tsag.generate_myw4_list()
    m = len(w_list)
    if m == 1:
        num_exp = S_2_gr(clique_list, t, n_qubits, w_list[0], qc)
        #print(qc)
        return num_exp
    #w_m
    num_exp = S_2_trotter_left_gr(clique_list, t, n_qubits, w_list[m-1], w_list[m-2], qc)

    #w_m-1 ~ w_1
    for i in reversed(range(1, m-1)):
        num_exp = S_2_trotter_gr(clique_list, t, n_qubits, w_list[i], w_list[i-1], qc, num_exp)

    #w_0 ~ w_m-1
    for i in range(0,m-1):
        num_exp = S_2_trotter_gr(clique_list, t, n_qubits, w_list[i], w_list[i+1], qc, num_exp)

    #w_m
    num_exp = S_2_trotter_right_gr(clique_list, t, n_qubits, w_list[m-1], qc, num_exp)
    return num_exp


def create_eigenstate_circuit(eigenvector: np.ndarray) -> QuantumCircuit:
    """
    eigenvector: 2^n 次元の複素振幅ベクトル（正規化済み） 
    返り値: 固有状態に初期化する QuantumCircuit
    """
    n_qubits = int(np.log2(len(eigenvector)))
    init_qc = QuantumCircuit(n_qubits)
    # Initialize: reset後に指定状態へ準備
    sv = Statevector(eigenvector)
    init_gate = qiskit.circuit.library.Initialize(sv)  
    init_qc.append(init_gate, init_qc.qubits)      
    return init_qc

def error_from_perture(E, t, ori_vec, state):
    state = state.data.reshape(-1,1)
    tevolution = np.exp(1j*E*t)
    delta_psi = (state - (tevolution * ori_vec)) / (1j*t)
    innerproduct = (ori_vec.conj().T @ delta_psi)
    innerproduct = innerproduct.real / np.cos(E*t)
    error = abs((innerproduct.real))
    free_var('final_state',locals())
    return error

def apply_time_evolution(eigenvector, time_evolution_circuit: QuantumCircuit, n_qubits):
    """
    基底状態を量子回路に変換し、回路上で時間発展演算子を作用させる。
    args:
        eigenvector : 基底状態のベクトル
        time_evolution_circuit : 量子回路に変換された時間発展演算子
    return:
        e^(iHt)|Ψ_0> のベクトル
    """

    if AerSimulator is None:
        return evolve_with_statevector(eigenvector, time_evolution_circuit)

    # 初期状態回路作成
    init_qc = create_eigenstate_circuit(eigenvector)

    # 時間発展演算子を合成
    all_qc = init_qc.compose(time_evolution_circuit)

    full_qc = QuantumCircuit(n_qubits)
    for instr, qargs, cargs in all_qc.data:
        full_qc.barrier()   # 各ゲートの間にバリアを入れる
        full_qc.append(instr, qargs, cargs)

    # 結果に statevector を含めるよう明示的に保存命令を追加
    full_qc.save_statevector()

    # シミュレーター
    simulator = AerSimulator()
    simulator.set_options(method="statevector")

    # トランスパイル
    tqc = transpile(full_qc, simulator, optimization_level=0)
    #print(tqc)
    # 実行
    result = simulator.run(tqc).result()

    # 最終状態ベクトル取得
    final_state = result.get_statevector()
    return final_state

def apply_time_evolution_gpu(eigenvector, time_evolution_circuit: QuantumCircuit, n_qubits, t, use_gpu):
    """
    基底状態を量子回路に変換し、回路上で時間発展演算子を作用させる。
    args:
        eigenvector : 基底状態のベクトル
        time_evolution_circuit : 量子回路に変換された時間発展演算子
    return:
        e^(iHt)|Ψ_0> のベクトル
    """

    if AerSimulator is None:
        print(f"qiskit_aer unavailable, falling back to CPU statevector simulation: {AER_IMPORT_ERROR}")
        final_state = evolve_with_statevector(eigenvector, time_evolution_circuit)
        path = f'/home/AbeHiromu/myproject/result/{t}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(final_state, f)
        return t, final_state

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(use_gpu)
    init = Statevector(eigenvector)
    full_qc = QuantumCircuit(n_qubits)
    full_qc.set_statevector(init)
    all_qc = full_qc.compose(time_evolution_circuit)
    
    # for instr, qargs, cargs in full_qc.data:
    #     all_qc.barrier()   # 各ゲートの間にバリアを入れる
    #     all_qc.append(instr, qargs, cargs)

    # 結果に statevector を含めるよう明示的に保存命令を追加
    all_qc.save_statevector()

    # シミュレーター
    simulator = AerSimulator(
    method="statevector",
    device = 'GPU'
    # ,blocking_enable = True,           # 分散／チャンク実行を有効化 :contentReference[oaicite:0]{index=0}
    # blocking_qubits = 9,             # １チャンクあたりの qubit 数
    )


    # トランスパイル
    tqc = transpile(all_qc, simulator, optimization_level=0)
    #print(tqc)
    try:
        # 実行
        now = datetime.now()
        print(f'circuit run {now.strftime("%Y-%m-%d %H:%M:%S")}')
        result = simulator.run(tqc).result()
        now = datetime.now()
        print(f'circuit applied {now.strftime("%Y-%m-%d %H:%M:%S")}')
        # 最終状態ベクトル取得
        now = datetime.now()
        final_state = result.get_statevector()
        path = f'/home/AbeHiromu/myproject/result/{t}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(final_state, f)
        # metadata = result.to_dict()
        # print(metadata)
    except Exception as e:
        print(f"シミュレーション失敗: {e}")
        return None
    return t, final_state

def apply_time_evolution_gpu_multi(eigenvector, time_evolution_circuit: QuantumCircuit, n_qubits,t, mol, num_w, ham_name,gpu):
    """
    基底状態を量子回路に変換し、回路上で時間発展演算子を作用させる。
    args:
        eigenvector : 基底状態のベクトル
        time_evolution_circuit : 量子回路に変換された時間発展演算子
    return:
        e^(iHt)|Ψ_0> のベクトル
    """

    if AerSimulator is None:
        print(f"qiskit_aer unavailable, falling back to CPU statevector simulation: {AER_IMPORT_ERROR}")
        final_state = evolve_with_statevector(eigenvector, time_evolution_circuit)
        rt = round(t,5)
        path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(final_state, f)
        return

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu)
    eigenvector = Statevector(eigenvector)
    full_qc = QuantumCircuit(n_qubits)
    full_qc.set_statevector(eigenvector)
    free_var('eigenvector',locals())
    full_qc = full_qc.compose(time_evolution_circuit)
    free_var('time_evolution_circuit',locals())
    
    # for instr, qargs, cargs in full_qc.data:
    #     all_qc.barrier()   # 各ゲートの間にバリアを入れる
    #     all_qc.append(instr, qargs, cargs)

    # 結果に statevector を含めるよう明示的に保存命令を追加
    full_qc.save_statevector()

    # シミュレーター
    simulator = AerSimulator(
    method="statevector",
    device = 'GPU'
    # ,blocking_enable = True,           # 分散／チャンク実行を有効化 :contentReference[oaicite:0]{index=0}
    # blocking_qubits = 21,             # １チャンクあたりの qubit 数
    )

    # トランスパイル
    tqc = transpile(full_qc, simulator, optimization_level=0)
    free_var('full_qc',locals())
    # tqc = transpile(all_qc, simulator)
    #print(tqc)
    # 実行
    now = datetime.now()
    print(f'circuit run {now.strftime("%Y-%m-%d %H:%M:%S")}')
    result = simulator.run(tqc).result()
    now = datetime.now()
    free_var('tqc',locals())
    print(f'circuit applied {now.strftime("%Y-%m-%d %H:%M:%S")}')
    # 最終状態ベクトル取得
    now = datetime.now()
    print(f'get statevector {now.strftime("%Y-%m-%d %H:%M:%S")}')
    final_state = result.get_statevector()
    rt = round(t,5)
    path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(final_state, f)

def apply_time_evolution_gpu_savememory(eigenvector, time_evolution_circuit: QuantumCircuit, n_qubits,t, mol, num_w, ham_name,gpu):
    from typing import List, Tuple

    def split_circuit(
        circuit: QuantumCircuit,
        num_splits: int
    ) -> List[QuantumCircuit]:
        """
        circuit.data（Instruction, qargs, cargs のリスト）を
        num_splits 個のサブサーキットにほぼ均等に分割して返す。
        元の回路が持つレジスタ構成（qregs, cregs）をそのままコピーします。
        """
        instructions: List[Tuple] = circuit.data
        total_instr = len(instructions)
        subcircuits: List[QuantumCircuit] = []

        for i in range(num_splits):
            start = (total_instr * i) // num_splits
            end   = (total_instr * (i + 1)) // num_splits

            # 元と同じレジスタ構成をコピー
            sub = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=f"{circuit.name}_part{i}")

            # 分割された命令をそのまま append する
            for inst, qargs, cargs in instructions[start:end]:
                sub.append(inst, qargs, cargs)

            subcircuits.append(sub)

        return subcircuits

    if AerSimulator is None:
        print(f"qiskit_aer unavailable, falling back to CPU statevector simulation: {AER_IMPORT_ERROR}")
        state = evolve_with_statevector(eigenvector, time_evolution_circuit)
        rt = round(t,5)
        path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu)
    part = split_circuit(time_evolution_circuit,10)

    for i, sub in enumerate(part):
        if i==0:
            eigenvector = Statevector(eigenvector)
            init_qc = QuantumCircuit(sub.num_qubits)
            init_qc.set_statevector(eigenvector)
            sub = init_qc.compose(sub) # compose で初期状態回路をセットする
            free_var('eigenvector',locals())
        else:
            init_qc = QuantumCircuit(sub.num_qubits)
            init_qc.set_statevector(state)
            sub = init_qc.compose(sub)
            free_var('state',locals())
        sub.save_statevector()
        simulator = AerSimulator()
        simulator.set_options(method="statevector",device = 'GPU')
        tqc = transpile(sub, simulator, optimization_level=0)
        free_var('sub',locals())
        result = simulator.run(tqc).result()
        free_var('tqc',locals())
        now = datetime.now()
        print(f'applyed_{i}_{now.strftime("%Y-%m-%d %H:%M:%S")}')
        state = result.get_statevector()
    rt = round(t,5)
    path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def tEvolution_vector(ham_list, t, n_qubits, ori_vec, num_w):
    evo_qc = QuantumCircuit(n_qubits)
    print('make circuit')
    w_trotter(evo_qc, ham_list, t, n_qubits, num_w)
    print('done')
    final_state = apply_time_evolution(ori_vec, evo_qc, n_qubits)
    return t, final_state

def tEvolution_vector_grouper(clique_list, t, n_qubits, ori_vec, num_w):
    evo_qc = QuantumCircuit(n_qubits)
    print('make circuit')
    num_exp = w_trotter_grouper(evo_qc, clique_list, t, n_qubits, num_w)
    print('done')
    final_state = apply_time_evolution(ori_vec, evo_qc, n_qubits)
    return t, final_state, num_exp

def make_circuit(ham_list, t, n_qubits, num_w):
    evo_qc = QuantumCircuit(n_qubits)
    print('make circuit')
    w_trotter(evo_qc, ham_list, t, n_qubits, num_w)
    print('done')
    return evo_qc, t

def make_circuit_gr(ham_list, t, n_qubits, num_w):
    evo_qc = QuantumCircuit(n_qubits)
    num_exp = w_trotter_grouper(evo_qc, ham_list, t, n_qubits, num_w)
    return evo_qc, t, num_exp
