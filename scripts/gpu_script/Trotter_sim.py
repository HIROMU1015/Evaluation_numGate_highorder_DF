import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig 
import scipy.sparse as sp
from scipy.sparse import eye, identity, save_npz, load_npz
from scipy.sparse.linalg import eigsh, matrix_power, eigs, norm, expm # type: ignore

from openfermion.transforms import get_fermion_operator, jordan_wigner, symmetry_conserving_bravyi_kitaev, bravyi_kitaev
from openfermion.linalg import get_sparse_operator
from openfermion.chem import MolecularData
from openfermion.ops import QubitOperator, FermionOperator
from openfermionpyscf import run_pyscf
from openfermion import count_qubits, commutator
import pyscf
from pyscf import gto, scf, mcscf

import multiprocessing
from multiprocessing import Pool

from functools import reduce
import math
import time
import gc
import os
import shutil
import tempfile
import gzip
import pickle
from datetime import datetime

import qiskit_tEvolutionOperator as qte

optimal_distance = {'LiH':1.51,'H2':0.71,'H3':0.85,'H3tri':0.97,'H4':1,'H5':1.01,'O2':1.2, 'Cr2':1.00,
                    'H5':1,'H6':1,'H8':1,'H9':1.0,'H10':1,'H11':1,'H12':1,'H13':1,'H14':1,'H15':1,'OH':1.07,'BeH2':1.36,'He2':1.16, 'HF':0.96, 'CH4':1.08, 'H2O':1.00, 'CO2':1.16}

def free_var(name, scope):
    if name in scope:
        del scope[name]
        import gc; gc.collect()

def call_geometry(mol_type, distance):
    """
    原子配置を関数から呼び出す。

    Args:
        mol_type(str): H2 のように分子式を入力
        distance: 原子間距離

    Returns:
        geometry: 原子配置
        multiplicity: スピン多重度
        charge: 電荷
    """

    def HF(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, 0)),("F", (0, 0, distance))]
        return geometry, multiplicity, charge

    def H2(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5)))]
        return geometry, multiplicity, charge

    def H3(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def H4(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5)))]
        return geometry, multiplicity, charge

    def H5(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2)))]
        return geometry, multiplicity, charge

    def H6(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-2.5))),("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5))),("H", (0, 0, distance * (2.5)))]
        return geometry, multiplicity, charge

    def H7(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-3))),("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2))),("H", (0, 0, distance * (3)))]
        return geometry, multiplicity, charge

    def H8(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-3.5))),("H", (0, 0, distance * (-2.5))),("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5))),("H", (0, 0, distance * (2.5))),("H", (0, 0, distance * (3.5)))]
        return geometry, multiplicity, charge

    def H9(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-4))),("H", (0, 0, distance * (-3))),("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2))),("H", (0, 0, distance * (3))),("H", (0, 0, distance * (4)))]
        return geometry, multiplicity, charge

    def H10(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-4.5))),("H", (0, 0, distance * (-3.5))),("H", (0, 0, distance * (-2.5))),("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5))),("H", (0, 0, distance * (2.5))),("H", (0, 0, distance * (3.5))),("H", (0, 0, distance * (4.5)))]
        return geometry, multiplicity, charge

    def H11(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-5))),("H", (0, 0, distance * (-4))),("H", (0, 0, distance * (-3))),("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2))),("H", (0, 0, distance * (3))),("H", (0, 0, distance * (4))),("H", (0, 0, distance * (5)))]
        return geometry, multiplicity, charge

    def H12(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-5.5))),("H", (0, 0, distance * (-4.5))),("H", (0, 0, distance * (-3.5))),("H", (0, 0, distance * (-2.5))),("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5))),("H", (0, 0, distance * (2.5))),("H", (0, 0, distance * (3.5))),("H", (0, 0, distance * (4.5))),("H", (0, 0, distance * (5.5)))]
        return geometry, multiplicity, charge

    def H13(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-6))),("H", (0, 0, distance * (-5))),("H", (0, 0, distance * (-4))),("H", (0, 0, distance * (-3))),("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2))),("H", (0, 0, distance * (3))),("H", (0, 0, distance * (4))),("H", (0, 0, distance * (5))),("H", (0, 0, distance * (6)))]
        return geometry, multiplicity, charge

    def H14(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-6.5))),("H", (0, 0, distance * (-5.5))),("H", (0, 0, distance * (-4.5))),("H", (0, 0, distance * (-3.5))),("H", (0, 0, distance * (-2.5))),("H", (0, 0, distance * (-1.5))),("H", (0, 0, distance * (-0.5))),("H", (0, 0, distance * (0.5))),("H", (0, 0, distance * (1.5))),("H", (0, 0, distance * (2.5))),("H", (0, 0, distance * (3.5))),("H", (0, 0, distance * (4.5))),("H", (0, 0, distance * (5.5))),("H", (0, 0, distance * (6.5)))]
        return geometry, multiplicity, charge
    
    def H15(distance):
        multiplicity = 3
        charge = +1 
        geometry = [("H", (0, 0, distance * (-7))),("H", (0, 0, distance * (-6))),("H", (0, 0, distance * (-5))),("H", (0, 0, distance * (-4))),("H", (0, 0, distance * (-3))),("H", (0, 0, distance * (-2))),("H", (0, 0, distance * (-1))),("H", (0, 0, 0)),("H", (0, 0, distance * (1))),("H", (0, 0, distance * (2))),("H", (0, 0, distance * (3))),("H", (0, 0, distance * (4))),("H", (0, 0, distance * (5))),("H", (0, 0, distance * (6))),("H", (0, 0, distance * (7)))]
        return geometry, multiplicity, charge

    def LiH(distance):
        multiplicity = 1
        charge = 0 
        geometry = [("Li", (0, 0, 0)),("H", (0, 0, distance))]
        return geometry, multiplicity, charge

    def OH(distance):
        multiplicity = 1
        charge = -1
        geometry = [("O", (0, 0, 0)),("H", (0, 0, distance))]
        return geometry, multiplicity, charge


    def BeH2(distance):
        multiplicity = 3
        charge = 0 
        geometry = [("H", (0, 0, distance * (-1))),("Be", (0, 0, 0)),("H", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def CO2(distance):
        multiplicity = 1
        charge = 0 
        geometry = [("O", (0, 0, distance * (-1))),("C", (0, 0, 0)),("O", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def He2(distance):
        multiplicity = 2
        charge = +1
        geometry = [("He", (0, 0, distance * (-0.5))),("He", (0, 0, distance * (0.5)))]
        return geometry, multiplicity, charge

    def CH4(distance):
        charge = 0 
        multiplicity = 1 
        a = distance / np.sqrt(3) 
        geometry = [
            ("C", (0.0, 0.0, 0.0)),  
            ("H", (a, a, a)),  
            ("H", (-a, -a, a)),  
            ("H", (-a, a, -a)),  
            ("H", (a, -a, -a)),  
        ]
        return geometry, multiplicity, charge

    def H2O(bond_length):
        charge = 0 
        multiplicity = 1  
        angle = np.radians(104.5)
        geometry = [
            ("O", (0.0, 0.0, 0.0)),  
            ("H", (bond_length, 0.0, 0.0)),  
            ("H", (bond_length * np.cos(angle), bond_length * np.sin(angle), 0.0)),  
        ]
        return geometry, multiplicity, charge

    def O2(distance):
        multiplicity = 3
        charge = 0
        geometry = [("O", (0, 0, -0.5*distance)),("O", (0, 0, 0.5*distance))]
        return geometry, multiplicity, charge

    def Cr2(distance):
        multiplicity = 1
        charge = 0
        geometry = [("Cr", (0, 0, 0)),("Cr", (0, 0, distance))]
        return geometry, multiplicity, charge


    if mol_type == 'H6':
        geometry, multiplicity, charge = H6(distance)
    if mol_type == 'H8':
        geometry, multiplicity, charge = H8(distance)
    if mol_type == 'H9':
        geometry, multiplicity, charge = H9(distance)
    if mol_type == 'H10':
        geometry, multiplicity, charge = H10(distance)
    if mol_type == 'H12':
        geometry, multiplicity, charge = H12(distance)
    if mol_type == 'H11':
        geometry, multiplicity, charge = H11(distance)
    if mol_type == 'H13':
        geometry, multiplicity, charge = H13(distance)
    if mol_type == 'H14':
        geometry, multiplicity, charge = H14(distance)
    if mol_type == 'H15':
        geometry, multiplicity, charge = H15(distance)
    elif mol_type == 'H5':
        geometry, multiplicity, charge = H5(distance)
    elif mol_type == 'H4':
        geometry, multiplicity, charge = H4(distance)
    elif mol_type == 'H2':
        geometry, multiplicity, charge = H2(distance)
    elif mol_type == 'H3':
        geometry, multiplicity, charge = H3(distance)
    elif mol_type == 'LiH':
        geometry, multiplicity, charge = LiH(distance)
    elif mol_type == 'BeH2':
        geometry, multiplicity, charge = BeH2(distance)
    elif mol_type == 'OH':
        geometry, multiplicity, charge = OH(distance)
    elif mol_type == 'He2':
        geometry, multiplicity, charge = He2(distance)
    elif mol_type == 'HF':
        geometry, multiplicity, charge = HF(distance)
    elif mol_type == 'CH4':
        geometry, multiplicity, charge = CH4(distance)
    elif mol_type == 'H2O':
        geometry, multiplicity, charge = H2O(distance)
    elif mol_type == 'CO2':
        geometry, multiplicity, charge = CO2(distance)
    elif mol_type == 'O2':
        geometry, multiplicity, charge = O2(distance)
    elif mol_type == 'Cr2':
        geometry, multiplicity, charge = Cr2(distance)

    return geometry, multiplicity, charge

def geo(mol_type, distance=None):
    if distance == None:
        distance = optimal_distance[mol_type]
    geometry, multiplicity, charge = call_geometry(mol_type, distance)
    return geometry, multiplicity, charge

def ham_ground_energy(jw_hamiltonian): 
    sum_matrix = get_sparse_operator(jw_hamiltonian)
    vals, vecs = eigsh(sum_matrix, k=1,return_eigenvectors=True, which="SA") 
    max_eig = eigsh(sum_matrix, k=1, return_eigenvectors=False, which="LA")
    exact_vals = vals[0]
    return exact_vals, vecs, max_eig[0]

def ham_list_maker(hamiltonian):
    """
    openfermion のハミルトニアンを項ごとにリストに格納する

    Args:
        hamiltonian: JW, BK変換されたハミルトニアン

    Returns:
        ham_list: 項ごとに格納したリスト
    """
    ham_list = []
    for term in hamiltonian:
        ham_list.append(term)
    #ham_list.pop(0)
    return ham_list

def jw_hamiltonian_maker(mol_type, distance = None):
    """
    JW変換されたハミルトニアンを構築する。

    Args:
        mol_type(str): H2 のように分子式を入力
        distance: 原子間距離

    Returns:
        jw_hamiltonian: JW変換されたハミルトニアン, 
        HF_energy: HFエネルギー, 
        ham_name: ハミルトニアンの保存名, 
        num_qubits: Qビット数
    """

    basis = "sto-3g"  #基底関数
    if distance == None:
        distance = optimal_distance[mol_type]
    geometry, multiplicity, charge = geo(mol_type,distance)
    name_distance = int(distance * 100)
    description = f"distance_{name_distance}_charge_{charge}"  #保存先のファイル名
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule,run_scf=1,run_fci=0)
    HF_energy = molecule.hf_energy
    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    num_qubits = count_qubits(jw_hamiltonian)
    file_path = molecule.filename
    ham_name = os.path.splitext(os.path.basename(file_path))[0]
    print(ham_name)
    return jw_hamiltonian, HF_energy, ham_name, num_qubits



"""
積公式の係数呼び出し用関数
"""
#Mauro's 8th order product formula
def generate_w8_list():
    w_1to8 = [0.29137384767986663096528500968049,
          0.26020394234904150277316667709864,
          0.18669648149540687549831902999911,
          -0.40049110428180105319963667975074,
          0.15982762208609923217390166127256,
          -0.38400573301491401473462588779099,
          0.56148845266356446893590729572808,
          0.12783360986284110837857554950443]

    w0_1to8 = [1 - 2*sum(w_1to8)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to8 + w_1to8
    return w

#Yoshida's 8th order product formula
def yoshida_list():
    w_1to7 = [-1.61582374150097,
          -2.44699182370524,
         -0.0071698941970812, 
         2.44002732616735, 
         0.157739928123617, 
         1.82020630970714, 
         1.04242620869991]
    w0_1to7 = [1 - 2*sum(w_1to7)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to7 + w_1to7
    return w

#Mauro's 10th order product formula(m=15)
def generate_w10_15_list():
    w_1to15 = [0.14552859955499429739088135596618,
          -0.48773512068133537309419933740564,
          0.12762011242429535909727342301656 ,
          0.70225450019485751220143080587959,
          -0.62035679146761710925756521405042,
          0.39099152412786178133688869373114,
        0.17860253604355465807791041367045,
        -0.80455783177921776295588528272593,
        0.053087216442758242118687385646283,
        0.86836307910275556258687030904753,
        -0.85326297197907834671536254437991 ,
        -0.11732457198874083224967699358383,
        0.03827345494186056632406947772047,
        0.74843529029532498233997793305357,
        0.30208715621975773712410948025906]

    w0_1to8 = [1 - 2*sum(w_1to15)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to8 + w_1to15
    return w

#Mauro's 10th order product formula(m=16)
def generate_w10_16_list():
    w_1to15 = [ -0.4945013179955571856347147977644,
          0.2904317222970121479878414292093,
          0.34781541068705330937913890281003,
          -0.98828132118546184603769781410676,
          0.98855187532756405235733957305613,
          -0.34622976933123177430694714630668,
        0.20218952619073117554714280367018,
        0.13064273069786247787208895471461,
        -0.26441199183146805554735845490359,
        0.060999140559210408869096992291531,
        -0.6855442489606141359108973267028,
        -0.15843692473786584550599206557006,
        0.15414691779958299150286452215575,
        0.66715205827214320371061839297055,
        0.20411874474696598289603677693511,
        0.081207318210272593225087711441684]

    w0_1to8 = [1 - 2*sum(w_1to15)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to8 + w_1to15
    return w

#Yoshida's 4th order product formula
def generate_w3_list(): #s3odr4
    w = [
         -1*(2**(1/3)) / (2-2**(1/3)),
         1/(2-2**(1/3))
    ]
    return w

def generate_w1_list():
    w = [
         1
    ]
    return w

def generate_myw1_list():
    w1 = [
          #1.52886228e-03, -2.14403532e+00 , 1.44778256e+00
          0.40653666 ,0.21638706 ,0.14924614
    ]
    w0_1to8 = [1 - 2*sum(w1)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to8 + w1
    return w

def generate_myw4_list():
    w1to3 = [
          #1.52886228e-03, -2.14403532e+00 , 1.44778256e+00
          0.42008729, 0.40899193
    ]
    w0_1to3 = [1 - 2*sum(w1to3)]
    # パラメータの設定
    #w_0 = 1 - 2sum(w_i)
    w = w0_1to3 + w1to3
    return w

def min_hamiltonian_grouper(hamiltonian, ham_name): # Qubithamiltonian を渡してグルーピング
    def are_commuting(op1: QubitOperator, op2: QubitOperator) -> bool:
        if len(op1.terms) != 1 or len(op2.terms) != 1:
            raise ValueError("Only single-term QubitOperators are supported.")

        term1 = list(op1.terms.keys())[0]
        term2 = list(op2.terms.keys())[0]

        n_anticommute = 0
        qubits = set(index for index, _ in term1).union(index for index, _ in term2)

        for q in qubits:
            op1_pauli = dict(term1).get(q, 'I')
            op2_pauli = dict(term2).get(q, 'I')
            if op1_pauli == 'I' or op2_pauli == 'I':
                continue
            if op1_pauli != op2_pauli:
                n_anticommute += 1

        return n_anticommute % 2 == 0

def trotter_error_plt_qc_gr(s_time, e_time, dividing, mol_type, num_w, use_gpu):
    start = time.time()
    t_values = list(np.arange(s_time, e_time, dividing))
    mi_t_values = [-1*t for t in t_values]

    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type) 
    
    path = f"/home/AbeHiromu/myproject/pkldir/{mol_type}"
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    clique_list,num_qubits,E, ori_vec = data

    ham_list = ham_list_maker(jw)
    for _, coeff in ham_list[0].terms.items():
        const = coeff.real

    # if mol_type == 'H2':
    #    jw_hamiltonian = jw
    #    E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
    #    clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    # else:
    #     clique_list,num_qubits,E, ori_vec = qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)

    E = E - const
    ham_name += '_grouping'
    task_args = [(clique_list, t, num_qubits, num_w) for t in mi_t_values]
    print('make circuit')
    with Pool(processes=len(mi_t_values)) as pool:
        qc_t_numexp_list = pool.starmap(qte.make_circuit_gr, task_args) # return evolve_qc, t
    print('done')
    now = datetime.now()
    print(f'circuit phase {now.strftime("%Y-%m-%d %H:%M:%S")}')
    
    task_args = [(ori_vec, evolve_qc, num_qubits, t, use_gpu) for evolve_qc, t, _ in qc_t_numexp_list]
    with Pool(processes=len(mi_t_values)) as pool:
        final_state_list = pool.starmap(qte.apply_time_evolution_gpu, task_args) # return t,final_state

    error_list_pertur = []
    t_values = []
    for t, vector in final_state_list:
        t *= -1
        vector = vector.data.reshape(-1,1)
        tevolution = np.exp(1j*E*t)
        delta_psi = (vector - (tevolution * ori_vec)) / (1j*t)
        innerproduct = (ori_vec.conj().T @ delta_psi)
        innerproduct = innerproduct.real / np.cos(E*t)
        error_list_pertur.extend(abs((innerproduct.real)))
        t_values.append(t)


    error_log = np.log10(error_list_pertur)
    time_log = np.log10(t_values)

    n_w_dir = {None:2,2:2,3:4,5_1:4,8:8,10_15:10,10_16:10,'yoshida':8,'my1':4,'my4':4}
    n_w = n_w_dir[num_w]
    set_expo_error = error_log - n_w*time_log
    ave_coeff = np.mean(set_expo_error)
    ave_coeff = 10**(ave_coeff)


    linear_error = np.polyfit(time_log,error_log,1)
    print("error exponent :" + str(linear_error[0]))
    #In Y = CX^a, logY = AlogX + B as 10^B = C
    coeff = 10**linear_error[1] 
    print("error coefficient :" + str(coeff))
    print(f'average coefficient : {ave_coeff}')

    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)

    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr = corr_matrix[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)

    print(f'error_list:{error_list_pertur}')
    end =time.time()
    print(f'execute time {end - start}')

def trotter_error_plt_qc_gr_multi(s_time, e_time, dividing, mol_type, num_w, gpu):
    start = time.time()
    t_values = list(np.arange(s_time, e_time, dividing))
    mi_t_values = [-1*t for t in t_values]
    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type) 
    ham_list = ham_list_maker(jw)
    for _, coeff in ham_list[0].terms.items():
        const = coeff.real
    if mol_type == 'H2':
       jw_hamiltonian = jw
       E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
       clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    else:
        clique_list,num_qubits,E, ori_vec = qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)

    E = E - const
    ham_name += '_grouping'
    task_args = [(clique_list, t, num_qubits, num_w) for t in mi_t_values]
    print('next')
    with Pool(processes=len(mi_t_values)) as pool:
        qc_t_numexp_list = pool.starmap(qte.make_circuit_gr, task_args) # return evolve_qc, t

    free_var('clique_list',locals())

    now = datetime.now()
    print(f'circuit phase {now.strftime("%Y-%m-%d %H:%M:%S")}')

    if len(gpu) > len(mi_t_values):
        part = len(gpu) // len(mi_t_values)
        split_gpu = [gpu[i*part:(i+1)*part] for i in range(len(mi_t_values))]
        task_args = [(ori_vec, evolve_qc, num_qubits, t, mol_type, num_w, ham_name, split_gpu[i]) for i, (evolve_qc, t, _) in enumerate(qc_t_numexp_list)]
        with Pool(processes=len(mi_t_values)) as pool:
            pool.starmap(qte.apply_time_evolution_gpu_savememory, task_args) # return t,final_state
    else:
        task_args = [(ori_vec, evolve_qc, num_qubits, t, mol_type, num_w, ham_name, gpu[i]) for i, (evolve_qc, t, _) in enumerate(qc_t_numexp_list)]
        with Pool(processes=len(mi_t_values)) as pool:
            pool.starmap(qte.apply_time_evolution_gpu_savememory, task_args) # return t,final_state

    final_state_list = []
    for t in mi_t_values:
        rt = round(t,5)
        path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
        with open(path,'rb') as f:
            state = pickle.load(f)
        final_state_list.append((t,state))


    for _ in range(2):
        ori_vec *= -1
        error_list_pertur = []
        t_values = []
        for t, vector in final_state_list:
            t *= -1
            vector = vector.data.reshape(-1,1)
            tevolution = np.exp(1j*E*t)
            delta_psi = (vector - (tevolution * ori_vec)) / (1j*t)
            innerproduct = (ori_vec.conj().T @ delta_psi)
            innerproduct = innerproduct.real / np.cos(E*t)
            error_list_pertur.extend(abs((innerproduct.real)))
            t_values.append(t)


        error_log = np.log10(error_list_pertur)
        time_log = np.log10(t_values)

        n_w_dir = {None:2,2:2,3:4,5_1:4,8:8,10_15:10,10_16:10,'yoshida':8,'my1':4,'my4':4}
        n_w = n_w_dir[num_w]
        set_expo_error = error_log - n_w*time_log
        ave_coeff = np.mean(set_expo_error)
        ave_coeff = 10**(ave_coeff)

        linear_error = np.polyfit(time_log,error_log,1)
        print("error exponent :" + str(linear_error[0]))
        #In Y = CX^a, logY = AlogX + B as 10^B = C
        coeff = 10**linear_error[1] 
        print("error coefficient :" + str(coeff))
        print(f'average coefficient : {ave_coeff}')
        print(f'error_list:{error_list_pertur}')
        
        data = {'expo':linear_error[0], 'coeff':coeff}
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}'
        with open(path, 'wb') as f:
            pickle.dump(data,f)
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}_ave'
        with open(path, 'wb') as f:
            pickle.dump(ave_coeff,f)
        end =time.time()
        print(f'execute time {end - start}')

def trotter_error_plt_qc_gr_ao(s_time, e_time, dividing, mol_type, n, num_w, gpu):
    start = time.time()
    num_w_dir = f'w{num_w}'
    t_values = list(np.arange(s_time, e_time, dividing))
    mi_t_values = [-1*t for t in t_values]
    print(f't_{t_values}')
    clique_list,num_qubits,E, ori_vec, ham_name = qte.make_fci_vector_from_pyscf_solver_grouper_ao(mol_type,n)
    print(f'qubit{num_qubits}')
    for term, coeff in clique_list[0].terms.items():
        if not term:
            const = coeff

    E = E - const

    task_args = [(clique_list, t, num_qubits, num_w) for t in mi_t_values]
    print('next')
    with Pool(processes=len(mi_t_values)) as pool:
        qc_t_numexp_list = pool.starmap(qte.make_circuit_gr, task_args) # return evolve_qc, t

    now = datetime.now()
    print(f'circuit phase {now.strftime("%Y-%m-%d %H:%M:%S")}')

    task_args = [(ori_vec, evolve_qc, num_qubits, t, mol_type, num_w, ham_name,gpu[i]) for i, (evolve_qc, t, _) in enumerate(qc_t_numexp_list)]
    with Pool(processes=len(mi_t_values)) as pool:
        pool.starmap(qte.apply_time_evolution_gpu_multi, task_args) # return t,final_state

    final_state_list = []
    for t in mi_t_values:
        rt = round(t,5)
        path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
        with open(path,'rb') as f:
            state = pickle.load(f)
        final_state_list.append((t,state))


    for _ in range(2):
        ori_vec *= -1
        error_list_pertur = []
        t_values = []
        for t, vector in final_state_list:
            t *= -1
            vector = vector.data.reshape(-1,1)
            tevolution = np.exp(1j*E*t)
            delta_psi = (vector - (tevolution * ori_vec)) / (1j*t)
            innerproduct = (ori_vec.conj().T @ delta_psi)
            innerproduct = innerproduct.real / np.cos(E*t)
            error_list_pertur.extend(abs((innerproduct.real)))
            t_values.append(t)


        error_log = np.log10(error_list_pertur)
        time_log = np.log10(t_values)

        n_w_dir = {None:2,2:2,3:4,5_1:4,8:8,10_15:10,10_16:10,'yoshida':8,'my1':4}
        n_w = n_w_dir[num_w]
        set_expo_error = error_log - n_w*time_log
        ave_coeff = np.mean(set_expo_error)
        ave_coeff = 10**(ave_coeff)

        linear_error = np.polyfit(time_log,error_log,1)
        print("error exponent :" + str(linear_error[0]))
        #In Y = CX^a, logY = AlogX + B as 10^B = C
        coeff = 10**linear_error[1] 
        print("error coefficient :" + str(coeff))
        print(f'average coefficient : {ave_coeff}')
        print(f'error_list:{error_list_pertur}')
        data = {'expo':linear_error[0], 'coeff':coeff}
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}'
        with open(path, 'wb') as f:
            pickle.dump(data,f)
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}_ave'
        with open(path, 'wb') as f:
            pickle.dump(ave_coeff,f)
        end =time.time()
        print(f'execute time {end - start}')
    print(ham_name)

def trotter_error_plt_qc_gr_ao_load(s_time, e_time, dividing, mol_type, num_w, file_path, gpu):
    import gzip
    print(f'gpu_{gpu}')
    start = time.time()
    t_values = list(np.arange(s_time, e_time, dividing))
    mi_t_values = [-1*t for t in t_values]
    print(f't_{t_values}')
    with gzip.open(file_path, 'rb') as f:
        clique_qubit_E_vec_name = pickle.load(f)
        
    clique_list,num_qubits,E, ori_vec, ham_name = clique_qubit_E_vec_name
    print(ham_name)
    print(f'qubit{num_qubits}')
    for term, coeff in clique_list[0].terms.items():
        if not term:
            const = coeff

    E = E - const

    task_args = [(clique_list, t, num_qubits, num_w) for t in mi_t_values]
    print('next')
    with Pool(processes=len(mi_t_values)) as pool:
        qc_t_numexp_list = pool.starmap(qte.make_circuit_gr, task_args) # return evolve_qc, t

    free_var('clique_list',locals())

    now = datetime.now()
    print(f'circuit phase {now.strftime("%Y-%m-%d %H:%M:%S")}')

    if len(gpu) > len(mi_t_values):
        part = len(gpu) // len(mi_t_values)
        split_gpu = [gpu[i*part:(i+1)*part] for i in range(len(mi_t_values))]
        task_args = [(ori_vec, evolve_qc, num_qubits, t, mol_type, num_w, ham_name, split_gpu[i]) for i, (evolve_qc, t, _) in enumerate(qc_t_numexp_list)]
        with Pool(processes=len(mi_t_values)) as pool:
            pool.starmap(qte.apply_time_evolution_gpu_savememory, task_args) # return t,final_state
    else:
        task_args = [(ori_vec, evolve_qc, num_qubits, t, mol_type, num_w, ham_name, gpu[i]) for i, (evolve_qc, t, _) in enumerate(qc_t_numexp_list)]
        with Pool(processes=len(mi_t_values)) as pool:
            pool.starmap(qte.apply_time_evolution_gpu_savememory, task_args) # return t,final_state

    free_var('task_args',locals())

    final_state_list = []
    for t in mi_t_values:
        rt = round(t,5)
        path = f'/home/abe/myproject/result/{ham_name}_{rt}_w{num_w}.pkl'
        with open(path,'rb') as f:
            state = pickle.load(f)
        final_state_list.append((t,state))


    for _ in range(2):
        ori_vec *= -1
        error_list_pertur = []
        t_values = []
        for t, vector in final_state_list:
            t *= -1
            vector = vector.data.reshape(-1,1)
            tevolution = np.exp(1j*E*t)
            delta_psi = (vector - (tevolution * ori_vec)) / (1j*t)
            innerproduct = (ori_vec.conj().T @ delta_psi)
            innerproduct = innerproduct.real / np.cos(E*t)
            error_list_pertur.extend(abs((innerproduct.real)))
            t_values.append(t)


        error_log = np.log10(error_list_pertur)
        time_log = np.log10(t_values)

        n_w_dir = {None:2,2:2,3:4,5_1:4,8:8,10_15:10,10_16:10,'yoshida':8,'my1':4}
        n_w = n_w_dir[num_w]
        set_expo_error = error_log - n_w*time_log
        ave_coeff = np.mean(set_expo_error)
        ave_coeff = 10**(ave_coeff)

        linear_error = np.polyfit(time_log,error_log,1)
        print("error exponent :" + str(linear_error[0]))
        #In Y = CX^a, logY = AlogX + B as 10^B = C
        coeff = 10**linear_error[1] 
        print("error coefficient :" + str(coeff))
        print(f'average coefficient : {ave_coeff}')
        print(f'error_list:{error_list_pertur}')
        data = {'expo':linear_error[0], 'coeff':coeff}
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}'
        with open(path, 'wb') as f:
            pickle.dump(data,f)
        path = f'/home/abe/myproject/expo_coeff/{ham_name}_Operator_w{num_w}_ave'
        with open(path, 'wb') as f:
            pickle.dump(ave_coeff,f)

        end =time.time()
        print(f'execute time {end - start}')
    print(ham_name)

def pretreatment(s_time, e_time, dividing, mol_type, num_w):
    start = time.time()
    num_w_dir = f'w{num_w}'
    t_values = list(np.arange(s_time, e_time, dividing))
    mi_t_values = [-1*t for t in t_values]
    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type) 
    ham_list = ham_list_maker(jw)
    for _, coeff in ham_list[0].terms.items():
        const = coeff.real
    if mol_type == 'H2':
       jw_hamiltonian = jw
       E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
       clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    else:
        clique_list,num_qubits,E, ori_vec = qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)

    E = E - const
    ham_name += '_grouping'
    task_args = [(clique_list, t, num_qubits, num_w) for t in mi_t_values]
    print('next')
    with Pool(processes=16) as pool:
        qc_t_numexp_list = pool.starmap(qte.make_circuit_gr, task_args) # return evolve_qc, t

    for qc, t,_ in qc_t_numexp_list:
        data = {'qc':qc, 'ori_vec':ori_vec, 'n_qubits':num_qubits}
        data_path = f"/home/abe/myproject/mpi_input/input_{t}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
    print('pretreatment done')
    return mi_t_values, E

def aftertreatment(t_values, E, num_w):
    print('start aftertreatment')
    final_state_list = []
    for t in range(t_values):
        data_path = f'/home/abe/myproject/mpi_output/output_{t}.pkl'
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        final_state_list.append(data['state'])

    for _ in range(2):
        ori_vec *= -1
        error_list_pertur = []
        t_values = []
        for t, vector in final_state_list:
            t *= -1
            vector = vector.data.reshape(-1,1)
            tevolution = np.exp(1j*E*t)
            delta_psi = (vector - (tevolution * ori_vec)) / (1j*t)
            innerproduct = (ori_vec.conj().T @ delta_psi)
            innerproduct = innerproduct.real / np.cos(E*t)
            error_list_pertur.extend(abs((innerproduct.real)))
            t_values.append(t)


        error_log = np.log10(error_list_pertur)
        time_log = np.log10(t_values)

        n_w_dir = {None:2,2:3,3:4,5_1:4,8:8,10_15:10,10_16:10,'yoshida':8}
        n_w = n_w_dir[num_w]
        set_expo_error = error_log - n_w*time_log
        ave_coeff = np.mean(set_expo_error)
        ave_coeff = 10**(ave_coeff)

        linear_error = np.polyfit(time_log,error_log,1)
        print("error exponent :" + str(linear_error[0]))
        #In Y = CX^a, logY = AlogX + B as 10^B = C
        coeff = 10**linear_error[1] 
        print("error coefficient :" + str(coeff))
        print(f'average coefficient : {ave_coeff}')
        print(f'error_list:{error_list_pertur}')
