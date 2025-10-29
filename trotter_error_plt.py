import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np
from collections import defaultdict
from numpy.linalg import eig
import scipy.sparse as sp
from scipy.sparse import eye, identity, save_npz, load_npz
from scipy.sparse.linalg import eigsh, matrix_power, eigs, norm, expm

from functools import reduce
import math
from typing import Optional, List, Dict, Tuple, Any

from openfermion.transforms import (
    get_fermion_operator,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
    bravyi_kitaev,
)
from openfermion.linalg import get_sparse_operator
from openfermion.chem import MolecularData
from openfermion.ops import QubitOperator, FermionOperator
from openfermionpyscf import run_pyscf
from openfermion import count_qubits, commutator
import pyscf
from pyscf import gto, scf, mcscf

import multiprocessing
from multiprocessing import Pool

import os
import shutil
import tempfile
import pickle

from Almost_optimal_grouping import Almost_optimal_grouper
import qiskit_time_evolution as qte

# =========================
# 設定セクション（魔法値の定数化）
# =========================
DEFAULT_BASIS = "sto-3g"  # 基底関数（数値は変更しない）
DEFAULT_DISTANCE = 1.0  # 原子間距離のデフォルト値（Å 相当の内部スケール）
PICKLE_DIR = "trotter_expo_coeff"
PICKLE_DIR_GROUPED = "trotter_expo_coeff_gr"
POOL_PROCESSES = 32  # 並列処理プロセス数（挙動は不変）


def call_geometry(mol_type: str, distance: float):
    """分子式と距離から原子配置・スピン多重度・電荷を返す。"""

    def HF(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, 0)), ("F", (0, 0, distance))]
        return geometry, multiplicity, charge

    def H2(distance):
        multiplicity = 1
        charge = 0
        geometry = [("H", (0, 0, distance * (-0.5))), ("H", (0, 0, distance * (0.5)))]
        return geometry, multiplicity, charge

    def H3(distance):
        multiplicity = 3
        charge = +1
        geometry = [("H", (0, 0, distance * (-1))), ("H", (0, 0, 0)), ("H", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def H4(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
        ]
        return geometry, multiplicity, charge

    def H5(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
        ]
        return geometry, multiplicity, charge

    def H6(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-2.5))),
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
            ("H", (0, 0, distance * (2.5))),
        ]
        return geometry, multiplicity, charge

    def H7(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-3))),
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
            ("H", (0, 0, distance * (3))),
        ]
        return geometry, multiplicity, charge

    def H8(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-3.5))),
            ("H", (0, 0, distance * (-2.5))),
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
            ("H", (0, 0, distance * (2.5))),
            ("H", (0, 0, distance * (3.5))),
        ]
        return geometry, multiplicity, charge

    def H9(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-4))),
            ("H", (0, 0, distance * (-3))),
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
            ("H", (0, 0, distance * (3))),
            ("H", (0, 0, distance * (4))),
        ]
        return geometry, multiplicity, charge

    def H10(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-4.5))),
            ("H", (0, 0, distance * (-3.5))),
            ("H", (0, 0, distance * (-2.5))),
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
            ("H", (0, 0, distance * (2.5))),
            ("H", (0, 0, distance * (3.5))),
            ("H", (0, 0, distance * (4.5))),
        ]
        return geometry, multiplicity, charge

    def H11(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-5))),
            ("H", (0, 0, distance * (-4))),
            ("H", (0, 0, distance * (-3))),
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
            ("H", (0, 0, distance * (3))),
            ("H", (0, 0, distance * (4))),
            ("H", (0, 0, distance * (5))),
        ]
        return geometry, multiplicity, charge

    def H12(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-5.5))),
            ("H", (0, 0, distance * (-4.5))),
            ("H", (0, 0, distance * (-3.5))),
            ("H", (0, 0, distance * (-2.5))),
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
            ("H", (0, 0, distance * (2.5))),
            ("H", (0, 0, distance * (3.5))),
            ("H", (0, 0, distance * (4.5))),
            ("H", (0, 0, distance * (5.5))),
        ]
        return geometry, multiplicity, charge

    def H13(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-6))),
            ("H", (0, 0, distance * (-5))),
            ("H", (0, 0, distance * (-4))),
            ("H", (0, 0, distance * (-3))),
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
            ("H", (0, 0, distance * (3))),
            ("H", (0, 0, distance * (4))),
            ("H", (0, 0, distance * (5))),
            ("H", (0, 0, distance * (6))),
        ]
        return geometry, multiplicity, charge

    def H14(distance):
        multiplicity = 1
        charge = 0
        geometry = [
            ("H", (0, 0, distance * (-6.5))),
            ("H", (0, 0, distance * (-5.5))),
            ("H", (0, 0, distance * (-4.5))),
            ("H", (0, 0, distance * (-3.5))),
            ("H", (0, 0, distance * (-2.5))),
            ("H", (0, 0, distance * (-1.5))),
            ("H", (0, 0, distance * (-0.5))),
            ("H", (0, 0, distance * (0.5))),
            ("H", (0, 0, distance * (1.5))),
            ("H", (0, 0, distance * (2.5))),
            ("H", (0, 0, distance * (3.5))),
            ("H", (0, 0, distance * (4.5))),
            ("H", (0, 0, distance * (5.5))),
            ("H", (0, 0, distance * (6.5))),
        ]
        return geometry, multiplicity, charge

    def H15(distance):
        multiplicity = 3
        charge = +1
        geometry = [
            ("H", (0, 0, distance * (-7))),
            ("H", (0, 0, distance * (-6))),
            ("H", (0, 0, distance * (-5))),
            ("H", (0, 0, distance * (-4))),
            ("H", (0, 0, distance * (-3))),
            ("H", (0, 0, distance * (-2))),
            ("H", (0, 0, distance * (-1))),
            ("H", (0, 0, 0)),
            ("H", (0, 0, distance * (1))),
            ("H", (0, 0, distance * (2))),
            ("H", (0, 0, distance * (3))),
            ("H", (0, 0, distance * (4))),
            ("H", (0, 0, distance * (5))),
            ("H", (0, 0, distance * (6))),
            ("H", (0, 0, distance * (7))),
        ]
        return geometry, multiplicity, charge

    def LiH(distance):
        multiplicity = 1
        charge = 0
        geometry = [("Li", (0, 0, 0)), ("H", (0, 0, distance))]
        return geometry, multiplicity, charge

    def OH(distance):
        multiplicity = 1
        charge = -1
        geometry = [("O", (0, 0, 0)), ("H", (0, 0, distance))]
        return geometry, multiplicity, charge

    def BeH2(distance):
        multiplicity = 3
        charge = 0
        geometry = [("H", (0, 0, distance * (-1))), ("Be", (0, 0, 0)), ("H", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def CO2(distance):
        multiplicity = 1
        charge = 0
        geometry = [("O", (0, 0, distance * (-1))), ("C", (0, 0, 0)), ("O", (0, 0, distance * (1)))]
        return geometry, multiplicity, charge

    def He2(distance):
        multiplicity = 2
        charge = +1
        geometry = [("He", (0, 0, distance * (-0.5))), ("He", (0, 0, distance * (0.5)))]
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
        geometry = [("O", (0, 0, -0.5 * distance)), ("O", (0, 0, 0.5 * distance))]
        return geometry, multiplicity, charge

    def Cr2(distance):
        multiplicity = 1
        charge = 0
        geometry = [("Cr", (0, 0, 0)), ("Cr", (0, 0, distance))]
        return geometry, multiplicity, charge

    # マップ化してガード節で早期return（分岐のネストを削減）
    _builders = {
        "H2": H2,
        "H3": H3,
        "H4": H4,
        "H5": H5,
        "H6": H6,
        "H7": H7,
        "H8": H8,
        "H9": H9,
        "H10": H10,
        "H11": H11,
        "H12": H12,
        "H13": H13,
        "H14": H14,
        "H15": H15,
        "LiH": LiH,
        "BeH2": BeH2,
        "OH": OH,
        "He2": He2,
        "HF": HF,
        "CH4": CH4,
        "H2O": H2O,
        "CO2": CO2,
        "O2": O2,
        "Cr2": Cr2,
    }
    if mol_type not in _builders:
        raise ValueError(f"Unknown mol_type '{mol_type}'. Supported keys: {sorted(_builders.keys())}")
    return _builders[mol_type](distance)


def geo(mol_type: str, distance: Optional[float] = None):
    """call_geometry の薄いラッパー（距離のデフォルトを適用）。"""
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
    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
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
        terms = [QubitOperator(term, coeff) for term, coeff in qubit_hamiltonian.terms.items()]
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


def morales_8th_list():
    w_1to8 = [0.29137384767986663096528500968049,
          0.26020394234904150277316667709864,
          0.18669648149540687549831902999911,
          -0.40049110428180105319963667975074,
          0.15982762208609923217390166127256,
          -0.38400573301491401473462588779099,
          0.56148845266356446893590729572808,
          0.12783360986284110837857554950443]

    w0_1to8 = [1 - 2*sum(w_1to8)]
    w = w0_1to8 + w_1to8
    return w

#Yoshida's 8th order product formula
def yoshida_8th_list():
    w_1to7 = [-1.61582374150097,
          -2.44699182370524,
         -0.0071698941970812, 
         2.44002732616735, 
         0.157739928123617, 
         1.82020630970714, 
         1.04242620869991]
    w0_1to7 = [1 - 2*sum(w_1to7)]
    # パラメータの設定
    w = w0_1to7 + w_1to7
    return w

#Mauro's 10th order product formula(m=15)
def morales_10th_m15_list():
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
def morales_10th_m16_list():
    w_1to16 = [ -0.4945013179955571856347147977644,
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

    w0 = [1 - 2*sum(w_1to16)]
    w = w0 + w_1to16
    return w

#Yoshida's 4th order product formula
def yoshida_4th_list(): #s3odr4
    w = [
         -1*(2**(1/3)) / (2-2**(1/3)),
         1/(2-2**(1/3))
    ]
    return w

def trotter_2nd_list():
    w = [
         1
    ]
    return w

def new_4th_m3_list():
    w1to3 = [
          0.40653666 ,0.21638706 ,0.14924614
    ]
    w0_1to3 = [1 - 2*sum(w1to3)]
    w = w0_1to3 + w1to3
    return w

def new_4th_m2_list():
    w1to2 = [
          0.42008729, 0.40899193
    ]
    w0_1to3 = [1 - 2*sum(w1to2)]
    w = w0_1to3 + w1to2
    return w


def _get_w_list(num_w: Any) -> List[float]:
    """積公式パラメータ w の系列を取得（分岐を関数化）。"""
    if num_w == '8th(Morales)':
        return morales_8th_list()
    if num_w == '10th(Morales)':
        return morales_10th_m16_list()
    if num_w == '4th':
        return yoshida_4th_list()
    if num_w == "8th(Yoshida)":
        return yoshida_8th_list()
    if num_w == '2nd':
        return trotter_2nd_list()
    if num_w == "4th(new_3)":
        return new_4th_m3_list()
    if num_w == "4th(new_2)":
        return new_4th_m2_list()
    raise ValueError(f"Unsupported num_w: {num_w}")


def save_data(file_name: str, data: Any, gr: Optional[bool] = None):
    """pickle で結果を保存する（保存先は定数で一元化）。"""
    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR if gr is None else PICKLE_DIR_GROUPED)
    os.makedirs(parent_dir, exist_ok=True)
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"データを {file_path} に保存しました。")


def load_data(file_name):
    current_dir = os.getcwd()  
    parent_dir = os.path.join(current_dir, "trotter_expo_coeff_gr")
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def label_replace(labelkey):
    replacedir = {'4th(new_2)':'4th (new_2)', '4th(new_3)':'4th (new_3)','8th(Morales)':'8th (Morales et al.)','10th(Morales)':'10th (Morales et al.)','8th(Yoshida)':'8th (Yoshida)'
                  }
    if labelkey in replacedir.keys():
        labelkey = replacedir[labelkey]
    return labelkey


def trotter_error_plt_qc_gr(
    s_time: float,
    e_time: float,
    dividing: float,
    mol_type: str,
    num_w,
    storage: bool, # フィッティング結果保存
    avestrage: bool, # fixed-p 保存
):
    """グルーピング版の時間発展誤差を log–log でプロット（挙動不変）。"""
    series_label = f"{num_w}"
    t_values = list(np.arange(s_time, e_time, dividing))
    neg_t_values = [-1.0 * t for t in t_values]

    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type)
    ham_list = ham_list_maker(jw)
    # 定数項（アイデンティティ項）の実数部を取得
    const = next(iter(ham_list[0].terms.values())).real

    if mol_type in ("H2", "H3"):
        jw_hamiltonian = jw
        E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
        clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    else:
        clique_list, num_qubits, E, ori_vec = qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)

    E = E - const
    print(f'energy_{E}')
    ham_name = f"{ham_name}_grouping"

    task_args = [(clique_list, t, num_qubits, ori_vec, num_w) for t in neg_t_values]
    with Pool(processes=POOL_PROCESSES) as pool:
        final_state_list = pool.starmap(qte.tEvolution_vector_grouper, task_args)

    error_list_pertur = []
    t_values = []
    for t, vector, _ in final_state_list:
        t *= -1
        vector = vector.data.reshape(-1, 1)
        tevolution = np.exp(1j * E * t)
        delta_psi = (vector - (tevolution * ori_vec)) / (1j * t)
        innerproduct = (ori_vec.conj().T @ delta_psi)
        innerproduct = innerproduct.real / np.cos(E * t)
        error_list_pertur.extend(abs(innerproduct.real))
        t_values.append(t)

    error_log = np.log10(error_list_pertur)
    time_log = np.log10(t_values)

    n_w = p_dir[num_w]

    set_expo_error = error_log - n_w * time_log
    ave_coeff = np.mean(set_expo_error)
    ave_coeff = 10 ** (ave_coeff)

    linear_error = np.polyfit(time_log, error_log, 1)
    print("error exponent :" + str(linear_error[0]))
    # In Y = CX^a, logY = AlogX + B as 10^B = C
    coeff = 10 ** linear_error[1]
    print("error coefficient :" + str(coeff))
    if storage is True:
        data = {"expo": linear_error[0], "coeff": coeff}
        if num_w is not None:
            target_path = f"{ham_name}_Operator_{num_w}"
        else:
            target_path = f"{ham_name}_Operator_normal"
        save_data(target_path, data, gr=True)

    if avestrage is True:
        if num_w is not None:
            target_path = f"{ham_name}_Operator_{num_w}_ave"
        else:
            target_path = f"{ham_name}_Operator_normal_ave"
        save_data(target_path, ave_coeff, True)
    print(f"average_coeff:{ave_coeff}")

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title(f"{ham_name}_{series_label}")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.plot(t_values, error_list_pertur, marker="s", linestyle="-")
    plt.show()


def calculation_cost(clique_list, num_w,ham_name):
    """
    分解数計算用(decompo_num) clique_num_dir にそのクリークに含まれる項をインデックス付きで登録
    """

    clique_num_dir = {}
    for i, clique in enumerate(clique_list):
        num_terms = 0
        for terms in clique:
            #term_list = [term for term in terms]
            term_list = [term for term in terms if list(term.terms.keys())[0] != ()]
            num_terms += len(term_list)
        clique_num_dir[i] = num_terms

    if num_w == 0:
        total_exp = sum(clique_num_dir.values())
        return total_exp, clique_num_dir
    
    w_list = _get_w_list(num_w)
    m = len(w_list)

    total_exp = 0
    J = len(clique_list)

    if m == 1:
        for i in range(J-1):
            total_exp += clique_num_dir[i]
        total_exp += clique_num_dir[J-1]
        for k in reversed(range(0,J-1)):
            total_exp += clique_num_dir[k]
        return total_exp, clique_num_dir

    #S2_left
    for i in range(J-1):
        total_exp += clique_num_dir[i]
    total_exp += clique_num_dir[J-1]
    for k in reversed(range(1,J-1)):
        total_exp += clique_num_dir[k]
    total_exp += clique_num_dir[0]

    #S2
    for _ in reversed(range(1, m-1)):
        for i in range(1,J-1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J-1]
        # 終端 - 1 個目まで
        for k in reversed(range(1,J-1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]
    for _ in range(0,m-1):
        for i in range(1,J-1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J-1]
        # 終端 - 1 個目まで
        for k in reversed(range(1,J-1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]

    #S2right
    for i in range(1,J-1):
        total_exp += clique_num_dir[i]
    # 折り返し
    total_exp += clique_num_dir[J-1]
    # 終端 まで
    for k in reversed(range(0,J-1)):
        total_exp += clique_num_dir[k]

    return total_exp,clique_num_dir


decompo_num = {'H2': {'8th(Yoshida)': 220, '2nd': 24, '4th': 52, '8th(Morales)': 248, '10th(Morales)': 472, '4th(new_3)': 108, '4th(new_2)': 80, '4(new_1)': 52, '6(new_3)': 108}, 'H3': {'8th(Yoshida)': 1476, '2nd': 118, '4th': 312, '8th(Morales)': 1670, '10th(Morales)': 3222, '4th(new_3)': 700, '4th(new_2)': 506, '4(new_1)': 312, '6(new_3)': 700}, 'H4': {'8th(Yoshida)': 5436, '2nd': 396, '4th': 1116, '8th(Morales)': 6156, '10th(Morales)': 11916, '4th(new_3)': 2556, '4th(new_2)': 1836, '4(new_1)': 1116, '6(new_3)': 2556}, 'H5': {'8th(Yoshida)': 14200, '2nd': 998, '4th': 2884, '8th(Morales)': 16086, '10th(Morales)': 31174, '4th(new_3)': 6656, '4th(new_2)': 4770, '4(new_1)': 2884, '6(new_3)': 6656}, 'H6': {'8th(Yoshida)': 30648, '2nd': 2116, '4th': 6192, '8th(Morales)': 34724, '10th(Morales)': 67332, '4th(new_3)': 14344, '4th(new_2)': 10268, '4(new_1)': 6192, '6(new_3)': 14344}, 'H7': {'8th(Yoshida)': 58920, '2nd': 4026, '4th': 11868, '8th(Morales)': 66762, '10th(Morales)': 129498, '4th(new_3)': 27552, '4th(new_2)': 19710, '4(new_1)': 11868, '6(new_3)': 27552}, 'H8': {'8th(Yoshida)': 102556, '2nd': 6964, '4th': 20620, '8th(Morales)': 116212, '10th(Morales)': 225460, '4th(new_3)': 47932, '4th(new_2)': 34276, '4(new_1)': 20620, '6(new_3)': 47932}, 'H9': {'8th(Yoshida)': 170016, '2nd': 11494, '4th': 34140, '8th(Morales)': 192662, '10th(Morales)': 373830, '4th(new_3)': 79432, '4th(new_2)': 56786, '4(new_1)': 34140, '6(new_3)': 79432}, 'H10': {'8th(Yoshida)': 261960, '2nd': 17660, '4th': 52560, '8th(Morales)': 296860, '10th(Morales)': 576060, '4th(new_3)': 122360, '4th(new_2)': 87460, '4(new_1)': 52560, '6(new_3)': 122360}, 'H11': {'8th(Yoshida)': 385648, '2nd': 25946, '4th': 77332, '8th(Morales)': 437034, '10th(Morales)': 848122, '4th(new_3)': 180104, '4th(new_2)': 128718, '4(new_1)': 77332, '6(new_3)': 180104}, 'H12': {'8th(Yoshida)': 550620, '2nd': 36988, '4th': 110364, '8th(Morales)': 623996, '10th(Morales)': 1211004, '4th(new_3)': 257116, '4th(new_2)': 183740, '4(new_1)': 110364, '6(new_3)': 257116}, 'H13': {'8th(Yoshida)': 767016, '2nd': 51462, '4th': 153684, '8th(Morales)': 869238, '10th(Morales)': 1687014, '4th(new_3)': 358128, '4th(new_2)': 255906, '4(new_1)': 153684, '6(new_3)': 358128}, 'H14': {'8th(Yoshida)': 1037656, '2nd': 69556, '4th': 207856, '8th(Morales)': 1175956, '10th(Morales)': 2282356, '4th(new_3)': 484456, '4th(new_2)': 346156, '4(new_1)': 207856, '6(new_3)': 484456}, 'H15': {'8th(Yoshida)': 1385520, '2nd': 92802, '4th': 277476, '8th(Morales)': 1570194, '10th(Morales)': 3047586, '4th(new_3)': 646824, '4th(new_2)': 462150, '4(new_1)': 277476, '6(new_3)': 646824}}

color_map = {'2nd':'g','4th(new_3)':'r','4th(new_1)':'lightcoral','4th(new_2)':'b','6th(new_4)':'darkgreen','4th':'c', '8th(Morales)':'m', '10th(Morales)':'greenyellow', '8th(Yoshida)':'orange'}
marker_map = {'2nd':'o','4th(new_3)':'v','4th(new_1)':'lightcoral','4th(new_2)':'b','6th(new_4)':'darkgreen','4th':'^', '8th(Morales)':'h', '10th(Morales et al.)':'H', '8th(Yoshida)':'>'}

p_dir = {"6th(new_4)":6,"6th(new_3)":6,"4th(new_1)":4,"4th(new_2)":4,"4th(new_3)":4,"2nd":2,"4th":4,"8th(Morales)":8,"10th(Morales)":10,'8th(Yoshida)':8}

def exp_extrapolation(Hchain, n_w_list, show_bands=True, band_height=0.06, band_alpha=0.28):
    from collections import defaultdict
    
    beta = 1.2

    mol_list = [f'H{i}' for i in range(2,Hchain+1)]

    CA = 1.59360010199040e-3
    CA1 = 1.59360010199040e-4
    eps = CA1

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, "trotter_expo_coeff_gr")
    total_dir = {}

    plt.figure(figsize=(8, 6), dpi=200)

    num_qubits = [i for i in range(4,(Hchain*2)+1,2)]

    lbcheck = 0
    for mol, qubit in zip(mol_list, num_qubits):
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(mol, distance)
        ham_name = ham_name + '_grouping'

        total_dir[n_qubits] = {} 


        for n_w in n_w_list:
            if n_w == '10th(Morales)' and n_qubits == 30: # 10th は H15 で未評価
                continue
            elif n_w == '4th(new_2)' and n_qubits >18:
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = p_dir[n_w]

            min_f = beta * (eps**(-(1+(1/expo)))) * (1/expo) * (coeff**(1/expo)) * (expo+1)**(1+(1/expo))

            # グルーピングあり
            unit_expo = decompo_num[mol][n_w]
            total_expo = unit_expo * min_f

            total_dir[n_qubits][n_w] = total_expo

    series = defaultdict(lambda: {"x": [], "y": []})
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(qubit, gate, ls='None', marker=marker_map[pf], color=color_map[pf],
                     label=lb if qubit == num_qubits[0] else None)
            # フィット用
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    XMAX = 100
    xmin_current, xmax_current = ax.get_xlim()
    ax.set_xlim(xmin_current, max(xmax_current, XMAX))

    # ---- フィット ----
    fit_params = {}
    for lb, d in series.items():
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        m = (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        B, log10A = np.polyfit(np.log10(x), np.log10(y), 1)
        A = 10**log10A
        fit_params[lb] = {"A": A, "B": B}

        x_right = ax.get_xlim()[1]
        xfit = np.logspace(np.log10(x.min()), np.log10(x_right), 400)
        yfit = A * (xfit ** B)
        ax.plot(xfit, yfit, '-', color=color_map.get(lb), alpha=0.9, linewidth=1.5)

    # ---- 色帯の表示（トグル）----
    if show_bands and fit_params:
        # 表示範囲に合わせて評価用グリッドを作成
        x_left, x_right = ax.get_xlim()
        xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)

        labels_fit = list(fit_params.keys())
        # 各ラベルの y(x)
        Y = np.vstack([fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit])
        imin = np.argmin(Y, axis=0)   # 各 x で最小の系列のインデックス

        # 勝者切替の境界
        switch_idx = np.where(np.diff(imin) != 0)[0] + 1
        bounds = np.r_[0, switch_idx, len(xgrid)-1]

        winners_in_order = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            lb = labels_fit[imin[s]]
            if lb not in winners_in_order:
                winners_in_order.append(lb)
            ax.axvspan(xgrid[s], xgrid[e],
                       ymin=0.0, ymax=band_height,  # ★ 高さを引数で
                       color=color_map.get(lb, "0.6"),
                       alpha=band_alpha,            # ★ 透明度を引数で
                       ec="none", zorder=0)

        # 縦点線（任意）
        for i in switch_idx:
            ax.axvline(xgrid[i], linestyle=":", linewidth=1.0, color="k", alpha=0.5, zorder=1)

    # ---- 凡例 ----
    from matplotlib.patches import Patch
    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u, labels_u = [], []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h); labels_u.append(lab); seen.add(lab)

    # 色帯も凡例に出すなら（show_bands True のときだけ）
    if show_bands and fit_params:
        # winners_in_order を上で作っているので、無ければ復元
        if 'winners_in_order' not in locals():
            x_left, x_right = ax.get_xlim()
            xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)
            labels_fit = list(fit_params.keys())
            Y = np.vstack([fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit])
            imin = np.argmin(Y, axis=0)
            switch_idx = np.where(np.diff(imin) != 0)[0] + 1
            bounds = np.r_[0, switch_idx, len(xgrid)-1]
            winners_in_order = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                lb = labels_fit[imin[s]]
                if lb not in winners_in_order:
                    winners_in_order.append(lb)

        proxies = [Patch(facecolor=color_map[lb], alpha=0.6, edgecolor="none", label=f"{lb} (lowest)")
                   for lb in winners_in_order
                   if f"{lb} (lowest)" not in seen]
        handles_u += proxies
        labels_u += [p.get_label() for p in proxies]

    ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)
    plt.show()
