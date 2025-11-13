import math
import multiprocessing
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np # type: ignore
import pyscf # type: ignore
import scipy.sparse as sp # type: ignore
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from numpy.linalg import eig
from openfermion import commutator, count_qubits # type: ignore
from openfermion.chem import MolecularData # type: ignore
from openfermion.linalg import get_sparse_operator # type: ignore
from openfermion.ops import FermionOperator, QubitOperator # type: ignore
from openfermion.transforms import ( # type: ignore
    bravyi_kitaev, # type: ignore
    get_fermion_operator, # type: ignore
    jordan_wigner, # type: ignore
    symmetry_conserving_bravyi_kitaev, # type: ignore
)
from openfermionpyscf import run_pyscf # type: ignore
from pyscf import gto, mcscf, scf # type: ignore
from scipy.sparse import eye, identity, load_npz, save_npz # type: ignore
from scipy.sparse.linalg import eigs, eigsh, expm, matrix_power, norm # type: ignore

import qiskit_time_evolution as qte # type: ignore
from Almost_optimal_grouping import Almost_optimal_grouper # type: ignore

# =========================
# 設定セクション（魔法値の定数化）
# =========================
DEFAULT_BASIS = "sto-3g"  # 基底関数（数値は変更しない）
DEFAULT_DISTANCE = 1.0  # 原子間距離のデフォルト値（Å 相当の内部スケール）
PICKLE_DIR = "trotter_expo_coeff"
PICKLE_DIR_GROUPED = "trotter_expo_coeff_gr"
POOL_PROCESSES = 32  # 並列処理プロセス数（挙動は不変）

# PF の次数
p_dir = {
    "6th(new_4)": 6,
    "6th(new_3)": 6,
    "4th(new_1)": 4,
    "4th(new_2)": 4,
    "4th(new_3)": 4,
    "2nd": 2,
    "4th": 4,
    "8th(Morales)": 8,
    "10th(Morales)": 10,
    "8th(Yoshida)": 8,
}


def call_geometry(Hchain, distance):
    if Hchain < 2:
        raise ValueError("Hchain must be >= 2")

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


def morales_8th_list():
    w_1to8 = [
        0.29137384767986663096528500968049,
        0.26020394234904150277316667709864,
        0.18669648149540687549831902999911,
        -0.40049110428180105319963667975074,
        0.15982762208609923217390166127256,
        -0.38400573301491401473462588779099,
        0.56148845266356446893590729572808,
        0.12783360986284110837857554950443,
    ]

    w0_1to8 = [1 - 2 * sum(w_1to8)]
    w = w0_1to8 + w_1to8
    return w


# Yoshida's 8th order product formula
def yoshida_8th_list():
    w_1to7 = [
        -1.61582374150097,
        -2.44699182370524,
        -0.0071698941970812,
        2.44002732616735,
        0.157739928123617,
        1.82020630970714,
        1.04242620869991,
    ]
    w0_1to7 = [1 - 2 * sum(w_1to7)]
    # パラメータの設定
    w = w0_1to7 + w_1to7
    return w


# Mauro's 10th order product formula(m=15)
def morales_10th_m15_list():
    w_1to15 = [
        0.14552859955499429739088135596618,
        -0.48773512068133537309419933740564,
        0.12762011242429535909727342301656,
        0.70225450019485751220143080587959,
        -0.62035679146761710925756521405042,
        0.39099152412786178133688869373114,
        0.17860253604355465807791041367045,
        -0.80455783177921776295588528272593,
        0.053087216442758242118687385646283,
        0.86836307910275556258687030904753,
        -0.85326297197907834671536254437991,
        -0.11732457198874083224967699358383,
        0.03827345494186056632406947772047,
        0.74843529029532498233997793305357,
        0.30208715621975773712410948025906,
    ]

    w0_1to8 = [1 - 2 * sum(w_1to15)]
    # パラメータの設定
    # w_0 = 1 - 2sum(w_i)
    w = w0_1to8 + w_1to15
    return w


# Mauro's 10th order product formula(m=16)
def morales_10th_m16_list():
    w_1to16 = [
        -0.4945013179955571856347147977644,
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
        0.081207318210272593225087711441684,
    ]

    w0 = [1 - 2 * sum(w_1to16)]
    w = w0 + w_1to16
    return w


# Yoshida's 4th order product formula
def yoshida_4th_list():  # s3odr4
    w = [-1 * (2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3))]
    return w


def trotter_2nd_list():
    w = [1]
    return w


def new_4th_m3_list():
    w1to3 = [0.40653666, 0.21638706, 0.14924614]
    w0_1to3 = [1 - 2 * sum(w1to3)]
    w = w0_1to3 + w1to3
    return w


def new_4th_m2_list():
    w1to2 = [0.42008729, 0.40899193]
    w0_1to3 = [1 - 2 * sum(w1to2)]
    w = w0_1to3 + w1to2
    return w


def _get_w_list(num_w: Any) -> List[float]:
    """積公式パラメータ w の系列を取得（分岐を関数化）。"""
    if num_w == "8th(Morales)":
        return morales_8th_list()
    if num_w == "10th(Morales)":
        return morales_10th_m16_list()
    if num_w == "4th":
        return yoshida_4th_list()
    if num_w == "8th(Yoshida)":
        return yoshida_8th_list()
    if num_w == "2nd":
        return trotter_2nd_list()
    if num_w == "4th(new_3)":
        return new_4th_m3_list()
    if num_w == "4th(new_2)":
        return new_4th_m2_list()
    raise ValueError(f"Unsupported num_w: {num_w}")


def save_data(file_name: str, data: Any, gr: Optional[bool] = None):
    """pickle で結果を保存する（保存先は定数で一元化）。"""
    current_dir = os.getcwd()
    parent_dir = os.path.join(
        current_dir, PICKLE_DIR if gr is None else PICKLE_DIR_GROUPED
    )
    os.makedirs(parent_dir, exist_ok=True)
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"データを {file_path} に保存しました。")


def load_data(file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, "trotter_expo_coeff_gr")
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def label_replace(labelkey):  # 凡例用
    replacedir = {
        "4th(new_2)": "4th (new_2)",
        "4th(new_3)": "4th (new_3)",
        "8th(Morales)": "8th (Morales et al.)",
        "10th(Morales)": "10th (Morales et al.)",
        "8th(Yoshida)": "8th (Yoshida)",
    }
    if labelkey in replacedir.keys():
        labelkey = replacedir[labelkey]
    return labelkey


def trotter_error_plt_qc_gr(
    s_time: float,
    e_time: float,
    dividing: float,
    mol_type: str,
    num_w: str,
    storage: bool,  # フィッティング結果保存
    avestrage: bool,  # fixed-p 保存
):
    """グルーピング版の時間発展誤差を log–log でプロット（挙動不変）。"""
    series_label = f"{num_w}"
    t_values = list(np.arange(s_time, e_time, dividing))
    neg_t_values = [-1.0 * t for t in t_values]

    jw, _, ham_name, num_qubits = jw_hamiltonian_maker(mol_type)
    ham_list = ham_list_maker(jw)
    # 定数項（アイデンティティ項）の実数部を取得
    const = next(iter(ham_list[0].terms.values())).real

    if mol_type in (2, 3):
        jw_hamiltonian = jw
        E, ori_vec, _ = ham_ground_energy(jw_hamiltonian)
        clique_list, _ = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
    else:
        clique_list, num_qubits, E, ori_vec = (
            qte.make_fci_vector_from_pyscf_solver_grouper(mol_type)
        )

    E = E - const
    print(f"energy_{E}")
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
        innerproduct = ori_vec.conj().T @ delta_psi
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
    print(f"average_coeff:{ave_coeff}")

    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)

    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr = corr_matrix[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)

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
    

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title(f"{ham_name}_{series_label}")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.plot(t_values, error_list_pertur, marker="s", linestyle="-")
    plt.show()


# =========================
# 摂動論誤差検証用
# =========================


def load_matrix_files(n_qubits, t, ham_name, num_w):
    """t に対応する疎行列ファイル群を読み込み順序で返す（挙動不変）。"""
    t = round(t, 10)
    sub = (
        f"{ham_name}_Operator_w{num_w}/{t}/{ham_name}_nostep_tOperator_w{num_w}"
        if num_w is not None
        else f"{ham_name}_Operator_normal/{t}/{ham_name}_nostep_tOperator"
    )
    directory_path = os.path.join(os.getcwd(), "matrix", sub)
    # "matrix_*.npz" を数値インデックス順にソート
    matrix_files = [
        name for name in os.listdir(directory_path) if name.startswith("matrix_")
    ]
    matrix_paths = sorted(
        matrix_files, key=lambda s: int(os.path.splitext(s)[0].split("_")[1])
    )
    return directory_path, matrix_paths


def eU_strage(jw_hamiltonian, t, n_qubits, w, idx, folder_path):
    """
    ハミルトニアンの項を疎行列の指数関数行列に変換し指定のフォルダに保存

    Args:
        jw_hamiltonian: JW変換されたハミルトニアン,
        t:  時間発展演算子のt
        n_qubits: ハミルトニアンのQビット数
        w: 積公式の係数
        idx: トロッター展開されたハミルトニアンの項のインデックス
        folder_path: 保存先のフォルダ

    Returns:
        idx: トロッター展開されたハミルトニアンの項のインデックス
    """

    def eU_exchanger(jw_hamiltonian, t, n_qubits, w):
        """各項の e^{i w t H_j} を構成（既存ロジックを保持）。"""
        dim = 2**n_qubits
        I = eye(dim, format="csc")

        def ham_to_cIsU(jw_hamiltonian, coefficient, t, n_qubits, w):
            # 係数で規格化した H を用い、cos・sin 展開で 1ステップ分の e^{iwtH} を構築
            for term in jw_hamiltonian:
                term_mat = get_sparse_operator(term, n_qubits)
                h_norm = term_mat / coefficient
                c = np.cos(w * t * coefficient)
                s = np.sin(w * t * coefficient)
                exp_op = complex(c, 0) * I + complex(0, s) * h_norm
            return exp_op

        for term, coefficient in jw_hamiltonian.terms.items():
            if term:
                eU = ham_to_cIsU(jw_hamiltonian, coefficient, t, n_qubits, w)
            else:
                eU = expm(1j * eye(2**n_qubits, format="csc") * coefficient * t * w)
        return eU

    eU = eU_exchanger(jw_hamiltonian, t, n_qubits, w)
    sp.save_npz(os.path.join(folder_path, f"matrix_{idx}.npz"), eU)
    idx += 1
    return idx


def S_2(ham_list, t, n_qubits, w, folder_path, idx):  # 左端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w / 2, idx, folder_path)
        print(f"idx {idx} ham {ham_list[i]}")
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w, idx, folder_path)
    # 終端まで
    for k in reversed(range(0, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w / 2, idx, folder_path)
    return idx


def S_2_trotter_left(ham_list, t, n_qubits, Max_w, nMax_w, folder_path, idx):  # 左端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, Max_w / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, Max_w, idx, folder_path)
    # 終端 - 1 個目まで
    for k in reversed(range(1, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, Max_w / 2, idx, folder_path)
    # 終端
    idx = eU_strage(ham_list[0], t, n_qubits, (Max_w + nMax_w) / 2, idx, folder_path)
    return idx


def S_2_trotter(ham_list, t, n_qubits, w_f, w_s, folder_path, idx):  # 左端、右端以外
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(1, J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w_f / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w_f, idx, folder_path)
    # 終端 - 1 個目まで
    for k in reversed(range(1, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w_f / 2, idx, folder_path)
    # 終端
    idx = eU_strage(ham_list[0], t, n_qubits, (w_f + w_s) / 2, idx, folder_path)
    return idx


def S_2_trotter_right(ham_list, t, n_qubits, w_i, folder_path, idx):  # 右端
    """
    Returns:
        idx: トロッター展開した項のインデックス
    """
    J = len(ham_list)
    # 折り返しの直前まで
    for i in range(1, J - 1):
        idx = eU_strage(ham_list[i], t, n_qubits, w_i / 2, idx, folder_path)
    # 折り返し
    idx = eU_strage(ham_list[J - 1], t, n_qubits, w_i, idx, folder_path)
    # 終端 まで
    for k in reversed(range(0, J - 1)):
        idx = eU_strage(ham_list[k], t, n_qubits, w_i / 2, idx, folder_path)
    return idx


def folder_maker_multiprocessing_values(
    t_values, jw_hamiltonian, n_qubits, ham_name, num_w
):
    """t ごとにフォルダと e^{iHt} 分解を並列生成。32 並列を固定（挙動不変）。"""
    workers = 32
    partition_size = (len(t_values) + workers - 1) // workers
    t_partitions = [
        t_values[i * partition_size : (i + 1) * partition_size] for i in range(workers)
    ]

    if num_w is not None:
        task_args = [
            (jw_hamiltonian, t, n_qubits, ham_name, num_w, i)
            for i, t in enumerate(t_partitions)
        ]
        with Pool(processes=workers) as pool:
            pool.starmap(w_trotter_folder_maker_multi, task_args)
    else:
        task_args = [
            (jw_hamiltonian, t, n_qubits, ham_name, i)
            for i, t in enumerate(t_partitions)
        ]
        with Pool(processes=workers) as pool:
            pool.starmap(normal_trotter_folder_maker_multi, task_args)
    print("done")


def normal_trotter_folder_maker_multi(
    jw_hamiltonian, t_list, n_qubits, ham_name, core_num
):
    """num_w=None の通常版フォルダ生成。"""
    base = os.path.join(os.getcwd(), "matrix", f"{ham_name}_Operator_normal")
    os.makedirs(base, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    for t in t_list:
        mt = round(t, 10)
        target_dir = os.path.join(base, str(mt))
        os.makedirs(target_dir, exist_ok=True)
        folder_path = os.path.join(target_dir, f"{ham_name}_nostep_tOperator")
        os.makedirs(folder_path, exist_ok=True)
        idx = 0
        for term in ham_list:
            idx = eU_strage(term, t, n_qubits, 1, idx, folder_path)


def w_trotter_folder_maker_multi(
    jw_hamiltonian, t_list, n_qubits, ham_name, num_w, core_num
):
    """重み付き（高次）PF のフォルダ生成。"""
    parent_dir = os.path.join(os.getcwd(), "matrix", f"{ham_name}_Operator_w{num_w}")
    os.makedirs(parent_dir, exist_ok=True)
    ham_list = ham_list_maker(jw_hamiltonian)
    w_list = _get_w_list(num_w)
    m = len(w_list)
    for t in t_list:
        mt = round(t, 10)
        target_dir = os.path.join(parent_dir, str(mt))
        os.makedirs(target_dir, exist_ok=True)
        folder_path = os.path.join(target_dir, f"{ham_name}_nostep_tOperator_w{num_w}")
        os.makedirs(folder_path, exist_ok=True)
        idx = 0
        if m == 1:
            print(f"m{m}")
            idx = S_2(ham_list, t, n_qubits, w_list[0], folder_path, idx)
            print(f"idx{idx}")
            continue
        # w_m
        idx = S_2_trotter_left(
            ham_list, t, n_qubits, w_list[m - 1], w_list[m - 2], folder_path, idx
        )
        # w_{m-1} ~ w_1
        for i in reversed(range(1, m - 1)):
            idx = S_2_trotter(
                ham_list, t, n_qubits, w_list[i], w_list[i - 1], folder_path, idx
            )
        # w_0 ~ w_{m-1}
        for i in range(0, m - 1):
            idx = S_2_trotter(
                ham_list, t, n_qubits, w_list[i], w_list[i + 1], folder_path, idx
            )
        # w_m
        idx = S_2_trotter_right(ham_list, t, n_qubits, w_list[m - 1], folder_path, idx)


def sparse_matrix_multiply_from_folder(core_num, folder_path, file_names):
    """フォルダ内の疎行列を順次乗算し、一時 npz に保存してパスを返す。"""
    if not file_names:
        return None
    result = sp.load_npz(os.path.join(folder_path, file_names[0]))
    for name in file_names[1:]:
        result = result @ sp.load_npz(os.path.join(folder_path, name))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_core_{core_num}.npz")
    sp.save_npz(tmp.name, result)
    return tmp.name


def multi_parallel_sparse_matrix_multiply_recursive(
    initial_folder_path, file_names, num_partitions
):
    """疎行列の多段並列乗算。分割→各分割を乗算→一時ファイルへ。最後に連鎖乗算して返す。"""
    current_folder_path = initial_folder_path
    step = 1
    while len(file_names) >= 2:
        print(f"{step}回目の処理開始")
        partition_size = (len(file_names) + num_partitions - 1) // num_partitions
        partitions = [
            file_names[i * partition_size : (i + 1) * partition_size]
            for i in range(num_partitions)
        ]
        with Pool(processes=num_partitions) as pool:
            results = pool.starmap(
                sparse_matrix_multiply_from_folder,
                [
                    (core, current_folder_path, part)
                    for core, part in enumerate(partitions)
                ],
            )
        # 有効結果だけに絞り、core番号順に並べ替え
        file_names = sorted(
            [p for p in results if p is not None],
            key=lambda p: int(p.split("_core_")[-1].split(".")[0]),
        )
        current_folder_path = tempfile.gettempdir()  # 以降は一時ファイル群を入力にする
        num_partitions = max(1, num_partitions // 2)
        step += 1

    # 最終段：残ったファイルを逐次乗算して返す
    final_result = sp.load_npz(file_names[0])
    print("final calculating")
    for fp in file_names[1:]:
        final_result = final_result @ sp.load_npz(fp)
    print("done")
    return final_result


def find_closest_value(E, values):
    """E に最も近い値とその誤差を返す。"""
    abs_diffs = [abs(E - v) for v in values]
    i_min = int(np.argmin(abs_diffs))
    return values[i_min], abs_diffs[i_min]


def error_cal_multi(t_list, terms_list, ori_vec, E, num_eig):
    """各 t での固有値誤差を計算（シフト付き反復法の既存ロジックを保持）。"""
    r_tlist, error_list = [], []
    for t, terms in zip(t_list, terms_list):
        Et = E * t
        n_wrap = int((-Et) // (2 * np.pi)) + 1
        # シフトは E の1点のみ（既存挙動）
        sigma = np.exp(1j * E * t)
        eigenvalues = eigs(
            terms, sigma=sigma, k=num_eig, v0=ori_vec, return_eigenvectors=False
        )
        phases = np.angle(eigenvalues)
        phases = np.where(phases > 0, phases - 2 * np.pi, phases) - (
            2 * (n_wrap - 1) * np.pi
        )
        en = sorted([ph.real / t for ph in phases])
        approx, _err = find_closest_value(E, en)
        error = abs(E - approx)
        r_tlist.append(t)
        error_list.append(error)
    return r_tlist, error_list


def trotter_error_plt(
    s_time,
    e_time,
    dividing,
    mol_type,
    num_w,
):
    """対角化版の時間発展誤差を log–log でプロット（GRスタイル、ヘルパ未使用）。挙動不変。"""
    series_label = f"{num_w}"
    t_values = [round(t, 5) for t in np.arange(s_time, e_time, dividing)]

    hamiltonian, _, ham_name, n_qubits = jw_hamiltonian_maker(mol_type)
    E, ori_vec, _ = ham_ground_energy(hamiltonian)

    # 出力先
    parent_dir = os.path.join(os.getcwd(), "calculation")
    num_w_dir = "normal" if num_w is None else f"w{num_w}"
    directory_path = os.path.join(parent_dir, ham_name)
    os.makedirs(directory_path, exist_ok=True)

    make_folder_t = []
    for t in t_values:
        ham_dir = os.path.join(
            parent_dir, "matrix", f"{ham_name}_Operator_{num_w_dir}", f"{t}"
        )
        term_dir_path = os.path.join(directory_path, f"t_{t}_{num_w_dir}.npz")
        if not os.path.exists(term_dir_path) and not os.path.exists(ham_dir):
            make_folder_t.append(t)
    if len(make_folder_t) > 0:
        folder_maker_multiprocessing_values(
            make_folder_t, hamiltonian, n_qubits, ham_name, num_w
        )

    term_list = []
    for t in t_values:
        term_dir_path = os.path.join(directory_path, f"t_{t}_{num_w_dir}.npz")
        if not os.path.exists(term_dir_path):
            folder_path, file_path = load_matrix_files(n_qubits, t, ham_name, num_w)
            term = multi_parallel_sparse_matrix_multiply_recursive(
                folder_path, file_path, 32
            )
            term_list.append(term)
            save_npz(term_dir_path, term)
            print("save")
        else:
            data = load_npz(term_dir_path)
            term_list.append(data)

    t_list, error_list = error_cal_multi(t_values, term_list, ori_vec, E, num_eig=1)
    print("multiprocessing done")

    error_log = np.log10(error_list)
    time_log = np.log10(t_list)

    n_w = p_dir[num_w]
    set_expo_error = error_log - n_w * time_log
    ave_coeff = 10 ** (np.mean(set_expo_error))

    # 回帰と r^2（log–log）
    linear_error = np.polyfit(time_log, error_log, 1)
    slope = linear_error[0]
    coeff = 10 ** linear_error[1]
    print("error exponent :" + str(slope))
    print("error coefficient :" + str(coeff))
    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)
    corr = np.corrcoef(x_flat, y_flat)[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)
    print(f"average_coeff:{ave_coeff}")

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title(f"{ham_name}_{series_label}")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.plot(t_list, error_list, marker="o", linestyle="-")
    plt.show()
    return t_list, error_list


def trotter_error_plt_qc(
    s_time,
    e_time,
    dividing,
    mol_type,
    num_w,
):
    """非グルーピングQC版の時間発展誤差を log–log でプロット"""
    t_values = list(np.arange(s_time, e_time, dividing))
    neg_t_values = [-1.0 * t for t in t_values]

    jw_hamiltonian, _, E, ori_vec, _ = qte.make_fci_vector_from_pyscf_solver(mol_type)
    _, _, _, num_qubits = jw_hamiltonian_maker(mol_type)

    clique_list = ham_list_maker(jw_hamiltonian)
    task_args = [(clique_list, t, num_qubits, ori_vec, num_w) for t in neg_t_values]
    with Pool(processes=32) as pool:
        final_state_list = pool.starmap(qte.tEvolution_vector, task_args)

    error_list_pertur = []
    t_values = []
    for t, vector in final_state_list:
        t *= -1
        vector = vector.data.reshape(-1, 1)
        tevolution = np.exp(1j * E * t)
        delta_psi = (vector - (tevolution * ori_vec)) / (1j * t)
        innerproduct = ori_vec.conj().T @ delta_psi
        innerproduct = innerproduct.real / np.cos(E * t)
        error_list_pertur.extend(abs((innerproduct.real)))
        t_values.append(t)

    error_log = np.log10(error_list_pertur)
    time_log = np.log10(t_values)

    # 指数は p_dir を使用（ヘルパ不使用）
    n_w = p_dir[num_w]
    set_expo_error = error_log - n_w * time_log
    ave_coeff = 10 ** (np.mean(set_expo_error))

    # 回帰と r^2（log–log）
    linear_error = np.polyfit(time_log, error_log, 1)
    slope = linear_error[0]
    coeff = 10 ** linear_error[1]
    print("error exponent :" + str(slope))
    print("error coefficient :" + str(coeff))
    x_flat = np.ravel(time_log).astype(float)
    y_flat = np.ravel(error_log).astype(float)
    corr = np.corrcoef(x_flat, y_flat)[0, 1]
    r2_loglog = corr**2
    print("r^2 (log-log):", r2_loglog)
    print(f"average_coeff:{ave_coeff}")

    # 対角化ベースの誤差も同図で比較
    t_list, error_list_ph = trotter_error_plt(
        s_time,
        e_time,
        dividing,
        mol_type,
        num_w
    )

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Eigenvalue error [Hartree]", fontsize=15)
    plt.plot(
        t_values, error_list_ph, marker="s", linestyle="-", label="Diagonalization"
    )
    plt.legend()
    plt.show()

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Eigenvalue error [Hartree]", fontsize=15)
    plt.plot(
        t_values,
        error_list_pertur,
        marker="s",
        linestyle="-",
        label="Perturbation",
        color="green",
    )
    plt.legend()
    plt.show()

    pertuer_error = []
    for ph, per in zip(error_list_ph, error_list_pertur):
        pertuer_error.append(abs(ph - per))

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Algorithm error [Hartree]", fontsize=15)
    plt.plot(t_values, pertuer_error, marker="o", linestyle="-", color="red")
    plt.show()


# =========================
# 総計算量外挿用
# =========================


def calculation_cost(clique_list, num_w, ham_name):
    """
    分解数計算用(decompo_num) clique_num_dir にそのクリークに含まれる項をインデックス付きで登録
    """

    clique_num_dir = {}
    for i, clique in enumerate(clique_list):
        num_terms = 0
        for terms in clique:
            # term_list = [term for term in terms]
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
        for i in range(J - 1):
            total_exp += clique_num_dir[i]
        total_exp += clique_num_dir[J - 1]
        for k in reversed(range(0, J - 1)):
            total_exp += clique_num_dir[k]
        return total_exp, clique_num_dir

    # S2_left
    for i in range(J - 1):
        total_exp += clique_num_dir[i]
    total_exp += clique_num_dir[J - 1]
    for k in reversed(range(1, J - 1)):
        total_exp += clique_num_dir[k]
    total_exp += clique_num_dir[0]

    # S2
    for _ in reversed(range(1, m - 1)):
        for i in range(1, J - 1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J - 1]
        # 終端 - 1 個目まで
        for k in reversed(range(1, J - 1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]
    for _ in range(0, m - 1):
        for i in range(1, J - 1):
            total_exp += clique_num_dir[i]
        # 折り返し
        total_exp += clique_num_dir[J - 1]
        # 終端 - 1 個目まで
        for k in reversed(range(1, J - 1)):
            total_exp += clique_num_dir[k]
        # 終端
        total_exp += clique_num_dir[0]

    # S2right
    for i in range(1, J - 1):
        total_exp += clique_num_dir[i]
    # 折り返し
    total_exp += clique_num_dir[J - 1]
    # 終端 まで
    for k in reversed(range(0, J - 1)):
        total_exp += clique_num_dir[k]

    return total_exp, clique_num_dir


# calculation_cost() による H-chain の PF ごとのパウリローテーションの個数
decompo_num = {
    "H2": {
        "8th(Yoshida)": 220,
        "2nd": 24,
        "4th": 52,
        "8th(Morales)": 248,
        "10th(Morales)": 472,
        "4th(new_3)": 108,
        "4th(new_2)": 80,
        "4(new_1)": 52,
        "6(new_3)": 108,
    },
    "H3": {
        "8th(Yoshida)": 1476,
        "2nd": 118,
        "4th": 312,
        "8th(Morales)": 1670,
        "10th(Morales)": 3222,
        "4th(new_3)": 700,
        "4th(new_2)": 506,
        "4(new_1)": 312,
        "6(new_3)": 700,
    },
    "H4": {
        "8th(Yoshida)": 5436,
        "2nd": 396,
        "4th": 1116,
        "8th(Morales)": 6156,
        "10th(Morales)": 11916,
        "4th(new_3)": 2556,
        "4th(new_2)": 1836,
        "4(new_1)": 1116,
        "6(new_3)": 2556,
    },
    "H5": {
        "8th(Yoshida)": 14200,
        "2nd": 998,
        "4th": 2884,
        "8th(Morales)": 16086,
        "10th(Morales)": 31174,
        "4th(new_3)": 6656,
        "4th(new_2)": 4770,
        "4(new_1)": 2884,
        "6(new_3)": 6656,
    },
    "H6": {
        "8th(Yoshida)": 30648,
        "2nd": 2116,
        "4th": 6192,
        "8th(Morales)": 34724,
        "10th(Morales)": 67332,
        "4th(new_3)": 14344,
        "4th(new_2)": 10268,
        "4(new_1)": 6192,
        "6(new_3)": 14344,
    },
    "H7": {
        "8th(Yoshida)": 58920,
        "2nd": 4026,
        "4th": 11868,
        "8th(Morales)": 66762,
        "10th(Morales)": 129498,
        "4th(new_3)": 27552,
        "4th(new_2)": 19710,
        "4(new_1)": 11868,
        "6(new_3)": 27552,
    },
    "H8": {
        "8th(Yoshida)": 102556,
        "2nd": 6964,
        "4th": 20620,
        "8th(Morales)": 116212,
        "10th(Morales)": 225460,
        "4th(new_3)": 47932,
        "4th(new_2)": 34276,
        "4(new_1)": 20620,
        "6(new_3)": 47932,
    },
    "H9": {
        "8th(Yoshida)": 170016,
        "2nd": 11494,
        "4th": 34140,
        "8th(Morales)": 192662,
        "10th(Morales)": 373830,
        "4th(new_3)": 79432,
        "4th(new_2)": 56786,
        "4(new_1)": 34140,
        "6(new_3)": 79432,
    },
    "H10": {
        "8th(Yoshida)": 261960,
        "2nd": 17660,
        "4th": 52560,
        "8th(Morales)": 296860,
        "10th(Morales)": 576060,
        "4th(new_3)": 122360,
        "4th(new_2)": 87460,
        "4(new_1)": 52560,
        "6(new_3)": 122360,
    },
    "H11": {
        "8th(Yoshida)": 385648,
        "2nd": 25946,
        "4th": 77332,
        "8th(Morales)": 437034,
        "10th(Morales)": 848122,
        "4th(new_3)": 180104,
        "4th(new_2)": 128718,
        "4(new_1)": 77332,
        "6(new_3)": 180104,
    },
    "H12": {
        "8th(Yoshida)": 550620,
        "2nd": 36988,
        "4th": 110364,
        "8th(Morales)": 623996,
        "10th(Morales)": 1211004,
        "4th(new_3)": 257116,
        "4th(new_2)": 183740,
        "4(new_1)": 110364,
        "6(new_3)": 257116,
    },
    "H13": {
        "8th(Yoshida)": 767016,
        "2nd": 51462,
        "4th": 153684,
        "8th(Morales)": 869238,
        "10th(Morales)": 1687014,
        "4th(new_3)": 358128,
        "4th(new_2)": 255906,
        "4(new_1)": 153684,
        "6(new_3)": 358128,
    },
    "H14": {
        "8th(Yoshida)": 1037656,
        "2nd": 69556,
        "4th": 207856,
        "8th(Morales)": 1175956,
        "10th(Morales)": 2282356,
        "4th(new_3)": 484456,
        "4th(new_2)": 346156,
        "4(new_1)": 207856,
        "6(new_3)": 484456,
    },
    "H15": {
        "8th(Yoshida)": 1385520,
        "2nd": 92802,
        "4th": 277476,
        "8th(Morales)": 1570194,
        "10th(Morales)": 3047586,
        "4th(new_3)": 646824,
        "4th(new_2)": 462150,
        "4(new_1)": 277476,
        "6(new_3)": 646824,
    },
}

color_map = {
    "2nd": "g",
    "4th(new_3)": "r",
    "4th(new_1)": "lightcoral",
    "4th(new_2)": "b",
    "6th(new_4)": "darkgreen",
    "4th": "c",
    "8th(Morales)": "m",
    "10th(Morales)": "greenyellow",
    "8th(Yoshida)": "orange",
}
marker_map = {
    "2nd": "o",
    "4th(new_3)": "v",
    "4th(new_1)": "lightcoral",
    "4th(new_2)": "<",
    "6th(new_4)": "darkgreen",
    "4th": "^",
    "8th(Morales)": "h",
    "10th(Morales)": "H",
    "8th(Yoshida)": ">",
}


def exp_extrapolation(
    Hchain, n_w_list, show_bands=True, band_height=0.06, band_alpha=0.28
):
    from collections import defaultdict

    beta = 1.2

    Hchain_list = [i for i in range(2, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]

    CA = 1.59360010199040e-3
    CA1 = 1.59360010199040e-4
    eps = CA1

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, "trotter_expo_coeff_gr")
    total_dir = {}

    plt.figure(figsize=(8, 6), dpi=200)

    num_qubits = [i for i in range(4, (Hchain * 2) + 1, 2)]

    for chain, qubit, mol in zip(Hchain_list, num_qubits, Hchain_str):
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"

        total_dir[n_qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and n_qubits == 30:  # 10th は H15 で未評価
                continue
            elif n_w == "4th(new_2)" and n_qubits > 28:
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = p_dir[n_w]

            min_f = (
                beta
                * (eps ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = decompo_num[mol][n_w]
            total_expo = unit_expo * min_f

            total_dir[n_qubits][n_w] = total_expo

    series = defaultdict(lambda: {"x": [], "y": []})
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(
                qubit,
                gate,
                ls="None",
                marker=marker_map[pf],
                color=color_map[pf],
                label=lb if qubit == num_qubits[0] else None,
            )
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
        yfit = A * (xfit**B)
        ax.plot(xfit, yfit, "-", color=color_map.get(lb), alpha=0.9, linewidth=1.5)

    # ---- 色帯の表示（トグル）----
    if show_bands and fit_params:
        # 表示範囲に合わせて評価用グリッドを作成
        x_left, x_right = ax.get_xlim()
        xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)

        labels_fit = list(fit_params.keys())
        # 各ラベルの y(x)
        Y = np.vstack(
            [fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"]) for lb in labels_fit]
        )
        imin = np.argmin(Y, axis=0)  # 各 x で最小の系列のインデックス

        # 勝者切替の境界
        switch_idx = np.where(np.diff(imin) != 0)[0] + 1
        bounds = np.r_[0, switch_idx, len(xgrid) - 1]

        winners_in_order = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            lb = labels_fit[imin[s]]
            if lb not in winners_in_order:
                winners_in_order.append(lb)
            ax.axvspan(
                xgrid[s],
                xgrid[e],
                ymin=0.0,
                ymax=band_height,  # ★ 高さを引数で
                color=color_map.get(lb, "0.6"),
                alpha=band_alpha,  # ★ 透明度を引数で
                ec="none",
                zorder=0,
            )

        # 縦点線（任意）
        for i in switch_idx:
            ax.axvline(
                xgrid[i], linestyle=":", linewidth=1.0, color="k", alpha=0.5, zorder=1
            )

    # ---- 凡例 ----
    from matplotlib.patches import Patch

    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u, labels_u = [], []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)

    # 色帯も凡例に出すなら（show_bands True のときだけ）
    if show_bands and fit_params:
        # winners_in_order を上で作っているので、無ければ復元
        if "winners_in_order" not in locals():
            x_left, x_right = ax.get_xlim()
            xgrid = np.logspace(np.log10(x_left), np.log10(x_right), 2000)
            labels_fit = list(fit_params.keys())
            Y = np.vstack(
                [
                    fit_params[lb]["A"] * (xgrid ** fit_params[lb]["B"])
                    for lb in labels_fit
                ]
            )
            imin = np.argmin(Y, axis=0)
            switch_idx = np.where(np.diff(imin) != 0)[0] + 1
            bounds = np.r_[0, switch_idx, len(xgrid) - 1]
            winners_in_order = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                lb = labels_fit[imin[s]]
                if lb not in winners_in_order:
                    winners_in_order.append(lb)

        proxies = [
            Patch(
                facecolor=color_map[lb],
                alpha=0.6,
                edgecolor="none",
                label=f"{lb} (lowest)",
            )
            for lb in winners_in_order
            if f"{lb} (lowest)" not in seen
        ]
        handles_u += proxies
        labels_u += [p.get_label() for p in proxies]

    ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

    # 軸など
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    plt.show()


def exp_extrapolation_diff(
    Hchain,
    n_w_list=("4th(new_2)", "8th(Morales)"),
    MIN_POS=1e-18,
    X_MIN_CALC=4,
    X_MAX_DISPLAY=100,
):
    """
    単一図（左右Y軸）:
      左Y: 総パウリ回転数（散布 + log–log フィット）
      右Y: 2本のフィット直線の絶対差 |Δ|
    依存：decompo_num, optimal_distance, jw_hamiltonian_maker, load_data, label_replace,
         marker_map, color_map, p_dir
    """
    from collections import defaultdict

    # 対象 H チェーン
    Hchain_list = [i for i in range(2, Hchain + 1)]
    Hchain_str = [f"H{i}" for i in range(2, Hchain + 1)]
    num_qubits = [i for i in range(4, (Hchain * 2) + 1, 2)]

    CA = 1.59360010199040e-3
    CA1 = 1.59360010199040e-4
    eps = CA1

    beta = 1.20

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, "trotter_expo_coeff_gr")

    # 総回転数の算出
    total_dir = {}
    for chain, qubit, mol in zip(Hchain_list, num_qubits, Hchain_str):
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"
        total_dir[n_qubits] = {}

        for n_w in n_w_list:
            if n_w == "10th(Morales)" and n_qubits == 30:  # 10th は H15 で未評価
                continue
            elif n_w == "4th(new_2)" and n_qubits > 28:
                continue

            target_path = f"{ham_name}_Operator_{n_w}_ave"
            file_path = os.path.join(parent_dir, target_path)
            data = load_data(file_path)

            coeff = data
            expo = p_dir[n_w]

            min_f = (
                beta
                * (eps ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = decompo_num[mol][n_w]
            total_expo = unit_expo * min_f

            total_dir[n_qubits][n_w] = total_expo

    # ---- プロット（単一図・双Y軸）----
    plt.figure(figsize=(8, 6), dpi=200)

    series = defaultdict(lambda: {"x": [], "y": []})
    for qubit, gate_dir in total_dir.items():
        for pf, gate in gate_dir.items():
            lb = label_replace(pf)
            # 散布
            plt.plot(
                qubit,
                gate,
                ls="None",
                marker=marker_map[pf],
                color=color_map[pf],
                label=lb if qubit == num_qubits[0] else None,
            )
            # フィット用
            series[pf]["x"].append(float(qubit))
            series[pf]["y"].append(float(gate))

    ax = plt.gca()
    ax2 = ax.twinx()

    # 軸
    ax.set_xscale("log")
    ax.set_yscale("log")

    # x の右端を固定
    x_left_auto, _ = ax.get_xlim()
    ax.set_xlim(x_left_auto, X_MAX_DISPLAY)

    # ---- フィット ----
    fit_params = {}
    xfit_lo = max(X_MIN_CALC, x_left_auto)
    xfit_hi = X_MAX_DISPLAY

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
        yfit = A * (xfit**B)
        ax.plot(xfit, yfit, "-", color=color_map.get(lb), alpha=0.9, linewidth=1.5)

    # ---- 左軸：凡例 ----
    handles_exist, labels_exist = ax.get_legend_handles_labels()
    seen = set()
    handles_u = []
    labels_u = []
    for h, lab in zip(handles_exist, labels_exist):
        if lab and lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)
    if handles_u:
        ax.legend(handles_u, labels_u, loc="upper left", framealpha=0.9)

    # ---- 右軸：差分 |Δ| ----
    # フィット区間と同じ x 範囲 [xfit_lo, xfit_hi] で評価
    # キーは pf 文字列（例: "4th(new_2)", "8(Mauro)"）で揃える
    if xfit_hi > xfit_lo:
        ax2.set_yscale("log")
        if len(n_w_list) == 2 and all(pf in fit_params for pf in n_w_list):
            pf_a, pf_b = n_w_list  # 表示順を n_w_list に揃える
            xx = np.logspace(np.log10(xfit_lo), np.log10(xfit_hi), 1200)

            A_a, B_a = fit_params[pf_a]["A"], fit_params[pf_a]["B"]
            A_b, B_b = fit_params[pf_b]["A"], fit_params[pf_b]["B"]
            ya = A_a * (xx**B_a)
            yb = A_b * (xx**B_b)

            diff = np.maximum(np.abs(yb - ya), MIN_POS)
            ax2.plot(
                xx,
                diff,
                "--",
                lw=2.0,
                alpha=0.9,
                color=color_map.get(pf_a, None),
                label=f"|Δ|: {label_replace(pf_b)} − {label_replace(pf_a)}",
            )
            # 左軸とレンジを一致させて目盛位置をそろえる
            ax2.set_ylim(ax.get_ylim())

        # 凡例（右軸）
        hr, lr = ax2.get_legend_handles_labels()
        if hr:
            ax2.legend(hr, lr, loc="upper right", framealpha=0.9)

    # 軸ラベル・グリッド
    ax.set_xlabel("Num qubits", fontsize=15)
    ax.set_ylabel("Number of Pauli rotations", fontsize=15)
    ax2.set_ylabel("Difference in number of Pauli rotations", fontsize=15)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


# =========================
# ベータ（β）計算用
# =========================


def eval_Fejer_kernel(T, x):
    """
    Generate the kernel of QPE
    """
    x_avoid = np.abs(x % (2 * np.pi)) < 1e-8
    numer = np.sin(0.5 * T * x) ** 2
    denom = np.sin(0.5 * x) ** 2
    denom += x_avoid
    ret = numer / denom
    ret = (1 - x_avoid) * ret + (T**2) * x_avoid
    return ret / T


def generate_QPE_distribution(spectrum, population, T):
    """
    Generate the index distribution of QPE
    """
    T = int(T)
    N = len(spectrum)
    dist = np.zeros(T)
    j_arr = 2 * np.pi * np.arange(T) / T - np.pi
    for k in range(N):
        dist += population[k] * eval_Fejer_kernel(T, j_arr - spectrum[k]) / T
    return dist


def draw_with_prob(measure, N):
    """
    Draw N indices independently from a given measure
    """
    L = measure.shape[0]
    cdf_measure = np.cumsum(measure)  # 累積和dis
    normal_fac = cdf_measure[-1]
    U = np.random.rand(N) * normal_fac  # 0-1のランダムな数をN個作製
    index = np.searchsorted(cdf_measure, U)
    return index


def estimate_phase(k, T):
    estimate = 2 * np.pi * k / (T) - np.pi
    return estimate


def QPE(spectrum, population, T, N):
    """
    QPE Main routine
    """
    discrete_energies = 2 * np.pi * np.arange(T) / (T) - np.pi
    index_dist = generate_QPE_distribution(
        spectrum, population, T
    )  # Generate QPE samples
    index_samp = draw_with_prob(index_dist, N)
    values, counts = np.unique(index_samp, return_counts=True)
    index_sort = np.argsort(counts)
    estimate_1 = estimate_phase(values[index_sort[-1]], T)
    ground_state_energy = estimate_1
    return ground_state_energy


def beta_plt(
    T_list_QPE=np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]),
    N_rep=10,  # 繰り返し回数
    N_QPE=int(100),  # サンプリング数
    spectrum_F=[-1.5],
):
    error_QPE = np.zeros(len(T_list_QPE), dtype="float")
    T_total_QPE = np.zeros(len(T_list_QPE), dtype="float")

    error_QPE_timeevo_all = np.zeros((N_rep, len(T_list_QPE)), dtype="float")
    for n in range(N_rep):
        for k in range(len(T_list_QPE)):
            T_max = T_list_QPE[k]
            output_energy = QPE([spectrum_F[0]], [1], T_max, N_QPE)
            T_total_QPE[k] += T_max * N_QPE
            ##---measure error--##
            error_QPE[k] += np.abs(spectrum_F[0] - output_energy)
            error_QPE_timeevo_all[n][k] += np.abs(spectrum_F[0] - output_energy)
    T_total_QPE = T_total_QPE / N_rep
    error_QPE = error_QPE / N_rep
    error_QPE_std = np.std(error_QPE_timeevo_all, axis=0)

    C = T_list_QPE  # コスト軸に合わせる
    eps = error_QPE

    # ----- α=1 固定フィット -----
    logC, logEps = np.log(C), np.log(eps)

    log_beta = np.average(logEps + logC)
    beta_fix = np.exp(log_beta)

    print(f"α (fixed) = 1.0")
    print(f"β (fitted) = {beta_fix:.3f}")

    # ----- プロット -----
    plt.figure(dpi=150)
    plt.xscale("log")
    plt.yscale("log")

    # データ点
    plt.errorbar(C, eps, yerr=error_QPE_std, fmt="^-", label="QPE")

    # フィット直線
    plt.loglog(
        C, beta_fix / C, "--", lw=3, label=rf"fit: $\varepsilon = {beta_fix:.2f}/M$"
    )

    plt.xlabel("$T$", fontsize=20)
    plt.ylabel(r"$\varepsilon$(T)", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(1e2, 1e5)
    plt.tight_layout()
    plt.legend(fontsize=15, loc="best")
    plt.tight_layout()
    plt.show()


def beta_scaling(
    T_list_QPE=np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]),  # M
    N_rep=10,  # 繰り返し回数
    N_QPE=int(100),  # サンプリング数
):
    # ---- 100 回の trial で beta_fix を求めて平均 ----
    N_trials = 100
    beta_fix_list = []

    np.random.seed(42)
    for trial in range(N_trials):
        # [-pi, pi] から一様ランダムに 1 点選択
        spectrum_F = [np.random.uniform(-np.pi, np.pi)]

        error_QPE = np.zeros(len(T_list_QPE), dtype="float")
        T_total_QPE = np.zeros(len(T_list_QPE), dtype="float")
        error_QPE_timeevo_all = np.zeros((N_rep, len(T_list_QPE)), dtype="float")

        for n in range(N_rep):
            for k in range(len(T_list_QPE)):
                T_max = T_list_QPE[k]
                output_energy = QPE([spectrum_F[0]], [1], T_max, N_QPE)
                T_total_QPE[k] += T_max * N_QPE

                # --- measure error ---
                err = np.abs(spectrum_F[0] - output_energy)
                error_QPE[k] += err
                error_QPE_timeevo_all[n][k] += err

        T_total_QPE = T_total_QPE / N_rep
        error_QPE = error_QPE / N_rep
        error_QPE_std = np.std(error_QPE_timeevo_all, axis=0)

        # ---- α=1 固定フィット（元の式を踏襲）----
        C = T_list_QPE  # コスト軸に合わせる
        eps = error_QPE

        # 数値安定化（log(0)回避）。元の式に +tiny を足す以外は変更しない
        tiny = 1e-300
        logC, logEps = np.log(C), np.log(eps + tiny)
        log_beta = np.average(logEps + logC)
        beta_fix = np.exp(log_beta)

        beta_fix_list.append(beta_fix)

    beta_fix_array = np.array(beta_fix_list, dtype=float)

    print("Mean beta_fix over 100 trials:", beta_fix_array.mean())
    print("Std  beta_fix over 100 trials:", beta_fix_array.std())


# =========================
# 許容誤差ごとの最良積公式
# =========================


def best_product_formula_all(mol, ham_name, n_w_list):
    CA = 1.59360010199040e-3
    CA_list = [CA * (10 ** (-0.01 * i)) for i in range(-200, 300)]
    beta = 1.2

    result = {str(pair): [] for pair in n_w_list}
    CA_exp = {str(pair): [] for pair in n_w_list}
    expo_dir = {str(pair): None for pair in n_w_list}
    coeff_dir = {str(pair): None for pair in n_w_list}
    cost_dir = {str(pair): None for pair in n_w_list}
    total_list = []

    for num_w in n_w_list:
        unit_expo = decompo_num[mol][num_w]

        cost_dir[str(num_w)] = unit_expo

        target_path = f"{ham_name}_Operator_{num_w}_ave"
        # target_path = f"{ham_name}_Operator_{num_w}"

        try:
            data = load_data(target_path)
            expo_dir[str(num_w)] = p_dir[num_w]
            coeff_dir[str(num_w)] = data
            # expo_dir[str(num_w)] = data['expo']
            # coeff_dir[str(num_w)] = data['coeff']

        except Exception as e:
            print(f"not found {target_path}")
            continue

    # 所望精度達成ゲート数計算
    for error_E in CA_list:
        min_total_expo = float("inf")
        best_trotter = None
        total = {}
        for num_w in n_w_list:
            expo = expo_dir[str(num_w)]
            coeff = coeff_dir[str(num_w)]
            if expo == None or coeff == None:
                continue
            expo = float(expo)
            coeff = float(coeff)
            min_f = (
                beta
                * (error_E ** (-(1 + (1 / expo))))
                * (1 / expo)
                * (coeff ** (1 / expo))
                * (expo + 1) ** (1 + (1 / expo))
            )

            # グルーピングあり
            unit_expo = cost_dir[str(num_w)]

            total_expo = unit_expo * min_f
            # print(f'minf{min_f} cost{unit_expo} w{num_w}')
            total[str(num_w)] = total_expo
            if error_E == CA:
                CA_exp[str(num_w)] = total_expo

            if total_expo < min_total_expo:
                min_total_expo = total_expo
                best_trotter = str(num_w)
        total_list.append(total)
        error_fac = math.log10(error_E / CA)
        result[best_trotter].append(error_fac)

    return result, total_list, CA_exp


def efficient_accuracy_range_plt_grouper(Hchain, n_w_list):
    CA = 1.59360010199040e-3
    xdic = {}
    dic = {str(n_w): {} for n_w in n_w_list}

    for chain in range(2, Hchain + 1):
        mol = f"H{chain}"
        distance = 1.0
        _, _, ham_name, n_qubits = jw_hamiltonian_maker(chain, distance)
        ham_name = ham_name + "_grouping"

        xdic.setdefault(mol, n_qubits)
        result, _, _ = best_product_formula_all(mol, ham_name, n_w_list)

        for label, error_range in result.items():
            if error_range:
                min_error_expo = min(error_range)
                max_error_expo = max(error_range)
                accuracy_range = []
                accuracy_range.append(CA * (10 ** (min_error_expo)))
                accuracy_range.append(CA * (10 ** (max_error_expo)))
                dic[label].setdefault(mol, accuracy_range)
            else:
                continue
    """
    data = {
        'w2': { 'H2':[1, 5], 'H4':[2,6]},
        'w4': { 'H2':[6, 7], 'H4':[7,8]}
    }
    xdic = {'H2':4, 'H4':8}
    """

    def plot_with_horizontal_offset(data, xdic, offset=0.2):

        plt.figure(figsize=(8, 6), dpi=200)

        # 凡例用のラベル管理
        color_labels = set()
        marker_labels = set()

        marker_map = {
            "H2": "o",
            "H3": "s",
            "H4": "^",
            "H5": "v",
            "H6": "<",
            "H7": ">",
            "H8": "D",
            "H9": "p",
            "H10": "h",
            "H11": "*",
            "H12": "x",
            "H13": "+",
            "H14": "P",
            "H15": "X",
        }

        x_offsets = {}  # 横軸のオフセット管理

        def get_unique_x(x_base, shape, label):
            # key = (shape, label)  # shape と label の組み合わせでユニークなキーを生成
            if x_base not in x_offsets:
                x_offsets[x_base] = {}
            if shape not in x_offsets[x_base]:
                x_offsets[x_base][shape] = len(x_offsets[x_base]) * offset
            return x_base + x_offsets[x_base][shape]

        for label, subsets in data.items():
            color = color_map[label]  # グループ (w2, w4, w8) の色を取得
            for shape, y_range in subsets.items():
                x_base = xdic[shape]  # 基準横軸値を取得
                x_unique = get_unique_x(x_base, shape, label)  # 一意の横軸位置を取得
                plt.plot(
                    [x_unique, x_unique], y_range, marker=marker_map[shape], color=color
                )

                if label not in color_labels:
                    plt.plot([], [], color=color, label=label)
                    color_labels.add(label)

                if shape not in marker_labels:
                    plt.plot(
                        [],
                        [],
                        marker=marker_map[shape],
                        color="black",
                        linestyle="None",
                        label=shape,
                    )
                    marker_labels.add(shape)

        # グループ化された凡例の設定
        handles, labels = plt.gca().get_legend_handles_labels()
        color_handles = [h for h, l in zip(handles, labels) if l in data.keys()]
        color_labels = [l for l in labels if l in data.keys()]
        marker_handles = [h for h, l in zip(handles, labels) if l in xdic.keys()]
        marker_labels = [l for l in labels if l in xdic.keys()]

        ca_handle = plt.axhline(y=CA, color="r", linestyle="--", label="CA")

        # 凡例にCAを追加
        combined_handles = color_handles + marker_handles + [ca_handle]
        combined_labels_0 = color_labels + marker_labels + ["CA"]
        combined_labels_1 = [s.replace("_gr", "") for s in combined_labels_0]
        combined_labels = [s.replace("my1", "4(new)") for s in combined_labels_1]
        combined_labels_2 = [label_replace(lb) for lb in combined_labels]

        plt.yscale("log")
        plt.xlabel("Spin orbitals", fontsize=15)
        plt.ylabel("Target error [Hartree]", fontsize=15)
        plt.tick_params(labelsize=15)
        # plt.title("Accuracy range of each product formula for spin orbitals",fontsize=15)
        plt.legend(
            combined_handles,
            combined_labels_2,
            title=f"product \nformula",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            ncol=1,
            fontsize=13,
        )

        plt.show()

    plot_with_horizontal_offset(dic, xdic, offset=0.2)
