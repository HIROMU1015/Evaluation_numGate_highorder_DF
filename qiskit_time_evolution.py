import logging
from functools import reduce, lru_cache
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

# openfermion / grouping
from openfermion import InteractionOperator
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner,get_fermion_operator
from openfermion.chem.molecular_data import spinorb_from_spatial  # noqa: F401 (参照のみ)

# qiskit
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate, GlobalPhaseGate
from qiskit_aer import AerSimulator

# pyscf
import pyscf
from pyscf import fci, gto, scf
from pyscf.fci import cistring

# external modules used in original file
import trotter_error_plt as tsag
from Almost_optimal_grouping import Almost_optimal_grouper


AER_METHOD_STATEVECTOR = "statevector"  # AerSimulator の method
DEFAULT_BASIS = "sto-3g"  # PySCF 基底関数

# ロガー（print の削減）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def free_var(name: str, scope: dict) -> None:
    """ローカル変数を解放して GC を促す（メモリ圧を下げるための補助）。"""
    if name in scope:
        del scope[name]
        import gc

        gc.collect()


def apply_time_evolution(eigenvector: np.ndarray, time_evolution_circuit: QuantumCircuit) -> Statevector:
    """|ψ⟩ を初期化せず、Statevector.evolve で時間発展させた最終状態を返す。"""
    # initialize 不使用
    sv = Statevector(eigenvector)
    final_sv = sv.evolve(time_evolution_circuit)
    return final_sv

def _term_to_sparse_pauli(term: Tuple[Tuple[int, str], ...], n_qubits: int) -> SparsePauliOp:
    """OpenFermionの term を Qiskit の SparsePauliOp に変換（キャッシュ付）。"""
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


def add_clique_to_circuit_grouper(
    clique: Sequence[QubitOperator], t: float, n_qubits: int, w: float, qc: QuantumCircuit
) -> int:
    """
    可換クリーク中の項を和に束ねて一括で進化ゲートを追加。
    戻り値は、元の「指数項の数」（ゲート数ではなく、物理評価用のカウント）を返す。
    """
    # 係数付きの SparsePauliOp を加算して合成ハミルトニアン H_clique を構築
    sum_op: SparsePauliOp | None = None
    num_exp = 0
    for hamiltonian in clique:
        for term, coeff in hamiltonian.terms.items():
            if not term:
                # 恒等項（定数項）は回路には入れない
                continue
            pauli_op = _term_to_sparse_pauli(tuple(term), n_qubits)
            pauli_op = coeff.real * pauli_op
            sum_op = pauli_op if sum_op is None else (sum_op + pauli_op)
            num_exp += 1
    if sum_op is None:
        return 0
    # 以前は各項 angle=coeff*w*t だったが、ここでは H=Σ coeff*P として time=w*t を与える
    evolution_gate = PauliEvolutionGate(sum_op, time=(w * t), synthesis=None)
    qc.append(evolution_gate, range(n_qubits))
    return num_exp


def S_2_gr(clique_list: Sequence[Sequence[QubitOperator]], t: float, n_qubits: int, w: float, qc: QuantumCircuit) -> int:
    """2次PFの基本シンメトリック分解（左端）。クリークを一括進化で実装。"""
    J = len(clique_list)
    num_exp = 0
    for i in range(J - 1):
        add_exp = add_clique_to_circuit_grouper(clique_list[i], t, n_qubits, w / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[J - 1], t, n_qubits, w, qc)
    num_exp += add_exp
    for k in reversed(range(0, J - 1)):
        add_exp = add_clique_to_circuit_grouper(clique_list[k], t, n_qubits, w / 2, qc)
        num_exp += add_exp
    return num_exp


def S_2_trotter_left_gr(
    clique_list: Sequence[Sequence[QubitOperator]], t: float, n_qubits: int, Max_w: float, nMax_w: float, qc: QuantumCircuit
) -> int:
    """高次PF 合成用の左端ブロック（クリーク一括進化）。"""
    J = len(clique_list)
    num_exp = 0
    for i in range(J - 1):
        add_exp = add_clique_to_circuit_grouper(clique_list[i], t, n_qubits, Max_w / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[J - 1], t, n_qubits, Max_w, qc)
    num_exp += add_exp
    for k in reversed(range(1, J - 1)):
        add_exp = add_clique_to_circuit_grouper(clique_list[k], t, n_qubits, Max_w / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[0], t, n_qubits, (Max_w + nMax_w) / 2, qc)
    num_exp += add_exp
    return num_exp


def S_2_trotter_gr(
    clique_list: Sequence[Sequence[QubitOperator]],
    t: float,
    n_qubits: int,
    w_f: float,
    w_s: float,
    qc: QuantumCircuit,
    num_exp: int,
) -> int:
    """高次PF 合成用の中央ブロック（クリーク一括進化）。"""
    J = len(clique_list)
    for i in range(1, J - 1):
        add_exp = add_clique_to_circuit_grouper(clique_list[i], t, n_qubits, w_f / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[J - 1], t, n_qubits, w_f, qc)
    num_exp += add_exp
    for k in reversed(range(1, J - 1)):
        add_exp = add_clique_to_circuit_grouper(clique_list[k], t, n_qubits, w_f / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[0], t, n_qubits, (w_f + w_s) / 2, qc)
    num_exp += add_exp
    return num_exp


def S_2_trotter_right_gr(
    clique_list: Sequence[Sequence[QubitOperator]], t: float, n_qubits: int, w_i: float, qc: QuantumCircuit, num_exp: int
) -> int:
    """高次PF 合成用の右端ブロック（クリーク一括進化）。"""
    J = len(clique_list)
    for i in range(1, J - 1):
        add_exp = add_clique_to_circuit_grouper(clique_list[i], t, n_qubits, w_i / 2, qc)
        num_exp += add_exp
    add_exp = add_clique_to_circuit_grouper(clique_list[J - 1], t, n_qubits, w_i, qc)
    num_exp += add_exp
    for k in reversed(range(0, J - 1)):
        add_exp = add_clique_to_circuit_grouper(clique_list[k], t, n_qubits, w_i / 2, qc)
        num_exp += add_exp
    return num_exp


def _get_w_list(num_w: Any) -> List[float]:
    """積公式パラメータ w の系列を取得（分岐を関数化）。"""
    if num_w == '8th(Morales)':
        return tsag.morales_8th_list()
    if num_w == '10th(Morales)':
        return tsag.morales_10th_m16_list()
    if num_w == '4th':
        return tsag.yoshida_4th_list()
    if num_w == "8th(Yoshida)":
        return tsag.yoshida_8th_list()
    if num_w == '2nd':
        return tsag.trotter_2nd_list()
    if num_w == "4th(new_3)":
        return tsag.new_4th_m3_list()
    if num_w == "4th(new_2)":
        return tsag.new_4th_m2_list()
    raise ValueError(f"Unsupported num_w: {num_w}")


def w_trotter_grouper(qc: QuantumCircuit, clique_list: Sequence[Sequence[QubitOperator]], t: float, n_qubits: int, num_w: Any) -> int:
    """与えられた w シリーズで PF 分解を回路に追加し、累計項数を返す。"""
    w_list = _get_w_list(num_w)

    m = len(w_list)
    if m == 1:
        num_exp = S_2_gr(clique_list, t, n_qubits, w_list[0], qc)
        return num_exp

    num_exp = S_2_trotter_left_gr(clique_list, t, n_qubits, w_list[m - 1], w_list[m - 2], qc)
    for i in reversed(range(1, m - 1)):
        num_exp = S_2_trotter_gr(clique_list, t, n_qubits, w_list[i], w_list[i - 1], qc, num_exp)
    for i in range(0, m - 1):
        num_exp = S_2_trotter_gr(clique_list, t, n_qubits, w_list[i], w_list[i + 1], qc, num_exp)
    num_exp = S_2_trotter_right_gr(clique_list, t, n_qubits, w_list[m - 1], qc, num_exp)
    return num_exp


def tEvolution_vector_grouper(
    clique_list: Sequence[Sequence[QubitOperator]],
    t: float,
    n_qubits: int,
    ori_vec: np.ndarray,
    num_w: Any,
) -> Tuple[float, Statevector, int]:
    """グルーピング済みハミルトニアンで時間発展回路を合成し、最終状態を返す。"""
    evo_qc = QuantumCircuit(n_qubits)
    num_exp = w_trotter_grouper(evo_qc, clique_list, t, n_qubits, num_w)
    final_state = apply_time_evolution(ori_vec, evo_qc)
    return t, final_state, num_exp


def make_fci_vector_from_pyscf_solver_grouper(mol_type: str) -> Tuple[List[List[QubitOperator]], int, float, np.ndarray]:
    """PySCF FCI から |ψ₀⟩ を構築し、グルーピング済み JW 演算子群とともに返す。"""
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = DEFAULT_BASIS
    mol.spin = multiplicity - 1
    mol.charge = molcharge
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    n_qubits_guess = 2 * mf.mo_coeff.shape[0]

    # integrals
    constant = mf.energy_nuc()
    mo = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body_integrals = reduce(np.dot, (mo.T, h_core, mo))

    eri = pyscf.ao2mo.kernel(mf.mol, mo)
    eri = pyscf.ao2mo.restore(1, eri, mo.shape[0])
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    almost_optimal_grouper = Almost_optimal_grouper(
        constant, one_body_integrals, two_body_integrals, fermion_qubit_mapping=jordan_wigner, validation=True
    )
    grouping_term_list = almost_optimal_grouper.group_term_list
    # 定数項をグループ先頭へ（OpenFermion FermionOperator の空文字は恒等項）
    grouping_term_list[0].insert(0, FermionOperator("", almost_optimal_grouper._const_fermion))
    grouping_jw_list: List[List[QubitOperator]] = [jordan_wigner(sum(group_term)) for group_term in grouping_term_list]

    pyscf_fci_solver = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = pyscf_fci_solver.kernel()
    n_qubits = pyscf_fci_solver.norb * 2
    n_orbitals = pyscf_fci_solver.norb
    nelec_alpha, nelec_beta = pyscf_fci_solver.nelec
    fci_vector = np.zeros(2**n_qubits, dtype=np.complex128)

    ci_strings_alpha = cistring.make_strings(range(n_orbitals), nelec_alpha)
    ci_strings_beta = cistring.make_strings(range(n_orbitals), nelec_beta)

    # 注意: 右が q0 となる Qiskit のビット順に合わせるにはビット反転が必要だが、
    # 既存コードは **反転しない** 前提で構築しているため、その挙動を維持する。
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

    vector = fci_vector.reshape(-1, 1)
    return grouping_jw_list, n_qubits, energy, vector


# =========================
# 摂動論誤差検証用
# =========================

def add_term_to_circuit(hamiltonian, n_qubits, t, w, qc):
    """単一ハミルトニアン（OpenFermion QubitOperator）の各項を回路に追加し、追加した指数項数を返す。
    注: Qiskit の Pauli ラベルは右端が q0。OpenFermion のインデックスに合わせて構築する際はこの反転規約に注意。
    """
    num_exp = 0
    for term, coeff in hamiltonian.terms.items():
        angle = coeff.real * w * t
        if not term:
            # 恒等項はグローバル位相のみ。物理的な指数項数としてはカウントしない。
            qc.append(GlobalPhaseGate(-angle))
            continue

        # ラベルを I で初期化し、(index, 'X'/'Y'/'Z') を配置
        label = ["I"] * n_qubits
        for idx, pauli_name in term:
            label[idx] = pauli_name

        # Qiskit では右端が q0 なのでラベルを反転して整合を取る
        pauli_label = "".join(label[::-1])  # ← ラベル反転（右端が q0）
        pauli_op = SparsePauliOp(pauli_label)

        evo = PauliEvolutionGate(pauli_op, time=angle, synthesis=None)
        qc.append(evo, range(n_qubits))
        num_exp += 1
    return num_exp



def add_term_to_circuit(hamiltonian, n_qubits, t, w, qc): #ハミルトニアンの項からゲートを回路に追加
    from qiskit.synthesis.evolution import SuzukiTrotter
    from qiskit.synthesis.evolution import MatrixExponential

    num_exp = 0
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
            qc.append(GlobalPhaseGate(-1*angle))
            return num_exp

        rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=None)
        #rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=SuzukiTrotter(order=4,reps=2))
        #rotation_gate = PauliEvolutionGate(pauli_op, time=angle, synthesis=MatrixExponential())

        qc.append(rotation_gate, range(n_qubits))
    return num_exp


def S_2(ham_list, t, n_qubits, w, qc, eU, speU):
    """2次PFの基本シンメトリック分解（左端）。指数項数を返す。"""
    J = len(ham_list)
    num_exp = 0
    for i in range(J - 1):
        num_exp += add_term_to_circuit(ham_list[i], n_qubits, t, w / 2, qc)
    num_exp += add_term_to_circuit(ham_list[J - 1], n_qubits, t, w, qc)
    for k in reversed(range(0, J - 1)):
        num_exp += add_term_to_circuit(ham_list[k], n_qubits, t, w / 2, qc)
    return num_exp


def S_2_trotter_left(A_list, n_qubits, t, Max_w, nMax_w, qc, eU, speU):
    """高次PF 合成用の左端ブロック。指数項数を返す。"""
    J = len(A_list)
    num_exp = 0
    for i in range(J - 1):
        num_exp += add_term_to_circuit(A_list[i], n_qubits, t, Max_w / 2, qc)
    num_exp += add_term_to_circuit(A_list[J - 1], n_qubits, t, Max_w, qc)
    for k in reversed(range(1, J - 1)):
        num_exp += add_term_to_circuit(A_list[k], n_qubits, t, Max_w / 2, qc)
    num_exp += add_term_to_circuit(A_list[0], n_qubits, t, (Max_w + nMax_w) / 2, qc)
    return num_exp


def S_2_trotter(A_list, n_qubits, t, w_f, w_s, qc, eU, speU, idx):
    """高次PF 合成用の中央ブロック。指数項数を返す。"""
    J = len(A_list)
    num_exp = 0
    for i in range(1, J - 1):
        num_exp += add_term_to_circuit(A_list[i], n_qubits, t, w_f / 2, qc)
    num_exp += add_term_to_circuit(A_list[J - 1], n_qubits, t, w_f, qc)
    for k in reversed(range(1, J - 1)):
        num_exp += add_term_to_circuit(A_list[k], n_qubits, t, w_f / 2, qc)
    num_exp += add_term_to_circuit(A_list[0], n_qubits, t, (w_f + w_s) / 2, qc)
    return num_exp


def S_2_trotter_right(A_list, n_qubits, t, w_i, qc, eU, speU, idx):
    """高次PF 合成用の右端ブロック。指数項数を返す。"""
    J = len(A_list)
    num_exp = 0
    for i in range(1, J - 1):
        num_exp += add_term_to_circuit(A_list[i], n_qubits, t, w_i / 2, qc)
    num_exp += add_term_to_circuit(A_list[J - 1], n_qubits, t, w_i, qc)
    for k in reversed(range(0, J - 1)):
        num_exp += add_term_to_circuit(A_list[k], n_qubits, t, w_i / 2, qc)
    return num_exp


def w_trotter(qc, ham_list, t, n_qubits, num_w):
    """与えられた w シリーズで PF 分解を回路に追加し、累計の指数項数を返す。"""
    eU = speU = idx = 0  # 未使用パラメータの互換維持
    w_list = _get_w_list(num_w)
    m = len(w_list)
    if m == 1:
        return S_2(ham_list, t, n_qubits, w_list[0], qc, eU, speU)

    num_exp = 0
    num_exp += S_2_trotter_left(ham_list, n_qubits, t, w_list[m - 1], w_list[m - 2], qc, eU, speU)
    for i in reversed(range(1, m - 1)):
        num_exp += S_2_trotter(ham_list, n_qubits, t, w_list[i], w_list[i - 1], qc, eU, speU, idx)
    for i in range(0, m - 1):
        num_exp += S_2_trotter(ham_list, n_qubits, t, w_list[i], w_list[i + 1], qc, eU, speU, idx)
    num_exp += S_2_trotter_right(ham_list, n_qubits, t, w_list[m - 1], qc, eU, speU, idx)
    return num_exp


def tEvolution_vector(ham_list, t, n_qubits, ori_vec, num_w):
    """非グルーピング版の時間発展回路を合成し、最終状態と指数項数を返す。"""
    evo_qc = QuantumCircuit(n_qubits)
    num_exp = w_trotter(evo_qc, ham_list, t, n_qubits, num_w)
    final_state = apply_time_evolution(ori_vec, evo_qc)
    return t, final_state

def make_fci_vector_from_pyscf_solver(mol_type):
    """PySCFからFCIベクトルとJWハミルトニアンを生成（元コードと同一の並びと位相）。"""
    # --- Geometry / SCF ---
    geometry, multiplicity, molcharge = tsag.geo(mol_type)
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
    constant = mf.energy_nuc()
    mo = mf.mo_coeff
    h_core = mf.get_hcore()
    one_body = reduce(np.dot, (mo.T, h_core, mo))
    eri = pyscf.ao2mo.kernel(mf.mol, mo)
    eri = pyscf.ao2mo.restore(1, eri, mo.shape[0])
    two_body = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    # spin-orbital Hamiltonian → Fermion → JW
    h1s, h2s = spinorb_from_spatial(one_body, two_body * 0.5)
    ham_fermion = get_fermion_operator(InteractionOperator(constant, h1s, h2s))
    jw_hamiltonian = jordan_wigner(ham_fermion)

    # --- FCI solve ---
    pyscf_fci = fci.FCI(mol, mf.mo_coeff)
    energy, ci_matrix = pyscf_fci.kernel()
    n_orb = pyscf_fci.norb
    n_qubits = n_orb * 2
    nelec_alpha, nelec_beta = pyscf_fci.nelec

    # --- CI state on qubits: interleave [beta_k, alpha_k] for k=0.. ---
    fci_vector = np.zeros(2 ** n_qubits, dtype=np.complex128)
    ci_alpha = cistring.make_strings(range(n_orb), nelec_alpha)
    ci_beta  = cistring.make_strings(range(n_orb), nelec_beta)

    for i, a_str in enumerate(ci_alpha):
        a_bits = format(a_str, f"0{n_qubits // 2}b")[::-1]  # LSBが軌道0
        for j, b_str in enumerate(ci_beta):
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
            fci_vector[index] = sign * ci_matrix[i][j]

    vector = fci_vector.reshape(-1, 1)
    return jw_hamiltonian, n_qubits, energy, vector, ci_matrix


