"""
qhandai グルーピング手法の再現実装 by 中川 2023年4月
https://arxiv.org/abs/2301.07335
"""

#!/usr/bin/env python
# coding: utf-8
import copy, itertools
from functools import reduce

import numpy as np
from sympy import isprime

from openfermion.ops import QubitOperator, FermionOperator, InteractionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, normal_ordered
from openfermion.linalg import get_sparse_operator, get_ground_state
from openfermion.utils import commutator, hermitian_conjugated, count_qubits
# from qulacs import QuantumState

def make_spinorb_ham_upthendown_order(constant, one_body_integrals, two_body_integrals, validation=True):
    """ (up全部)-(down全部) の添字順で第二量子化ハミルトニアンを作る. 
    スピン毎に電子積分が違う場合は未対応.
    """
    n_orb = one_body_integrals.shape[0]
    assert one_body_integrals.shape == (n_orb, n_orb)
    assert two_body_integrals.shape == (n_orb, n_orb, n_orb, n_orb)
    n_qubits = 2 * n_orb
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_orb):
        for q in range(n_orb):
            p_up = p
            q_up = q
            p_down = n_orb + p
            q_down = n_orb + q
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[p_up, q_up] = one_body_integrals[p, q]
            one_body_coefficients[p_down, q_down] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_orb):
                for s in range(n_orb):
                    r_up = r
                    s_up = s
                    r_down = n_orb + r
                    s_down = n_orb + s
                    # Mixed spin
                    two_body_coefficients[p_up, q_down, r_down, s_up] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[p_down, q_up, r_up, s_down] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    # Same spin
                    two_body_coefficients[p_up, q_up, r_up, s_up] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[
                        p_down, q_down, r_down, s_down
                    ] = (two_body_integrals[p, q, r, s] / 2.0)

    # Cast to InteractionOperator class and return.
    return InteractionOperator(constant, one_body_coefficients, two_body_coefficients)


def map_openfermion_fermionop_to_upthendown_order(fermi_op, n_qubits):
    """ up-down-...up-down order の添字順を, (up全部)-(down全部)にする
    """
    assert n_qubits % 2 == 0
    mapping = []
    for i in range(n_qubits // 2):
        mapping.append(i)
        mapping.append(i + n_qubits // 2)
    
    ret = FermionOperator()
    for key, item in fermi_op.terms.items():
        # print(key)
        # print(item)
        indices = list(key)
        if len(indices) == 0: ## 定数項で key が () の場合
            ret += FermionOperator(key, item)
            continue
        new_indices = [] ## 新たなkeyになるもの (最後はtupleにする)
        for index, create_or_annihilate in indices:
            new_indices.append( (mapping[index], create_or_annihilate) )
        ret += FermionOperator(tuple(new_indices), item) ## 係数はそのまま
    return ret

def schedule(n):
    """ 総当たりスケジュールを生成する. ChatGPT-3.5により生成.
    この関数は、引数としてチームの数nを受け取り、nが奇数の場合は、nを偶数に変更します。
    次に、0からn-1までの数字をリストteamsに格納し、各ラウンドでの対戦を行います。

    各ラウンドでは、まず、チームを2つのグループに分けます。次に、片方のグループのチームを反転して、グループ1とグループ2の各チーム同士で対戦します。対戦が終わったら、グループ2のチームを先頭に移動して、次のラウンドでの対戦を行います。

    最後に、ラウンドごとの対戦結果をリストmatchesに格納し、これを返します。
    """
    n_is_odd = False ## 1を足したかどうかの処理を覚えておく.
    if n % 2 == 1:
        n += 1
        n_is_odd = True
    teams = list(range(n))
    matches = []
    for i in range(n-1):
        mid = n // 2
        group1 = teams[:mid]
        group2 = teams[mid:]
        group2.reverse()
        round = []
        for j in range(mid):
            match = (group1[j], group2[j])
            ## 存在しないチームは入れない処理を追加
            if not n_is_odd:
                round.append(match) 
            elif group1[j] != n-1 and group2[j] != n-1: ## 奇数の時はn-1が架空のチーム
                round.append(match)
        matches.append(round)
        teams.insert(1, teams.pop()) ## 最後のチームを取り出して1番目に挿入.
    return matches

def make_clique_by_finite_projective_plane(n):
    """ 有限射影平面を使い, 空間軌道 n のハミルトニアンに対するクリーク(整数のタプルのリスト)を返す.
    """
    assert n >= 4
    pi = n - 1
    while not isprime(pi):
        pi += 1
    assert isprime(pi)

    ## qubitを表す点
    s_points_in_Pgamma = set([(i, i*i % pi) for i in range(pi)]) ## S_gamma

    clique_list = []

    ## S が乗っていない P_gamma の点(a,b)に対応するクリーク
    for a in range(pi):
        for b in range(pi):
            if b == (a*a % pi):
                continue
            clique = [(a,pi)] ## L_betaからくる項
            ## 傾きslopeの直線 L_gamma を引いて S がどれか含まれるかチェック
            for slope in range(pi):
                points_on_line = set([(i, (slope*(i-a)+b) % pi) for i in range(pi)])
                intersection = points_on_line & s_points_in_Pgamma ## L_gamma と S_gamma との交点
                if len(intersection) == 2:
                    p1 = intersection.pop()[0]
                    p2 = intersection.pop()[0]
                    if p1 < p2:
                        clique.append( (p1,p2) )
                    elif p2 < p1:
                        clique.append( (p2,p1) )
                    else:
                        raise ValueError
                elif len(intersection) == 1:
                    p1 = intersection.pop()[0]
                    clique.append( (p1,p1) )
                else:
                    pass
                
            clique_list.append(clique)
    # print('checked P_gamma clique')
    # print(clique_list)

    ## P_beta(b)に対応するクリーク
    for b in range(pi):
        clique = [(pi,pi)] ## 直線 L_alpha
        for intercept in range(pi): ## (0, intercept) を通る傾きbの直線を調べる
            points_on_line = set([(i, (b*i + intercept) % pi) for i in range(pi)]) ## {(),(),...}
            intersection = points_on_line & s_points_in_Pgamma ## 一致した()を保持
            if len(intersection) == 2:
                p1 = intersection.pop()[0] ## 削除した要素を取得 
                p2 = intersection.pop()[0]
                if p1 < p2:
                    clique.append( (p1,p2) )
                elif p2 < p1:
                    clique.append( (p2,p1) )
                else:
                    raise ValueError
            elif len(intersection) == 1:
                p1 = intersection.pop()[0]
                clique.append( (p1,p1) )
            else:
                pass
        clique_list.append(clique)
    # print('checked P_beta clique')
    # print(clique_list)

    ## piが大きい場合は本来のサイズより大きい点を削除. 
    ## 動的に更新するとなぜかバグるので, 静的コピーを用意して代入する.
    if pi > n-1:
        clique_new_list = [] 
        for clique in clique_list:
            # print(clique)
            clique_new = []
            for p,q in clique:
                if p <= n-1 and q <= n-1:
                    clique_new.append((p,q))
            clique_new_list.append(clique_new)
        return clique_new_list
    else:
        return clique_list

def validiate_group_term_list(group_term_list):
    """ 各グループについて, (1) 全ての項がエルミートなこと (2) 全ての項が互いに交換することを確認
    """
    for ind, group_term in enumerate(group_term_list):
        m = len(group_term)
        for i in range(m):
            term1 = group_term[i]
            #term1 /= term1.induced_norm() ## 小さくなりすぎるのを防ぐ
            assert normal_ordered(commutator(term1, hermitian_conjugated(term1))) == FermionOperator("",0.), term1
            for j in range(i+1, m):
                term2 = group_term[j]
                # term2 /= term2.induced_norm()
                assert normal_ordered(commutator(term1, term2)) == FermionOperator("",0.), f"{ind},{i},{j},\n{term1},\n{term2},\n{normal_ordered(commutator(term1, term2))}"
    print("all groups are hermite and mutually commute!")

class Almost_optimal_grouper(object):
    """ qhandai法を使った grouper. 
    """
    def __init__(self, const: float, one_body_integrals:np.array, two_body_integrals: np.array, fermion_qubit_mapping, validation=False):
        """
        第二量子化ハミルトニアンの情報(空間軌道添字)を受け取り,
        up-then-down の順番のハミルトニアンを作成.
        その後フェルミオン演算子のままグルーピングを行う.
        H = const 
        + \sum_{p,q=1}^N \sum_{spin=up,down} one_body_integrals[p,q] c_{p,spin}^\dag c_{p,spin}
        + 1/2 * \sum_{p,q,r,s=1}^N \sum_{spin,spin'=up,down}
                two_body_integrals[p,q,r,s] c_{p,spin}^\dag c_{q,spin'}^\dag c_{r,spin'} c_{s,spin}
        という notation. 論文は chemist notationなので注意.

        validation = True にすると, 出来上がったグループの検証(グループの和が元のハミルトニアンに一致するか, グループ内の演算子が交換するか)を行う.
        qubit演算子への変換も, validationがあるときだけにする?

        """
        self.ham_fermion_upthendown_original = normal_ordered(get_fermion_operator(make_spinorb_ham_upthendown_order(const, one_body_integrals, two_body_integrals)))
        self._ham_fermion_upthendown = copy.deepcopy(self.ham_fermion_upthendown_original)
        self._one_body_integrals = one_body_integrals
        self._two_body_integrals = two_body_integrals

        self._const_fermion = self.get_and_remove_const()
        self.group_term_list = self.generate_group_nakagawa(validation, fermion_qubit_mapping)
        ## qubit 演算子への変換が重いとき用
        # print("skip making qubit operators for groups.")
        self._group_list = self.create_group_list(fermion_qubit_mapping) 


    def get_and_remove_const(self):
        """ フェルミオンハミルトニアンの定数部分を取り出す.
        """
        return self._ham_fermion_upthendown.terms.pop((), 0.0)

    # def get_group_count(self):
    #     return len(self._group_list)
        
    # def get_group(self, index):
    #     return self._group_list[index]
    
    def generate_group_nakagawa(self, validation, fermion_qubit_mapping):
        """ フェルミオン演算子のグルーピングを行う.
        Return:
            group_term_list:
                list of list of FermionOperator.
                group_term_list[i][j] は i 番目のグループの j 番目のエルミート演算子(論文のA_{pq,spin}などで書かれる).
        """
        n_orb = self._one_body_integrals.shape[0] ## 空間軌道の数. 論文のN.
        n_qubits = 2 * n_orb
        group_term_list = []
        self._ham_fermion_upthendown = copy.deepcopy(self.ham_fermion_upthendown_original)
        self._ham_fermion_upthendown.terms.pop((), 0.0)
        zero_fermion_operator = FermionOperator("", 0.) ## 項がゼロかどうかに判定に使う. 本当は自分で閾値を決めるべき.

        """
        group for number operator (which can be measured by C^part in the paper)
        n_{p,spin} の係数は spinによらず h_{pp}.
        n_{p,spin}*n_{q,spin'} の係数は同一スピンと逆スピンで少し変わるので注意.
        """
        group_term = []
        for spin in [0,1]:
            for p in range(n_orb):
                p_spin = spin * n_orb + p ## スピン軌道の添字
                ## n_{p, spin}
                coef = self._one_body_integrals[p,p]
                term = FermionOperator(f"[{p_spin}^ {p_spin}]", coef)
                if term != zero_fermion_operator:
                    group_term.append(term)
                    self._ham_fermion_upthendown -= term

                ## n_{p,spin}*n_{p,1-spin}
                p_diff_spin = (1-spin) * n_orb + p
                coef = 0.5 * self._two_body_integrals[p,p,p,p]
                term = FermionOperator(f"[{p_spin}^ {p_spin} {p_diff_spin}^ {p_diff_spin}]", coef)
                if term != zero_fermion_operator:
                    group_term.append(term)
                    self._ham_fermion_upthendown -= normal_ordered(term)

                for q in range(p+1, n_orb): ## p != q の n_{p,spin}*n_{q,spin'}
                    ## 同じスピン
                    q_same_spin = spin * n_orb + q
                    coef_same_spin = self._two_body_integrals[p,q,q,p] - self._two_body_integrals[p,q,p,q]
                    term = FermionOperator(f"[{p_spin}^ {p_spin} {q_same_spin}^ {q_same_spin}]", coef_same_spin)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        self._ham_fermion_upthendown -= normal_ordered(term)
                    ## 逆スピン
                    q_diff_spin = (1-spin) * n_orb + q
                    coef_diff_spin = self._two_body_integrals[p,q,q,p]
                    term = FermionOperator(f"[{p_spin}^ {p_spin} {q_diff_spin}^ {q_diff_spin}]", coef_diff_spin)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        self._ham_fermion_upthendown -= normal_ordered(term)

        if len(group_term) != 0:
            group_term_list.append(group_term) ## 1つ目のグループ

        """
        A_{pq,spin}, A_{pq,spin} * n_{r,1-spin} の項をとる. 論文のU^{[1]}
        論文の I_i (なるべく被らないで, 0<= p < q <= N-1 である(p,q) の全通りを尽くす)
        """
        I_indices_list = schedule(n_orb)

        for spin in [0, 1]:
            for matches in I_indices_list:
                group_term = [] 
                for p,q in matches:
                    ## indices for spin orbitals (up-then-down order)
                    p_spin = spin*n_orb + p
                    q_spin = spin*n_orb + q

                    coef = self._one_body_integrals[p,q]
                    term =  FermionOperator(f"[{p_spin}^ {q_spin}]", coef)
                    term += FermionOperator(f"[{q_spin}^ {p_spin}]", coef)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        self._ham_fermion_upthendown -= normal_ordered(term)

                    for r in range(n_orb):
                        r_spin = (1-spin)*n_orb + r
                        ## A_{pq,spin} * n_{r, 1-spin}の項.                        
                        coef = self._two_body_integrals[p,r,r,q]
                        term =  FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {r_spin}]", coef)
                        term += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {r_spin}]", coef)
                        if term != zero_fermion_operator:
                            group_term.append(term)
                            self._ham_fermion_upthendown -= normal_ordered(term)
                            
                if len(group_term) != 0:
                    group_term_list.append(group_term)

        # print("# of groups for C^part, U^1:", len(group_term_list))

        """
        A_{pq,spin}, A_{rs,1-spin} の項をとる. 論文のU^{[2,diff]}.
        """
        for i in range(len(I_indices_list)):
            matches_i = I_indices_list[i]
            for j in range(len(I_indices_list)):
                matches_j = I_indices_list[j]
                group_term = []
                for (p,q), (r,s) in itertools.product(matches_i, matches_j):
                    ## I_i, I_j の整数ペアに関する直積をとる.
                    p_up = p
                    q_up = q
                    r_down = n_orb + r
                    s_down = n_orb + s

                    ## A_{pq,spin}, A_{rs,1-spin}
                    coef = self._two_body_integrals[p,r,s,q]
                    term =  FermionOperator(f"[{p_up}^ {q_up} {r_down}^ {s_down}]", coef)
                    term += FermionOperator(f"[{q_up}^ {p_up} {r_down}^ {s_down}]", coef)
                    term += FermionOperator(f"[{p_up}^ {q_up} {s_down}^ {r_down}]", coef)
                    term += FermionOperator(f"[{q_up}^ {p_up} {s_down}^ {r_down}]", coef)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        self._ham_fermion_upthendown -= normal_ordered(term)

                if len(group_term) != 0:
                    group_term_list.append(group_term)

        # print("# of groups for C^part, U^1, U^{2,diff}:", len(group_term_list))

        """
        同一スピン内の A_{pq,spin} * A_{rs,spin} を列挙する. 論文の U^{[2,same]}.
        論文(5)式にあるように, ハミルトニアンの二電子項は
        1/8 \sum_{pqrs,spin} g_{pqrs,spin} A_{pq,spin} A_{rs,spin}
        とかける. (U^{[2,same]} でカバーすべきなのはspinが同じ項なので以下ではスピンを省略する)
        クリークを使って取り出せるのは, ((p<q),(r<s)) あるいは ((p=q),(r<s)), ((p<q),(r=s))
        という整数の組.  
        ((p<q),(r<s)) に対応するハミルトニアンの項はp,q,r,sが全て異なる項で, 対称性を考慮すると
        1/2 \sum_{p<q,r<s} g_{pqrs} A_{pq} A_{rs}
        とまとまり, さらに (p<q) と (r<s) の入れ替え対称性も考えて, 同時測定groupに取り込む演算子としては
        g_{pqrs} A_{pq} A_{rs}
        を入れる.
        ((p=q),(r<s))の場合に対応するハミルトニアンの項は
        1/4 \sum_{p=q,r<s} g_{pqrs} A_{pq} A_{rs} = 1/4 \sum_{p=q,r<s} g_{pqrs} 2 n_p A_{rs} 
        だけかと思いきや, A_{pr} A_{ps}  という項からも n_p A_{rs} が出てくる:
        A_{pr} A_{ps} = a_r^\dag a_s - n_p * (a_r^dag a_s + a_s^dag a_r).
        これの係数は 1/8 * g_{pprs} だが, (p,r)の入れ替えと(p,s)の入れ替えで factor 4が出てくる一方,
        計算すると n_p = 1/2 * A_{pp} になるので, 結果的には 1/8 * 4 * 1/2 = 1/4 になる.
        さらに, A_{ps} A_{pr} からにも同様に n_p A_{rs} が出てきて, 係数は 1/4 * g_{ppsr} になる.
        """
        #print(make_clique_by_finite_projective_plane(n_orb))
        #clique_list = np.array(make_clique_by_finite_projective_plane(n_orb))
        clique_list = make_clique_by_finite_projective_plane(n_orb)

        checked_pqrs = []
        for clique in clique_list:
            group_term = []
            for i in range(len(clique)):
                p, q = clique[i]
                for j in range(i+1, len(clique)):
                    r, s = clique[j]
                    ## 粒子数演算子だけでかけるものは既に考慮済み
                    if p == q and r == s:
                        continue 
                    ## 既に考慮したかチェック
                    if ((p,q),(r,s)) in checked_pqrs or ((r,s),(p,q)) in checked_pqrs:
                        continue
                    checked_pqrs.append(((p,q),(r,s)))
                    checked_pqrs.append(((r,s),(p,q)))

                    term = FermionOperator("[]", 0.) ## クリーク内の1つのedgeに対応する項.
                    if p != q and r != s: ## g_{pqrs} A_{pq} A_{rs} に対応
                        coef = self._two_body_integrals[p,r,s,q]
                    else: ## ここで少し非自明な寄与がある(上記コメント参照)
                        coef = 1. * (self._two_body_integrals[p,r,s,q]/2. - self._two_body_integrals[p,q,r,s]/4. - self._two_body_integrals[p,q,s,r]/4.)
                    for spin in [0,1]:
                        p_spin = spin * n_orb + p
                        q_spin = spin * n_orb + q
                        r_spin = spin * n_orb + r
                        s_spin = spin * n_orb + s

                        target  = FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{p_spin}^ {q_spin} {s_spin}^ {r_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {s_spin}^ {r_spin}]", coef)
                        term += target
                    
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        self._ham_fermion_upthendown -= normal_ordered(term)

            # print(group_term)
            if len(group_term) != 0:
                group_term_list.append(group_term)
        #print(group_term_list)
        # print("# of all groups:", len(group_term_list))
        # print("remaining terms:", self._ham_fermion_upthendown)

        if validation:
            ## 各グループの項を足すと元のハミルトニアンに戻るか確認
            print("sum of all grouped terms == original Hamiltonian?")
            print(f'diffinision{normal_ordered( sum([sum(group_term) for group_term in group_term_list]) ) + self._ham_fermion_upthendown + self._const_fermion - self.ham_fermion_upthendown_original}')
            print( normal_ordered( sum([sum(group_term) for group_term in group_term_list]) ) + self._ham_fermion_upthendown + self._const_fermion == self.ham_fermion_upthendown_original)

            ## 各グループのエルミート性と交換性を確認
            print("validation of the group:")
            # validiate_group_term_list(group_term_list)
        #remain_term = jordan_wigner(self._ham_fermion_upthendown + self._const_fermion)
        #print(f'remain {remain_term}')
        return group_term_list

    def generate_clique_list(self):
        """
        O(N^2) 個のclique (一つのcliqueはインデックスペアのリスト) を全列挙する関数
        """
        clique_list = []
        n_orb = self._one_body_integrals.shape[0] ## 空間軌道の数. 論文のN.
        n_qubits = 2 * n_orb
        """
        論文の C^part
        """
        clique_list.append([(i,i) for i in range(n_qubits)]) ## C^part= n_i =a_i^dag a_i 
        """
        論文のU^{[1]}
        論文の I_i (なるべく被らないで, 0<= p < q <= N-1 である(p,q) の全通りを尽くす)
        """
        I_indices_list = schedule(n_orb)
        for spin in [0,1]:
            for matches in I_indices_list:
                clique = []
                for p,q in matches:
                    clique.append((spin*n_orb + p, spin*n_orb + q))
                for r in range(n_orb):
                    clique.append(((1-spin)*n_orb + r, (1-spin)*n_orb + r))
                clique_list.append(clique)

        # print("# of cliques for C^part, U^1:", len(clique_list))

        """
        A_{pq,spin}, A_{rs,1-spin} の項をとる. 論文のU^{[2,diff]}.
        """
        for i in range(len(I_indices_list)):
            matches_i = np.array(I_indices_list[i])
            for j in range(len(I_indices_list)):
                matches_j = np.array(I_indices_list[j])
                clique = np.vstack((matches_i, matches_j+n_orb))
                clique_list.append(clique.tolist())
        

        print("# of groups for C^part, U^1, U^{2,diff}:", len(clique_list))

        """
        同一スピン内の A_{pq,spin} * A_{rs,spin} を列挙する. 論文の U^{[2,same]}.
        """
        clique_list_same_spin = np.array(make_clique_by_finite_projective_plane(n_orb))
        for c in clique_list_same_spin:
            clique = np.vstack((c, c+n_orb))
            clique_list.append(clique.tolist())
        
        # print("# of all groups:", len(clique_list))

    def generate_group_mitarai(self, validation, fermion_qubit_mapping):
        """ フェルミオン演算子のグルーピングを行う.
        Return:
            group_term_list:
                list of list of FermionOperator.
                group_term_list[i][j] は i 番目のグループの j 番目のエルミート演算子(論文のA_{pq,spin}などで書かれる).
        """
        n_orb = self._one_body_integrals.shape[0] ## 空間軌道の数. 論文のN.
        n_qubits = 2 * n_orb
        group_term_list = []
        zero_fermion_operator = FermionOperator("", 0.) ## 項がゼロかどうかに判定に使う. 本当は自分で閾値を決めるべき.
        self._ham_fermion_upthendown = copy.deepcopy(self.ham_fermion_upthendown_original)
        self._ham_fermion_upthendown.terms.pop((), 0.0)

        """
        group for number operator (which can be measured by C^part in the paper)
        n_{p,spin} の係数は spinによらず h_{pp}.
        n_{p,spin}*n_{q,spin'} の係数は同一スピンと逆スピンで少し変わるので注意.
        """
        group_term = []
        for spin in [0,1]:
            for p in range(n_orb):
                p_spin = spin * n_orb + p ## スピン軌道の添字
                ## n_{p, spin}
                coef = self._one_body_integrals[p,p]
                term = FermionOperator(f"[{p_spin}^ {p_spin}]", coef)
                if term != zero_fermion_operator:
                    group_term.append(term)
                    # self._ham_fermion_upthendown -= term

                ## n_{p,spin}*n_{p,1-spin}
                p_diff_spin = (1-spin) * n_orb + p
                coef = 0.5 * self._two_body_integrals[p,p,p,p]
                term = FermionOperator(f"[{p_spin}^ {p_spin} {p_diff_spin}^ {p_diff_spin}]", coef)
                if term != zero_fermion_operator:
                    group_term.append(term)
                    # self._ham_fermion_upthendown -= normal_ordered(term)

                for q in range(p+1, n_orb): ## p != q の n_{p,spin}*n_{q,spin'}
                    ## 同じスピン
                    q_same_spin = spin * n_orb + q
                    coef_same_spin = self._two_body_integrals[p,q,q,p] - self._two_body_integrals[p,q,p,q]
                    term = FermionOperator(f"[{p_spin}^ {p_spin} {q_same_spin}^ {q_same_spin}]", coef_same_spin)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        # self._ham_fermion_upthendown -= normal_ordered(term)
                    ## 逆スピン
                    q_diff_spin = (1-spin) * n_orb + q
                    coef_diff_spin = self._two_body_integrals[p,q,q,p]
                    term = FermionOperator(f"[{p_spin}^ {p_spin} {q_diff_spin}^ {q_diff_spin}]", coef_diff_spin)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        # self._ham_fermion_upthendown -= normal_ordered(term)

        if len(group_term) != 0:
            group_term_list.append(group_term) ## 1つ目のグループ

        """
        A_{pq,spin}, A_{pq,spin} * n_{r,1-spin} の項をとる. 論文のU^{[1]}
        論文の I_i (なるべく被らないで, 0<= p < q <= N-1 である(p,q) の全通りを尽くす)
        """
        I_indices_list = schedule(n_orb)

        for spin in [0, 1]:
            for matches in I_indices_list:
                group_term = [] 
                for p,q in matches:
                    ## indices for spin orbitals (up-then-down order)
                    p_spin = spin*n_orb + p
                    q_spin = spin*n_orb + q

                    coef = self._one_body_integrals[p,q]
                    term =  FermionOperator(f"[{p_spin}^ {q_spin}]", coef)
                    term += FermionOperator(f"[{q_spin}^ {p_spin}]", coef)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        # self._ham_fermion_upthendown -= normal_ordered(term)

                    for r in range(n_orb):
                        r_spin = (1-spin)*n_orb + r
                        ## A_{pq,spin} * n_{r, 1-spin}の項.                        
                        coef = self._two_body_integrals[p,r,r,q]
                        term =  FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {r_spin}]", coef)
                        term += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {r_spin}]", coef)
                        if term != zero_fermion_operator:
                            group_term.append(term)
                            # self._ham_fermion_upthendown -= normal_ordered(term)
                            
                if len(group_term) != 0:
                    group_term_list.append(group_term)

        print("# of groups for C^part, U^1:", len(group_term_list))

        """
        A_{pq,spin}, A_{rs,1-spin} の項をとる. 論文のU^{[2,diff]}.
        """
        for i in range(len(I_indices_list)):
            matches_i = I_indices_list[i]
            for j in range(len(I_indices_list)):
                matches_j = I_indices_list[j]
                group_term = []
                for (p,q), (r,s) in itertools.product(matches_i, matches_j):
                    ## I_i, I_j の整数ペアに関する直積をとる.
                    p_up = p
                    q_up = q
                    r_down = n_orb + r
                    s_down = n_orb + s

                    ## A_{pq,spin}, A_{rs,1-spin}
                    coef = self._two_body_integrals[p,r,s,q]
                    term =  FermionOperator(f"[{p_up}^ {q_up} {r_down}^ {s_down}]", coef)
                    term += FermionOperator(f"[{q_up}^ {p_up} {r_down}^ {s_down}]", coef)
                    term += FermionOperator(f"[{p_up}^ {q_up} {s_down}^ {r_down}]", coef)
                    term += FermionOperator(f"[{q_up}^ {p_up} {s_down}^ {r_down}]", coef)
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        # self._ham_fermion_upthendown -= normal_ordered(term)

                if len(group_term) != 0:
                    group_term_list.append(group_term)

        print("# of groups for C^part, U^1, U^{2,diff}:", len(group_term_list))

        """
        同一スピン内の A_{pq,spin} * A_{rs,spin} を列挙する. 論文の U^{[2,same]}.
        論文(5)式にあるように, ハミルトニアンの二電子項は
        1/8 \sum_{pqrs,spin} g_{pqrs,spin} A_{pq,spin} A_{rs,spin}
        とかける. (U^{[2,same]} でカバーすべきなのはspinが同じ項なので以下ではスピンを省略する)
        クリークを使って取り出せるのは, ((p<q),(r<s)) あるいは ((p=q),(r<s)), ((p<q),(r=s))
        という整数の組.  
        ((p<q),(r<s)) に対応するハミルトニアンの項はp,q,r,sが全て異なる項で, 対称性を考慮すると
        1/2 \sum_{p<q,r<s} g_{pqrs} A_{pq} A_{rs}
        とまとまり, さらに (p<q) と (r<s) の入れ替え対称性も考えて, 同時測定groupに取り込む演算子としては
        g_{pqrs} A_{pq} A_{rs}
        を入れる.
        ((p=q),(r<s))の場合に対応するハミルトニアンの項は
        1/4 \sum_{p=q,r<s} g_{pqrs} A_{pq} A_{rs} = 1/4 \sum_{p=q,r<s} g_{pqrs} 2 n_p A_{rs} 
        だけかと思いきや, A_{pr} A_{ps}  という項からも n_p A_{rs} が出てくる:
        A_{pr} A_{ps} = a_r^\dag a_s - n_p * (a_r^dag a_s + a_s^dag a_r).
        これの係数は 1/8 * g_{pprs} だが, (p,r)の入れ替えと(p,s)の入れ替えで factor 4が出てくる一方,
        計算すると n_p = 1/2 * A_{pp} になるので, 結果的には 1/8 * 4 * 1/2 = 1/4 になる.
        さらに, A_{ps} A_{pr} からにも同様に n_p A_{rs} が出てきて, 係数は 1/4 * g_{ppsr} になる.
        """

        ## 4/18 書き換え
        clique_list = np.array(make_clique_by_finite_projective_plane(n_orb))
        weight_list = np.array(self.get_weight_list(clique_list, fermion_qubit_mapping))
        clique_list = clique_list[np.argsort(weight_list)[::-1]]



        checked_pqrs = []
        for clique in clique_list:
            group_term = []
            for i in range(len(clique)):
                p, q = clique[i]
                for j in range(i+1, len(clique)):
                    r, s = clique[j]
                    ## 粒子数演算子だけでかけるものは既に考慮済み
                    if p == q and r == s:
                        continue 
                    ## 既に考慮したかチェック
                    if ((p,q),(r,s)) in checked_pqrs or ((r,s),(p,q)) in checked_pqrs:
                        continue
                    checked_pqrs.append(((p,q),(r,s)))
                    checked_pqrs.append(((r,s),(p,q)))

                    term = FermionOperator("[]", 0.) ## クリーク内の1つのedgeに対応する項.
                    if p != q and r != s: ## g_{pqrs} A_{pq} A_{rs} に対応
                        coef = self._two_body_integrals[p,r,s,q]
                    else: ## ここで少し非自明な寄与がある(上記コメント参照)
                        coef = 1. * (self._two_body_integrals[p,r,s,q]/2. - self._two_body_integrals[p,q,r,s]/4. - self._two_body_integrals[p,q,s,r]/4.)
                    for spin in [0,1]:
                        p_spin = spin * n_orb + p
                        q_spin = spin * n_orb + q
                        r_spin = spin * n_orb + r
                        s_spin = spin * n_orb + s

                        target  = FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{p_spin}^ {q_spin} {s_spin}^ {r_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {s_spin}^ {r_spin}]", coef)
                        term += target
                    
                    if term != zero_fermion_operator:
                        group_term.append(term)
                        # self._ham_fermion_upthendown -= normal_ordered(term)

            # print(group_term)
            if len(group_term) != 0:
                group_term_list.append(group_term)

        # print("# of all groups:", len(group_term_list))
        # print("remaining terms:", self._ham_fermion_upthendown)

        if validation:
            ## 各グループの項を足すと元のハミルトニアンに戻るか確認
            print("sum of all grouped terms == original Hamiltonian?")
            print( normal_ordered( sum([sum(group_term) for group_term in group_term_list]) ) + self._ham_fermion_upthendown + self._const_fermion == self.ham_fermion_upthendown_original)

            ## 各グループのエルミート性と交換性を確認
            print("validation of the group:")
            # validiate_group_term_list(group_term_list)

        return group_term_list
    
    def create_group_list(self, fermion_qubit_mapping):
        """ 他のgrouperと同様に qubit 演算子としてのgroup listを返す.
        そして group のすべてのtermを足しておく.定数項がもしあれば除く.
        """
        group_list = [fermion_qubit_mapping(sum(group_term)) for group_term in self.group_term_list]
        constant_in_all_group = 0.
        for group in group_list:
            constant_in_all_group += group.terms.pop((), 0.0) ## 第二引数はdictのデフォルト値
        self._const = self._const_fermion + constant_in_all_group
        # print(constant_in_all_group)
        return group_list

    def create_group_list_mitarai(self, fermion_qubit_mapping):
        """ 他のgrouperと同様に qubit 演算子としてのgroup listを返す.
        そして group のすべてのtermを足しておく.定数項がもしあれば除く.
        """
        group_term_list = self.generate_group_mitarai(False, fermion_qubit_mapping)
        group_list = [fermion_qubit_mapping(sum(group_term)) for group_term in group_term_list]
        constant_in_all_group = 0.
        for group in group_list:
            constant_in_all_group += group.terms.pop((), 0.0) ## 第二引数はdictのデフォルト値
        self._const = self._const_fermion + constant_in_all_group
        # print(constant_in_all_group)
        return group_list

    

    def get_weight_list(self, clique_list, fermion_qubit_mapping):
        """
        For each clique generated by qhandai, compute \sqrt(\sum_i c_i^2). 
        Sort the clique_list in descending order
        """
        weight_list = []
        zero_fermion_operator = FermionOperator("", 0.) ## 項がゼロかどうかに判定に使う. 本当は自分で閾値を決めるべき.
        n_orb = self._one_body_integrals.shape[0]
        for clique in clique_list:
            group_term = []
            for i in range(len(clique)):
                p, q = clique[i]
                for j in range(i+1, len(clique)):
                    r, s = clique[j]
                    ## 粒子数演算子だけでかけるものは既に考慮済み
                    if p == q and r == s:
                        continue 
                    
                    term = FermionOperator("[]", 0.) ## クリーク内の1つのedgeに対応する項.
                    if p != q and r != s: ## g_{pqrs} A_{pq} A_{rs} に対応
                        coef = self._two_body_integrals[p,r,s,q]
                    else: ## ここで少し非自明な寄与がある(上記コメント参照)
                        coef = 1. * (self._two_body_integrals[p,r,s,q]/2. - self._two_body_integrals[p,q,r,s]/4. - self._two_body_integrals[p,q,s,r]/4.)
                    for spin in [0,1]:
                        p_spin = spin * n_orb + p
                        q_spin = spin * n_orb + q
                        r_spin = spin * n_orb + r
                        s_spin = spin * n_orb + s

                        target  = FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{p_spin}^ {q_spin} {s_spin}^ {r_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {s_spin}^ {r_spin}]", coef)
                        term += target
                    
                    if term != zero_fermion_operator:
                        group_term.append(term)
            # print(clique, group_term)
            if len(group_term) != 0:
                group_total_qubit_hamiltonian = fermion_qubit_mapping(sum(group_term))
                group_total_qubit_hamiltonian.terms.pop((), 0.0)
                w = np.sum( np.abs(list(group_total_qubit_hamiltonian.terms.values()))**2)
                weight_list.append(w)
        return weight_list
    
    def get_detailed_weight_list(self, clique_list, fermion_qubit_mapping):
        """
        For each clique generated by qhandai, compute \sqrt(\sum_i c_i^2). 
        Sort the clique_list in descending order
        """
        weight_list = []
        w_detailed_list = []
        zero_fermion_operator = FermionOperator("", 0.) ## 項がゼロかどうかに判定に使う. 本当は自分で閾値を決めるべき.
        n_orb = self._one_body_integrals.shape[0]
        for clique in clique_list:
            group_term = []
            w_list_inner = []
            for i in range(len(clique)):
                p, q = clique[i]
                for j in range(i+1, len(clique)):
                    r, s = clique[j]
                    ## 粒子数演算子だけでかけるものは既に考慮済み
                    if p == q and r == s:
                        continue 
                    
                    term = FermionOperator("[]", 0.) ## クリーク内の1つのedgeに対応する項.
                    if p != q and r != s: ## g_{pqrs} A_{pq} A_{rs} に対応
                        coef = self._two_body_integrals[p,r,s,q]
                    else: ## ここで少し非自明な寄与がある(上記コメント参照)
                        coef = 1. * (self._two_body_integrals[p,r,s,q]/2. - self._two_body_integrals[p,q,r,s]/4. - self._two_body_integrals[p,q,s,r]/4.)
                    for spin in [0,1]:
                        p_spin = spin * n_orb + p
                        q_spin = spin * n_orb + q
                        r_spin = spin * n_orb + r
                        s_spin = spin * n_orb + s

                        target  = FermionOperator(f"[{p_spin}^ {q_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {r_spin}^ {s_spin}]", coef)
                        target += FermionOperator(f"[{p_spin}^ {q_spin} {s_spin}^ {r_spin}]", coef)
                        target += FermionOperator(f"[{q_spin}^ {p_spin} {s_spin}^ {r_spin}]", coef)
                        term += target
                    
                    if term != zero_fermion_operator:
                        term_qubit = fermion_qubit_mapping(term)
                        # print("inner ham:", term_qubit)
                        w_list_inner.append(np.sum( np.abs(list(term_qubit.terms.values()))**2))
                        group_term.append(term)
            # print(group_term)
            if len(group_term) != 0:
                group_total_qubit_hamiltonian = fermion_qubit_mapping(sum(group_term))
                group_total_qubit_hamiltonian.terms.pop((), 0.0)
                w_detailed_list.append(list(group_total_qubit_hamiltonian.terms.values()))
        return w_detailed_list


if __name__ == '__main__':
    pass




