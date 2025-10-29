import numpy as np
import math
import itertools
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import expm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import multiprocessing as mp


def generate_c_list(W_full):
    """
    W_full = [w0, w1, ..., wm]
    対称な高次積公式の標準的な展開に対応する係数列 c_j を返す。
    c_j は X, Y, X, Y, ..., X と交互に対応させる想定。
    """
    m = len(W_full) - 1
    c = []

    # 前半
    c.append(W_full[m] / 2)
    c.append(W_full[m])
    for i in range(m, 0, -1):
        c.append((W_full[i] + W_full[i - 1]) / 2)
        c.append(W_full[i - 1])

    # 後半
    for i in range(0, m, 1):
        c.append((W_full[i] + W_full[i + 1]) / 2)
        c.append(W_full[i + 1])
    c.append((W_full[m]) / 2)

    return c


def return_S(A, n):
    """
    A: 例 [0,0,2,5,...] のようなマルチセット
    n: 長さ
    -> 各インデックス出現数を数えた長さ n のベクトルを返す
    """
    S = [0] * n
    for a in A:
        S[a] += 1
    return S


def r_quick_gene(k, n):
    """
    sum(r_i) <= k となる n次元の非負整数ベクトル r を全列挙。
    (弱合成を全列挙)
    """
    S_list = []
    for i in range(k + 1):
        for p in combinations_with_replacement(list(range(n)), i):
            S_list.append(return_S(list(p), n))
    return S_list


def Trotter_calculate_sum(k, t, W_full):
    """
    積公式 S(W_full) の展開を、次数 k まで
    全ての語(X/Yの並び)の係数をまとめたベクトルとして返す。
    """
    c = generate_c_list(W_full)
    vec = np.zeros(2 ** (k + 1) - 1, dtype=float)

    for r in r_quick_gene(k, len(c)):  # r: 各 c_i の使用回数
        # 係数 (t^|r| / Π factorial(r_i)) * Π c_i^{r_i}
        term0 = ((t ** sum(r)) /
                 math.prod([math.factorial(r_i) for r_i in r])) * \
                math.prod([c[i] ** r[i] for i in range(len(c))])

        # 語(ワード)を bitstring にエンコード
        # X -> "0", Y -> "1" で、"0b1..." の形にして int(...,2)-1 をインデックスに
        s = "0b1"
        for i in range(0, len(c) - 2, 2):
            for _ in range(r[i]):
                s += "0"  # X
            for _ in range(r[i + 1]):
                s += "1"  # Y
        # 最後は X
        for _ in range(r[len(c) - 1]):
            s += "0"

        vec_index = int(s, 2) - 1
        vec[vec_index] += term0

    return vec


def Origin_calculate_sum(k, t):
    """
    e^{t(X+Y)} の展開を、同様に次数 k まで並べたベクトルを返す。
    """
    vec = np.zeros(2 ** (k + 1) - 1, dtype=float)
    vec[0] = 1.0
    for p in range(1, k + 1):
        term0 = (t ** p) / math.factorial(p)
        # r は長さ p の 0/1 列。1ならX,0ならYとして扱う
        for r in itertools.product(range(2), repeat=p):
            s = "0b1"
            for bit in r:
                if bit == 1:
                    s += "0"  # X
                else:
                    s += "1"  # Y
            vec_index = int(s, 0) - 1
            vec[vec_index] += term0
    return vec


def residual_for_least_squares(w_free, k):
    """
    w_free = [w1, w2, ..., wm]
    w0 は拘束 w0 = 1 - 2 * sum(w_free)
    目的: Origin_calculate_sum(k,1) と PF の展開を一致させる。
    """
    w0 = 1.0 - 2.0 * np.sum(w_free)
    W_full = [w0] + list(w_free)
    return Origin_calculate_sum(k, 1.0) - Trotter_calculate_sum(k, 1.0, W_full)


def solve_one_candidate(m, k, rng):
    """
    1回の最適化:
      入力:
        m: w1..wm の本数
        k: t^0..t^k を一致させたい次数
      出力:
        {
          "w_free": np.array([w1,...,wm]),  # w0以外
          "residual_vec": result.fun,
          "residual_total": sum(|residual|)
        }
    """
    w_init = rng.normal(0.0, 0.8, size=m)
    result = least_squares(
        residual_for_least_squares,
        w_init,
        args=(k,),
        method='lm',
        xtol=2.5e-16,
        ftol=2.5e-16,
        gtol=2.5e-16
    )

    w_free_opt = result.x
    residual_vec = result.fun
    residual_total = float(np.sum(np.abs(residual_vec)))
    return {
        "w_free": w_free_opt,
        "residual_vec": residual_vec,
        "residual_total": residual_total,
    }


def product_formula_unitary(A, B, t_eval, W_full):
    """
    U_pf(t) = Π_j exp(-i c_j t H_j)
    H_j は A, B, A, B, ..., A （偶数→A, 奇数→B）
    """
    c_list = generate_c_list(W_full)
    U = np.eye(A.shape[0], dtype=complex)
    for idx, coeff in enumerate(c_list):
        H_term = A if (idx % 2 == 0) else B
        U = U @ expm(-1j * coeff * t_eval * H_term)
    return U


def spectral_norm_sparse(M_dense):
    """
    ||M||_2 = 最大特異値。
    scipy.sparse.linalg.svds を使って計算する。
    """
    M_sparse = sp.csr_matrix(M_dense)
    svals = spla.svds(M_sparse, k=1, return_singular_vectors=False)
    return float(np.abs(svals[-1]))


def random_hermitian(dim, rng):
    """
    dim x dim のランダムエルミティック行列 H を作り、
    スペクトルノルムで1に正規化して返す。
    """
    M = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (M + M.conj().T) / 2.0
    nrm = spectral_norm_sparse(H)
    if nrm == 0.0:
        return random_hermitian(dim, rng)
    return H / nrm


def estimate_chi_for_wfree(
    w_free,
    k,
    num_trials=32,
    dim=4,
    t_eval=1e-3,
    seed=None
):
    """
    w_free = [w1,...,wm] だけを受け取る。
    w0 = 1 - 2*sum(w_free) を復元し、W_full を作る。

    χ ≈ || e^{-i t (A+B)} - U_pf(t) ||_2 / t^(k+1)
    をランダムな (A,B) で num_trials 回サンプル。
    幾何平均などを返す。
    """
    w0 = 1.0 - 2.0 * np.sum(w_free)
    W_full = [w0] + list(w_free)

    rng = np.random.default_rng(seed)
    p = k + 1  # leading exponent

    chi_samples = []
    for _ in range(num_trials):
        A = random_hermitian(dim, rng)
        B = random_hermitian(dim, rng)
        H = A + B

        U_exact = expm(-1j * t_eval * H)
        U_pf = product_formula_unitary(A, B, t_eval, W_full)

        Delta = U_exact - U_pf
        delta_norm = spectral_norm_sparse(Delta)
        chi_val = delta_norm / (t_eval ** p)
        chi_samples.append(chi_val)

    chi_samples = np.array(chi_samples, dtype=float)
    chi_geo  = float(np.exp(np.mean(np.log(chi_samples))))
    chi_mean = float(np.mean(chi_samples))
    chi_std  = float(np.std(chi_samples, ddof=1))

    return {
        "chi_geo": chi_geo,
        "chi_mean": chi_mean,
        "chi_std": chi_std,
        "all_samples": chi_samples,
    }


def _worker(stop_event: mp.Event,
            result_queue: mp.Queue,
            threshold: float,
            m: int,
            k: int,
            seed: int):
    """
    stop_event が set されるまでループ:
      - solve_one_candidate(m,k,rng)
      - 残差ノルム sum(|residual|) < threshold なら
        (residual_total, w_free) を result_queue に送る
    """
    rng = np.random.default_rng(seed)
    while not stop_event.is_set():
        cand = solve_one_candidate(m, k, rng)
        total = cand["residual_total"]
        if total < threshold:
            # w_free のみ返す（w0は返さない）
            result_queue.put((total, cand["w_free"]))


def parallel_search_and_select_best(
    m: int,
    k: int,
    n_targets: int = 100,
    threshold: float = 1e-12,
    chi_trials: int = 32,
    dim: int = 4,
    t_eval: float = 1e-3,
    base_seed: int = 1234
):
    """
    フロー:
      1. 並列で w_free 候補を集める（residual_total < threshold）
         -> n_targets 個集まるまで result_queue.get()
      2. stop_event を立ててワーカー停止
      3. 親プロセス側で各 w_free について χ を推定
      4. χ_geo が最小の w_free を返す

    戻り値:
      {
        "best_w_free": np.array([...]),   # w0以外
        "best_chi_geo": float,
        "best_chi_info": {...},
        "all_candidates": [
            {
              "w_free": ...,
              "residual_total": ...,
              "chi_info": {...}
            }, ...
        ]
      }
    """
    n_procs = mp.cpu_count()
    stop_event = mp.Event()
    result_queue = mp.Queue()

    # ワーカープロセス起動
    procs = []
    for i in range(n_procs):
        p = mp.Process(
            target=_worker,
            args=(
                stop_event,
                result_queue,
                threshold,
                m,
                k,
                base_seed + i
            )
        )
        p.start()
        procs.append(p)

    # 候補を収集
    collected = []
    while len(collected) < n_targets:
        total, w_free = result_queue.get()
        collected.append({
            "w_free": np.array(w_free),
            "residual_total": float(total)
        })

    # 停止シグナル
    stop_event.set()
    for p in procs:
        p.join()

    # 各候補に対して χ を計算
    for cand in collected:
        chi_info = estimate_chi_for_wfree(
            cand["w_free"],
            k,
            num_trials=chi_trials,
            dim=dim,
            t_eval=t_eval,
            seed=base_seed  # 評価側の乱数は固定でもOK
        )
        cand["chi_info"] = chi_info

    # χ_geo が最小の候補を選ぶ
    chi_list = [c["chi_info"]["chi_geo"] for c in collected]
    best_idx = int(np.argmin(chi_list))
    best_cand = collected[best_idx]

    return {
        "best_w_free": best_cand["w_free"],
        "best_chi_geo": best_cand["chi_info"]["chi_geo"],
        "best_chi_info": best_cand["chi_info"],
        "all_candidates": collected,
    }


def build_W_full_from_wfree(w_free):
    """
    w_free = [w1, w2, ..., wm] から
    w0 = 1 - 2*sum(w_free) を復元して
    W_full = [w0, w1, ..., wm] を返す。
    """
    w_free = np.asarray(w_free, dtype=float)
    w0 = 1.0 - 2.0 * np.sum(w_free)
    W_full = np.concatenate(([w0], w_free))
    return W_full


def estimate_chi_for_Wfull(
    W_full,
    k,
    num_trials=32,
    dim=4,
    t_eval=1e-3,
    seed=None
):
    """
    直接 W_full = [w0, w1, ..., wm] を受け取り、
    χ を推定するバージョン。
    これは estimate_chi_for_wfree とほぼ同じだが、
    既に W_full を手で与えたい場合用。

    戻り値は { "chi_geo", "chi_mean", "chi_std", "all_samples" }
    """
    rng = np.random.default_rng(seed)
    p = k + 1  # leading exponent

    chi_samples = []
    for _ in range(num_trials):
        A = random_hermitian(dim, rng)
        B = random_hermitian(dim, rng)
        H = A + B

        U_exact = expm(-1j * t_eval * H)
        U_pf    = product_formula_unitary(A, B, t_eval, W_full)

        Delta = U_exact - U_pf
        delta_norm = spectral_norm_sparse(Delta)
        chi_val = delta_norm / (t_eval ** p)
        chi_samples.append(chi_val)

    chi_samples = np.array(chi_samples, dtype=float)
    chi_geo  = float(np.exp(np.mean(np.log(chi_samples))))
    chi_mean = float(np.mean(chi_samples))
    chi_std  = float(np.std(chi_samples, ddof=1))

    return {
        "chi_geo": chi_geo,
        "chi_mean": chi_mean,
        "chi_std": chi_std,
        "all_samples": chi_samples,
    }


def compare_known_formulas(
    known_formulas,
    k,
    num_trials=32,
    dim=4,
    t_eval=1e-3,
    seed=999
):
    """
    既に構築済みの積公式たちについて χ を横並び比較する。

    known_formulas: list[dict]
        例:
        [
          {
            "name": "Yoshida-4th",
            "w_free": [w1, w2, w3],   # w0は入れない
          },
          {
            "name": "MyHandTuned",
            "W_full": [w0, w1, w2, w3, w4],  # w0含めて直接指定する版
          },
          ...
        ]

    返り値:
        results: list[dict]
        各要素は
        {
          "name": ...,
          "w_free": np.ndarray([...]) or None,
          "W_full": np.ndarray([...]),
          "chi_info": {
              "chi_geo": ...,
              "chi_mean": ...,
              "chi_std": ...,
              "all_samples": np.array([...])
          }
        }
    あわせて、どれが最小の chi_geo かを簡単に見たいときは
    呼び出し側で min() してください。
    """
    rng = np.random.default_rng(seed)
    results = []

    for item in known_formulas:
        name = item.get("name", "unnamed")

        # 1. W_full を用意
        if "W_full" in item:
            W_full = np.asarray(item["W_full"], dtype=float)
            w_free = None
        elif "w_free" in item:
            w_free = np.asarray(item["w_free"], dtype=float)
            W_full = build_W_full_from_wfree(w_free)
        else:
            raise ValueError(
                f"Formula '{name}' must have 'w_free' or 'W_full'."
            )

        # 2. χ を評価
        #    乱数シードは formula ごとにずらしてもいいし、固定でもいい。
        #    ここでは formula ごとに rng から引くことで変える。
        chi_seed = int(rng.integers(0, 2**32 - 1))
        chi_info = estimate_chi_for_Wfull(
            W_full,
            k,
            num_trials=num_trials,
            dim=dim,
            t_eval=t_eval,
            seed=chi_seed
        )

        # 3. 結果格納
        results.append({
            "name": name,
            "w_free": None if w_free is None else w_free.copy(),
            "W_full": W_full.copy(),
            "chi_info": chi_info,
        })

    return results
