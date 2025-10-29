import multiprocessing as mp
import numpy as np
import math
import itertools
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def generate_c_list(W_full, dtype=float):
    """
    W_full = [w0, w1, ..., wm]
    対称な高次PFの係数列 c_j を返す。
    c_j は X,Y,X,Y,...,X の順で使う想定。
    dtype には float / np.longdouble などを渡す。
    """
    m = len(W_full) - 1
    # 明示的にdtypeキャスト
    W_full = np.asarray(W_full, dtype=dtype)

    c_list = []
    c_list.append(W_full[m] / dtype(2))
    c_list.append(W_full[m])
    for i in range(m, 0, -1):
        c_list.append((W_full[i] + W_full[i - 1]) / dtype(2))
        c_list.append(W_full[i - 1])

    for i in range(0, m, 1):
        c_list.append((W_full[i] + W_full[i + 1]) / dtype(2))
        c_list.append(W_full[i + 1])
    c_list.append(W_full[m] / dtype(2))

    return c_list


def return_S(A, n):
    """マルチセット A -> カウントベクトル (通常精度でOK。整数なので問題なし)"""
    S = [0] * n
    for a in A:
        S[a] += 1
    return S


def r_quick_gene(k, n):
    """
    sum(r_i) <= k となる n次元の非負整数ベクトル r を全列挙。
    これは組合せ (弱合成) の列挙なので基本整数だけ。
    """
    S_list = []
    for i in range(k + 1):
        for p in combinations_with_replacement(range(n), i):
            S_list.append(return_S(list(p), n))
    return S_list


def Trotter_calculate_sum_float64(k, t, w_free):
    """
    既存の float64 版。最適化中(least_squares)はこれを使う。
    w_free: [w1,...,wm] (float64)
    """
    if isinstance(w_free, np.ndarray):
        w_free = w_free.tolist()
    w0 = [1 - 2 * sum(w_free)]
    W_full = w0 + w_free
    c = generate_c_list(W_full, dtype=float)

    vec = np.zeros(2 ** (k + 1) - 1, dtype=float)

    for r in r_quick_gene(k, len(c)):
        # term0 = (t^|r| / Π factorial(r_i)) * Π c_i^{r_i}
        term0 = (t ** sum(r)) / math.prod(math.factorial(r_i) for r_i in r)
        for i_ci, power in enumerate(r):
            if power != 0:
                term0 *= c[i_ci] ** power

        s = "0b1"
        for i in range(0, len(c) - 2, 2):
            s += "0" * r[i]      # X
            s += "1" * r[i + 1]  # Y
        s += "0" * r[len(c) - 1] # 最後はX

        vec_index = int(s, 2) - 1
        vec[vec_index] += term0

    return vec


def Origin_calculate_sum_float64(k, t):
    """
    e^{t(X+Y)} 側の float64 版
    """
    vec = np.zeros(2 ** (k + 1) - 1, dtype=float)
    vec[0] = 1.0
    for p in range(1, k + 1):
        term0 = (t ** p) / math.factorial(p)
        # r は長さpの0/1列 (bit=1 -> X, 0 -> Y)
        for r in itertools.product(range(2), repeat=p):
            s = "0b1"
            for bit in r:
                s += "0" if bit == 1 else "1"
            vec_index = int(s, 0) - 1
            vec[vec_index] += term0
    return vec


def fun_float64(w_free, k):
    """
    least_squares に渡す目的関数 (float64)
    w_free = [w1,...,wm]
    """
    return Origin_calculate_sum_float64(k, 1.0) - Trotter_calculate_sum_float64(k, 1.0, w_free)


# ====== 高精度（longdouble）評価用 ======
def Trotter_calculate_sum_hp(k, t, w_free):
    """
    high precision版 (np.longdouble)。
    最適化後の評価/判定用だけに使う。
    """
    dtype = np.longdouble

    w_free = np.asarray(w_free, dtype=dtype)
    w0_val = dtype(1) - dtype(2) * np.sum(w_free, dtype=dtype)
    W_full = np.concatenate(([w0_val], w_free)).astype(dtype)

    c = generate_c_list(W_full, dtype=dtype)

    vec = np.zeros(2 ** (k + 1) - 1, dtype=dtype)

    for r in r_quick_gene(k, len(c)):
        # term0 = (t^|r| / Π factorial(r_i)) * Π c_i^{r_i}  (高精度)
        power_sum = sum(r)
        term0 = (dtype(t) ** dtype(power_sum))

        # divide by factorials
        for r_i in r:
            term0 /= dtype(math.factorial(r_i))

        # multiply c_i^{r_i}
        for i_ci, power in enumerate(r):
            if power != 0:
                term0 *= (dtype(c[i_ci]) ** dtype(power))

        # ビット列エンコード
        s = "0b1"
        for i in range(0, len(c) - 2, 2):
            s += "0" * r[i]      # X
            s += "1" * r[i + 1]  # Y
        s += "0" * r[len(c) - 1]

        vec_index = int(s, 2) - 1
        vec[vec_index] += term0

    return vec


def Origin_calculate_sum_hp(k, t):
    """
    e^{t(X+Y)} 側の high precision版 (np.longdouble)
    """
    dtype = np.longdouble
    vec = np.zeros(2 ** (k + 1) - 1, dtype=dtype)
    vec[0] = dtype(1)
    t_ld = dtype(t)

    for p in range(1, k + 1):
        term0 = (t_ld ** dtype(p)) / dtype(math.factorial(p))
        for r in itertools.product(range(2), repeat=p):
            s = "0b1"
            for bit in r:
                s += "0" if bit == 1 else "1"
            vec_index = int(s, 0) - 1
            vec[vec_index] += term0

    return vec


def fun_hp(w_free, k):
    """
    high precision residual:
    Origin_hp - Trotter_hp, dtype=np.longdouble
    """
    return Origin_calculate_sum_hp(k, 1.0) - Trotter_calculate_sum_hp(k, 1.0, w_free)


# ====== 最適化＋高精度残差評価 ======
def optimal_trotter(m, k):
    """
    1. float64 で最適化 (scipy.least_squares)
    2. 解 w_free を high precision(np.longdouble)に入れ直して fun_hp を評価
    3. high precision の残差ベクトル rest_hp を返す
    """
    # ステップ1: 通常倍精度で最適化
    w_init = np.random.normal(0, 0.8, m)  # float64
    result = least_squares(
        fun_float64,
        w_init,
        args=(k,),
        method='lm',
        xtol=2.5e-16,
        ftol=2.5e-16,
        gtol=2.5e-16
    )

    w_free_opt = result.x  # float64, shape (m,)

    # ステップ2: high precision で残差を再計算
    rest_hp = fun_hp(w_free_opt, k)  # np.longdouble vector

    return w_free_opt, rest_hp


def worker(m, k, stop_event: mp.Event, result_queue: mp.Queue, threshold: float):
    """
    stop_event が set されるまでループし、
    high precision の残差ノルム合計が threshold 未満の解をキューに送る
    """
    while not stop_event.is_set():
        w_free_opt, rest_hp = optimal_trotter(m, k)

        # high precision の残差をもとに total を計算
        # rest_hp は np.longdouble 配列
        total_hp = np.sum(np.abs(rest_hp, dtype=np.longdouble))

        # total_hp は np.longdouble なので float() に落とす
        total_val = float(total_hp)

        if total_val < threshold:
            result_queue.put((total_val, w_free_opt))


def parallel_find(m: int,
                  k: int,
                  n_targets: int = 100,
                  threshold: float = 1e-12):
    """
    並列に optimal_trotter を呼び出し、
    high precision 残差合計 total < threshold の解を n_targets 個集める。
    total が最小のものを返す。
    """
    n_procs = mp.cpu_count()
    stop_event = mp.Event()
    result_queue = mp.Queue()

    procs = [
        mp.Process(
            target=worker,
            args=(m, k, stop_event, result_queue, threshold)
        )
        for _ in range(n_procs)
    ]
    for p in procs:
        p.start()

    collected = []
    while len(collected) < n_targets:
        total, w_free_opt = result_queue.get()
        collected.append((total, w_free_opt))

    stop_event.set()
    for p in procs:
        p.join()

    best_total, best_w = min(collected, key=lambda x: x[0])
    return best_w, best_total


def compute_residual_metrics_hp(w_free, k):
    """
    1個の積公式候補 (w_free = [w1,...,wm]) について、
    high precision 残差 rest_hp = fun_hp(w_free, k) を使って
    残差の大きさを評価する。

    返り値:
        {
          "rest_vec": rest_hp (np.longdouble配列),
          "l1":   L1ノルム   = sum |rest_hp|,
          "l2":   L2ノルム   = sqrt(sum |rest_hp|^2),
          "linf": L∞ノルム  = max |rest_hp|
        }

    ※戻り値の l1/l2/linf は float にキャストして返す。
    """
    # あなたの既存コード内の high precision 残差関数を使う
    rest_hp = fun_hp(w_free, k)  # np.longdouble のベクトル

    abs_rest = np.abs(rest_hp, dtype=np.longdouble)

    l1_val   = float(np.sum(abs_rest))
    l2_val   = float(np.sqrt(np.sum(abs_rest ** np.longdouble(2))))
    linf_val = float(np.max(abs_rest))

    return {
        "rest_vec": rest_hp,
        "l1":   l1_val,
        "l2":   l2_val,
        "linf": linf_val,
    }


def compare_residuals(formulas, k, sort_by="l1"):
    """
    複数の積公式を high precision 残差で比較する。
    """
    allowed = {"l1", "l2", "linf"}
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of {allowed}, got {sort_by}")

    results = []
    for item in formulas:
        name = item.get("name", "unnamed")
        w_free = np.asarray(item["w_free"], dtype=float)

        # 既存の high precision 評価ロジックを流用
        metrics = compute_residual_metrics_hp(w_free, k)

        results.append({
            "name": name,
            "w_free": w_free,
            "l1": metrics["l1"],
            "l2": metrics["l2"],
            "linf": metrics["linf"],
            "rest_vec": metrics["rest_vec"],  # np.longdouble
        })

    results_sorted = sorted(results, key=lambda x: x[sort_by])
    return results_sorted
