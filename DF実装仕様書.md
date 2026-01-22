# 仕様書（既存関数流用版）：DF + 任意積公式の時間発展（Qiskit）

## 0. 方針（最重要）
- **ハミルトニアン構築・回路実装で既存の関数が使える部分は必ず流用する。**
- 新規実装は「DF分解」「DFブロックの (U, D) 生成」「DFブロックを既存Trotter枠へ挿す」など、**差分に限定**する。
- 既存関数のI/O（引数・戻り値の形）を変えない。必要なら **薄いラッパー関数**を追加して整合させる。
- OpenFermion/Cirq回路の変換は避け、**分解データからQiskitで再構築**する（変換の不確実性回避）。

---

## 1. 目的（ゴール）
PySCFから得た電子積分から構築したハミルトニアンについて、OpenFermionで2体項を低ランク分解（DF）し、得られた分解結果を用いて **Qiskitで時間発展回路**  
\[
U(t)\approx e^{iHt}
\]
を任意の積公式（Trotter/Suzuki/カスタム係数列）で構成する。

回路で得た
\[
|\psi(t)\rangle = U(t)|\psi_0\rangle
\]
から、既存のエネルギー推定手法（あなたがすでに持つ摂動論的推定など）で基底エネルギー推定値 \(\hat E\) を求め、基準値 \(E_{\mathrm{ref}}\)（FCI等）との差 \(|\hat E - E_{\mathrm{ref}}|\) を評価できるようにする。

---

## 2. 前提・スコープ
- 量子ビット数は **~30 qubits 程度**を想定（Step2の対角化は軽い）。
- statevectorシミュレーションが可能な範囲（2^Nが重い場合は将来別対応）。
- フェルミオン→量子ビット変換は **Jordan-Wigner (JW)** を前提。
- DFは OpenFermion の low-rank 分解（`low_rank_two_body_decomposition`）を基本ルートとする。
- 既存の「時間発展回路作成・状態取得・エネルギー推定」関数があれば、それを優先して利用する。

---

## 3. 既存資産（前提：すでにあるもの）
### 3.1 ハミルトニアン構築（既存を入口にする）
既存関数（例）：
- `make_fci_vector_from_pyscf_solver_grouper(mol_type)`
  - PySCFで `one_body_integrals`, `two_body_integrals`, `constant` を作っている
  - `grouping_jw_list`（JW変換後の grouped Hamiltonian）を返している
  - `n_qubits`, `energy (FCI)`, `vector (psi0)` を返している

**方針**
- この関数は改造しない／最小改造にする。
- DF用に積分が必要なので、既存関数の返り値を壊さずに、追加情報を返す **互換ラッパー**を新設する。

---

## 4. 成果物（アウトプット）
1. Pythonモジュール（例：`df_trotter/`）として以下のAPIを提供
   - `build_df_trotter_circuit(...) -> QuantumCircuit`
   - `simulate_statevector(qc, psi0) -> np.ndarray`（既存があれば流用）
   - `estimate_energy(...) -> float`（既存推定関数を流用）
   - `report_cost(qc, *, basis_gates=None) -> dict`（count_ops / transpile後count）

2. 最低限のユニットテスト
   - 小系（H2等）で、DFブロックが小次元の行列指数と一致（許容誤差）
   - Uの向き（U vs U†）が仕様として固定される

3. サンプルスクリプト
   - PySCF → DF分解 → 回路生成 → statevector → エネルギー推定 → 誤差出力

---

## 5. 全体設計（差分だけ追加）

### 5.1 DFで扱う数学形（実装の基準）
- 2体項のDF（low-rank）分解で
  \[
  H_2 \approx \sum_{\ell=1}^L \lambda_\ell (N_\ell)^2 + (\text{one-body correction}) + (\text{constant correction})
  \]
  \[
  N_\ell = \sum_{pq} (G_\ell)_{pq} a_p^\dagger a_q
  \]
- 各 \(G_\ell\) を対角化して
  \[
  G_\ell = U_\ell \operatorname{diag}(\eta_{\ell,k}) U_\ell^\dagger
  \]
  すると
  \[
  N_\ell = \sum_k \eta_{\ell,k} n_k
  \]
- よって
  \[
  e^{i\tau\lambda_\ell N_\ell^2} = U_\ell \, D_\ell(\tau)\, U_\ell^\dagger
  \]
  ここで \(D_\ell(\tau)\) は **RZ（単体）＋ZZ位相（ペア）＋定数位相**のみで実装可能：
  \[
  N_\ell^2 = \sum_k \eta_k^2 n_k + 2\sum_{k<j}\eta_k\eta_j\, n_k n_j
  \]

### 5.2 1体項もガウシアン化する場合（オプション）
- 1体行列 \(h\) を対角化して
  \[
  e^{i\tau H_1} = U_0\, D_0(\tau)\, U_0^\dagger
  \]
- DFで返る one-body correction がある場合は
  \[
  h_{\text{eff}} = h + h_{\text{corr}}
  \]
  を用いる。

---

## 6. 新規実装（差分）一覧

### 6.1 互換ラッパー（既存関数は壊さない）
#### 新関数（提案）
- `make_integrals_and_fci(mol_type) -> dict`
  - 既存 `make_fci_vector_from_pyscf_solver_grouper` の処理を **分割して再利用**する（コピーは避ける）
  - 返り値に以下を含める：
    - `grouping_jw_list`, `n_qubits`, `E_fci`, `psi0_vector`（既存と同等）
    - `constant`, `one_body_integrals`, `two_body_integrals`（DF用に追加）

> 既存コードへの影響を最小化すること。既存関数の戻り値は変えない。

---

### 6.2 DF分解（OpenFermion）
#### 新関数
- `df_decompose_from_integrals(one_body_integrals, two_body_integrals, constant, *, rank=None, tol=None) -> DFModel`

#### 入力
- `one_body_integrals`：空間軌道（MO基底）行列
- `two_body_integrals`：空間軌道2体テンソル（順序に注意）
- `constant`：核反発などの定数

#### 出力（dataclass推奨：`DFModel`）
- `lambdas: np.ndarray shape (L,)`
- `G_list: list[np.ndarray]`（各 `(N,N)`）
- `one_body_correction: np.ndarray (N,N)`（あれば）
- `constant_correction: float`（あれば）
- `N: int`（スピン軌道数=量子ビット数）

#### 実装要点
- 空間軌道 → **スピン軌道化**（N=2*n_orb）を実装する
- two-body テンソルの **chemist/physicist順**と係数（1/2）を明示的に合わせる
- 小系での一致テストを必須にする（テンソル順の地雷回避）

---

### 6.3 Step2：対角化（U, eta）
#### 新関数
- `diag_hermitian(mat: np.ndarray) -> (U: np.ndarray, evals: np.ndarray)`
  - `evals, U = np.linalg.eigh(mat)`（Uは列ベクトル）
  - ソート順を固定（必要なら降順/絶対値順など）

---

### 6.4 Step3：UをQiskitゲート列へ（Uのゲート構成が見えるルート）
#### 優先ルート：Qiskit Nature を使う
- `U_to_qiskit_ops_jw(U: np.ndarray, qr: QuantumRegister) -> list[tuple[Gate, tuple[Qubit,...]]]`
  - `fermionic_gaussian_decomposition_jw` を使い、戻り値のゲート列をそのまま採用
  - ゲートは主に `XXPlusYYGate` になる（コスト見積もりしやすい）

#### 注意（必須テスト）
- 「Uを与えたら回路がUを実装するのか、U†を実装するのか」を **ユニットテストで仕様として固定**する

---

### 6.5 Step3：Dの実装（簡単：RZ/RZZ）
#### 新関数
- `apply_D_one_body(qc, eps: np.ndarray, tau: float) -> phase_shift`
  - `exp(i tau Σ eps_k n_k)` を RZ列で実装
  - 定数位相（global phase）を返すか `qc.global_phase` に加える

- `apply_D_squared(qc, eta: np.ndarray, lam: float, tau: float) -> phase_shift`
  - `exp(i tau lam (Σ eta_k n_k)^2)` を RZ + RZZ で実装
  - RZZが使えない場合は `cx-rz-cx` で代替
  - 定数位相は返す or global_phaseへ

---

### 6.6 DFブロックを回路化
#### 新関数
- `apply_df_block(qc, U_ops, eta, lam, tau)`
  - `U_ops` を順に適用
  - `apply_D_squared(...)`
  - `U_ops` を逆順で dagger 適用（`gate.inverse()`）

---

## 7. 既存“積公式ループ”への統合（最小diff）
### 7.1 Block抽象（薄いラッパー）
既存が「Pauli和のリストを回す」形なら、DFブロックは別種なので、
- 既存：`apply_pauli_block(qc, qubit_op, tau)`（既存関数を流用）
- 新規：`apply_df_block(qc, df_block, tau)`
を同じ形式で呼べるようにする。

#### 仕様（提案）
- `Block` を dataclass にして以下を持つ：
  - `kind: str`（"pauli" / "df" / "one_body_gaussian"）
  - `payload`（pauliならQubitOperator、dfなら(ops,eta,lam)など）
  - `apply(qc, tau)`（中で既存関数を呼ぶだけ）

既存の積公式ループは最小改造で：
```python
for (block, coeff) in sequence:
    block.apply(qc, coeff*dt)
