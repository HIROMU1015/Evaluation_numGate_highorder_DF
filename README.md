# Evaluation_numGate_highorder

高次Trotter積公式による時間発展シミュレーションの誤差スケーリングとコスト評価を行う研究用コード。OpenFermion/PySCFで水素鎖のハミルトニアンを生成し、Qiskitで時間発展を実装、誤差を摂動論で計算、log-logフィットと外挿で比較する。積公式構築アルゴリズムの参考文献として `Greatly-improved-higher-order-product-formulae-for-quantum-simulation.pdf` を同梱する。

## できること
- 2nd/4th/8th/10thなどの積公式係数に基づく誤差評価
- グルーピングあり/なしの時間発展誤差プロット
- QPEのβ評価
- 誤差のlog-logフィット結果を保存
- パウリ回転数/RZ・T-depth外挿

## ディレクトリ構成
- `src/trotterlib/`: 実験用ライブラリ本体
- `abe_trotter_project.ipynb`: 解析の一連の流れをまとめたノートブック
- `artifacts/`: 生成物（係数、行列キャッシュ、スケーリング結果）
- `Greatly-improved-higher-order-product-formulae-for-quantum-simulation.pdf`: 参照論文

## セットアップ
Python 3.11 を前提とする。

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` には `-e .` が含まれているため、`trotterlib` がインストールされる。

## 使い方（ノートブック）
`abe_trotter_project.ipynb` を開いて、Error plt → extrapolation の流れで実行する。論文と同様の外挿を再現する場合は `artifacts/trotter_expo_coeff_gr_original` を参照する（例: `exp_extrapolation(..., use_original=True)`）。既存の挙動で実行したい場合は従来通り `artifacts/trotter_expo_coeff_gr` のデータを使う。

## 主要関数（trotterlib.__all__）
`pf_label` は積公式ラベル（例: `"2nd"`, `"4th(new_2)"`, `"8th(Morales)"`, `"10th(Morales)"`）を指定する。

- `jw_hamiltonian_maker(mol_type, distance=None)`: 水素鎖のJWハミルトニアンを構築し、`(jw_hamiltonian, HFエネルギー, ham_name, num_qubits)` を返す。`mol_type` はH鎖の原子数、`distance` は原子間距離（省略時は `config.DEFAULT_DISTANCE`）。
- `trotter_error_plt(t_start, t_end, t_step, molecule_type, pf_label)`: 対角化ベースの時間発展誤差をlog-logでプロットし、`(t_list, error_list)` を返す。`t_start`/`t_end` は時間範囲、`t_step` は刻み幅、`molecule_type` はH鎖原子数、`pf_label` は積公式ラベル。
- `trotter_error_plt_qc(t_start, t_end, t_step, molecule_type, pf_label)`: 非グルーピングのQiskit時間発展で誤差を評価し、対角化結果とも比較プロットする。引数は `trotter_error_plt` と同じ。
- `trotter_error_plt_qc_gr(t_start, t_end, t_step, molecule_type, pf_label, save_fit_params, save_avg_coeff)`: グルーピングありの誤差評価。`save_fit_params=True` でフィット結果（expo/coeff）を保存、`save_avg_coeff=True` で次数固定の平均係数を保存する。
- `exp_extrapolation(Hchain, n_w_list, show_bands=True, band_height=0.06, band_alpha=0.28)`: 保存済みフィット係数からパウリ回転数を外挿し、H鎖サイズに対するスケーリングを比較する。`Hchain` は最大H鎖原子数、`n_w_list` は比較する積公式ラベル一覧。
  - `scaling_data` / `decompo_counts` を渡すと `scripts.df_trotter_energy_plot_perturb.df_trotter_fixed_order_coeff` などの df_trotter スケーリング結果を min_f とパウリ回転数に使えます。
  - `trotterlib.cost_extrapolation.total_pauli_rotations_from_scaling(mol_type, pf_label, ...)` を使えば単一の分子＋PFについて df_trotter の α（または artifacts）をもとに `total_expo` を返せます。
  - `trotterlib.cost_extrapolation.df_trotter_total_rotations(...)` を使えば df_trotter の sweep → α → `total_expo` を一括で実行し、`exp_extrapolation` に渡す scaling_data やそのまま表示する `total_rotations` にくわえ、1 PF ステップ中の実測パウリ回転数（decompo_num）も返せます。`show_plot` を True にすればその場で log-log plot も表示します。
  - `trotterlib.cost_extrapolation.trotter_qc_gr_total_rotations(...)` を使えば `trotter_error_plt_qc_gr` の摂動論誤差から α（avg coeff）と実際のグループ化回路のパウリ回転数を取得し、`total_expo` を直接求められます（必要なら `scaling_data` / `decompo_counts` で上書き）。
- `exp_extrapolation_diff(Hchain, n_w_list=("4th(new_2)", "8th(Morales)"), MIN_POS=1e-18, X_MIN_CALC=4, X_MAX_DISPLAY=100)`: 2つの積公式の外挿曲線と差分を同一図（双Y軸）で表示する。`n_w_list` は2要素推奨。
- `t_depth_extrapolation(Hchain, n_w_list, rz_layer=None, show_bands=True, band_height=0.06, band_alpha=0.28)`: QPE全体のT-depth（またはRZレイヤ数）を外挿する。`rz_layer=True` でRZレイヤ数を表示する。
- `t_depth_extrapolation_diff(Hchain, rz_layer=None, n_w_list=("4th(new_2)", "8th(Morales)"), MIN_POS=1e-18, X_MIN_CALC=4, X_MAX_DISPLAY=100)`: 2つの積公式のT-depth（またはRZレイヤ数）と差分を比較する。
- `efficient_accuracy_range_plt_grouper(Hchain, n_w_list)`: 各H鎖サイズで最適となる誤差範囲を評価し、積公式ごとの有効範囲を可視化する。
- `scripts.df_trotter_energy_plot_perturb.df_trotter_fixed_order_coeff(times, errors, pf_label, ...)`: `α t^p` の形で誤差スケーリングを仮定し、pf_label（p）が決まっているときの α を返す。`exp_extrapolation` の `scaling_data` に渡す値を準備できます。

## 主要な設定
`src/trotterlib/config.py` で出力先やプロセス数を調整できる。
- `ARTIFACTS_DIR`: 生成物の保存先（`MATRIX_DIR`, `CALCULATION_DIR`, `PICKLE_DIR_PATH` の基点）
- `PICKLE_DIR` / `PICKLE_DIR_GROUPED`: 係数保存フォルダ名（`artifacts/` 配下に作成）
- `POOL_PROCESSES`: 並列プロセス数（時間発展の並列評価に使用）
- `DEFAULT_BASIS` / `DEFAULT_DISTANCE`: 分子基底と原子間距離のデフォルト値
- `BETA`: QPE のユニタリ作用回数 M の比例定数（`M = β / ε`）
- `CA`: 化学的精度
- `TARGET_ERROR`: 外挿や T-depth 計算で使う target error（既定は `CA / 10`）
- `P_DIR`: 積公式ラベルと次数の対応（`num_w` 指定に必須）
- `DECOMPO_NUM`: H-chain のハミルトニアンを PF で分解した際の、1ステップあたりのパウリ回転数
- `PF_RZ_LAYER`: H-chain のハミルトニアンを PF で分解した際の、1ステップあたりの RZ レイヤー数（FermionOperator→RZ の見積りは Inoue 論文の変換に準拠）
- `validate_config()`: 設定キーの整合チェック（必要なら呼び出して検証）

作業ディレクトリを変える場合は `TROTTER_PROJECT_ROOT` を設定すると保存先が崩れない。

```bash
export TROTTER_PROJECT_ROOT=/path/to/Evaluation_numGate_highorder
```

## 注意事項
- PySCF/Qiskitの計算は重いため、H鎖のサイズや `POOL_PROCESSES` は環境に合わせて調整する。
- `artifacts/` 配下に大量のキャッシュが生成される。
