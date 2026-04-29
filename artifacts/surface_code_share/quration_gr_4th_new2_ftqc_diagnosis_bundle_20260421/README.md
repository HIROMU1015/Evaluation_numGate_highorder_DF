# quration FTQC Compile 調査用共有資料

この共有資料は、公開されている `quration` を使って grouped `4th(new_2)` の 1-step Trotter 回路に対し `qret parse -> qret opt -> qret compile` を実行したとき、どの処理に時間がかかっているかを確認していただくためのものです。

主眼は、上流の化学計算や product-formula 回路生成そのものではなく、回路を OpenQASM2 として書き出した後の `qret` 側の挙動です。

## 使用した quration build

- リポジトリ: `https://github.com/quration/quration`
- ローカルの build 先: `/home/abe/Project/quration`
- commit: `293912c`
- `qret --help` が示す version: `1.0.2`

このローカル build では Ubuntu 22.04 / GCC 11 環境に合わせて、次の 2 つの小さい patch を入れています。

- `quration-core/src/qret/target/sc_ls_fixed_v0/state.h`
  `#include <span>` を追加
- `quration-core/src/qret/target/sc_ls_fixed_v0/topology.h`
  defaulted `operator<=>` から `constexpr` を外す

差分そのものは [metadata/quration_local_patch.diff](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/quration_local_patch.diff) に入れています。

## 含めている対象

この共有資料には、product formula `4th(new_2)` を使った grouped の対象を 2 つ入れています。

- `H3`
  小さめの成功例
- `H4`
  より大きく、すでにかなり遅くなっている例

各対象には次のファイルを入れています。

- `step.qasm`
- `step_ir.json`
- `step_opt_pass_00.json`
  `ir::recursive_inliner` 後の IR
- `step_opt_pass_02.json`
  `ir::decompose_inst` 後の IR
- `step_opt.json`
- `compile_info.json`
- `opt_pass_timings.json`
- 再実行用の `opt.yaml`, `opt_pass_*.yaml`, `compile.yaml`
- `timing.log`

`step_sc_ls_fixed_v0.json` はかなり大きく、共有効率が悪いため含めていません。必要であれば `step_opt.json` と `compile.yaml` から再生成できます。省略したファイルサイズは [metadata/file_sizes.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/file_sizes.txt) に記録しています。

## 主な観測結果

- `H3` と `H4` はどちらも `qret compile` まで成功しています。
- 実行時間の大半は `qret compile` ではなく `qret opt` に費やされています。
- `qret opt` の中では `ir::decompose_inst` が支配的です。
- 次に重いのは `ir::recursive_inliner` です。

観測した実行時間は次の通りです。

- `H3`
  総時間: `357.2 s`
  `recursive_inliner`: `15.7 s`
  `decompose_inst`: `337.8 s`
  `qret compile`: `1.9 s`
- `H4`
  総時間: `2090.0 s`
  `recursive_inliner`: `510.1 s`
  `decompose_inst`: `1565.1 s`
  `qret compile`: `7.3 s`

## IR の増え方

instruction count の変化は次の通りです。

- `H3`
  `17,931 -> 30,251 -> 166,649`
  対応:
  `step_ir.json -> step_opt_pass_00.json -> step_opt_pass_02.json`
- `H4`
  `82,097 -> 140,257 -> 658,467`
  対応:
  `step_ir.json -> step_opt_pass_00.json -> step_opt_pass_02.json`

今回の対象では、

- `recursive_inliner` である程度増える
- `decompose_inst` でさらに大きく増える

という振る舞いになっています。

詳しい数値は次にまとめています。

- [metadata/timing_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/timing_summary.json)
- [metadata/ir_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/ir_summary.json)
- [metadata/observed_behavior.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/observed_behavior.txt)

## timing の取り方

`qret opt` の pass 別 timing は、1 回の実行の内部 profiler から取ったものではありません。

こちらでは、同じ pass 順序を使って `qret opt` を 1 pass ずつ個別に実行し、実行時間を測っています。

- `ir::recursive_inliner`
- `ir::static_condition_pruning`
- `ir::decompose_inst`
- `ir::ignore_global_phase`
- `ir::delete_consecutive_same_pauli`
- `ir::delete_opt_hint`

そのため、この pass 別 timing は「この順序で個別実行したとき、どこで時間を使っているか」を見るための実用的な内訳です。

## ご確認いただきたい点

特にご意見をいただきたいのは次の点です。

1. `ir::recursive_inliner` と、特に `ir::decompose_inst` の実行時間は、今回の input size と回路形状に対して想定内か。
2. 目的が `compile_info.json` のような FTQC の要約情報の取得であれば、今より軽い `opt` pipeline があるか。
3. `step_ir.json` から `step_opt_pass_02.json` への増え方は自然か、それとも避けられる expansion pattern を示しているか。
4. 根本原因の切り分けに有効な profiler hook, debug flag, tracing mode があるか。
5. 欲しい指標が magic-state 関連の summary に限られるなら、現在の `parse -> opt -> compile` より軽い経路があるか。

## ディレクトリ構成

- [payload/common/tutorial_topology.yaml](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/payload/common/tutorial_topology.yaml)
  `compile.yaml` で使っている topology file
- [payload/H3](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/payload/H3)
  小さめの成功例
- [payload/H4](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/payload/H4)
  より遅い大きめの対象
- [metadata/reproduction_commands.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/reproduction_commands.txt)
  `parse`, `opt`, `compile` の再現コマンド例
- [metadata/environment.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/environment.txt)
  実行環境, compiler, toolchain 情報
- [metadata/quration_repo_status.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/quration_repo_status.txt)
  commit と local modified-file status

## 補足

- この共有資料は、input, 中間 IR, timing log, summary output をまとめて確認できるようにしています。
- 上流の Hamiltonian 生成 code や Trotter 回路構成 code は含めていません。今回の論点は、OpenQASM2 の input が与えられた後の `qret` の挙動だからです。
