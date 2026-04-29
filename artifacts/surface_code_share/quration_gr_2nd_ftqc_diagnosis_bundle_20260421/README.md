# quration FTQC Compile 調査用共有資料

この共有資料は、公開されている `quration` を使って `qret parse -> qret opt -> qret compile` を実行したとき、どの処理に時間がかかっているかを確認していただくためのものです。

共有相手がこちらのプロジェクトを全く知らない前提で、必要な背景も簡単に書いています。

## 背景

こちらのプロジェクトでは、量子化学由来の Hamiltonian から Trotter 法の 1-step 回路を作り、その回路をフォールトトレラント量子計算向けに見積もることを目的にしています。

今回ご確認いただきたいのは、Hamiltonian 生成そのものではなく、**すでに OpenQASM2 として書き出した 1-step 回路を `quration` に入力した後** の処理です。

つまり、この共有資料の主眼は次の部分です。

- `qret parse`
- `qret opt`
- `qret compile`

上流の化学計算コードや product-formula 回路生成コードを知らなくても、この共有資料に含まれている `step.qasm` を起点に同じ処理を追えるようにしています。

## 今回の回路

今回は、分かりやすさを優先して、product formula として **通常の 2 次 (`2nd`)** を使っています。

含めている対象は 2 つです。

- `H3`
- `H4`

どちらも grouped Hamiltonian から作った **1-step Trotter 回路** です。`H3` は小さめの成功例、`H4` は少し大きくして処理時間の伸びを見やすくした例です。

## 使用した quration build

- リポジトリ: `https://github.com/quration/quration`
- commit: `293912c`
- `qret --help` が示す version: `1.0.2`

この環境では Ubuntu 22.04 / GCC 11 の都合で、ローカル build に 2 つだけ patch を入れています。

- `quration-core/src/qret/target/sc_ls_fixed_v0/state.h`
  `#include <span>` を追加
- `quration-core/src/qret/target/sc_ls_fixed_v0/topology.h`
  defaulted `operator<=>` から `constexpr` を外す

差分そのものは [metadata/quration_local_patch.diff](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/quration_local_patch.diff) に入れています。

## この共有資料から見ていただきたい点

特に確認していただきたいのは次の点です。

1. `qret parse`, `qret opt`, `qret compile` のうち、実際に時間の支配要因になっているのはどこか。
2. `qret opt` の中では、どの pass が特に重いか。
3. 今回の IR size と instruction count の増え方は想定内か、それとも避けられる展開が起きていそうか。
4. もし欲しいものが `compile_info.json` のような要約情報だけなら、今より軽い pipeline や設定があるか。
5. 根本原因を調べるために、追加で有効な profiler / debug option / trace option があるか。

## 先に分かっていること

今回の 2 つの対象は、どちらも `qret compile` まで成功しています。

また、実測結果からは次が分かっています。

- ボトルネックは `qret compile` ではなく、ほぼ `qret opt`
- `qret opt` の中では `ir::decompose_inst` が最も重い
- 次に重いのは `ir::recursive_inliner`

実測した実行時間は次の通りです。

- `H3`
  総時間: `70.1 s`
  `recursive_inliner`: `0.4 s`
  `decompose_inst`: `68.8 s`
  `qret compile`: `0.4 s`
- `H4`
  総時間: `331.1 s`
  `recursive_inliner`: `13.0 s`
  `decompose_inst`: `314.5 s`
  `qret compile`: `1.5 s`

## IR の増え方

instruction count の変化は次の通りです。

- `H3`
  `3631 -> 6095 -> 37493`
  対応:
  `step_ir.json -> step_opt_pass_00.json -> step_opt_pass_02.json`
- `H4`
  `16497 -> 28129 -> 138541`
  対応:
  `step_ir.json -> step_opt_pass_00.json -> step_opt_pass_02.json`

つまり、今回の対象では

- `recursive_inliner` である程度増える
- `decompose_inst` でさらに大きく増える

という振る舞いになっています。

詳しい数値は次を見ていただければと思います。

- [metadata/timing_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/timing_summary.json)
- [metadata/ir_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/ir_summary.json)
- [metadata/observed_behavior.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/observed_behavior.txt)

## timing の取り方

`qret opt` の pass 別 timing は、1 回の実行の内部 profiler から取ったものではありません。

こちらでは、次の pass を 1 つずつ個別に実行して実行時間を測っています。

- `ir::recursive_inliner`
- `ir::static_condition_pruning`
- `ir::decompose_inst`
- `ir::ignore_global_phase`
- `ir::delete_consecutive_same_pauli`
- `ir::delete_opt_hint`

そのため、今回の値は、この pass 順序で個別実行したとき、どこに時間がかかるか、を見るための内訳です。

## 共有資料に含めたもの

各対象 (`payload/H3`, `payload/H4`) には次を入れています。

- `step.qasm`
  `qret parse` に入力した OpenQASM2
- `step_ir.json`
  parse 後の IR
- `step_opt_pass_00.json`
  `ir::recursive_inliner` 後の IR
- `step_opt_pass_02.json`
  `ir::decompose_inst` 後の IR
- `step_opt.json`
  opt 完了後の IR
- `compile_info.json`
  compile の要約情報
- `opt_pass_timings.json`
  pass 別 timing
- `timing.log`
  実際の実行ログ
- `opt.yaml`, `opt_pass_*.yaml`, `compile.yaml`
  この共有資料だけで再実行できる YAML

共通ファイルとして [payload/common/tutorial_topology.yaml](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/payload/common/tutorial_topology.yaml) も入れています。

## あえて含めていないもの

`step_sc_ls_fixed_v0.json` はかなり大きく、共有効率が悪いのでこの共有資料には含めていません。

ただし、再生成に必要な

- `step_opt.json`
- `compile.yaml`
- `tutorial_topology.yaml`

は含めてあります。

## 再現方法

実行例は [metadata/reproduction_commands.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/reproduction_commands.txt) にまとめています。

最低限であれば以下を確認していただければと思います。

- `qret parse --input payload/H3/step.qasm ...`
- `qret opt --pipeline payload/H3/opt.yaml ...`
- `qret compile --pipeline payload/H3/compile.yaml ...`

pass ごとに見たい場合は `opt_pass_00.yaml` から `opt_pass_05.yaml` を順番に実行してください。

## 環境情報

実行環境は [metadata/environment.txt](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_2nd_ftqc_diagnosis_bundle_20260421/metadata/environment.txt) に入れています。

主な点だけ書くと、

- OS: Ubuntu 22.04
- compiler: GCC/G++ 11.4
- CPU: Intel Core i9-13900

です。

## 補足

この共有資料は、**私のプロジェクト全体を知らなくても、`qret` に入った後の重さを調べられる** ことを想定して作っています。

そのため、Hamiltonian 生成 code や Trotter 回路の実装は含めていません。そのため、 `step.qasm` を起点にした `qret` 側の挙動を確認していただければ幸いです。
