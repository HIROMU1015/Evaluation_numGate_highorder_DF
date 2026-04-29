# この共有資料について

この共有資料は、`quration` を使って grouped `4th(new_2)` の 1-step Trotter 回路を `qret parse -> qret opt -> qret compile` と処理したとき、**どこに時間がかかっているか** を確認していただくためのものです。

こちらで特に知りたいのは、`qret compile` 自体が重いのか、それともその前段の `qret opt`、特に特定の pass が重いのか、という点です。

## 何を確認してほしいか

次の点をご確認いただきたいです。

1. 実行時間の支配要因はどこか。
2. `qret opt` の中で特に重い pass は妥当か。
3. IR の増え方が自然か、それとも避けられる展開が起きていそうか。
4. `compile_info.json` のような要約情報だけが欲しい場合に、もっと軽い実行経路があるか。
5. 原因調査のために有効な profiler, debug option, trace option があるか。

## この資料に入っているもの

この共有資料には、`H3` と `H4` の 2 つの例について、次のものを入れています。

- `step.qasm`
- `step_ir.json`
- `step_opt_pass_00.json`
- `step_opt_pass_02.json`
- `step_opt.json`
- `compile_info.json`
- `opt_pass_timings.json`
- `timing.log`

つまり、`qret` への入力、途中の IR、実行時間の記録、最終的な要約情報までを一通り含めています。

## 先に分かっていること

今回含めた `4th(new_2)` の 2 例では、どちらも `qret compile` まで成功しています。

現時点では、重いのは `qret compile` 本体ではなく、主に `qret opt` の `ir::decompose_inst` と `ir::recursive_inliner` だと見えています。

## まず見ていただきたいファイル

最初に見るなら、次の 3 つで十分です。

- [README.md](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/README.md)
- [metadata/timing_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/timing_summary.json)
- [metadata/ir_summary.json](/home/abe/Project/Evaluation_numGate_DF/artifacts/surface_code_share/quration_gr_4th_new2_ftqc_diagnosis_bundle_20260421/metadata/ir_summary.json)

詳細な背景や再現方法は `README.md` にまとめています。
