# H4 Grouped 4th(new_2) QCSF Share Bundle


対象は、次の比較的小規模な successful case です。

- Hamiltonian: `H4_sto-3g_singlet_distance_100_charge_0_grouping`
- Product formula: `4th(new_2)`
- Source: grouped (`gr`)

## Circuit provenance

この input circuit は、chemistry Hamiltonian そのものを直接渡したものではなく、こちらで事前に構成した one-step Trotter circuit です。

- 対象は `H4_sto-3g_singlet_distance_100_charge_0_grouping` という grouped Hamiltonian です。
- この Hamiltonian に対して、product formula `4th(new_2)` の 1 step 分の circuit を構成しています。
- `payload/parse_and_opt/step.qasm` は、その 1-step circuit を OpenQASM2 に export したものです。
- この bundle には、Hamiltonian の元データや PF circuit を生成する上流コード自体は含めていません。
- 今回ご確認いただきたい対象は、上流の chemistry / circuit synthesis ではなく、`qcsf` に渡した後の `parse` / `opt` / `compile` の処理コストです。

したがって、この bundle のみで「`qcsf` に入力された回路」と「その後の IR / compile 結果」を追える構成にしています。

この bundle には 2 つの processing route を含めています。

1. `parse + opt` route
   OpenQASM2 input を IR に parse し、optimization/decomposition pass を実行した後、こちらで optimized IR から T-count / T-depth を集計する route です。

2. `parse + opt + FTQC compile` route
   optimized IR を入力として `qcsf compile` を FTQC target 付きで実行し、`compile_info.json` と `step.asm` を生成する route です。

こちらの internal label では、次のように呼んでいました。

- `decompose_only` = route 1
- `lightweight` = route 2 with a minimal FTQC compile-info pass set

ただし、これらの label は repo 内部向けのものです。この directory では、processing stage ごとにファイルを整理しています。

## この bundle の共有目的

本 bundle を共有する目的は、`qcsf` を使ったこの一連の処理に時間がかかる理由、特にどの stage が wall-clock time の支配要因になっているのかを確認することです。

この H4 case は small successful case であり、

- `qcsf parse`
- `qcsf opt`
- `qcsf compile`

のすべてが通るため、まず code path と data shape を確認していただくためのケースとして用意しています。

## 調べていただきたいこと

次の点をご確認いただけますと幸いです。

1. この case では、`qcsf parse` / `qcsf opt` / `qcsf compile` のうち、どの stage が本質的に重くなりやすいでしょうか。
2. この input size と IR size に対して、処理時間が長く感じられるのは想定内でしょうか。それとも、何らかの非効率や回避可能な使い方が含まれていそうでしょうか。
3. `compile_info.json` のような summary だけを取得したい場合、現在の `compile.yaml` より軽い設定や pass 構成はあるでしょうか。
4. `OpenQASM2 -> IR -> optimized IR -> FTQC compile` の流れの中で、特に scaling が悪化しやすい opcode, pattern, or pass があればご教示いただきたいです。
5. wall-clock time の原因を切り分けるために、追加で有効な log, profiler, or diagnostic option があればご教示いただきたいです。

特に確認したいのは、取得したい metric に応じて、どこまでの処理が必要になるかという点です。

現時点では、例えば次のように考えています。

- T-count / T-depth の proxy が欲しいだけなら、`parse + opt` までで十分かもしれない
- `compile_info.json` に入る FTQC-specific な summary が欲しいなら、`FTQC compile` が必要かもしれない

ただし、この切り分けが妥当かどうかについては確信がないため、次の点についてご意見を伺えれば幸いです。

- どの metric を得たい場合に `FTQC compile` が本当に必要になるのか
- どの metric であれば `parse + opt` までで十分なのか
- `FTQC compile` が必要な場合、summary だけを得るために、より軽い pass 構成や設定があるのか

## Directory layout

- `payload/parse_and_opt/step.qasm`
  この one-step Trotter circuit に対する OpenQASM2 input です。
- `payload/parse_and_opt/opt.yaml`
  この bundle のみで `qcsf opt` を rerun するための portable config です。
- `payload/parse_and_opt/step_ir.json`
  parse 後の IR output です。
- `payload/parse_and_opt/step_opt.json`
  optimization / decomposition 後の IR output です。
- `payload/ftqc_compile/compile.yaml`
  この bundle のみで `qcsf compile` を rerun するための portable config です。
- `payload/ftqc_compile/tutorial_topology.yaml`
  `compile.yaml` から参照される topology file です。
- `payload/ftqc_compile/compile_info.json`
  `qcsf compile` が出力した FTQC compile summary です。
- `payload/ftqc_compile/step.asm`
  `qcsf compile` が出力した FTQC-target assembly です。
- `metadata/bundle_overview.json`
  この case と key metrics の短い summary です。
- `metadata/reproduction_commands.txt`
  この bundle 内のファイルのみを使う example command です。
- `metadata/file_sizes.txt`
  file size の summary です。
- `metadata/original_opt.yaml`
  runtime directory 内で出力された、repo path を含む元の `opt.yaml` です。
- `metadata/original_compile.yaml`
  runtime directory 内で出力された、repo path を含む元の `compile.yaml` です。

## Key observations

- この H4 case では `parse + opt` route は正常に完了しました。
- この H4 case では `FTQC compile` route も正常に完了しました。
- ただし、H4 case であっても実行時間は比較的長くなっています。
- そのため、この small successful case を用いて、どの処理が重さの主因になっているのかをご確認いただきたいです。

## Selected metrics

- Target error: `1.5936001019904e-04`
- Rotation precision used in our run: `1.0e-05`
- `parse + opt` T-count proxy: `91300`
- `parse + opt` T-depth proxy: `71034`
- `FTQC compile` magic-state consumption count: `91300`
- `FTQC compile` magic-state consumption depth: `71034`
- `FTQC compile` gate_count: `490534`
- `FTQC compile` runtime_without_topology: `419747`
