# Surface-code Debug Bundle

This directory bundles a smaller successful grouped decompose-only run for code-path inspection.

## Target case
- ham_name: `H4_sto-3g_singlet_distance_100_charge_0_grouping`
- pf_label: `4th(new_2)`
- source: `gr`
- compile_mode: `decompose_only`
- generator: `decompose_only_ir`
- target_error: `0.00015936001019904`
- rotation_precision: `1e-05`

## Why this case
- Same grouped `4th(new_2)` path as the larger H12 case.
- Much smaller monolithic OpenQASM input, so it is easier to inspect directly.
- This run completed successfully without chunking.

## Input / output size
- `step.qasm`: `1064974` bytes, `82095` lines
- `step_ir.json`: `8516951` bytes
- `step_opt.json`: `8913275` bytes

## Final metrics
- T-count / magic-state count proxy: `91300`
- T-depth / magic-state depth proxy: `71034`
- gate_count: `401906`
- gate_depth: `236096`

## Contents
- `metadata/settings.json`: execution settings and resolved paths
- `metadata/step_metrics.json`: stored `surface_code_step` payload
- `metadata/fit_payload.json`: grouped fit payload (`coeff`, `expo`)
- `metadata/file_sizes.txt`: size summary for the main generated files
- `metadata/environment.txt`: machine and `qcsf --help` output
- `links/runtime_root`: symlink to the full runtime directory
- `links/step_artifact`: symlink to the dedicated `surface_code_step_gr` artifact
- `links/fit_artifact`: symlink to the grouped fit artifact
- `links/step_qasm`: symlink to the monolithic `step.qasm`
- `links/step_ir`: symlink to `step_ir.json`
- `links/step_opt`: symlink to `step_opt.json`
