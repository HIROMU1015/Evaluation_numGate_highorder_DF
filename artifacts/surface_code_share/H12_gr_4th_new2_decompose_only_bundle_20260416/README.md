# Surface-code Debug Bundle

This directory bundles the information needed to discuss the H12 grouped 4th(new_2) decompose-only surface-code run.

## Target case
- ham_name: `H12_sto-3g_singlet_distance_100_charge_0_grouping`
- pf_label: `4th(new_2)`
- source: `gr`
- compile_mode: `decompose_only`
- generator: `decompose_only_ir_chunked`
- target_error: `0.00015936001019904`
- rotation_precision: `1e-05`

## What happened
- The original monolithic `qcsf parse` failed with `code=-9` while constructing the OpenQASM2 AST.
- The monolithic input `step.qasm` is about `158705075` bytes and `11557055` lines.
- The current code now uses chunked `decompose_only` for large basis circuits.
- The chunked run completed with `47` chunks, `max_ops=250000`, `workers=2`.

## Final chunked metrics
- T-count / magic-state count proxy: `9181176`
- T-depth / magic-state depth proxy: `6781466`
- gate_count: `45711196`
- gate_depth: `23936866`

## Contents
- `metadata/settings.json`: execution settings and resolved paths
- `metadata/step_metrics.json`: stored `surface_code_step` payload
- `metadata/chunk_summary.json`: aggregated chunk summary
- `metadata/chunk_inventory.json`: chunk file inventory
- `metadata/environment.txt`: machine and `qcsf --help` output
- `metadata/observed_failure_excerpt.txt`: notebook error excerpt from the monolithic failure
- `links/runtime_root`: symlink to the full runtime directory
- `links/step_artifact`: symlink to the dedicated `surface_code_step_gr` artifact
- `links/fit_artifact`: symlink to the grouped fit artifact
- `links/step_qasm`: symlink to the original monolithic `step.qasm`
- `links/chunks`: symlink to the chunk directory with per-chunk `step.qasm`, `step_ir.json`, and `step_opt.json`
