from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

from trotterlib import cost_extrapolation as ce


def test_normalize_surface_code_step_metrics_extracts_required_fields() -> None:
    raw = {
        "magic_state_consumption_count": 123,
        "magic_state_consumption_depth": 45,
        "runtime": 678,
        "runtime_without_topology": 640,
        "qubit_volume": 999,
        "gate_count": 111,
        "source": "compile_info.json",
        "compile_mode": ce.SURFACE_CODE_COMPILE_MODE,
    }

    metrics = ce.normalize_surface_code_step_metrics(raw)

    assert metrics["magic_state_consumption_count"] == 123
    assert metrics["magic_state_consumption_depth"] == 45
    assert metrics["runtime"] == 678
    assert metrics["runtime_without_topology"] == 640
    assert metrics["qubit_volume"] == 999
    assert metrics["gate_count"] == 111
    assert metrics["source"] == "compile_info.json"


def test_attach_df_surface_code_step_metrics_updates_payload(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ce, "PICKLE_DIR_DF_PATH", tmp_path)
    monkeypatch.setattr(ce, "surface_code_step_dir", lambda *args, **kwargs: tmp_path / "surface")

    target = tmp_path / "H3_test_Operator_2nd"
    with target.open("wb") as f:
        pickle.dump({"expo": 2.0, "coeff": 1.0e-3}, f)

    metrics = {
        "magic_state_consumption_count": 10,
        "magic_state_consumption_depth": 7,
        "runtime": 25,
        "runtime_without_topology": 22,
        "qubit_volume": 80,
    }

    stored = ce.attach_df_surface_code_step_metrics("H3_test", "2nd", metrics)

    with (tmp_path / "surface" / "H3_test_Operator_2nd").open("rb") as f:
        payload = pickle.load(f)
    with target.open("rb") as f:
        original_payload = pickle.load(f)

    assert stored["magic_state_consumption_count"] == 10
    assert payload["surface_code_step"]["runtime"] == 25
    assert payload["surface_code_step"]["qubit_volume"] == 80
    assert len(payload["surface_code_step_cache"]) == 1
    assert original_payload["expo"] == 2.0
    assert "surface_code_step" not in original_payload


def test_attach_grouped_surface_code_step_metrics_updates_payload(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ce, "surface_code_step_dir", lambda *args, **kwargs: tmp_path / "surface")
    monkeypatch.setattr(
        ce,
        "_load_grouped_artifact_payload",
        lambda *args, **kwargs: {"coeff": 1.0e-3},
    )

    metrics = {
        "magic_state_consumption_count": 12,
        "magic_state_consumption_depth": 8,
        "runtime": 28,
        "runtime_without_topology": 24,
        "qubit_volume": 90,
    }

    stored = ce.attach_grouped_surface_code_step_metrics("H3_test", "2nd", metrics)

    with (tmp_path / "surface" / "H3_test_Operator_2nd").open("rb") as f:
        payload = pickle.load(f)

    assert stored["magic_state_consumption_count"] == 12
    assert payload["surface_code_step"]["runtime"] == 28
    assert payload["surface_code_step"]["qubit_volume"] == 90
    assert len(payload["surface_code_step_cache"]) == 1


def test_load_surface_code_artifact_payload_falls_back_to_legacy_embedded_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "surface_code_step_dir",
        lambda *args, **kwargs: Path("/nonexistent/surface"),
    )
    monkeypatch.setattr(
        ce,
        "_load_grouped_artifact_payload",
        lambda *args, **kwargs: {
            "coeff": 1.0e-3,
            "surface_code_step": {
                "magic_state_consumption_count": 12,
                "magic_state_consumption_depth": 8,
                "runtime": 28,
                "runtime_without_topology": 24,
                "qubit_volume": 90,
            },
        },
    )

    payload = ce._load_surface_code_artifact_payload("H3_test", "2nd", source="gr")

    assert payload["surface_code_step"]["runtime"] == 28
    assert "coeff" not in payload


def test_load_surface_code_step_metrics_uses_target_error_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_surface_code_artifact_payload",
        lambda *args, **kwargs: {
            "surface_code_step_cache": [
                {
                    "magic_state_consumption_count": 10,
                    "magic_state_consumption_depth": 7,
                    "runtime": 25,
                    "runtime_without_topology": 22,
                    "qubit_volume": 80,
                    "target_error": 1.0e-2,
                    "cache_key": ce._surface_code_cache_key(1.0e-2),
                    "compile_mode": ce.SURFACE_CODE_COMPILE_MODE,
                },
                {
                    "magic_state_consumption_count": 20,
                    "magic_state_consumption_depth": 9,
                    "runtime": 35,
                    "runtime_without_topology": 30,
                    "qubit_volume": 90,
                    "target_error": 1.0e-3,
                    "cache_key": ce._surface_code_cache_key(1.0e-3),
                    "compile_mode": ce.SURFACE_CODE_COMPILE_MODE,
                },
            ]
        },
    )

    metrics = ce._load_surface_code_step_metrics(
        "H3_test",
        "2nd",
        source="df",
        target_error=1.0e-3,
        auto_generate=False,
    )

    assert metrics["magic_state_consumption_count"] == 20
    assert metrics["target_error"] == 1.0e-3


def test_load_surface_code_step_metrics_auto_populates_when_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_surface_code_artifact_payload",
        lambda *args, **kwargs: {"coeff": 1.0e-3},
    )
    monkeypatch.setattr(
        ce,
        "_auto_populate_surface_code_step_metrics",
        lambda *args, **kwargs: {
            "magic_state_consumption_count": 30,
            "magic_state_consumption_depth": 11,
            "runtime": 40,
            "runtime_without_topology": 34,
            "qubit_volume": 95,
            "target_error": 1.0e-3,
        },
    )

    metrics = ce._load_surface_code_step_metrics(
        "H3_test",
        "2nd",
        source="gr",
        target_error=1.0e-3,
        auto_generate=True,
    )

    assert metrics["magic_state_consumption_count"] == 30
    assert metrics["target_error"] == 1.0e-3


def test_load_surface_code_step_metrics_uses_runtime_cache_before_auto_generate(
    tmp_path,
    monkeypatch,
) -> None:
    compile_info_path = tmp_path / "compile_info.json"
    compile_info_path.write_text(
        (
            "{"
            '"magic_state_consumption_count": 77, '
            '"magic_state_consumption_depth": 33, '
            '"runtime": 120, '
            '"runtime_without_topology": 111, '
            '"qubit_volume": 999'
            "}"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ce,
        "_load_surface_code_artifact_payload",
        lambda *args, **kwargs: {"coeff": 1.0e-3},
    )
    monkeypatch.setattr(
        ce,
        "_surface_code_runtime_compile_info_path",
        lambda *args, **kwargs: compile_info_path,
    )
    monkeypatch.setattr(ce, "_surface_code_step_time", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        ce,
        "_surface_code_rotation_precision",
        lambda *args, **kwargs: 1.0e-9,
    )

    attached: list[dict[str, float | int | str | bool]] = []

    def fake_attach(*args, **kwargs):
        metrics = dict(args[2])
        attached.append(metrics)
        return metrics

    monkeypatch.setattr(ce, "_attach_surface_code_step_metrics", fake_attach)

    def fail_auto_populate(*args, **kwargs):
        raise AssertionError("auto_populate should not be called")

    monkeypatch.setattr(
        ce,
        "_auto_populate_surface_code_step_metrics",
        fail_auto_populate,
    )

    metrics = ce._load_surface_code_step_metrics(
        "H3_test",
        "2nd",
        source="gr",
        target_error=1.0e-3,
        auto_generate=True,
    )

    assert metrics["magic_state_consumption_count"] == 77
    assert metrics["target_error"] == 1.0e-3
    assert metrics["step_time"] == 0.5
    assert metrics["generator"] == "runtime_cache"
    assert len(attached) == 1


def test_backfill_grouped_surface_code_step_cache_from_runtime_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_surface_code_step_metrics",
        lambda ham_name, pf_label, **kwargs: {
            "ham_name": ham_name,
            "pf_label": pf_label,
            "target_error": kwargs["target_error"],
            "runtime": 10,
            "runtime_without_topology": 9,
            "magic_state_consumption_count": 8,
            "magic_state_consumption_depth": 7,
            "qubit_volume": 6,
        },
    )

    results = ce.backfill_grouped_surface_code_step_cache_from_runtime_cache(
        3,
        ["2nd", "4th(new_2)"],
        target_error=1.0e-3,
    )

    assert 6 in results
    assert results[6]["2nd"]["ham_name"] == "H3_sto-3g_triplet_1+_distance_100_charge_1_grouping"
    assert results[6]["4th(new_2)"]["pf_label"] == "4th(new_2)"


def test_prepare_surface_code_runtime_env_adds_library_dirs(tmp_path) -> None:
    env = ce._prepare_surface_code_runtime_env(
        tmp_path,
        library_dirs=[Path("/tmp/qcsf-bin")],
    )

    assert env["TMPDIR"] == str(tmp_path / "tmp")
    assert env["MPLCONFIGDIR"] == str(tmp_path / "mplconfig")
    assert env["LD_LIBRARY_PATH"].split(":")[0] == "/tmp/qcsf-bin"


def test_prepare_surface_code_runtime_env_sets_rotation_precision(tmp_path) -> None:
    env = ce._prepare_surface_code_runtime_env(
        tmp_path,
        rotation_precision=2.5e-7,
    )

    assert float(env["QSVT_OPENQASM_ROTATION_PRECISION"]) == 2.5e-7


def test_surface_code_rotation_precision_fixed_mode(monkeypatch) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "fixed")
    monkeypatch.setattr(ce, "SURFACE_CODE_FIXED_ROTATION_PRECISION", 2.5e-7)

    precision = ce._surface_code_rotation_precision(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
    )

    assert precision == 2.5e-7


def test_surface_code_rotation_precision_task_budget_grouped(monkeypatch) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "task_budget")
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION", 1.0e-2)
    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (2.0e-3, 2.0),
    )
    monkeypatch.setattr(
        ce,
        "DECOMPO_NUM",
        {"H3": {"2nd": 50}},
    )

    precision = ce._surface_code_rotation_precision(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
        step_time=0.25,
    )

    qpe = ce._qpe_iteration_factor(2.0e-3, 2.0, 1.0e-3)
    expected = (0.25 * 1.0e-2 * 1.0e-3) / (50 * qpe)
    assert precision == expected


def test_surface_code_rotation_precision_layer_linear_floor_grouped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "layer_linear_floor")
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION", 1.0e-2)
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_FLOOR", 1.0e-9)
    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (2.0e-3, 2.0),
    )
    monkeypatch.setattr(
        ce,
        "PF_RZ_LAYER",
        {"H3": {"2nd": 10}},
    )

    precision = ce._surface_code_rotation_precision(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
        step_time=0.25,
    )

    qpe = ce._qpe_iteration_factor(2.0e-3, 2.0, 1.0e-3)
    raw_expected = (0.25 * 1.0e-2 * 1.0e-3) / (10 * qpe)
    assert raw_expected < 1.0e-9
    assert precision == 1.0e-9


def test_surface_code_rotation_precision_layer_linear_floor_df(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "layer_linear_floor")
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION", 5.0e-2)
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_FLOOR", 1.0e-12)
    monkeypatch.setattr(
        ce,
        "SURFACE_CODE_DF_ROTATION_LAYER_PREFERRED_KEY",
        "total_nonclifford_z_coloring_depth",
    )
    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (1.0e-2, 1.0),
    )
    monkeypatch.setattr(
        ce,
        "_load_df_artifact_payload",
        lambda *args, **kwargs: {
            "rz_layers": {"total_nonclifford_z_coloring_depth": 20}
        },
    )

    precision = ce._surface_code_rotation_precision(
        "H3_dummy",
        "2nd",
        source="df",
        target_error=1.0e-3,
        step_time=0.4,
    )

    qpe = ce._qpe_iteration_factor(1.0e-2, 1.0, 1.0e-3)
    expected = (0.4 * 5.0e-2 * 1.0e-3) / (20 * qpe)
    assert precision == expected


def test_surface_code_proxy_step_metrics_grouped(monkeypatch) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_COMPILE_MODE", "proxy")
    monkeypatch.setattr(ce, "SURFACE_CODE_PROXY_DEPTH_MULTIPLIER", 3.0)
    monkeypatch.setattr(ce, "SURFACE_CODE_PROXY_GATE_COUNT_PER_MAGIC", 5.0)
    monkeypatch.setattr(ce, "SURFACE_CODE_PROXY_RUNTIME_PER_MAGIC", 4.5)
    monkeypatch.setattr(ce, "DECOMPO_NUM", {"H3": {"2nd": 118}})
    monkeypatch.setattr(ce, "PF_RZ_LAYER", {"H3": {"2nd": 39}})

    metrics = ce._surface_code_proxy_step_metrics(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
        step_time=0.25,
        rotation_precision=1.0e-5,
    )

    assert metrics["magic_state_consumption_count"] == 5900
    assert metrics["magic_state_consumption_depth"] == 5850
    assert metrics["runtime_without_topology"] == 26550
    assert metrics["runtime"] == 26550
    assert metrics["gate_count"] == 29500
    assert metrics["generator"] == "proxy_formula"
    assert metrics["compile_mode"] == "proxy"


def test_surface_code_ir_summary_from_opt_json(tmp_path) -> None:
    opt_path = tmp_path / "step_opt.json"
    opt_path.write_text(
        (
            "{"
            '"circuit_list":['
            '{'
            '"name":"main",'
            '"argument":{"num_qubits":2},'
            '"bb_list":[{"inst_list":['
            '{"opcode":"Call","callee":"rot","operate":[0],"input":[],"output":[]},'
            '{"opcode":"CX","q0":0,"q1":1},'
            '{"opcode":"Call","callee":"rot","operate":[1],"input":[],"output":[]},'
            '{"opcode":"Return"}'
            "]}]"
            "},"
            '{'
            '"name":"rot",'
            '"argument":{"num_qubits":1},'
            '"bb_list":[{"inst_list":['
            '{"opcode":"H","q":0},'
            '{"opcode":"T","q":0},'
            '{"opcode":"H","q":0},'
            '{"opcode":"Return"}'
            "]}]"
            "}"
            "]"
            "}"
        ),
        encoding="utf-8",
    )

    summary = ce._surface_code_ir_summary_from_opt_json(opt_path)

    assert summary["magic_count"] == 2
    assert summary["magic_depth"] == 2
    assert summary["gate_count"] == 7
    assert summary["gate_depth"] == 7


def test_surface_code_ir_summary_from_opt_json_ignores_identity(tmp_path) -> None:
    opt_path = tmp_path / "step_opt.json"
    opt_path.write_text(
        (
            "{"
            '"circuit_list":['
            '{'
            '"name":"main",'
            '"argument":{"num_qubits":1},'
            '"bb_list":[{"inst_list":['
            '{"opcode":"I","q":0},'
            '{"opcode":"T","q":0},'
            '{"opcode":"Return"}'
            "]}]"
            "}"
            "]"
            "}"
        ),
        encoding="utf-8",
    )

    summary = ce._surface_code_ir_summary_from_opt_json(opt_path)

    assert summary["magic_count"] == 1
    assert summary["magic_depth"] == 1
    assert summary["gate_count"] == 1
    assert summary["gate_depth"] == 1


def test_generate_surface_code_step_metrics_proxy_skips_qcsf(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_COMPILE_MODE", "proxy")
    monkeypatch.setattr(ce, "SURFACE_CODE_QCSF_PATH", Path("/nonexistent/qcsf"))
    monkeypatch.setattr(ce, "SURFACE_CODE_TOPOLOGY_PATH", Path("/nonexistent/topology.yaml"))
    monkeypatch.setattr(ce, "_surface_code_step_time", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(
        ce,
        "_surface_code_rotation_precision",
        lambda *args, **kwargs: 1.0e-5,
    )
    monkeypatch.setattr(
        ce,
        "_surface_code_proxy_step_metrics",
        lambda *args, **kwargs: {
            "magic_state_consumption_count": 100,
            "magic_state_consumption_depth": 80,
            "runtime": 450,
            "runtime_without_topology": 450,
            "qubit_volume": 0,
            "target_error": 1.0e-3,
            "step_time": 0.25,
            "rotation_precision": 1.0e-5,
            "generator": "proxy_formula",
            "compile_mode": "proxy",
        },
    )

    metrics = ce._generate_surface_code_step_metrics(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
    )

    assert metrics["generator"] == "proxy_formula"
    assert metrics["magic_state_consumption_count"] == 100


def test_generate_surface_code_step_metrics_decompose_only_skips_compile(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_COMPILE_MODE", "decompose_only")
    qcsf = tmp_path / "qcsf"
    qcsf.write_text("", encoding="utf-8")
    topo = tmp_path / "topology.yaml"
    topo.write_text("", encoding="utf-8")
    monkeypatch.setattr(ce, "SURFACE_CODE_QCSF_PATH", qcsf)
    monkeypatch.setattr(ce, "SURFACE_CODE_TOPOLOGY_PATH", topo)
    monkeypatch.setattr(ce, "_surface_code_step_time", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(
        ce,
        "_surface_code_rotation_precision",
        lambda *args, **kwargs: 1.0e-5,
    )
    monkeypatch.setattr(
        ce,
        "_build_grouped_surface_code_step_circuit",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        ce,
        "_surface_code_runtime_root",
        lambda *args, **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        ce,
        "_surface_code_qasm_text_from_circuit",
        lambda *args, **kwargs: "OPENQASM 2.0;",
    )

    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])

    monkeypatch.setattr(ce, "_run_surface_code_command_logged", fake_run)
    monkeypatch.setattr(ce, "_ensure_surface_code_binary_usable", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ce,
        "_surface_code_decompose_only_step_metrics",
        lambda *args, **kwargs: {
            "magic_state_consumption_count": 100,
            "magic_state_consumption_depth": 80,
            "runtime": 123,
            "runtime_without_topology": 123,
            "qubit_volume": 0,
            "target_error": 1.0e-3,
            "step_time": 0.25,
            "rotation_precision": 1.0e-5,
            "generator": "decompose_only_ir",
            "compile_mode": "decompose_only",
        },
    )

    metrics = ce._generate_surface_code_step_metrics(
        "H3_dummy",
        "2nd",
        source="gr",
        target_error=1.0e-3,
    )

    assert calls == ["parse", "opt"]
    assert metrics["generator"] == "decompose_only_ir"
    assert metrics["runtime_without_topology"] == 123
    opt_yaml_text = (tmp_path / "opt.yaml").read_text(encoding="utf-8")
    assert "- ir::decompose_inst" in opt_yaml_text
    assert "- ir::recursive_inliner" not in opt_yaml_text
    assert "- ir::static_condition_pruning" not in opt_yaml_text


def test_surface_code_opt_passes_lightweight_include_compile_shortcuts(monkeypatch) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_COMPILE_MODE", "lightweight")

    passes = ce._surface_code_opt_passes()

    assert passes[:4] == [
        "ir::recursive_inliner",
        "ir::static_condition_pruning",
        "ir::decompose_inst",
        "ir::ignore_global_phase",
    ]


def test_match_surface_code_step_metrics_respects_cache_key(monkeypatch) -> None:
    monkeypatch.setattr(ce, "SURFACE_CODE_COMPILE_MODE", "lightweight")
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "fixed")
    monkeypatch.setattr(ce, "SURFACE_CODE_FIXED_ROTATION_PRECISION", 1.0e-9)

    target_error = 1.0e-3
    metrics = {
        "compile_mode": "lightweight",
        "target_error": target_error,
        "cache_key": ce._surface_code_cache_key(target_error),
    }

    assert ce._match_surface_code_step_metrics(metrics, target_error=target_error)

    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_MODE", "layer_linear_floor")
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_PRECISION_FLOOR", 1.0e-9)
    monkeypatch.setattr(ce, "SURFACE_CODE_ROTATION_ERROR_BUDGET_FRACTION", 1.0e-2)

    assert not ce._match_surface_code_step_metrics(metrics, target_error=target_error)


def test_run_surface_code_command_reports_binary_abi_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    class DummyCompletedProcess:
        returncode = 127
        stdout = ""
        stderr = (
            "/tmp/qcsf: /lib/x86_64-linux-gnu/libc.so.6: "
            "version `GLIBC_2.38' not found"
        )

    def fake_run(*args, **kwargs):
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        ce._run_surface_code_command(
            ["/tmp/qcsf", "--help"],
            runtime_root=tmp_path,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert "not runnable on this machine" in message
    assert "SURFACE_CODE_QCSF_PATH" in message
    assert "GLIBC_2.38" in message


def test_estimate_df_surface_code_task_resources_finds_min_code_distance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (1.0e-3, 2.0),
    )
    monkeypatch.setattr(
        ce,
        "_load_surface_code_step_metrics",
        lambda *args, **kwargs: {
            "magic_state_consumption_count": 50,
            "magic_state_consumption_depth": 20,
            "runtime": 120,
            "runtime_without_topology": 100,
            "qubit_volume": 10,
        },
    )

    result = ce.estimate_df_surface_code_task_resources(
        "H3_dummy",
        "2nd",
        target_error=1.0e-2,
        p_th=1.0e-2,
        a_eff_values=[1.0],
        p_phys_values=[1.0e-3, 1.0e-4],
        delta_fail_values=[1.0e-2],
        code_distances=[3, 5, 7, 9, 11],
    )

    assert result["effective_block_count"] > 0.0
    assert result["totals"]["total_magic_state_count"] > 0.0
    assert result["totals"]["total_runtime"] > 0.0
    assert result["totals"]["total_qubit_volume"] > 0.0

    by_p_phys = {entry["p_phys"]: entry for entry in result["scenarios"]}
    assert by_p_phys[1.0e-4]["d_min"] == 5
    assert by_p_phys[1.0e-4]["meets_target"] is True
    assert by_p_phys[1.0e-3]["d_min"] == 9
    assert by_p_phys[1.0e-3]["meets_target"] is True


def test_estimate_grouped_surface_code_task_resources_finds_min_code_distance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "_load_compare_alpha_and_exponent",
        lambda *args, **kwargs: (1.0e-3, 2.0),
    )
    monkeypatch.setattr(
        ce,
        "_load_surface_code_step_metrics",
        lambda *args, **kwargs: {
            "magic_state_consumption_count": 40,
            "magic_state_consumption_depth": 16,
            "runtime": 90,
            "runtime_without_topology": 80,
            "qubit_volume": 8,
        },
    )

    result = ce.estimate_grouped_surface_code_task_resources(
        "H3_dummy",
        "2nd",
        target_error=1.0e-2,
        p_th=1.0e-2,
        a_eff_values=[1.0],
        p_phys_values=[1.0e-3, 1.0e-4],
        delta_fail_values=[1.0e-2],
        code_distances=[3, 5, 7, 9, 11],
    )

    assert result["source"] == "gr"
    assert result["effective_block_count"] > 0.0
    assert result["totals"]["total_magic_state_count"] > 0.0
    assert result["totals"]["total_runtime"] > 0.0
    assert result["totals"]["total_qubit_volume"] > 0.0

    by_p_phys = {entry["p_phys"]: entry for entry in result["scenarios"]}
    assert by_p_phys[1.0e-4]["d_min"] == 5
    assert by_p_phys[1.0e-4]["meets_target"] is True
    assert by_p_phys[1.0e-3]["d_min"] == 9
    assert by_p_phys[1.0e-3]["meets_target"] is True


def test_surface_code_task_resource_extrapolation_uses_grouped_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "surface_code_task_resource_sweep_grouped",
        lambda *args, **kwargs: {
            6: {
                "2nd": {
                    "ham_name": "H3_dummy",
                    "totals": {"total_magic_state_count": 100.0},
                    "scenarios": [],
                }
            },
            8: {
                "2nd": {
                    "ham_name": "H4_dummy",
                    "totals": {"total_magic_state_count": 200.0},
                    "scenarios": [],
                }
            },
        },
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    series = ce.surface_code_task_resource_extrapolation(
        Hchain=4,
        n_w_list=["2nd"],
        source="gr",
        metric="total_magic_state_count",
        show_bands=False,
    )

    assert series == {"2nd": {"x": [6.0, 8.0], "y": [100.0, 200.0]}}


def test_surface_code_task_resource_extrapolation_supports_d_min_metric(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ce,
        "surface_code_task_resource_sweep_df",
        lambda *args, **kwargs: {
            6: {
                "2nd": {
                    "ham_name": "H3_dummy",
                    "totals": {"total_magic_state_count": 100.0},
                    "scenarios": [
                        {
                            "a_eff": 1.0,
                            "p_phys": 1.0e-3,
                            "delta_fail": 1.0e-2,
                            "d_min": 7,
                        }
                    ],
                }
            },
            8: {
                "2nd": {
                    "ham_name": "H4_dummy",
                    "totals": {"total_magic_state_count": 200.0},
                    "scenarios": [
                        {
                            "a_eff": 1.0,
                            "p_phys": 1.0e-3,
                            "delta_fail": 1.0e-2,
                            "d_min": 9,
                        }
                    ],
                }
            },
        },
    )
    monkeypatch.setattr(ce.plt, "show", lambda: None)

    series = ce.surface_code_task_resource_extrapolation(
        Hchain=4,
        n_w_list=["2nd"],
        source="df",
        metric="d_min",
        a_eff=1.0,
        p_phys=1.0e-3,
        delta_fail=1.0e-2,
        show_bands=False,
    )

    assert series == {"2nd": {"x": [6.0, 8.0], "y": [7.0, 9.0]}}
