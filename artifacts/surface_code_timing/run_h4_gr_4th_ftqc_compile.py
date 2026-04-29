import importlib
import time
import traceback

from trotterlib import cost_extrapolation as ce


def main() -> None:
    print(f"[job] start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    start = time.time()
    importlib.reload(ce)
    ce.SURFACE_CODE_COMPILE_MODE = "ftqc_compile"
    try:
        metrics = ce._generate_surface_code_step_metrics(
            "H4_sto-3g_singlet_distance_100_charge_0_grouping",
            "4th(new_2)",
            source="gr",
            target_error=1.5936001019904e-4,
            use_original=False,
        )
        print("[python] done", flush=True)
        for key in (
            "generator",
            "compile_mode",
            "magic_state_consumption_count",
            "magic_state_consumption_depth",
            "qubit_volume",
            "runtime_without_topology",
            "runtime_with_topology",
        ):
            if key in metrics:
                print(f"[python] {key}={metrics[key]}", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        end = time.time()
        print(f"[job] end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        print(f"[job] elapsed_seconds={end - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
