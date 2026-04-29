import argparse
import importlib
import time
import traceback

from trotterlib import cost_extrapolation as ce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run surface-code FTQC compile timing for a single step circuit."
    )
    parser.add_argument("--ham-name", required=True)
    parser.add_argument("--pf-label", required=True)
    parser.add_argument("--source", default="gr")
    parser.add_argument("--target-error", type=float, default=1.5936001019904e-4)
    parser.add_argument("--use-original", action="store_true")
    parser.add_argument("--compile-mode", default="ftqc_compile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[job] start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"[job] ham_name={args.ham_name}", flush=True)
    print(f"[job] pf_label={args.pf_label}", flush=True)
    print(f"[job] source={args.source}", flush=True)
    print(f"[job] target_error={args.target_error:.16e}", flush=True)
    print(f"[job] compile_mode={args.compile_mode}", flush=True)
    start = time.time()
    importlib.reload(ce)
    ce.SURFACE_CODE_COMPILE_MODE = str(args.compile_mode)
    try:
        metrics = ce._generate_surface_code_step_metrics(
            args.ham_name,
            args.pf_label,
            source=args.source,
            target_error=float(args.target_error),
            use_original=bool(args.use_original),
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
