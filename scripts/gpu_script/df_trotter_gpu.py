#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import asdict, dataclass
import multiprocessing as mp
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence


_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_PATH.parents[2]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_CUDA_BOOTSTRAP_ENV = "TROTTER_DF_GPU_BOOTSTRAPPED"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplconfig"))


def _collect_cuda_lib_dirs(project_root: Path) -> list[str]:
    cuda_dirs: list[str] = []
    site_packages_list = glob(str(project_root / ".venv" / "lib" / "python*" / "site-packages"))
    rel_paths = [
        os.path.join("nvidia", "nvjitlink", "lib"),
        os.path.join("nvidia", "cusparse", "lib"),
        os.path.join("nvidia", "cusolver", "lib"),
        os.path.join("nvidia", "cublas", "lib"),
        os.path.join("nvidia", "cuda_runtime", "lib"),
        os.path.join("cuquantum", "lib"),
        os.path.join("cutensor", "lib"),
    ]
    for site_packages in site_packages_list:
        for rel_path in rel_paths:
            lib_dir = os.path.join(site_packages, rel_path)
            if os.path.isdir(lib_dir):
                cuda_dirs.append(lib_dir)
    return cuda_dirs


def _prepend_library_path(env: dict[str, str], lib_dirs: Sequence[str]) -> None:
    current = env.get("LD_LIBRARY_PATH", "")
    current_dirs = [path for path in current.split(":") if path]
    merged: list[str] = []
    for path in [*lib_dirs, *current_dirs]:
        if path and path not in merged:
            merged.append(path)
    if merged:
        env["LD_LIBRARY_PATH"] = ":".join(merged)


_cuda_lib_dirs = _collect_cuda_lib_dirs(_PROJECT_ROOT)
if _VENV_PYTHON.exists() and (
    Path(sys.executable).resolve() != _VENV_PYTHON.resolve()
    or os.environ.get(_CUDA_BOOTSTRAP_ENV) != "1"
):
    env = os.environ.copy()
    _prepend_library_path(env, _cuda_lib_dirs)
    env[_CUDA_BOOTSTRAP_ENV] = "1"
    os.execve(
        str(_VENV_PYTHON),
        [str(_VENV_PYTHON), str(_SCRIPT_PATH), *sys.argv[1:]],
        env,
    )

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
from openfermion.chem import MolecularData
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermionpyscf import run_pyscf
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
except Exception as exc:  # pragma: no cover - depends on GPU server env
    AerSimulator = None
    AER_IMPORT_ERROR = exc
else:
    AER_IMPORT_ERROR = None

from scripts.df_trotter_energy_plot import (
    _resolve_plot_time_range,
    df_ground_state_physical_sector,
)
from trotterlib.analysis_utils import loglog_average_coeff, loglog_fit, print_loglog_fit
from trotterlib.chemistry_hamiltonian import geo
from trotterlib.config import (
    ARTIFACTS_DIR,
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    PFLabel,
    ensure_artifact_dirs,
    get_df_rank_fraction_for_molecule,
    normalize_pf_label,
    pf_order,
)
from trotterlib.df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from trotterlib.df_trotter.decompose import df_decompose_from_integrals
from trotterlib.df_trotter.model import Block
from trotterlib.df_trotter.ops import build_df_blocks, build_one_body_gaussian_block


@dataclass(frozen=True)
class DFGPUTrotterSetup:
    blocks: tuple[Block, ...]
    energy_ref: float
    energy_shift: float
    num_qubits: int
    psi0: np.ndarray
    rank: int | None
    rank_fraction: float | None
    tol: float | None
    ground_state_info: dict[str, object]


@dataclass(frozen=True)
class DFGPURunConfig:
    molecule_type: int
    pf_label: str
    t_start: float
    t_end: float
    t_step: float
    rank: int | None
    rank_fraction: float | None
    tol: float | None
    distance: float
    basis: str
    gpu_ids: tuple[str, ...]
    chunk_splits: int
    optimization_level: int


@dataclass(frozen=True)
class DFGPUFitResult:
    exponent: float
    coeff: float
    avg_coeff: float
    r2: float | None


_GPU_WORKER_SETUP: DFGPUTrotterSetup | None = None
_GPU_WORKER_PF_LABEL: str | None = None
_GPU_WORKER_DEBUG = False


def _set_gpu_worker_context(
    setup: DFGPUTrotterSetup,
    pf_label: str,
    debug: bool,
) -> None:
    global _GPU_WORKER_SETUP, _GPU_WORKER_PF_LABEL, _GPU_WORKER_DEBUG
    _GPU_WORKER_SETUP = setup
    _GPU_WORKER_PF_LABEL = pf_label
    _GPU_WORKER_DEBUG = bool(debug)


def _assign_gpu_groups(num_times: int, gpu_ids: Sequence[str]) -> list[tuple[str, ...]]:
    gpus = [str(g) for g in gpu_ids if str(g) != ""]
    if not gpus:
        return [("0",)] * max(0, int(num_times))
    if num_times <= 0:
        return []
    if len(gpus) >= num_times:
        base = len(gpus) // num_times
        rem = len(gpus) % num_times
        out: list[tuple[str, ...]] = []
        idx = 0
        for i in range(num_times):
            size = base + (1 if i < rem else 0)
            out.append(tuple(gpus[idx : idx + size]))
            idx += size
        return out
    return [(gpus[i % len(gpus)],) for i in range(num_times)]


def _resolve_parallel_processes(num_times: int, processes: int | None) -> int:
    if num_times <= 0:
        return 0
    if processes is None:
        return int(num_times)
    return max(1, min(int(processes), int(num_times)))


def _get_pool_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _gpu_time_task(
    args: tuple[float, tuple[str, ...], int, int],
) -> tuple[float, float, float, float, tuple[str, ...]]:
    time, gpu_group, chunk_splits, optimization_level = args
    setup = _GPU_WORKER_SETUP
    pf_label = _GPU_WORKER_PF_LABEL
    if setup is None or pf_label is None:
        raise RuntimeError("GPU worker context is not initialized.")

    t0 = perf_counter()
    qc = build_df_trotter_circuit(
        setup.blocks,
        time=float(time),
        num_qubits=setup.num_qubits,
        pf_label=pf_label,
        energy_shift=setup.energy_shift,
    )
    build_s = perf_counter() - t0

    t0 = perf_counter()
    psi_t = simulate_statevector_gpu(
        qc,
        setup.psi0,
        gpu_ids=gpu_group,
        chunk_splits=int(chunk_splits),
        optimization_level=int(optimization_level),
        debug=_GPU_WORKER_DEBUG,
    )
    simulate_s = perf_counter() - t0

    err = _perturbation_error(float(time), setup.energy_ref, setup.psi0, psi_t)
    return float(time), float(err), float(build_s), float(simulate_s), tuple(gpu_group)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return token.strip("_") or "value"


def _result_stem(
    molecule_type: int,
    pf_label: str,
    *,
    distance: float,
    basis: str,
    rank: int | None,
    rank_fraction: float | None,
) -> str:
    parts = [
        f"H{int(molecule_type)}",
        _safe_token(basis),
        f"d{int(round(float(distance) * 100))}",
        _safe_token(pf_label),
        "df_gpu",
    ]
    if rank is not None:
        parts.append(f"rank{int(rank)}")
    if rank_fraction is not None:
        parts.append(f"rankfrac_{str(rank_fraction).replace('.', 'p')}")
    return "_".join(parts)


def _h_chain_integrals(
    molecule_type: int,
    *,
    distance: float | None,
    basis: str | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    distance_value = DEFAULT_DISTANCE if distance is None else float(distance)
    basis_value = DEFAULT_BASIS if basis is None else str(basis)
    geometry, multiplicity, charge = geo(molecule_type, distance_value)
    description = f"distance_{int(distance_value * 100)}_charge_{charge}"
    molecule = MolecularData(
        geometry,
        basis_value,
        multiplicity,
        charge,
        description=description,
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    one_body, two_body = molecule.get_integrals()
    constant = float(molecule.nuclear_repulsion)
    return constant, one_body, two_body


def _symmetrize_two_body(two_body: np.ndarray) -> np.ndarray:
    t = np.asarray(two_body)
    return 0.25 * (
        t
        + np.transpose(t, (1, 0, 3, 2))
        + np.transpose(t, (2, 3, 0, 1))
        + np.transpose(t, (3, 2, 1, 0))
    )


def _resolve_rank_inputs(
    molecule_type: int,
    one_body: np.ndarray,
    *,
    rank: int | None,
    rank_fraction: float | None,
    tol: float | None,
) -> tuple[int | None, float | None, float | None]:
    if rank is not None and rank_fraction is not None:
        raise ValueError("rank and rank_fraction are mutually exclusive.")
    if rank is None and rank_fraction is None:
        config_fraction = get_df_rank_fraction_for_molecule(int(molecule_type))
        if config_fraction is not None:
            rank_fraction = float(config_fraction)
    if rank_fraction is None:
        return rank, rank_fraction, tol
    if rank_fraction <= 0:
        raise ValueError("rank_fraction must be positive.")
    n_spatial = int(one_body.shape[0])
    full_rank = int(n_spatial**2)
    if rank_fraction >= 1.0:
        return full_rank, float(rank_fraction), 0.0 if tol is None else tol
    resolved_rank = int(round(full_rank * rank_fraction))
    resolved_rank = max(1, min(resolved_rank, full_rank))
    return resolved_rank, float(rank_fraction), tol


def _split_circuit(circuit: QuantumCircuit, num_splits: int) -> list[QuantumCircuit]:
    if num_splits <= 1:
        return [circuit]
    instructions = list(circuit.data)
    total_instr = len(instructions)
    subcircuits: list[QuantumCircuit] = []
    for i in range(num_splits):
        start = (total_instr * i) // num_splits
        end = (total_instr * (i + 1)) // num_splits
        sub = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=f"{circuit.name or 'df'}_part{i}")
        for inst, qargs, cargs in instructions[start:end]:
            sub.append(inst, qargs, cargs)
        subcircuits.append(sub)
    return subcircuits


def _run_statevector_backend(
    qc: QuantumCircuit,
    psi0: np.ndarray,
    *,
    gpu_ids: Sequence[str],
    optimization_level: int,
) -> np.ndarray:
    if AerSimulator is None:
        return np.asarray(simulate_statevector(qc, psi0), dtype=np.complex128)

    visible_devices = [str(g) for g in gpu_ids if str(g) != ""]
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    init_state = Statevector(np.asarray(psi0, dtype=np.complex128).reshape(-1))
    full_qc = QuantumCircuit(qc.num_qubits)
    full_qc.set_statevector(init_state)
    full_qc = full_qc.compose(qc)
    full_qc.save_statevector()

    simulator = AerSimulator(method="statevector", device="GPU")
    tqc = transpile(full_qc, simulator, optimization_level=int(optimization_level))
    result = simulator.run(tqc).result()
    return np.asarray(result.get_statevector(), dtype=np.complex128)


def simulate_statevector_gpu(
    qc: QuantumCircuit,
    psi0: np.ndarray,
    *,
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
) -> np.ndarray:
    if chunk_splits <= 1:
        return _run_statevector_backend(
            qc,
            psi0,
            gpu_ids=gpu_ids,
            optimization_level=optimization_level,
        )

    state = np.asarray(psi0, dtype=np.complex128).reshape(-1)
    subcircuits = _split_circuit(qc, int(chunk_splits))
    for idx, subcircuit in enumerate(subcircuits):
        if debug:
            debug_print(
                f"gpu chunk {idx + 1}/{len(subcircuits)}: "
                f"instructions={len(subcircuit.data)}"
            )
        state = _run_statevector_backend(
            subcircuit,
            state,
            gpu_ids=gpu_ids,
            optimization_level=optimization_level,
        )
    return state


def _perturbation_error(
    time: float,
    energy: float,
    psi0: np.ndarray,
    psi_t: np.ndarray,
) -> float:
    if time == 0.0:
        return 0.0
    phase_factor = np.exp(-1j * energy * time)
    delta_state = psi_t - phase_factor * psi0
    denom = time * np.sin(energy * time)
    if abs(denom) < 1e-12:
        denom = energy * (time**2)
    if denom == 0.0:
        return 0.0
    delta_e = np.vdot(psi0, delta_state).real / denom
    return float(abs(delta_e))


def build_df_gpu_setup(
    *,
    molecule_type: int,
    pf_label: PFLabel,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    load_ground_state: bool = True,
    save_ground_state: bool = True,
    ground_state_solver_tol: float = 1e-10,
    ground_state_solver_maxiter: int | None = None,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
) -> DFGPUTrotterSetup:
    pf_label = normalize_pf_label(pf_label)
    requested_rank = rank
    requested_rank_fraction = rank_fraction
    constant, one_body, two_body = _h_chain_integrals(
        molecule_type,
        distance=distance,
        basis=basis,
    )
    two_body = _symmetrize_two_body(two_body)
    rank, rank_fraction, tol = _resolve_rank_inputs(
        molecule_type,
        one_body,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
    )
    one_body_spin, _ = spinorb_from_spatial(one_body, two_body * 0.5)
    model = df_decompose_from_integrals(
        one_body,
        two_body,
        constant=constant,
        rank=rank,
        tol=tol,
    ).hermitize()
    h_eff = one_body_spin + model.one_body_correction
    blocks: list[Block] = [Block.from_one_body_gaussian(build_one_body_gaussian_block(h_eff))]
    blocks.extend(Block.from_df(block) for block in build_df_blocks(model))

    ground_state_rank = requested_rank
    ground_state_rank_fraction = requested_rank_fraction
    if ground_state_rank is None and ground_state_rank_fraction is None:
        ground_state_rank_fraction = rank_fraction
    if ground_state_rank_fraction is not None:
        ground_state_rank = None

    energy_ref, psi0, ground_state_info = df_ground_state_physical_sector(
        molecule_type=int(molecule_type),
        rank=ground_state_rank,
        rank_fraction=ground_state_rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
        solver="eigsh",
        solver_tol=ground_state_solver_tol,
        solver_maxiter=ground_state_solver_maxiter,
        matrix_free=True,
        compare_with_sparse_eigsh=False,
        load_artifact=load_ground_state,
        save_artifact=save_ground_state,
        debug=debug,
        debug_print=debug_print,
    )
    return DFGPUTrotterSetup(
        blocks=tuple(blocks),
        energy_ref=float(energy_ref),
        energy_shift=float(constant + model.constant_correction),
        num_qubits=int(model.N),
        psi0=np.asarray(psi0, dtype=np.complex128).reshape(-1),
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        ground_state_info=dict(ground_state_info),
    )


def df_trotter_energy_error_curve_sector_gpu(
    t_start: float | None = None,
    t_end: float | None = None,
    t_step: float | None = None,
    *,
    molecule_type: int,
    pf_label: PFLabel,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    load_ground_state: bool = True,
    save_ground_state: bool = True,
    ground_state_solver_tol: float = 1e-10,
    ground_state_solver_maxiter: int | None = None,
    parallel_times: bool = True,
    processes: int | None = None,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
) -> tuple[list[float], list[float], dict[str, object]]:
    pf_label = normalize_pf_label(pf_label)
    t_start, t_end, t_step = _resolve_plot_time_range(
        molecule_type=int(molecule_type),
        pf_label=pf_label,
        t_start=t_start,
        t_end=t_end,
        t_step=t_step,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
        ground_state_solver_tol=ground_state_solver_tol,
        ground_state_solver_maxiter=ground_state_solver_maxiter,
        load_ground_state_artifact=load_ground_state,
        save_ground_state_artifact=save_ground_state,
        debug=debug,
        debug_print=debug_print,
    )
    if t_step <= 0:
        raise ValueError("t_step must be positive.")
    setup = build_df_gpu_setup(
        molecule_type=molecule_type,
        pf_label=pf_label,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance,
        basis=basis,
        load_ground_state=load_ground_state,
        save_ground_state=save_ground_state,
        ground_state_solver_tol=ground_state_solver_tol,
        ground_state_solver_maxiter=ground_state_solver_maxiter,
        debug=debug,
        debug_print=debug_print,
    )
    times = [float(t) for t in np.arange(t_start, t_end, t_step) if t != 0.0]
    gpu_groups = _assign_gpu_groups(len(times), gpu_ids)
    total_start = perf_counter()

    task_args = [
        (float(time), tuple(gpu_groups[idx]), int(chunk_splits), int(optimization_level))
        for idx, time in enumerate(times)
    ]
    raw_results: list[tuple[float, float, float, float, tuple[str, ...]]] = []
    resolved_processes = _resolve_parallel_processes(len(task_args), processes)

    if task_args:
        if parallel_times and resolved_processes > 1:
            ctx = _get_pool_context()
            with ctx.Pool(
                processes=resolved_processes,
                initializer=_set_gpu_worker_context,
                initargs=(setup, str(pf_label), bool(debug)),
            ) as pool:
                raw_results = list(pool.map(_gpu_time_task, task_args, chunksize=1))
        else:
            resolved_processes = 1 if task_args else 0
            for time, gpu_group, task_chunk_splits, task_optimization_level in task_args:
                t0 = perf_counter()
                qc = build_df_trotter_circuit(
                    setup.blocks,
                    time=time,
                    num_qubits=setup.num_qubits,
                    pf_label=pf_label,
                    energy_shift=setup.energy_shift,
                )
                build_s = perf_counter() - t0

                t0 = perf_counter()
                psi_t = simulate_statevector_gpu(
                    qc,
                    setup.psi0,
                    gpu_ids=gpu_group,
                    chunk_splits=task_chunk_splits,
                    optimization_level=task_optimization_level,
                    debug=debug,
                    debug_print=debug_print,
                )
                simulate_s = perf_counter() - t0
                err = _perturbation_error(time, setup.energy_ref, setup.psi0, psi_t)
                raw_results.append((time, err, build_s, simulate_s, tuple(gpu_group)))
                if debug:
                    debug_print(
                        f"t={time:.6e} error={err:.6e} "
                        f"build_qc={build_s:.3f}s simulate={simulate_s:.3f}s gpu={','.join(gpu_group)}"
                    )

    raw_results.sort(key=lambda item: item[0])
    times = [float(time) for time, *_ in raw_results]
    errors = [float(err) for _, err, *_ in raw_results]
    circuit_times = [float(build_s) for _, _, build_s, _, _ in raw_results]
    simulate_times = [float(simulate_s) for _, _, _, simulate_s, _ in raw_results]
    used_gpu_groups = [list(gpu_group) for _, _, _, _, gpu_group in raw_results]

    info: dict[str, object] = {
        "num_qubits": setup.num_qubits,
        "energy_ref": float(setup.energy_ref),
        "energy_shift": float(setup.energy_shift),
        "rank": setup.rank,
        "rank_fraction": setup.rank_fraction,
        "tol": setup.tol,
        "gpu_ids": [str(g) for g in gpu_ids],
        "gpu_groups": used_gpu_groups,
        "chunk_splits": int(chunk_splits),
        "optimization_level": int(optimization_level),
        "parallel_times": bool(parallel_times),
        "processes": int(resolved_processes),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "t_step": float(t_step),
        "num_time_points": int(len(times)),
        "ground_state_info": setup.ground_state_info,
        "mean_build_circuit_s": float(np.mean(circuit_times)) if circuit_times else 0.0,
        "mean_simulate_s": float(np.mean(simulate_times)) if simulate_times else 0.0,
        "total_runtime_s": perf_counter() - total_start,
    }
    return times, errors, info


def _fit_errors(times: Sequence[float], errors: Sequence[float], pf_label: PFLabel) -> DFGPUFitResult:
    fit = loglog_fit(times, errors, mask_nonpositive=True, compute_r2=True)
    avg_coeff = loglog_average_coeff(
        times,
        errors,
        float(pf_order(pf_label)),
        mask_nonpositive=True,
    )
    return DFGPUFitResult(
        exponent=float(fit.slope),
        coeff=float(fit.coeff),
        avg_coeff=float(avg_coeff),
        r2=fit.r2,
    )


def df_trotter_energy_error_plot_sector_gpu(
    t_start: float | None = None,
    t_end: float | None = None,
    t_step: float | None = None,
    *,
    molecule_type: int,
    pf_label: PFLabel,
    rank: int | None = None,
    rank_fraction: float | None = None,
    tol: float | None = None,
    distance: float | None = None,
    basis: str | None = None,
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    load_ground_state: bool = True,
    save_ground_state: bool = True,
    ground_state_solver_tol: float = 1e-10,
    ground_state_solver_maxiter: int | None = None,
    parallel_times: bool = True,
    processes: int | None = None,
    fit: bool = True,
    show_plot: bool = False,
    save_plot: bool = False,
    output_dir: str | os.PathLike[str] | None = None,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
) -> dict[str, object]:
    pf_label = normalize_pf_label(pf_label)
    distance_value = DEFAULT_DISTANCE if distance is None else float(distance)
    basis_value = DEFAULT_BASIS if basis is None else str(basis)
    times, errors, info = df_trotter_energy_error_curve_sector_gpu(
        t_start=t_start,
        t_end=t_end,
        t_step=t_step,
        molecule_type=molecule_type,
        pf_label=pf_label,
        rank=rank,
        rank_fraction=rank_fraction,
        tol=tol,
        distance=distance_value,
        basis=basis_value,
        gpu_ids=gpu_ids,
        chunk_splits=chunk_splits,
        optimization_level=optimization_level,
        load_ground_state=load_ground_state,
        save_ground_state=save_ground_state,
        ground_state_solver_tol=ground_state_solver_tol,
        ground_state_solver_maxiter=ground_state_solver_maxiter,
        parallel_times=parallel_times,
        processes=processes,
        debug=debug,
        debug_print=debug_print,
    )

    result: dict[str, object] = {
        "times": list(times),
        "errors": list(errors),
        "info": info,
    }
    fit_result: DFGPUFitResult | None = None
    if fit:
        fit_result = _fit_errors(times, errors, pf_label)
        result["fit"] = asdict(fit_result)
        print_loglog_fit(
            loglog_fit(times, errors, mask_nonpositive=True, compute_r2=True),
            ave_coeff=fit_result.avg_coeff,
        )

    if output_dir is None:
        ensure_artifact_dirs(include_pickle_dirs=False)
        out_dir = ARTIFACTS_DIR / "df_gpu"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _result_stem(
        molecule_type,
        pf_label,
        distance=distance_value,
        basis=basis_value,
        rank=rank,
        rank_fraction=rank_fraction,
    )
    pickle_path = out_dir / f"{stem}.pkl"
    json_path = out_dir / f"{stem}.json"
    plot_path = out_dir / f"{stem}.png"

    with pickle_path.open("wb") as f:
        pickle.dump(result, f)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if show_plot or save_plot:
        import matplotlib.pyplot as plt
        from trotterlib.plot_utils import set_loglog_axes

        ax = plt.gca()
        set_loglog_axes(
            ax,
            xlabel="time",
            ylabel="energy error [Hartree]",
            title=f"H{int(molecule_type)}_DF_GPU_{pf_label}",
        )
        ax.plot(times, errors, marker="o", linestyle="-", label="error")
        if fit_result is not None:
            fit_curve = [fit_result.coeff * (t ** fit_result.exponent) for t in times]
            ax.plot(
                times,
                fit_curve,
                linestyle="--",
                label=f"fit: α={fit_result.coeff:.2e}, p={fit_result.exponent:.2f}",
            )
        ax.legend()
        if save_plot:
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()

    result["output_pickle"] = str(pickle_path)
    result["output_json"] = str(json_path)
    if save_plot:
        result["output_plot"] = str(plot_path)
    return result


def _parse_gpu_ids(value: str) -> tuple[str, ...]:
    values = tuple(v.strip() for v in str(value).split(",") if v.strip())
    return values or ("0",)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DF Trotter statevector simulation on GPU and fit perturbation errors.",
    )
    parser.add_argument("--molecule-type", type=int, required=True)
    parser.add_argument("--pf-label", type=str, default="8th(Morales)")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--t-step", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--rank-fraction", type=float, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE)
    parser.add_argument("--basis", type=str, default=DEFAULT_BASIS)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--chunk-splits", type=int, default=1)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--disable-parallel-times", action="store_true")
    parser.add_argument("--ground-state-solver-tol", type=float, default=1e-10)
    parser.add_argument("--ground-state-solver-maxiter", type=int, default=None)
    parser.add_argument("--disable-ground-state-cache", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = df_trotter_energy_error_plot_sector_gpu(
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        molecule_type=int(args.molecule_type),
        pf_label=str(args.pf_label),
        rank=args.rank,
        rank_fraction=args.rank_fraction,
        tol=args.tol,
        distance=float(args.distance),
        basis=str(args.basis),
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        chunk_splits=int(args.chunk_splits),
        optimization_level=int(args.optimization_level),
        load_ground_state=not bool(args.disable_ground_state_cache),
        save_ground_state=not bool(args.disable_ground_state_cache),
        ground_state_solver_tol=float(args.ground_state_solver_tol),
        ground_state_solver_maxiter=args.ground_state_solver_maxiter,
        parallel_times=not bool(args.disable_parallel_times),
        processes=args.processes,
        fit=not bool(args.no_fit),
        show_plot=bool(args.show_plot),
        save_plot=bool(args.save_plot),
        output_dir=args.output_dir,
        debug=bool(args.debug),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
