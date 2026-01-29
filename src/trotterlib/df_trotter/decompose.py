from __future__ import annotations

from typing import Any, Tuple

import inspect
import numpy as np

from openfermion.chem.molecular_data import spinorb_from_spatial

from .model import DFModel


def diag_hermitian(
    mat: np.ndarray, *, sort: str = "descending_abs", assume_hermitian: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize a Hermitian matrix and return (U, evals)."""
    mat = np.asarray(mat)
    herm = mat if assume_hermitian else 0.5 * (mat + mat.conj().T)
    evals, U = np.linalg.eigh(herm)
    if sort == "descending_abs":
        order = np.argsort(np.abs(evals))[::-1]
        evals = evals[order]
        U = U[:, order]
    elif sort == "descending":
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        U = U[:, order]
    elif sort == "ascending":
        order = np.argsort(evals)
        evals = evals[order]
        U = U[:, order]
    return U, evals


def _import_low_rank_decomposition() -> Any:
    try:
        from openfermion import low_rank_two_body_decomposition
    except ImportError:
        try:
            from openfermion.circuits import low_rank_two_body_decomposition
        except ImportError:
            try:
                from openfermion.linalg import low_rank_two_body_decomposition
            except ImportError:
                try:
                    from openfermion.transforms import low_rank_two_body_decomposition
                except ImportError as exc:
                    raise ImportError(
                        "low_rank_two_body_decomposition is not available in openfermion."
                    ) from exc
    return low_rank_two_body_decomposition


def _call_low_rank_decomposition(
    two_body_integrals: np.ndarray,
    *,
    rank: int | None,
    tol: float | None,
) -> Any:
    low_rank_two_body_decomposition = _import_low_rank_decomposition()
    params = inspect.signature(low_rank_two_body_decomposition).parameters
    kwargs: dict[str, Any] = {}
    if rank is not None:
        if "final_rank" in params:
            kwargs["final_rank"] = rank
        elif "max_rank" in params:
            kwargs["max_rank"] = rank
        elif "rank" in params:
            kwargs["rank"] = rank
    if tol is not None:
        if "truncation_threshold" in params:
            kwargs["truncation_threshold"] = tol
        elif "tol" in params:
            kwargs["tol"] = tol
    if "spin_basis" in params:
        kwargs["spin_basis"] = True
    elif "spin_orbital" in params:
        kwargs["spin_orbital"] = True
    return low_rank_two_body_decomposition(two_body_integrals, **kwargs)


def _is_matrix_list(obj: Any) -> bool:
    if isinstance(obj, list):
        if not obj:
            return False
        arr = np.asarray(obj[0])
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1]
    arr = np.asarray(obj)
    return arr.ndim == 3


def _normalize_g_list(raw: Any, n: int) -> list[np.ndarray]:
    if isinstance(raw, list):
        mats = [np.asarray(m) for m in raw]
    else:
        arr = np.asarray(raw)
        if arr.ndim == 2:
            mats = [arr]
        elif arr.ndim == 3:
            if arr.shape[1] == n and arr.shape[2] == n:
                mats = [arr[i] for i in range(arr.shape[0])]
            elif arr.shape[0] == n and arr.shape[1] == n:
                mats = [arr[:, :, i] for i in range(arr.shape[2])]
            else:
                raise ValueError("Unsupported G matrix stack shape.")
        else:
            raise ValueError("Unsupported G matrix container.")
    for mat in mats:
        if mat.shape != (n, n):
            raise ValueError("G matrix has incompatible shape.")
    return mats


def _parse_low_rank_result(result: Any, n: int) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
    lambdas = None
    g_list = None
    one_body_correction = None
    constant_correction = None

    for name in ("lambdas", "eigenvalues", "eigvals"):
        if hasattr(result, name):
            lambdas = np.asarray(getattr(result, name))
            break
    for name in ("one_body_squares", "one_body_squared", "one_body_square"):
        if hasattr(result, name):
            g_list = getattr(result, name)
            break
    if hasattr(result, "one_body_correction"):
        one_body_correction = np.asarray(result.one_body_correction)
    if hasattr(result, "constant_correction"):
        constant_correction = np.asarray(result.constant_correction)

    if isinstance(result, (tuple, list)):
        items = list(result)
        if lambdas is None:
            for item in items:
                arr = np.asarray(item)
                if arr.ndim == 1 and arr.size > 0:
                    lambdas = arr
                    items.remove(item)
                    break
        if g_list is None:
            for item in items:
                if _is_matrix_list(item):
                    g_list = item
                    items.remove(item)
                    break
        if one_body_correction is None:
            for item in items:
                arr = np.asarray(item)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    one_body_correction = arr
                    items.remove(item)
                    break
        if constant_correction is None and len(items) > 0 and len(result) != 4:
            for item in items:
                arr = np.asarray(item)
                if arr.ndim == 0:
                    constant_correction = arr
                    items.remove(item)
                    break

    if lambdas is None or g_list is None:
        raise ValueError("Unsupported low-rank decomposition return format.")

    lambdas = np.asarray(lambdas).reshape(-1)
    g_list = _normalize_g_list(g_list, n)

    if len(lambdas) != len(g_list):
        raise ValueError("Mismatch between lambdas and G_list length.")

    if one_body_correction is None:
        one_body_correction = np.zeros((n, n), dtype=np.complex128)
    elif one_body_correction.shape != (n, n):
        raise ValueError("one_body_correction has incompatible shape.")
    if constant_correction is None:
        constant_correction = 0.0
    else:
        constant_correction = float(np.real_if_close(constant_correction))

    return lambdas, g_list, one_body_correction, constant_correction


def df_decompose_from_integrals(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    constant: float,
    *,
    rank: int | None = None,
    tol: float | None = None,
) -> DFModel:
    """
    Low-rank DF decomposition from spatial integrals.

    Conventions:
    - two_body_integrals uses chemist notation (p, q, r, s).
    - 0.5 factor is applied before spin-orbital expansion, matching InteractionOperator.
    """
    one_body_integrals = np.asarray(one_body_integrals)
    two_body_integrals = _symmetrize_two_body_spatial(two_body_integrals)
    if one_body_integrals.ndim != 2 or one_body_integrals.shape[0] != one_body_integrals.shape[1]:
        raise ValueError("one_body_integrals must be a square matrix.")
    if two_body_integrals.ndim != 4:
        raise ValueError("two_body_integrals must be a rank-4 tensor.")
    n_orb = one_body_integrals.shape[0]
    if two_body_integrals.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("two_body_integrals has incompatible shape.")
    _ = constant

    h1s, h2s = spinorb_from_spatial(one_body_integrals, two_body_integrals * 0.5)
    n_spin_orb = h1s.shape[0]

    result = _call_low_rank_decomposition(h2s, rank=rank, tol=tol)
    lambdas, g_list, one_body_correction, constant_correction = _parse_low_rank_result(
        result, n_spin_orb
    )

    return DFModel(
        lambdas=lambdas,
        G_list=g_list,
        one_body_correction=one_body_correction,
        constant_correction=constant_correction,
        N=n_spin_orb,
    )


def _symmetrize_two_body_spatial(two_body: np.ndarray) -> np.ndarray:
    t = np.asarray(two_body, dtype=np.complex128)
    parts = [
        t,
        np.transpose(t, (1, 0, 2, 3)),
        np.transpose(t, (0, 1, 3, 2)),
        np.transpose(t, (1, 0, 3, 2)),
        np.transpose(t, (2, 3, 0, 1)),
        np.transpose(t, (3, 2, 0, 1)),
        np.transpose(t, (2, 3, 1, 0)),
        np.transpose(t, (3, 2, 1, 0)),
    ]
    sym = sum(parts) / len(parts)
    return np.real_if_close(sym, tol=1e-8)
