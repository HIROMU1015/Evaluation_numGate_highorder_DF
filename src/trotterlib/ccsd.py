# coverage:ignore
""" Pretty-print a table comparing DF vector thresh vs accuracy and cost """
from typing import Any, Sequence

import numpy as np
from pyscf import scf

from .config import (
    DEFAULT_DF_CCSD_TARGET_ERROR_HA,
    set_df_rank_fraction_for_molecule,
)

try:
    from openfermion.resource_estimates import df as _df_pkg
except ImportError:
    _df_pkg = None

try:
    # Older OpenFermion builds re-export helpers from this module.
    from openfermion.resource_estimates.molecule import (
        factorized_ccsd_t,
        cas_to_pyscf,
        pyscf_to_cas,
    )
except ImportError:
    # Newer builds keep them under molecule.pyscf_utils.
    from openfermion.resource_estimates.molecule.pyscf_utils import (
        factorized_ccsd_t,
        cas_to_pyscf,
        pyscf_to_cas,
    )


def _df_compute_lambda_local(pyscf_mf, df_factors):
    """Version-agnostic lambda evaluator for DF factors."""
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)
    t_mat = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full)
    lambda_t = float(np.sum(np.abs(np.linalg.eigvalsh(t_mat))))

    lambda_f = 0.0
    for idx in range(df_factors.shape[2]):
        eigvals = np.linalg.eigvalsh(df_factors[:, :, idx])
        lambda_f += 0.25 * float(np.sum(np.abs(eigvals)) ** 2)
    return lambda_t + lambda_f


if _df_pkg is not None and hasattr(_df_pkg, "factorize"):
    _df_factorize = _df_pkg.factorize
else:
    from openfermion.resource_estimates.df.factorize_df import factorize as _df_factorize

if _df_pkg is not None and hasattr(_df_pkg, "compute_cost"):
    _df_compute_cost = _df_pkg.compute_cost
else:
    from openfermion.resource_estimates.df.compute_cost_df import (
        compute_cost as _df_compute_cost,
    )

if _df_pkg is not None and hasattr(_df_pkg, "compute_lambda"):
    _df_compute_lambda = _df_pkg.compute_lambda
else:
    try:
        from openfermion.resource_estimates.df.compute_lambda_df import (
            compute_lambda as _df_compute_lambda,
        )
    except Exception:
        _df_compute_lambda = _df_compute_lambda_local


def generate_costing_table(
    pyscf_mf,
    name='molecule',
    thresh_range=None,
    dE=0.001,
    chi=10,
    beta=20,
    use_kernel=True,
    no_triples=False,
):
    """Print a table to file for testing how various DF thresholds impact cost,
        accuracy, etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'double_factorization_<name>.txt'
        thresh_range (list of floats) - list of thresholds to try for DF alg
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients
                    (default: 10)
        beta (int) - number of bits for rotations (default: 20)
        use_kernel (bool) - re-do SCF prior to estimating CCSD(T) error?
            Will use canonical orbitals and full ERIs for the one-body
            contributions, using DF reconstructed ERIs for two-body
        no_triples (bool) - if True, skip the (T) correction, doing only CCSD

    Returns:
       None
    """

    if thresh_range is None:
        thresh_range = [0.0001]

    DE = dE  # max allowable phase error
    CHI = chi  # number of bits for representation of coefficients
    BETA = beta  # number of bits for rotations

    if isinstance(pyscf_mf, scf.rohf.ROHF):
        num_alpha, num_beta = pyscf_mf.nelec
        assert num_alpha + num_beta == pyscf_mf.mol.nelectron
    else:
        assert pyscf_mf.mol.nelectron % 2 == 0
        num_alpha = pyscf_mf.mol.nelectron // 2
        num_beta = num_alpha

    num_orb = len(pyscf_mf.mo_coeff)
    num_spinorb = num_orb * 2

    cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)

    try:
        assert num_orb**4 == len(pyscf_mf._eri.flatten())
    except AssertionError:
        # ERIs are not in correct form in pyscf_mf._eri, so this is a quick prep
        _, pyscf_mf = cas_to_pyscf(*pyscf_to_cas(pyscf_mf))

    # Reference calculation (eri_rr= None is full rank / exact ERIs)
    escf, ecor, etot = factorized_ccsd_t(
        pyscf_mf, eri_rr=None, use_kernel=use_kernel, no_triples=no_triples
    )

    # exact_ecor = ecor
    exact_etot = etot

    filename = 'double_factorization_' + name + '.txt'

    with open(filename, 'w') as f:
        print("\n Double low rank factorization data for '" + name + "'.", file=f)
        print("    [*] using " + cas_info, file=f)
        print("        [+]                      E(SCF): %18.8f" % escf, file=f)
        if no_triples:
            print("        [+]    Active space CCSD E(cor): %18.8f" % ecor, file=f)
            print("        [+]    Active space CCSD E(tot): %18.8f" % etot, file=f)
        else:
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor, file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot, file=f)
        print("{}".format('=' * 139), file=f)
        if no_triples:
            print(
                "{:^12} {:^18} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format(
                    'threshold',
                    '||ERI - DF||',
                    'L',
                    'eigenvectors',
                    'lambda',
                    'CCSD error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        else:
            print(
                "{:^12} {:^18} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format(
                    'threshold',
                    '||ERI - DF||',
                    'L',
                    'eigenvectors',
                    'lambda',
                    'CCSD(T) error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        print("{}".format('-' * 139), file=f)
    for thresh in thresh_range:
        # First, up: lambda and CCSD(T)
        eri_rr, LR, L, Lxi = _df_factorize(pyscf_mf._eri, thresh=thresh)
        lam = _df_compute_lambda(pyscf_mf, LR)
        escf, ecor, etot = factorized_ccsd_t(
            pyscf_mf, eri_rr, use_kernel=use_kernel, no_triples=no_triples
        )
        error = (etot - exact_etot) * 1e3  # to mEh
        l2_norm_error_eri = np.linalg.norm(eri_rr - pyscf_mf._eri)  # ERI reconstruction error

        # now do costing
        stps1 = _df_compute_cost(
            num_spinorb,
            lam,
            DE,
            L=L,
            Lxi=Lxi,
            chi=CHI,
            beta=BETA,
            stps=20000,
        )[
            0
        ]
        _, df_total_cost, df_logical_qubits = _df_compute_cost(
            num_spinorb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=stps1
        )

        with open(filename, 'a') as f:
            print(
                "{:^12.6f} {:^18.4e} {:^12} {:^12} {:^12.1f} {:^24.2f} {:^20} \
                 {:^20.1e}".format(
                    thresh, l2_norm_error_eri, L, Lxi, lam, error, df_logical_qubits, df_total_cost
                ),
                file=f,
            )
    with open(filename, 'a') as f:
        print("{}".format('=' * 139), file=f)


def _prepare_mf_with_full_eri(pyscf_mf):
    num_orb = len(pyscf_mf.mo_coeff)
    required_size = num_orb**4
    eri = getattr(pyscf_mf, "_eri", None)
    has_full_eri = False
    if eri is not None:
        try:
            has_full_eri = int(np.asarray(eri).size) == required_size
        except Exception:
            has_full_eri = False
    if not has_full_eri:
        # ERIs are missing or not in full s1 form in pyscf_mf._eri, so rebuild once.
        _, pyscf_mf = cas_to_pyscf(*pyscf_to_cas(pyscf_mf))
        rebuilt_eri = getattr(pyscf_mf, "_eri", None)
        if rebuilt_eri is None:
            raise RuntimeError("Failed to construct full ERIs for CCSD DF scan.")
        rebuilt_size = int(np.asarray(rebuilt_eri).size)
        if rebuilt_size != required_size:
            raise RuntimeError(
                "Unexpected ERI shape after reconstruction: "
                f"size={rebuilt_size}, expected={required_size}."
            )
    return pyscf_mf


def scan_df_ccsd_error_by_threshold(
    pyscf_mf,
    *,
    thresh_range: Sequence[float] | None = None,
    use_kernel: bool = True,
    no_triples: bool = False,
    scan_order: str = "high_rank_first",
    target_error_ha: float | None = None,
    stop_when_target_met: bool = False,
) -> dict[str, Any]:
    """Evaluate DF-approximated CCSD(T) energy error for a threshold grid.

    Returns:
        dict with exact energy, full rank, scan metadata, and per-threshold records.
    """
    if thresh_range is None:
        thresh_values = np.geomspace(1e-8, 1e-2, num=17)
    else:
        if not thresh_range:
            raise ValueError("thresh_range must not be empty.")
        thresh_values = np.asarray(list(thresh_range), dtype=float)
    if np.any(thresh_values <= 0):
        raise ValueError("All thresholds must be positive.")
    thresh_values = np.unique(np.sort(thresh_values))
    if scan_order == "low_rank_first":
        ordered_thresholds = thresh_values[::-1]
    elif scan_order == "high_rank_first":
        ordered_thresholds = thresh_values
    else:
        raise ValueError("scan_order must be 'high_rank_first' or 'low_rank_first'.")
    if target_error_ha is not None and target_error_ha <= 0:
        raise ValueError("target_error_ha must be positive when provided.")
    target_abs_error = (
        float(target_error_ha) if target_error_ha is not None else None
    )

    pyscf_mf = _prepare_mf_with_full_eri(pyscf_mf)
    num_orb = len(pyscf_mf.mo_coeff)
    full_rank = int(num_orb**2)

    _, _, exact_etot = factorized_ccsd_t(
        pyscf_mf, eri_rr=None, use_kernel=use_kernel, no_triples=no_triples
    )

    records: list[dict[str, Any]] = []
    stopped_early = False
    for thresh in ordered_thresholds:
        eri_rr, _, rank, _ = _df_factorize(pyscf_mf._eri, thresh=float(thresh))
        _, _, etot = factorized_ccsd_t(
            pyscf_mf, eri_rr, use_kernel=use_kernel, no_triples=no_triples
        )
        abs_error = float(abs(etot - exact_etot))
        records.append(
            {
                "threshold": float(thresh),
                "rank": int(rank),
                "rank_fraction": float(int(rank) / full_rank),
                "ccsd_error_ha": float(etot - exact_etot),
                "abs_ccsd_error_ha": abs_error,
            }
        )
        if (
            stop_when_target_met
            and target_abs_error is not None
            and abs_error <= target_abs_error
        ):
            stopped_early = True
            break
    return {
        "full_rank": full_rank,
        "exact_ccsd_energy_ha": float(exact_etot),
        "scan_order": scan_order,
        "thresholds_total": int(len(thresh_values)),
        "thresholds_evaluated": int(len(records)),
        "stopped_early": stopped_early,
        "records": records,
    }


def select_rank_fraction_for_ccsd_error(
    pyscf_mf,
    *,
    target_error_ha: float = 1e-2,
    thresh_range: Sequence[float] | None = None,
    use_kernel: bool = False,
    no_triples: bool = False,
) -> dict[str, Any]:
    """Pick a DF rank fraction that targets a CCSD(T) error budget.

    Selection rule:
      - Scan starts from low-rank side and stops at first point satisfying target.
      - Otherwise choose the point with the smallest |CCSD error|.
    """
    if target_error_ha <= 0:
        raise ValueError("target_error_ha must be positive.")

    target = float(target_error_ha)
    scan = scan_df_ccsd_error_by_threshold(
        pyscf_mf,
        thresh_range=thresh_range,
        use_kernel=use_kernel,
        no_triples=no_triples,
        scan_order="low_rank_first",
        target_error_ha=target,
        stop_when_target_met=True,
    )
    records = list(scan["records"])
    if not records:
        raise RuntimeError("CCSD threshold scan returned no records.")

    feasible = [row for row in records if float(row["abs_ccsd_error_ha"]) <= target]
    if feasible:
        selected = feasible[0]
        target_met = True
    else:
        selected = min(records, key=lambda row: float(row["abs_ccsd_error_ha"]))
        target_met = False

    return {
        "target_error_ha": target,
        "target_met": target_met,
        "selected_threshold": float(selected["threshold"]),
        "selected_rank": int(selected["rank"]),
        "selected_rank_fraction": float(selected["rank_fraction"]),
        "selected_ccsd_error_ha": float(selected["ccsd_error_ha"]),
        "selected_abs_ccsd_error_ha": float(selected["abs_ccsd_error_ha"]),
        "full_rank": int(scan["full_rank"]),
        "exact_ccsd_energy_ha": float(scan["exact_ccsd_energy_ha"]),
        "scan_order": str(scan["scan_order"]),
        "thresholds_total": int(scan["thresholds_total"]),
        "thresholds_evaluated": int(scan["thresholds_evaluated"]),
        "stopped_early": bool(scan["stopped_early"]),
        "records": records,
    }


def select_rank_fraction_for_molecule(
    molecule_type: int,
    ccsd_target_error_ha: float = DEFAULT_DF_CCSD_TARGET_ERROR_HA,
    *,
    thresh_range: Sequence[float] | None = None,
    use_kernel: bool = False,
    no_triples: bool = False,
    record_in_config: bool = False,
) -> dict[str, Any]:
    """Estimate DF rank_fraction for a molecule_type from CCSD error target.

    Args:
        molecule_type: H-chain index used in this project.
        ccsd_target_error_ha: target absolute CCSD error in Hartree.
        thresh_range/use_kernel/no_triples: forwarded to CCSD scan.
        record_in_config: when True, write selected fraction into config map.
    """
    from .qiskit_time_evolution_pyscf import _run_scf_and_integrals

    if ccsd_target_error_ha <= 0:
        raise ValueError("ccsd_target_error_ha must be positive.")

    _, pyscf_mf, _, _, _ = _run_scf_and_integrals(int(molecule_type))
    selection = select_rank_fraction_for_ccsd_error(
        pyscf_mf,
        target_error_ha=float(ccsd_target_error_ha),
        thresh_range=thresh_range,
        use_kernel=use_kernel,
        no_triples=no_triples,
    )
    selection["molecule_type"] = int(molecule_type)
    if record_in_config:
        set_df_rank_fraction_for_molecule(
            int(molecule_type),
            float(selection["selected_rank_fraction"]),
            selected_rank=int(selection["selected_rank"]),
            full_rank=int(selection["full_rank"]),
        )
    return selection


def populate_df_rank_fraction_config(
    molecule_types: Sequence[int],
    ccsd_target_error_ha: float = DEFAULT_DF_CCSD_TARGET_ERROR_HA,
    *,
    thresh_range: Sequence[float] | None = None,
    use_kernel: bool = False,
    no_triples: bool = False,
) -> dict[int, dict[str, float | int | str]]:
    """Populate config map for multiple molecule_types and return rank metadata."""
    rank_map: dict[int, dict[str, float | int | str]] = {}
    for molecule_type in molecule_types:
        selection = select_rank_fraction_for_molecule(
            int(molecule_type),
            ccsd_target_error_ha=ccsd_target_error_ha,
            thresh_range=thresh_range,
            use_kernel=use_kernel,
            no_triples=no_triples,
            record_in_config=True,
        )
        selected_rank = int(selection["selected_rank"])
        full_rank = int(selection["full_rank"])
        fraction = float(selection["selected_rank_fraction"])
        rank_map[int(molecule_type)] = {
            "rank_fraction": fraction,
            "selected_rank": selected_rank,
            "full_rank": full_rank,
            "rank_ratio": f"{selected_rank}/{full_rank}",
        }
    return rank_map
