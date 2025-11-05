"""CLI utility to regenerate optimisation figures from existing outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .optimisation import (
        OptimisationConstraints,
        regenerate_optimisation_plots,
    )
except ImportError:  # Fallback when executed as a script, not module
    CURRENT_DIR = Path(__file__).resolve().parent
    PACKAGE_ROOT = CURRENT_DIR.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from scripts.optimisation import (  # type: ignore
        OptimisationConstraints,
        regenerate_optimisation_plots,
    )


def _default_paths() -> tuple[Path | None, Path | None, OptimisationConstraints | None]:
    try:
        from main import OPTIMISATION_OUTPUTS, OPTIMISATION_CONSTRAINTS
    except Exception:  # pragma: no cover - defensive fallback
        return None, None, None

    return (
        OPTIMISATION_OUTPUTS.results_csv_path,
        OPTIMISATION_OUTPUTS.best_result_path,
        OPTIMISATION_CONSTRAINTS,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Regenerate optimisation plots from saved output files.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=defaults[0] if defaults[0] is not None else None,
        help="Path to optimisation results CSV (defaults to configuration if available).",
    )
    parser.add_argument(
        "--best",
        type=Path,
        default=defaults[1] if defaults[1] is not None else None,
        help="Path to best result JSON (defaults to configuration if available).",
    )
    parser.add_argument(
        "--penetration-target",
        type=float,
        default=defaults[2].min_renewable_penetration if defaults[2] is not None else 0.8,
        help="Minimum renewable penetration target (default 0.8).",
    )
    parser.add_argument(
        "--max-unserved-energy",
        type=float,
        default=defaults[2].max_unserved_energy_mwh if defaults[2] is not None else 1.0,
        help="Maximum unserved energy constraint in MWh (default 1.0).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.results is None or args.best is None:
        parser.error("Both --results and --best must be provided when defaults are unavailable.")

    constraints = OptimisationConstraints(
        min_renewable_penetration=args.penetration_target,
        max_unserved_energy_mwh=args.max_unserved_energy,
    )

    regenerate_optimisation_plots(args.results, args.best, constraints)


if __name__ == "__main__":
    main()
