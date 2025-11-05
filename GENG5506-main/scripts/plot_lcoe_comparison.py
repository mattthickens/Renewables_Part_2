"""Plot LCOE comparisons.

Two modes:
1) Reference comparison (default): compares your simulation LCOE (dashed line) to a set of reference technologies.
2) Component mode: computes LCOE for each simulated component (PV, Wind, BESS, PHES, Diesel) from your outputs
   and plots them as bars, with a dashed line for the whole-system LCOE.

Examples
  # Compare the optimised design LCOE against default reference technologies
  python -m scripts.plot_lcoe_comparison

  # Use system run instead of optimisation
  python -m scripts.plot_lcoe_comparison --source system

  # Provide your own reference set (JSON path)
  python -m scripts.plot_lcoe_comparison --references data/lcoe_refs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_references() -> Dict[str, float]:
    # AUD/MWh reference LCOEs – feel free to adjust as needed
    return {
        "Solar": 96.1,
        "Wind": 77.5,
        "Natural Gas": 116.2,
        "Geothermal": 131.8,
        "Coal": 178.2,
        "Nuclear": 279.0,
    }


def _paths_for(source: str) -> Tuple[Path, Path, Path, Path]:
    if source == "system":
        return (
            PROJECT_ROOT / "outputs/system/cost_summary.json",
            PROJECT_ROOT / "outputs/system/dispatch_summary.json",
            PROJECT_ROOT / "outputs/system/dispatch_timeseries.csv",
            PROJECT_ROOT / "outputs/optimisation/component_summary.json",
        )
    return (
        PROJECT_ROOT / "outputs/optimisation/best_cost_summary.json",
        PROJECT_ROOT / "outputs/system/dispatch_summary.json",  # best candidate re-sim writes here
        PROJECT_ROOT / "outputs/system/dispatch_timeseries.csv",
        PROJECT_ROOT / "outputs/optimisation/component_summary.json",
    )


def plot_lcoe_comparison(
    lcoe_sim: float,
    references: Dict[str, float],
    out_path: Path,
    currency: str = "AUD",
) -> None:
    import matplotlib.pyplot as plt

    # Filter out components with NaN/inf LCOE (e.g., zero throughput)
    import math
    filtered = [(k, float(v)) for k, v in references.items() if v is not None and math.isfinite(float(v))]
    labels = [k for k, _ in filtered]
    values = [v for _, v in filtered]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Choose a pleasant color cycle
    colors = [
        "#f1c40f",  # solar yellow
        "#5bc0de",  # wind blue
        "#bdc3c7",  # gas grey
        "#c9b37e",  # geothermal tan
        "#636363",  # coal dark grey
        "#9b8cc4",  # nuclear purple
    ]
    while len(colors) < len(values):
        colors += colors

    bars = ax.bar(labels, values, color=colors[: len(values)], alpha=0.95)

    # Annotate bars
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + max(values) * 0.02,
            f"{val:.1f} {currency}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    # Simulation LCOE reference line
    ax.axhline(lcoe_sim, color="#d62728", linestyle="--", linewidth=2.0)
    ax.text(
        0.98,
        lcoe_sim,
        f"  Simulation LCOE: {lcoe_sim:.0f} {currency}/MWh",
        color="#d62728",
        fontsize=11,
        va="center",
        ha="right",
        transform=ax.get_yaxis_transform(),
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
    )

    ax.set_ylabel(f"LCOE ({currency} per MWh)")
    ax.set_title("Levelised Cost of Energy comparison by source")
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _present_worth_factor(discount_rate: float, years: int) -> float:
    if discount_rate == 0:
        return float(years)
    return (1 - (1 + discount_rate) ** -years) / discount_rate


def _compute_component_lcoes(source: str) -> Tuple[Dict[str, float], float, str]:
    """Return (component_lcoes, system_lcoe, currency).

    Component LCOE bases:
    - PV, Wind: NPC = capex + PV(annual fixed O&M); Energy = annual component MWh * lifetime_years
    - BESS, PHES: NPC = capex + PV(annual fixed O&M) [+ BESS replacement PV]; Energy = annual discharge MWh * lifetime_years
    - Diesel: NPC = capex + PV(annual fixed O&M + fuel + variable O&M); Energy = annual generator MWh * lifetime_years
    """

    # Import site config to retrieve lifetime years and discount rate
    try:
        from main import COST_MODEL_CONFIG  # type: ignore
    except ModuleNotFoundError:
        # If script was executed directly (not with -m), ensure project root on sys.path
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from main import COST_MODEL_CONFIG  # type: ignore
        except Exception:
            COST_MODEL_CONFIG = None  # type: ignore

    # Reasonable fallbacks if import fails
    lifetime = int(getattr(COST_MODEL_CONFIG, "lifetime_years", 25) if COST_MODEL_CONFIG else 25)
    rate = float(getattr(COST_MODEL_CONFIG, "discount_rate", 0.08) if COST_MODEL_CONFIG else 0.08)
    pw = _present_worth_factor(rate, lifetime)

    cost_path, dispatch_path, timeseries_path, component_path = _paths_for(source)
    cost = _read_json(cost_path)
    dispatch = _read_json(dispatch_path)
    component = _read_json(component_path)

    if not cost:
        raise SystemExit(f"Cost summary not found: {cost_path}")
    if not dispatch:
        raise SystemExit(f"Dispatch summary not found: {dispatch_path}")

    currency = str(cost.get("currency", "AUD"))
    system_lcoe = float(cost.get("lcoe_per_mwh", 0.0))

    # Annual energies from timeseries (for robustness)
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required for component LCOE mode")

    if not Path(timeseries_path).exists():
        raise SystemExit(f"Timeseries not found: {timeseries_path}")
    df = pd.read_csv(timeseries_path, parse_dates=["timestamp"])  # timestep is hourly in this project
    timestep_hours = 1.0

    def energy_mwh(col: str) -> float:
        return float(df[col].sum() * timestep_hours / 1000.0) if col in df.columns else 0.0

    pv_mwh = energy_mwh("pv_kw")
    wind_mwh = energy_mwh("wind_kw")
    gen_mwh = energy_mwh("generator_kw")
    bess_dis_mwh = energy_mwh("bess_discharge_kw")
    phes_dis_mwh = energy_mwh("phes_discharge_kw")

    # Pull capex and opex
    pv_capex = float(cost.get("pv_capex", 0.0))
    wind_capex = float(cost.get("wind_capex", 0.0))
    storage_capex = float(cost.get("storage_capex", 0.0))
    phes_capex = float(cost.get("phes_capex", 0.0))
    gen_capex = float(cost.get("generator_capex", 0.0))

    # Annual fixed O&M by component (prefer component summary; fall back to zero)
    pv_om = float(component.get("pv", {}).get("annual_opex_aud", 0.0))
    wind_om = float(component.get("wind", {}).get("annual_opex_aud", 0.0))
    bess_om = float(component.get("battery_storage", {}).get("annual_opex_aud", 0.0))
    phes_om = float(component.get("pumped_hydro", {}).get("annual_opex_aud", 0.0))
    gen_fixed_om = float(component.get("generator", {}).get("annual_opex_aud", 0.0))

    # Generator variable O&M + fuel from dispatch (annual numbers)
    gen_var_om = float(dispatch.get("generator_variable_om_cost", 0.0))
    gen_fuel = float(dispatch.get("fuel_cost", 0.0))

    # BESS replacement PV cost (present value) from cost summary
    bess_repl_pv = float(cost.get("storage_replacement_present_cost", 0.0))

    def lcoe(npc: float, annual_mwh: float) -> float:
        denom = annual_mwh * lifetime if annual_mwh > 0 else 0.0
        return npc / denom if denom > 0 else float("nan")

    comp_lcoes: Dict[str, float] = {}

    # PV
    pv_npc = pv_capex + pv_om * pw
    comp_lcoes["PV"] = lcoe(pv_npc, pv_mwh)

    # Wind
    wind_npc = wind_capex + wind_om * pw
    comp_lcoes["Wind"] = lcoe(wind_npc, wind_mwh)

    # BESS (per discharged MWh)
    bess_npc = storage_capex + bess_om * pw + bess_repl_pv
    comp_lcoes["BESS (per MWh discharged)"] = lcoe(bess_npc, bess_dis_mwh)

    # PHES (per discharged MWh)
    phes_npc = phes_capex + phes_om * pw
    comp_lcoes["PHES (per MWh discharged)"] = lcoe(phes_npc, phes_dis_mwh)

    # Diesel generator
    gen_annual_total = gen_fixed_om + gen_var_om + gen_fuel
    gen_npc = gen_capex + gen_annual_total * pw
    comp_lcoes["Diesel Generator"] = lcoe(gen_npc, gen_mwh)

    return comp_lcoes, system_lcoe, currency


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot LCOE comparison bar chart.")
    parser.add_argument("--source", choices=["optimisation", "system"], default="optimisation")
    # Default to components so running the script uses simulated component LCOEs
    parser.add_argument("--mode", choices=["references", "components"], default="components",
                        help="Plot reference technologies or compute component LCOEs from simulation outputs")
    parser.add_argument("--references", type=str, default="",
                        help="Optional JSON file mapping label -> LCOE (AUD/MWh) for references mode")
    parser.add_argument("--currency", type=str, default="AUD")
    parser.add_argument("--out", type=str, default="",
                        help="Output image path. If omitted, picks a sensible default by mode.")
    args = parser.parse_args(argv)

    if args.mode == "components":
        comp_lcoes, system_lcoe, currency = _compute_component_lcoes(args.source)
        out_path = Path(args.out) if args.out else (PROJECT_ROOT / "images/finance/lcoe_components.png")
        plot_lcoe_comparison(system_lcoe, comp_lcoes, out_path, currency=currency)
        print(f"Wrote component LCOE comparison -> {out_path}")
        return

    # references mode
    cost_path, _, _, _ = _paths_for(args.source)
    cost = _read_json(cost_path)
    if not cost:
        raise SystemExit(f"Cost summary not found: {cost_path}")
    lcoe = float(cost.get("lcoe_per_mwh", 0.0))
    if lcoe <= 0:
        raise SystemExit("lcoe_per_mwh missing or zero in cost summary")

    if args.references:
        ref_path = Path(args.references)
        refs = _read_json(ref_path)
        references = {str(k): float(v) for k, v in refs.items()}
    else:
        references = _default_references()

    out_path = Path(args.out) if args.out else (PROJECT_ROOT / "images/finance/lcoe_comparison.png")
    plot_lcoe_comparison(lcoe, references, out_path, currency=args.currency)
    print(f"Wrote LCOE reference comparison -> {out_path}")


if __name__ == "__main__":
    main()
