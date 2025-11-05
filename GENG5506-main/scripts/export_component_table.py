"""Generate a LaTeX table for system components and costs.

Usage:
  cd GENG5506-main
  python -m scripts.export_component_table                             # uses optimisation outputs
  python -m scripts.export_component_table --source system              # uses system outputs
  python -m scripts.export_component_table --out outputs/finance/components_table.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_number(v: float | int, decimals: int = 0) -> str:
    try:
        if decimals <= 0:
            return f"{float(v):,.0f}"
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return str(v)


def _aud(v: float | int) -> str:
    return f"${_fmt_number(v)}"


def _paths_for(source: str) -> Tuple[Path, Path, Path]:
    if source == "system":
        return (
            PROJECT_ROOT / "outputs/system/dispatch_summary.json",
            PROJECT_ROOT / "outputs/system/cost_summary.json",
            PROJECT_ROOT / "outputs/optimisation/component_summary.json",  # fallback if system component summary is absent
        )
    # optimisation
    return (
        PROJECT_ROOT / "outputs/optimisation/best_result.json",
        PROJECT_ROOT / "outputs/optimisation/best_cost_summary.json",
        PROJECT_ROOT / "outputs/optimisation/component_summary.json",
    )


def _latex_table(component: Dict, cost: Dict, dispatch: Dict, title: str) -> str:
    lines: list[str] = []
    add = lines.append

    def row(label: str, value: str) -> None:
        add(f"    {label} & {value} \\")

    add("\\begin{table}[h]")
    add("\\centering")
    add("\\begin{tabular}{llr}")
    add("\\hline")
    add("Component & Parameter & Value \\")
    add("\\hline")

    # PV
    pv = component.get("pv", {})
    add("\\textbf{PV} & & \\")
    row("", f"Panel Count & {_fmt_number(pv.get('panel_count', 0))}")
    row("", f"Total Area (ha) & {_fmt_number(float(pv.get('total_area_m2', 0))/10000.0, 2)}")
    row("", f"CAPEX (AUD) & {_aud(pv.get('capex_aud', 0))}")
    row("", f"Annual OPEX (AUD) & {_aud(pv.get('annual_opex_aud', 0))}")
    add("\\hline")

    # Wind
    wind = component.get("wind", {})
    add("\\textbf{Wind} & & \\")
    row("", f"Turbine Count & {_fmt_number(wind.get('turbine_count', 0))}")
    row("", f"Total Capacity (kW) & {_fmt_number(wind.get('total_capacity_kw', 0))}")
    row("", f"CAPEX (AUD) & {_aud(wind.get('capex_aud', 0))}")
    row("", f"Annual OPEX (AUD) & {_aud(wind.get('annual_opex_aud', 0))}")
    add("\\hline")

    # BESS
    bess = component.get("battery_storage", {})
    add("\\textbf{BESS} & & \\")
    row("", f"Unit Name & {bess.get('unit_name', '')}")
    row("", f"Unit Energy (MWh) & {_fmt_number(bess.get('unit_energy_mwh', 0), 1)}")
    row("", f"Unit Power (MW) & {_fmt_number(bess.get('unit_power_mw', 0), 1)}")
    row("", f"Unit Count & {_fmt_number(bess.get('unit_count', 0))}")
    row("", f"Total Energy (MWh) & {_fmt_number(bess.get('total_energy_mwh', 0), 1)}")
    row("", f"Total Power (MW) & {_fmt_number(bess.get('total_power_mw', 0), 1)}")
    row("", f"CAPEX (AUD) & {_aud(bess.get('capex_aud', 0))}")
    row("", f"Annual OPEX (AUD) & {_aud(bess.get('annual_opex_aud', 0))}")
    repl = cost.get("storage_replacement_present_cost", 0.0)
    if repl and float(repl) > 0:
        row("", f"Replacement Cost @ Yr{int(cost.get('storage_replacement_year', 20))} (present value) & {_aud(repl)}")
    add("\\hline")

    # PHES
    phes = component.get("pumped_hydro", {})
    add("\\textbf{PHES} & & \\")
    row("", f"Total Energy (MWh) & {_fmt_number(phes.get('total_energy_mwh', 0), 0)}")
    row("", f"Total Power (MW) & {_fmt_number(phes.get('total_power_mw', 0), 0)}")
    row("", f"CAPEX (AUD) & {_aud(phes.get('capex_aud', 0))}")
    row("", f"Annual OPEX (AUD) & {_aud(phes.get('annual_opex_aud', 0))}")
    add("\\hline")

    # Generator
    gen = component.get("generator", {})
    add("\\textbf{Generator} & & \\")
    row("", f"Unit Name & {gen.get('unit_name', '')}")
    row("", f"Unit Capacity (kW) & {_fmt_number(gen.get('unit_capacity_kw', 0))}")
    row("", f"Unit Count & {_fmt_number(gen.get('unit_count', 0))}")
    row("", f"Total Capacity (kW) & {_fmt_number(gen.get('total_capacity_kw', 0))}")
    row("", f"CAPEX (AUD) & {_aud(gen.get('capex_aud', 0))}")
    row("", f"Annual OPEX (AUD) & {_aud(gen.get('annual_opex_aud', 0))}")
    fuel_cost = dispatch.get("fuel_cost", dispatch.get("annual_fuel_cost", 0))
    if fuel_cost and float(fuel_cost) > 0:
        row("", f"Annual Fuel (AUD) & {_aud(fuel_cost)}")
    add("\\hline")

    # Summary
    add("\\textbf{Summary} & & \\")
    row("", f"NPC Total & {_aud(cost.get('npc_total', 0))}")
    row("", f"Initial CAPEX & {_aud(cost.get('initial_capex', 0))}")
    row("", f"Total OPEX (NPC Adjusted) & {_aud(cost.get('present_value_annual_costs', 0))}")

    add("\\hline")
    add("\\end{tabular}")
    add(f"\\caption{{{title}}}")
    add("\\end{table}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table for components and costs.")
    parser.add_argument("--source", choices=["optimisation", "system"], default="optimisation")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs/finance/components_table.tex"))
    args = parser.parse_args(argv)

    dispatch_path, cost_path, component_path = _paths_for(args.source)
    dispatch = _read_json(dispatch_path)
    cost = _read_json(cost_path)
    component = _read_json(component_path)

    if not component:
        raise SystemExit("Component summary JSON not found; run optimisation first.")
    if not cost:
        raise SystemExit("Cost summary JSON not found.")

    title = "System Components and Costs"
    table_tex = _latex_table(component, cost, dispatch, title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote LaTeX table -> {out_path}")


if __name__ == "__main__":
    main()

