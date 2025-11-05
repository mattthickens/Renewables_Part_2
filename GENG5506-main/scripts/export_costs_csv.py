"""Export consolidated cost summaries to CSV.

Collects available outputs and writes a flat CSV for quick analysis:
  - System run: outputs/system/cost_summary.json (+ dispatch_summary.json)
  - Best optimisation: outputs/optimisation/best_cost_summary.json (+ best_result.json)
  - Sensitivity runs: outputs/analysis/sensitivity/*/cost_summary.json (+ dispatch_summary.json)

Usage:
  cd GENG5506-main
  python -m scripts.export_costs_csv               # export all available
  python -m scripts.export_costs_csv --out my.csv  # custom path
  python -m scripts.export_costs_csv --only system # restrict to one group
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _row_with_defaults(label: str) -> Dict[str, float | str]:
    # Common schema columns (any missing values remain blank)
    cols: Dict[str, float | str] = {
        "label": label,
        # capacities/config
        "pv_capacity_kw": "",
        "wind_capacity_kw": "",
        "storage_energy_kwh": "",
        "storage_power_kw": "",
        "phes_energy_kwh": "",
        "phes_power_kw": "",
        "generator_capacity_kw": "",
        # high-level metrics
        "renewable_penetration": "",
        "lcoe_per_mwh": "",
        # totals and capex/opex
        "npc_total": "",
        "initial_capex": "",
        "pv_capex": "",
        "wind_capex": "",
        "storage_capex": "",
        "generator_capex": "",
        "phes_capex": "",
        "present_value_annual_costs": "",
        "storage_replacement_present_cost": "",
        "annual_cost": "",
        "annual_curtailment_cost": "",
        # energy context
        "load_energy_mwh": "",
        "pv_energy_mwh": "",
        "wind_energy_mwh": "",
        "generator_energy_mwh": "",
        "curtailed_energy_mwh": "",
        "unserved_energy_mwh": "",
    }
    return cols


def _merge_from(d: Dict[str, float | str], src: Dict[str, float | str]) -> None:
    for k, v in src.items():
        if k in d and v is not None:
            d[k] = v


def _export(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    import csv

    if not rows:
        print("No cost outputs found; nothing to export.")
        return
    # Stable column order using keys from the template
    header = list(_row_with_defaults("template").keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def _system_rows() -> List[Dict[str, float | str]]:
    cost = _read_json(PROJECT_ROOT / "outputs/system/cost_summary.json")
    disp = _read_json(PROJECT_ROOT / "outputs/system/dispatch_summary.json")
    if not cost and not disp:
        return []
    row = _row_with_defaults("system")
    _merge_from(row, cost)
    _merge_from(row, disp)
    # Map PHES capacity keys from dispatch summary if present
    if "phes_energy_capacity_kwh" in disp:
        row["phes_energy_kwh"] = disp.get("phes_energy_capacity_kwh", "")
    if "phes_power_discharge_kw" in disp:
        row["phes_power_kw"] = disp.get("phes_power_discharge_kw", "")
    return [row]


def _best_rows() -> List[Dict[str, float | str]]:
    cost = _read_json(PROJECT_ROOT / "outputs/optimisation/best_cost_summary.json")
    best = _read_json(PROJECT_ROOT / "outputs/optimisation/best_result.json")
    if not cost and not best:
        return []
    row = _row_with_defaults("best")
    _merge_from(row, cost)
    # best_result contains capacities; map keys if needed
    mapping = {
        "pv_capacity_kw": best.get("pv_capacity_kw"),
        "wind_capacity_kw": best.get("wind_capacity_kw"),
        "storage_energy_kwh": best.get("storage_energy_kwh"),
        "storage_power_kw": best.get("storage_power_kw"),
        "phes_energy_kwh": best.get("phes_energy_kwh"),
        "phes_power_kw": best.get("phes_power_kw"),
        "generator_capacity_kw": best.get("generator_capacity_kw"),
    }
    _merge_from(row, mapping)
    return [row]


def _sensitivity_rows() -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    root = PROJECT_ROOT / "outputs/analysis/sensitivity"
    if not root.exists():
        return rows
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cost = _read_json(scenario_dir / "cost_summary.json")
        disp = _read_json(scenario_dir / "dispatch_summary.json")
        if not cost and not disp:
            continue
        label = f"sensitivity:{scenario_dir.name}"
        row = _row_with_defaults(label)
        _merge_from(row, cost)
        _merge_from(row, disp)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export consolidated cost summaries to CSV.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs/finance/cost_summary.csv"), help="Output CSV path")
    parser.add_argument(
        "--only",
        choices=["system", "best", "sensitivity", "all"],
        default="all",
        help="Which summaries to export",
    )
    args = parser.parse_args(argv)

    rows: List[Dict[str, float | str]] = []
    if args.only in ("system", "all"):
        rows.extend(_system_rows())
    if args.only in ("best", "all"):
        rows.extend(_best_rows())
    if args.only in ("sensitivity", "all"):
        rows.extend(_sensitivity_rows())

    _export(rows, Path(args.out))


if __name__ == "__main__":
    main()
