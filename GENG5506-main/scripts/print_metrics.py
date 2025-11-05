"""Print key metrics from dispatch outputs.

Reads outputs/system/dispatch_summary.json (default) and prints:
- Renewable penetration
- Annual energies (MWh): load, PV, wind, generator, curtailed, unserved
- Installed capacities

Usage:
  - From project root:  python -m scripts.print_metrics
  - Custom summary:     python -m scripts.print_metrics path/to/dispatch_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    project_root = Path(__file__).resolve().parents[1]
    default_summary = project_root / "outputs" / "system" / "dispatch_summary.json"
    path = Path(argv[0]).resolve() if argv else default_summary

    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        s = json.load(f)

    def g(key: str, default: float = 0.0) -> float:
        return float(s.get(key, default))

    ren_pen = g("renewable_penetration")
    print(f"Renewable penetration: {ren_pen:.3%}")

    print("Energies (MWh):")
    print(f"  Load:       {g('load_energy_mwh'):.1f}")
    print(f"  PV:         {g('pv_energy_mwh'):.1f}")
    print(f"  Wind:       {g('wind_energy_mwh'):.1f}")
    print(f"  Generator:  {g('generator_energy_mwh'):.1f}")
    print(f"  Curtailment:{g('curtailed_energy_mwh'):.1f}")
    print(f"  Unserved:   {g('unserved_energy_mwh'):.1f}")

    print("Capacities:")
    print(f"  PV (kW):        {g('pv_capacity_kw'):.0f}")
    print(f"  Wind (count):   {g('wind_turbine_count'):.0f}")
    print(f"  Battery (kWh):  {g('storage_energy_capacity_kwh'):.0f}")
    print(f"  Battery (kW):   {g('storage_power_discharge_kw'):.0f}")
    print(f"  Generator (kW): {g('generator_capacity_kw'):.0f}")


if __name__ == "__main__":
    main()

