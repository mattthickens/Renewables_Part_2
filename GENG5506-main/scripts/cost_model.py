"""Cost and NPV evaluation utilities for the Burnie microgrid."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class CostModelInputs:
    """Capacity inputs used to scale capital and operating expenses."""

    pv_capacity_kw: float
    wind_capacity_kw: float
    storage_energy_kwh: float
    storage_power_kw: float
    generator_capacity_kw: float
    load_energy_mwh_per_year: float
    # Optional PHES asset sizing (energy-based capex and opex)
    phes_energy_kwh: float = 0.0
    phes_power_kw: float = 0.0


@dataclass(frozen=True)
class PVCostParameters:
    """Component-level inputs for computing PV costs per kW."""

    panel_power_kw: float
    panel_cost: float
    panel_area_m2: float
    land_cost_per_m2: float
    bos_cost_per_kw: float
    fixed_om_per_kw_year: float

    def capex_per_kw(self) -> float:
        """Return the effective capex per kW DC including panel and land costs."""
        if self.panel_power_kw <= 0:
            return 0.0
        panel_cost_per_kw = self.panel_cost / self.panel_power_kw
        land_cost_per_kw = (self.panel_area_m2 * self.land_cost_per_m2) / self.panel_power_kw
        return panel_cost_per_kw + land_cost_per_kw + self.bos_cost_per_kw

    def panel_count(self, total_capacity_kw: float) -> float:
        return total_capacity_kw / self.panel_power_kw if self.panel_power_kw else 0.0

    def land_area_m2(self, total_capacity_kw: float) -> float:
        return self.panel_area_m2 * self.panel_count(total_capacity_kw)


@dataclass(frozen=True)
class WindCostParameters:
    """Component-level inputs for computing wind costs per kW."""

    turbine_rated_kw: float
    turbine_cost: float
    balance_of_plant_per_kw: float
    fixed_om_per_kw_year: float

    def capex_per_kw(self) -> float:
        if self.turbine_rated_kw <= 0:
            return 0.0
        turbine_cost_per_kw = self.turbine_cost / self.turbine_rated_kw
        return turbine_cost_per_kw + self.balance_of_plant_per_kw


@dataclass(frozen=True)
class CostModelConfig:
    lifetime_years: int
    discount_rate: float
    pv_costs: PVCostParameters
    wind_costs: WindCostParameters
    storage_capex_per_kwh: float
    storage_capex_per_kw: float
    storage_fixed_om_per_kwh_year: float
    storage_replacement_year: int
    storage_replacement_fraction: float
    generator_capex_per_kw: float
    generator_fixed_om_per_kw_year: float
    write_output_path: Path
    currency: str = "AUD"
    curtailment_cost_per_mwh: float = 0.0
    # Optional fixed capex for large, non-scaled assets (e.g., PHES)
    phes_capex_fixed: float = 0.0
    # Energy-scaled PHES costs (per kWh installed and per kWh-year O&M)
    phes_capex_per_kwh: float = 0.0
    phes_fixed_om_per_kwh_year: float = 0.0


def _present_worth_factor(discount_rate: float, years: int) -> float:
    if discount_rate == 0:
        return float(years)
    return (1 - (1 + discount_rate) ** -years) / discount_rate


def evaluate_costs(
    dispatch_summary: Dict[str, float],
    inputs: CostModelInputs,
    config: CostModelConfig,
) -> Dict[str, float]:
    pv_capex_per_kw = config.pv_costs.capex_per_kw()
    pv_capex = inputs.pv_capacity_kw * pv_capex_per_kw
    pv_panel_count = config.pv_costs.panel_count(inputs.pv_capacity_kw)
    pv_land_area = config.pv_costs.land_area_m2(inputs.pv_capacity_kw)

    wind_capex_per_kw = config.wind_costs.capex_per_kw()
    wind_capex = inputs.wind_capacity_kw * wind_capex_per_kw

    storage_capex = (
        inputs.storage_energy_kwh * config.storage_capex_per_kwh
        + inputs.storage_power_kw * config.storage_capex_per_kw
    )
    generator_capex = inputs.generator_capacity_kw * config.generator_capex_per_kw
    # PHES costs: fixed component plus energy-scaled component
    phes_capex_variable = inputs.phes_energy_kwh * config.phes_capex_per_kwh
    phes_capex = config.phes_capex_fixed + phes_capex_variable
    initial_capex = pv_capex + wind_capex + storage_capex + generator_capex + phes_capex

    annual_pv_om = inputs.pv_capacity_kw * config.pv_costs.fixed_om_per_kw_year
    annual_wind_om = inputs.wind_capacity_kw * config.wind_costs.fixed_om_per_kw_year
    annual_storage_om = inputs.storage_energy_kwh * config.storage_fixed_om_per_kwh_year
    annual_generator_om = inputs.generator_capacity_kw * config.generator_fixed_om_per_kw_year
    annual_phes_om = inputs.phes_energy_kwh * config.phes_fixed_om_per_kwh_year

    annual_variable_costs = (
        dispatch_summary.get("generator_variable_om_cost", 0.0)
        + dispatch_summary.get("fuel_cost", 0.0)
    )
    curtailed_energy_mwh = dispatch_summary.get("curtailed_energy_mwh", 0.0)
    annual_curtailment_cost = curtailed_energy_mwh * config.curtailment_cost_per_mwh

    annual_cost = (
        annual_pv_om
        + annual_wind_om
        + annual_storage_om
        + annual_generator_om
        + annual_phes_om
        + annual_variable_costs
        + annual_curtailment_cost
    )

    lifetime_years = config.lifetime_years
    discount_rate = config.discount_rate
    pw_factor = _present_worth_factor(discount_rate, lifetime_years)
    present_value_annual_costs = annual_cost * pw_factor

    storage_replacement_pv = 0.0
    replacement_year = config.storage_replacement_year
    if 0 < replacement_year < lifetime_years:
        replacement_cost = storage_capex * config.storage_replacement_fraction
        storage_replacement_pv = replacement_cost / ((1 + discount_rate) ** replacement_year)

    total_npc = initial_capex + present_value_annual_costs + storage_replacement_pv
    total_lifetime_energy_mwh = inputs.load_energy_mwh_per_year * lifetime_years
    lcoe = total_npc / total_lifetime_energy_mwh if total_lifetime_energy_mwh > 0 else float("inf")

    renewable_penetration = dispatch_summary.get("renewable_penetration", 0.0)

    results = {
        "currency": config.currency,
        "npc_total": float(total_npc),
        "initial_capex": float(initial_capex),
        "pv_capex": float(pv_capex),
        "wind_capex": float(wind_capex),
        "storage_capex": float(storage_capex),
        "generator_capex": float(generator_capex),
        "phes_capex": float(phes_capex),
        "present_value_annual_costs": float(present_value_annual_costs),
        "storage_replacement_present_cost": float(storage_replacement_pv),
        "annual_cost": float(annual_cost),
        "lcoe_per_mwh": float(lcoe),
        "renewable_penetration": float(renewable_penetration),
        "pv_panel_count": float(pv_panel_count),
        "pv_land_area_m2": float(pv_land_area),
        "pv_capex_per_kw": float(pv_capex_per_kw),
        "wind_capex_per_kw": float(wind_capex_per_kw),
        "curtailed_energy_mwh": float(curtailed_energy_mwh),
        "annual_curtailment_cost": float(annual_curtailment_cost),
        "curtailment_cost_per_mwh": float(config.curtailment_cost_per_mwh),
        "annual_phes_om": float(annual_phes_om),
        "phes_capex_per_kwh": float(config.phes_capex_per_kwh),
        "phes_fixed_om_per_kwh_year": float(config.phes_fixed_om_per_kwh_year),
    }

    config.write_output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.write_output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results


__all__ = [
    "CostModelInputs",
    "PVCostParameters",
    "WindCostParameters",
    "CostModelConfig",
    "evaluate_costs",
]
