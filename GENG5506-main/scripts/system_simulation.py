"""System dispatch simulation driven by configuration from main."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StorageConfig:
    energy_capacity_kwh: float
    charge_power_kw: float
    discharge_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float


@dataclass(frozen=True)
class GeneratorConfig:
    max_power_kw: float
    min_power_kw: float
    fuel_litre_per_kwh: float
    fuel_cost_per_litre: float
    variable_om_per_kwh: float


@dataclass(frozen=True)
class SimulationInputsConfig:
    load_profile_path: Path
    pv_profile_path: Path
    wind_profile_path: Path
    output_timeseries_path: Path
    output_summary_path: Path


@dataclass(frozen=True)
class DispatchConfig:
    pv_capacity_kw: float
    wind_turbine_count: int
    storage: StorageConfig
    generator: GeneratorConfig
    timestep_hours: float
    # Optional second storage asset (e.g., pumped hydro)
    pumped_hydro: StorageConfig | None = None


@dataclass(frozen=True)
class SystemSimulationConfig:
    inputs: SimulationInputsConfig
    dispatch: DispatchConfig
    float_format: str


def _load_timeseries(inputs: SimulationInputsConfig) -> pd.DataFrame:
    load_df = pd.read_csv(inputs.load_profile_path, parse_dates=["timestamp"], index_col="timestamp")
    pv_df = pd.read_csv(inputs.pv_profile_path, parse_dates=["timestamp"], index_col="timestamp")
    wind_df = pd.read_csv(inputs.wind_profile_path, parse_dates=["timestamp"], index_col="timestamp")

    df = load_df[[load_df.columns[0]]].rename(columns={load_df.columns[0]: "load_mw"})
    df["pv_per_kw"] = pv_df["pv_ac_kw_per_kw"]
    df["wind_per_turbine_kw"] = wind_df["turbine_ac_kw_per_unit"]
    return df


def _simulate_dispatch(df: pd.DataFrame, config: DispatchConfig) -> pd.DataFrame:
    timestep = config.timestep_hours
    load_kw = df["load_mw"].to_numpy() * 1000.0
    pv_kw = df["pv_per_kw"].to_numpy() * config.pv_capacity_kw
    wind_kw = df["wind_per_turbine_kw"].to_numpy() * config.wind_turbine_count

    n = len(df)
    storage = config.storage
    phes = config.pumped_hydro
    generator = config.generator

    energy_capacity = storage.energy_capacity_kwh
    soc_min = storage.min_soc_fraction * energy_capacity
    soc_max = storage.max_soc_fraction * energy_capacity
    soc = storage.initial_soc_fraction * energy_capacity

    # PHES state (if configured)
    if phes is not None:
        phes_energy_capacity = phes.energy_capacity_kwh
        phes_soc_min = phes.min_soc_fraction * phes_energy_capacity
        phes_soc_max = phes.max_soc_fraction * phes_energy_capacity
        phes_soc = phes.initial_soc_fraction * phes_energy_capacity

    results = {
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "wind_kw": wind_kw,
        "net_load_kw": np.zeros(n),
        "bess_charge_kw": np.zeros(n),
        "bess_discharge_kw": np.zeros(n),
        "bess_soc_kwh": np.zeros(n),
        # PHES traces (zero if not used)
        "phes_charge_kw": np.zeros(n),
        "phes_discharge_kw": np.zeros(n),
        "phes_soc_kwh": np.zeros(n),
        "generator_kw": np.zeros(n),
        "curtailment_kw": np.zeros(n),
        "unserved_kw": np.zeros(n),
        "renewable_served_kw": np.zeros(n),
    }

    for i in range(n):
        net = load_kw[i] - pv_kw[i] - wind_kw[i]
        results["net_load_kw"][i] = net

        if net < 0.0:
            surplus = -net
            # Charge BESS first
            charge_cap_bess = min(
                storage.charge_power_kw,
                (soc_max - soc) / (storage.charge_efficiency * timestep),
            )
            charge_kw_bess = min(surplus, max(charge_cap_bess, 0.0))
            energy_added_bess = charge_kw_bess * storage.charge_efficiency * timestep
            soc = min(soc + energy_added_bess, soc_max)
            results["bess_charge_kw"][i] = charge_kw_bess

            remaining = surplus - charge_kw_bess
            # Then charge PHES with remaining surplus
            if phes is not None and remaining > 0.0:
                charge_cap_phes = min(
                    phes.charge_power_kw,
                    (phes_soc_max - phes_soc) / (phes.charge_efficiency * timestep),
                )
                charge_kw_phes = min(remaining, max(charge_cap_phes, 0.0))
                energy_added_phes = charge_kw_phes * phes.charge_efficiency * timestep
                phes_soc = min(phes_soc + energy_added_phes, phes_soc_max)
                results["phes_charge_kw"][i] = charge_kw_phes
                remaining -= charge_kw_phes
            
            curtailment = remaining
            results["curtailment_kw"][i] = curtailment
            renewable_served = load_kw[i]
        else:
            deficit = net
            # Discharge PHES first (bulk), then BESS
            discharge_kw_phes = 0.0
            if phes is not None:
                available_phes = min(
                    phes.discharge_power_kw,
                    max((phes_soc - phes_soc_min) * phes.discharge_efficiency / timestep, 0.0),
                )
                discharge_kw_phes = min(deficit, available_phes)
                energy_removed_phes = discharge_kw_phes / phes.discharge_efficiency * timestep
                phes_soc = max(phes_soc - energy_removed_phes, phes_soc_min)
                results["phes_discharge_kw"][i] = discharge_kw_phes

            residual_after_phes = deficit - discharge_kw_phes

            available_bess = min(
                storage.discharge_power_kw,
                max((soc - soc_min) * storage.discharge_efficiency / timestep, 0.0),
            )
            discharge_kw_bess = min(residual_after_phes, available_bess)
            energy_removed_bess = discharge_kw_bess / storage.discharge_efficiency * timestep
            soc = max(soc - energy_removed_bess, soc_min)
            results["bess_discharge_kw"][i] = discharge_kw_bess

            residual = residual_after_phes - discharge_kw_bess
            generator_kw = min(max(residual, 0.0), generator.max_power_kw)
            if generator_kw < generator.min_power_kw and residual > 0:
                generator_kw = min(max(generator.min_power_kw, 0.0), generator.max_power_kw)
                generator_kw = min(generator_kw, residual)
            residual_after_gen = residual - generator_kw
            results["generator_kw"][i] = generator_kw
            results["unserved_kw"][i] = max(residual_after_gen, 0.0)
            renewable_served = load_kw[i] - (generator_kw + results["unserved_kw"][i])
            renewable_served = max(renewable_served, 0.0)

        results["renewable_served_kw"][i] = renewable_served
        results["bess_soc_kwh"][i] = soc
        if phes is not None:
            results["phes_soc_kwh"][i] = phes_soc

    return pd.DataFrame(results, index=df.index)


def _summarise(results: pd.DataFrame, config: DispatchConfig) -> Dict[str, float]:
    timestep = config.timestep_hours
    load_energy = results["load_kw"].sum() * timestep / 1000.0  # MWh
    renewable_served = results["renewable_served_kw"].sum() * timestep / 1000.0
    renewable_penetration = renewable_served / load_energy if load_energy > 0 else 0.0

    generator_energy_mwh = results["generator_kw"].sum() * timestep / 1000.0
    generator = config.generator
    generator_energy_kwh = generator_energy_mwh * 1000.0
    fuel_litres = generator_energy_kwh * generator.fuel_litre_per_kwh
    fuel_cost = fuel_litres * generator.fuel_cost_per_litre
    generator_om_cost = generator_energy_kwh * generator.variable_om_per_kwh

    curtailed_energy = results["curtailment_kw"].sum() * timestep / 1000.0
    unserved_energy = results["unserved_kw"].sum() * timestep / 1000.0

    summary = {
        "pv_capacity_kw": config.pv_capacity_kw,
        "wind_turbine_count": config.wind_turbine_count,
        "storage_energy_capacity_kwh": config.storage.energy_capacity_kwh,
        "storage_power_charge_kw": config.storage.charge_power_kw,
        "storage_power_discharge_kw": config.storage.discharge_power_kw,
        "load_energy_mwh": load_energy,
        "renewable_served_mwh": renewable_served,
        "renewable_penetration": renewable_penetration,
        "generator_energy_mwh": generator_energy_mwh,
        "fuel_litres": fuel_litres,
        "fuel_cost": fuel_cost,
        "generator_variable_om_cost": generator_om_cost,
        "curtailed_energy_mwh": curtailed_energy,
        "unserved_energy_mwh": unserved_energy,
    }

    if config.pumped_hydro is not None:
        summary.update(
            {
                "phes_energy_capacity_kwh": config.pumped_hydro.energy_capacity_kwh,
                "phes_power_charge_kw": config.pumped_hydro.charge_power_kw,
                "phes_power_discharge_kw": config.pumped_hydro.discharge_power_kw,
            }
        )

    return summary


def run_system_simulation(config: SystemSimulationConfig) -> Dict[str, float]:
    df_inputs = _load_timeseries(config.inputs)
    dispatch_results = _simulate_dispatch(df_inputs, config.dispatch)

    dispatch_results.index.name = "timestamp"
    dispatch_results.sort_index(inplace=True)
    config.inputs.output_timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    dispatch_results.to_csv(config.inputs.output_timeseries_path, float_format=config.float_format)

    summary = _summarise(dispatch_results, config.dispatch)
    config.inputs.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with config.inputs.output_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


__all__ = [
    "StorageConfig",
    "GeneratorConfig",
    "SimulationInputsConfig",
    "DispatchConfig",
    "SystemSimulationConfig",
    "run_system_simulation",
]

# --------- Lightweight helpers for optimisation (no file I/O) ---------

def load_inputs_dataframe(inputs: SimulationInputsConfig) -> pd.DataFrame:
    """Load and return the merged input dataframe once for reuse.

    This mirrors the internal _load_timeseries but is exposed for optimisation
    to avoid repeated CSV reads for each candidate.
    """
    return _load_timeseries(inputs)


def run_dispatch_with_loaded(df_inputs: pd.DataFrame, dispatch: DispatchConfig) -> Dict[str, float]:
    """Run dispatch simulation using a preloaded inputs dataframe and return summary only.

    - No files are written in this path (faster for optimisation inner loop).
    - Uses the same physics as run_system_simulation.
    """
    results = _simulate_dispatch(df_inputs, dispatch)
    return _summarise(results, dispatch)
