"""Sensitivity analysis runner for the optimised Burnie microgrid design."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as base_config
from scripts.cost_model import CostModelConfig, CostModelInputs, evaluate_costs
from scripts.system_simulation import SystemSimulationConfig, run_system_simulation


@dataclass(frozen=True)
class SensitivityScenario:
    """Single point sensitivity around the best optimisation result."""

    name: str
    fuel_price_multiplier: float = 1.0
    fuel_price_override: float | None = None
    discount_rate: float | None = None
    load_growth_rate: float = 0.0  # Interpreted as total proportional change in load


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "scenario"


def _load_best_result(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Best result not found at {path}. Run optimisation before the sensitivity analysis."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _prepare_dispatch_config(best_result: dict[str, float], scenario: SensitivityScenario):
    base_dispatch = base_config.SYSTEM_SIMULATION_CONFIG.dispatch
    storage_cfg = replace(
        base_dispatch.storage,
        energy_capacity_kwh=float(best_result.get("storage_energy_kwh", 0.0)),
        charge_power_kw=float(best_result.get("storage_power_kw", 0.0)),
        discharge_power_kw=float(best_result.get("storage_power_kw", 0.0)),
    )

    base_generator = base_dispatch.generator
    generator_capacity = float(best_result.get("generator_capacity_kw", base_generator.max_power_kw))
    generator_min_fraction = base_config.OPTIMISATION_SEARCH_SPACE.generator_min_power_fraction

    if scenario.fuel_price_override is not None:
        fuel_price = scenario.fuel_price_override
    else:
        fuel_price = base_generator.fuel_cost_per_litre * scenario.fuel_price_multiplier

    generator_cfg = replace(
        base_generator,
        max_power_kw=generator_capacity,
        min_power_kw=generator_capacity * generator_min_fraction,
        fuel_cost_per_litre=fuel_price,
    )

    return replace(
        base_dispatch,
        pv_capacity_kw=float(best_result.get("pv_capacity_kw", base_dispatch.pv_capacity_kw)),
        wind_turbine_count=int(best_result.get("wind_turbine_count", base_dispatch.wind_turbine_count)),
        storage=storage_cfg,
        generator=generator_cfg,
    ), fuel_price


def _prepare_simulation_config(
    dispatch_cfg,
    scenario: SensitivityScenario,
    scenario_dir: Path,
    base_load_df: pd.DataFrame,
) -> SystemSimulationConfig:
    base_inputs = base_config.SYSTEM_SIMULATION_CONFIG.inputs
    load_column = base_config.LOAD_PROFILE_CONFIG.load_column_name

    if abs(scenario.load_growth_rate) < 1e-9:
        load_path = base_inputs.load_profile_path
    else:
        adjusted = base_load_df.copy()
        adjusted[load_column] *= 1.0 + scenario.load_growth_rate
        load_path = scenario_dir / "load_profile_hourly.csv"
        adjusted.to_csv(
            load_path,
            index=False,
            float_format=base_config.LOAD_PROFILE_CONFIG.float_format,
        )

    scenario_inputs = replace(
        base_inputs,
        load_profile_path=load_path,
        output_timeseries_path=scenario_dir / "dispatch_timeseries.csv",
        output_summary_path=scenario_dir / "dispatch_summary.json",
    )

    return replace(base_config.SYSTEM_SIMULATION_CONFIG, inputs=scenario_inputs, dispatch=dispatch_cfg)


def _prepare_cost_config(cost_config: CostModelConfig, scenario: SensitivityScenario, scenario_dir: Path) -> CostModelConfig:
    discount_rate = scenario.discount_rate
    if discount_rate is None:
        discount_rate = cost_config.discount_rate

    return replace(
        cost_config,
        discount_rate=discount_rate,
        write_output_path=scenario_dir / "cost_summary.json",
    )


def _build_npv_trajectory(
    cost_summary: dict[str, float],
    cost_cfg: CostModelConfig,
) -> List[Tuple[int, float]]:
    lifetime = cost_cfg.lifetime_years
    discount_rate = cost_cfg.discount_rate
    initial_capex = float(cost_summary.get("initial_capex", 0.0))
    annual_cost = float(cost_summary.get("annual_cost", 0.0))
    storage_capex = float(cost_summary.get("storage_capex", 0.0))
    replacement_year = cost_cfg.storage_replacement_year
    replacement_fraction = cost_cfg.storage_replacement_fraction

    trajectory: List[Tuple[int, float]] = []
    cumulative = initial_capex
    trajectory.append((0, cumulative))

    discount_factor = 1.0
    for year in range(1, lifetime + 1):
        discount_factor *= 1.0 / (1.0 + discount_rate)
        cumulative += annual_cost * discount_factor
        if 0 < replacement_year == year:
            cumulative += storage_capex * replacement_fraction * discount_factor
        trajectory.append((year, cumulative))

    return trajectory


def _run_scenario(
    best_result: dict[str, float],
    scenario: SensitivityScenario,
    scenario_dir: Path,
    base_load_df: pd.DataFrame,
) -> dict[str, float]:
    dispatch_cfg, fuel_price = _prepare_dispatch_config(best_result, scenario)
    sim_cfg = _prepare_simulation_config(dispatch_cfg, scenario, scenario_dir, base_load_df)
    dispatch_summary = run_system_simulation(sim_cfg)

    cost_cfg = _prepare_cost_config(base_config.COST_MODEL_CONFIG, scenario, scenario_dir)
    cost_inputs = CostModelInputs(
        pv_capacity_kw=float(best_result.get("pv_capacity_kw", dispatch_cfg.pv_capacity_kw)),
        wind_capacity_kw=base_config.WIND_TURBINE_RATED_POWER_KW * int(best_result.get("wind_turbine_count", 0)),
        storage_energy_kwh=float(best_result.get("storage_energy_kwh", dispatch_cfg.storage.energy_capacity_kwh)),
        storage_power_kw=float(best_result.get("storage_power_kw", dispatch_cfg.storage.discharge_power_kw)),
        generator_capacity_kw=float(best_result.get("generator_capacity_kw", dispatch_cfg.generator.max_power_kw)),
        load_energy_mwh_per_year=dispatch_summary.get("load_energy_mwh", 0.0),
    )
    cost_summary = evaluate_costs(dispatch_summary, cost_inputs, cost_cfg)

    summary_path = sim_cfg.inputs.output_summary_path
    combined_summary = {**dispatch_summary, **cost_summary, "fuel_cost_per_litre": fuel_price}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(combined_summary, file, indent=2)

    trajectory = _build_npv_trajectory(cost_summary, cost_cfg)
    trajectory_path = scenario_dir / "npv_trajectory.csv"
    pd.DataFrame(trajectory, columns=["year", "cumulative_npv_aud"]).to_csv(trajectory_path, index=False)

    return {
        "scenario": scenario.name,
        "fuel_cost_per_litre": fuel_price,
        "discount_rate": cost_cfg.discount_rate,
        "load_growth_rate": scenario.load_growth_rate,
        "npc_total": cost_summary.get("npc_total", float("nan")),
        "lcoe_per_mwh": cost_summary.get("lcoe_per_mwh", float("nan")),
        "renewable_penetration": dispatch_summary.get("renewable_penetration", float("nan")),
        "generator_energy_mwh": dispatch_summary.get("generator_energy_mwh", float("nan")),
        "fuel_cost": dispatch_summary.get("fuel_cost", float("nan")),
        "trajectory": trajectory,
    }


def run_sensitivity(scenarios: Iterable[SensitivityScenario]) -> List[dict[str, float]]:
    project_root = base_config.PROJECT_ROOT
    sensitivity_dir = project_root / "outputs" / "analysis" / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    best_result_path = base_config.OPTIMISATION_OUTPUTS.best_result_path
    best_result = _load_best_result(best_result_path)
    base_load_df = pd.read_csv(base_config.LOAD_PROFILE_CONFIG.output_hourly_path)

    results: List[dict[str, float]] = []
    trajectory_records: List[dict[str, float]] = []
    for scenario in scenarios:
        scenario_slug = _slugify(scenario.name)
        scenario_dir = sensitivity_dir / scenario_slug
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running sensitivity scenario: {scenario.name}")
        result = _run_scenario(best_result, scenario, scenario_dir, base_load_df)
        result["scenario_slug"] = scenario_slug
        results.append(result)
        for year, value in result.pop("trajectory"):
            trajectory_records.append(
                {
                    "scenario": result["scenario"],
                    "year": year,
                    "cumulative_npv_aud": value,
                }
            )

    results_path = sensitivity_dir / "summary.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    if trajectory_records:
        trajectories_df = pd.DataFrame(trajectory_records)
        trajectories_df.to_csv(sensitivity_dir / "npv_trajectories.csv", index=False)

        plt.switch_backend("Agg")
        plt.figure(figsize=(10, 6))
        for scenario_name, group in trajectories_df.groupby("scenario"):
            plt.plot(group["year"], group["cumulative_npv_aud"] / 1e6, label=scenario_name)

        plt.xlabel("Year")
        plt.ylabel("Cumulative NPV (million AUD)")
        plt.title("Cumulative Net Present Cost Trajectories")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(sensitivity_dir / "npv_trajectories.png", dpi=150)
        plt.close()

        # Scenario-specific plots with annotations
        for scenario_name, group in trajectories_df.groupby("scenario"):
            scenario_slug = _slugify(scenario_name)
            plt.figure(figsize=(8, 5))
            plt.plot(group["year"], group["cumulative_npv_aud"] / 1e6, color="#1f77b4", linewidth=2)
            final_year = int(group["year"].iloc[-1])
            final_value = group["cumulative_npv_aud"].iloc[-1] / 1e6
            plt.scatter([final_year], [final_value], color="#d62728")
            plt.annotate(
                f"{final_value:,.2f} M AUD",
                xy=(final_year, final_value),
                xytext=(final_year - max(2, final_year // 10), final_value * 0.95 if final_value > 0 else final_value * 1.05),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=10,
            )
            plt.xlabel("Year")
            plt.ylabel("Cumulative NPV (million AUD)")
            plt.title(f"Cumulative NPV Trajectory – {scenario_name}")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(sensitivity_dir / f"npv_trajectory_{scenario_slug}.png", dpi=150)
            plt.close()

        # Final NPC summary bar chart
        summary_df = pd.DataFrame(results)
        summary_df["npc_million"] = summary_df["npc_total"] / 1e6
        summary_df = summary_df.sort_values("npc_million")

        plt.figure(figsize=(9, 5))
        bars = plt.barh(summary_df["scenario"], summary_df["npc_million"], color="#6baed6")
        for bar, value in zip(bars, summary_df["npc_million"]):
            offset = max(1.0, 0.01 * abs(value))
            x_pos = value + offset if value >= 0 else value - offset
            plt.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.1f} M",
                va="center",
                ha="left" if value >= 0 else "right",
            )
        plt.xlabel("Net Present Cost (million AUD)")
        plt.title("Scenario Comparison – Net Present Cost")
        plt.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(sensitivity_dir / "npc_sensitivity_bar.png", dpi=150)
        plt.close()

    print("\nSensitivity summary (NPC in million AUD):")
    for record in results:
        npc_million = record["npc_total"] / 1e6 if record["npc_total"] == record["npc_total"] else float("nan")
        load_growth_pct = record["load_growth_rate"] * 100
        print(
            f"- {record['scenario']}: NPC={npc_million:,.2f} M AUD, "
            f"fuel={record['fuel_cost_per_litre']:.2f} AUD/L, "
            f"discount={record['discount_rate']:.3f}, load growth={load_growth_pct:+.1f}%"
        )

    return results


def main() -> None:
    scenarios = [
        SensitivityScenario(name="Base case"),
        SensitivityScenario(name="Fuel price +20%", fuel_price_multiplier=1.2),
        SensitivityScenario(name="Fuel price -20%", fuel_price_multiplier=0.8),
        SensitivityScenario(name="Load growth +10%", load_growth_rate=0.10),
        SensitivityScenario(name="Load growth -10%", load_growth_rate=-0.10),
    ]

    run_sensitivity(scenarios)


if __name__ == "__main__":
    main()
