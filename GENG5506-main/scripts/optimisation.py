"""Brute-force optimisation over PV, wind, storage, and generator sizes."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
import math
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import Index

import numpy as np
import os
import multiprocessing as mp

from .cost_model import CostModelConfig, CostModelInputs, evaluate_costs
from .system_simulation import (
    SystemSimulationConfig,
    run_system_simulation,
    load_inputs_dataframe,
    run_dispatch_with_loaded,
)


@dataclass(frozen=True)
class FloatRange:
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class IntRange:
    minimum: int
    maximum: int
    step: int


@dataclass(frozen=True)
class OptimisationSearchSpace:
    pv_capacity_kw: FloatRange
    wind_turbine_count: IntRange
    storage_energy_kwh: FloatRange
    storage_power_kw: FloatRange
    generator_capacity_kw: FloatRange
    generator_min_power_fraction: float
    # PHES search ranges (similar to BESS)
    phes_energy_kwh: FloatRange | None = None
    phes_power_kw: FloatRange | None = None


@dataclass(frozen=True)
class OptimisationConstraints:
    min_renewable_penetration: float
    max_unserved_energy_mwh: float


@dataclass(frozen=True)
class OptimisationOutputs:
    results_csv_path: Path
    best_result_path: Path
    best_cost_path: Path
    cost_candidate_path: Path
    component_summary_path: Path


@dataclass(frozen=True)
class OptimisationConfig:
    base_simulation_config: SystemSimulationConfig
    base_cost_config: CostModelConfig
    search_space: OptimisationSearchSpace
    constraints: OptimisationConstraints
    turbine_rated_kw: float
    outputs: OptimisationOutputs
    bess_unit_energy_mwh: float
    bess_unit_power_mw: float
    bess_unit_name: str
    generator_unit_capacity_kw: float
    generator_unit_name: str


def _combine_dicts(capacities: Dict[str, float], dispatch: Dict[str, float], cost: Dict[str, float]) -> Dict[str, float]:
    combined = {**capacities}
    combined.update(dispatch)
    combined.update(cost)
    return combined


def _float_values(range_cfg: FloatRange) -> Iterable[float]:
    values = []
    current = range_cfg.minimum
    while current <= range_cfg.maximum + 1e-9:
        values.append(round(current, 6))
        current += range_cfg.step
    return values


def _int_values(range_cfg: IntRange) -> Iterable[int]:
    return range(range_cfg.minimum, range_cfg.maximum + range_cfg.step, range_cfg.step)


# ------------ Parallel evaluation helpers ------------
_DF_INPUTS = None  # type: ignore[var-annotated]


def _init_pool(inputs: SimulationInputsConfig) -> None:  # type: ignore[name-defined]
    """Worker initializer: load inputs once per process."""
    global _DF_INPUTS
    _DF_INPUTS = load_inputs_dataframe(inputs)


def _evaluate_candidate(args: tuple) -> Dict[str, float]:
    (
        pv_kw,
        wind_count,
        storage_energy,
        storage_power,
        generator_capacity,
        phes_energy,
        phes_power,
        base_sim_config,
        base_cost_config,
        constraints,
        turbine_rated_kw,
        generator_min_power_fraction,
    ) = args

    base_dispatch = base_sim_config.dispatch
    storage_cfg = replace(
        base_dispatch.storage,
        energy_capacity_kwh=storage_energy,
        charge_power_kw=storage_power,
        discharge_power_kw=storage_power,
    )
    base_phes = base_dispatch.pumped_hydro if base_dispatch.pumped_hydro is not None else base_dispatch.storage
    phes_cfg = None
    if phes_energy > 0 and phes_power > 0:
        phes_cfg = replace(
            base_phes,
            energy_capacity_kwh=phes_energy,
            charge_power_kw=phes_power,
            discharge_power_kw=phes_power,
        )
    generator_min_power = generator_capacity * generator_min_power_fraction
    generator_cfg = replace(
        base_dispatch.generator,
        max_power_kw=generator_capacity,
        min_power_kw=generator_min_power,
    )

    dispatch_cfg = replace(
        base_dispatch,
        pv_capacity_kw=pv_kw,
        wind_turbine_count=wind_count,
        storage=storage_cfg,
        generator=generator_cfg,
        pumped_hydro=phes_cfg,
    )

    # Use preloaded inputs stored in this worker
    summary = run_dispatch_with_loaded(_DF_INPUTS, dispatch_cfg)  # type: ignore[arg-type]

    cost_inputs = CostModelInputs(
        pv_capacity_kw=pv_kw,
        wind_capacity_kw=turbine_rated_kw * wind_count,
        storage_energy_kwh=storage_energy,
        storage_power_kw=storage_power,
        generator_capacity_kw=generator_capacity,
        load_energy_mwh_per_year=summary.get("load_energy_mwh", 0.0),
        phes_energy_kwh=phes_energy,
        phes_power_kw=phes_power,
    )

    # Avoid file I/O contention: write costs to OS null device
    temp_cost_config = replace(base_cost_config, write_output_path=Path(os.devnull))
    cost_summary = evaluate_costs(summary, cost_inputs, temp_cost_config)

    capacities = {
        "pv_capacity_kw": pv_kw,
        "wind_turbine_count": wind_count,
        "storage_energy_kwh": storage_energy,
        "storage_power_kw": storage_power,
        "phes_energy_kwh": phes_energy,
        "phes_power_kw": phes_power,
        "generator_capacity_kw": generator_capacity,
    }
    combined = _combine_dicts(capacities, summary, cost_summary)

    renewable_penetration = combined.get("renewable_penetration", 0.0)
    unserved_energy = combined.get("unserved_energy_mwh", float("inf"))
    feasible = (
        renewable_penetration >= constraints.min_renewable_penetration
        and unserved_energy <= constraints.max_unserved_energy_mwh
    )
    combined["feasible"] = bool(feasible)
    return combined


def _create_penetration_cost_plot(
    results: list[Dict[str, float]],
    best_result: Dict[str, float] | None,
    constraints: OptimisationConstraints,
) -> None:
    if not results or best_result is None:
        return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.DataFrame(results)
    required_columns = {"renewable_penetration", "npc_total"}
    if not required_columns.issubset(df.columns):
        return

    df = df.dropna(subset=required_columns)
    if df.empty:
        return

    df = df.copy()
    if "feasible" not in df.columns:
        df["feasible"] = True
    if "feasible" not in df.columns:
        df["feasible"] = True
    df["penetration_pct"] = df["renewable_penetration"] * 100
    df["npc_million"] = df["npc_total"] / 1e6

    best_pen_pct = best_result.get("renewable_penetration", 0.0) * 100
    best_cost_million = best_result.get("npc_total", 0.0) / 1e6

    plot_path = Path("images/optimisation/penetration_vs_cost.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    feasible_df = df[df["feasible"]]
    infeasible_df = df[~df["feasible"]]

    if not infeasible_df.empty:
        ax.scatter(
            infeasible_df["penetration_pct"],
            infeasible_df["npc_million"],
            color="#d3d3d3",
            edgecolor="#9e9e9e",
            linewidth=0.5,
            alpha=0.6,
            s=45,
            label="Infeasible candidates",
        )

        if "unserved_energy_mwh" in infeasible_df.columns:
            load_not_served_df = infeasible_df[infeasible_df["unserved_energy_mwh"] > constraints.max_unserved_energy_mwh]
            if load_not_served_df.empty:
                load_not_served_df = infeasible_df
            if len(load_not_served_df) >= 2:
                x_vals = load_not_served_df["penetration_pct"].to_numpy()
                y_vals = load_not_served_df["npc_million"].to_numpy()
                degree = 2 if len(load_not_served_df) >= 3 else 1
                coeffs = np.polyfit(x_vals, y_vals, degree)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                y_line = np.polyval(coeffs, x_line)
                ax.plot(
                    x_line,
                    y_line,
                    linestyle="--",
                    linewidth=3.5,
                    color="#1f77b4",
                    solid_capstyle="round",
                    label="Unserved-energy cutoff",
                )

    if not feasible_df.empty:
        ax.scatter(
            feasible_df["penetration_pct"],
            feasible_df["npc_million"],
            color="#6baed6",
            edgecolor="#ffffff",
            linewidth=0.6,
            alpha=0.85,
            s=65,
            label="Feasible candidates",
        )

    ax.scatter(
        [best_pen_pct],
        [best_cost_million],
        color="#d62728",
        edgecolor="#000000",
        linewidth=0.8,
        s=120,
        zorder=5,
        label="Lowest-cost design",
    )

    ax.axvline(
        constraints.min_renewable_penetration * 100,
        color="#c71f1f",
        linestyle="--",
        linewidth=2.8,
        label=f"Penetration target ({constraints.min_renewable_penetration*100:.0f}%)",
    )

    ax.set_xlabel("Renewable penetration (%)")
    ax.set_ylabel("Net present cost (million AUD)")
    ax.set_title("Optimisation Results: Cost vs Renewable Penetration")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _create_design_variable_plots(
    results: list[Dict[str, float]],
    best_result: Dict[str, float] | None,
    constraints: OptimisationConstraints,
) -> None:
    if not results or best_result is None:
        return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.lines import Line2D
    except ImportError:
        return

    df = pd.DataFrame(results)
    required = {
        "npc_total",
        "renewable_penetration",
        "pv_capacity_kw",
        "wind_turbine_count",
        "storage_energy_kwh",
        "generator_capacity_kw",
    }
    if not required.issubset(df.columns):
        return

    df = df.dropna(subset=required)
    if df.empty:
        return

    df = df.copy()
    if "feasible" not in df.columns:
        df["feasible"] = True
    df["penetration_pct"] = df["renewable_penetration"] * 100
    df["npc_million"] = df["npc_total"] / 1e6
    df["pv_mw"] = df["pv_capacity_kw"] / 1000
    df["storage_mwh"] = df["storage_energy_kwh"] / 1000
    df["generator_mw"] = df["generator_capacity_kw"] / 1000

    best_pen_pct = best_result.get("renewable_penetration", 0.0) * 100
    best_cost_million = best_result.get("npc_total", 0.0) / 1e6
    best_pv_mw = best_result.get("pv_capacity_kw", 0.0) / 1000
    best_storage_mwh = best_result.get("storage_energy_kwh", 0.0) / 1000
    best_generator_mw = best_result.get("generator_capacity_kw", 0.0) / 1000
    best_wind = int(round(best_result.get("wind_turbine_count", 0)))

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=df["pv_mw"].min(), vmax=df["pv_mw"].max())

    storage = df["storage_mwh"].to_numpy()
    min_storage = np.min(storage) if storage.size else 0.0
    max_storage = np.max(storage) if storage.size else 0.0
    if max_storage - min_storage <= 0:
        df["marker_size"] = np.full_like(storage, 120, dtype=float)
    else:
        normed_storage = (storage - min_storage) / (max_storage - min_storage)
        df["marker_size"] = 40 + normed_storage * 200

    marker_cycle = ["o", "s", "^", "D", "P", "X"]
    df["wind_label"] = df["wind_turbine_count"].astype(int).astype(str) + " turbines"
    unique_wind_labels = list(dict.fromkeys(df["wind_label"].tolist()))
    marker_map = {label: marker_cycle[i % len(marker_cycle)] for i, label in enumerate(unique_wind_labels)}

    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    cutoff_handle: Line2D | None = None
    if "unserved_energy_mwh" in df.columns:
        load_not_served_df = df[(~df["feasible"]) & (df["unserved_energy_mwh"] > constraints.max_unserved_energy_mwh)]
        if load_not_served_df.empty:
            load_not_served_df = df[~df["feasible"]]
        if len(load_not_served_df) >= 2:
            x_vals = load_not_served_df["penetration_pct"].to_numpy()
            y_vals = load_not_served_df["npc_million"].to_numpy()
            degree = 2 if len(load_not_served_df) >= 3 else 1
            coeffs = np.polyfit(x_vals, y_vals, degree)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=3.5,
                color="#1f77b4",
                solid_capstyle="round",
                zorder=4,
            )
            cutoff_handle = Line2D([], [], linestyle="--", linewidth=3.5, color="#1f77b4", label="Unserved-energy cutoff")

    infeasible_handle: Line2D | None = None
    has_feasible_points = False

    for wind_label in unique_wind_labels:
        subset = df[df["wind_label"] == wind_label]
        feasible_subset = subset[subset["feasible"]]
        infeasible_subset = subset[~subset["feasible"]]

        if not infeasible_subset.empty:
            ax.scatter(
                infeasible_subset["penetration_pct"],
                infeasible_subset["npc_million"],
                c="#bcbcbc",
                s=infeasible_subset["marker_size"],
                alpha=0.55,
                marker=marker_map[wind_label],
                edgecolor="#999999",
                linewidth=0.4,
                label=None,
                zorder=2,
            )
            if infeasible_handle is None:
                infeasible_handle = Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#bcbcbc", markeredgecolor="#999999", label="Infeasible candidate")

        if not feasible_subset.empty:
            has_feasible_points = True
            ax.scatter(
                feasible_subset["penetration_pct"],
                feasible_subset["npc_million"],
                c=feasible_subset["pv_mw"],
                cmap=cmap,
                norm=norm,
                s=feasible_subset["marker_size"],
                alpha=0.85,
                marker=marker_map[wind_label],
                edgecolor="#ffffff",
                linewidth=0.55,
                label=wind_label,
                zorder=3,
            )

    ax.axvline(
        constraints.min_renewable_penetration * 100,
        color="#c71f1f",
        linestyle="--",
        linewidth=2.8,
        alpha=0.8,
    )

    ax.scatter(
        best_pen_pct,
        best_cost_million,
        s=260,
        facecolors="none",
        edgecolors="red",
        linewidth=2.5,
        zorder=6,
    )

    ax.set_xlabel("Renewable penetration (%)")
    ax.set_ylabel("Net present cost (million AUD)")
    ax.set_title("Optimisation Landscape: Cost vs Renewable Penetration")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(left=70)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.018)
    cbar.set_label("PV capacity (MW)")

    wind_handles = [
        Line2D([], [], marker=marker_map[label], linestyle="", markersize=8, color="#555555", label=label)
        for label in unique_wind_labels
    ]
    best_handle = Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="none", markeredgecolor="red", linewidth=2, label="Lowest-cost design")
    target_handle = Line2D([], [], linestyle="--", color="#c71f1f", linewidth=1.2, label=f"Penetration target ({constraints.min_renewable_penetration*100:.0f}%)")

    legend_entries: list[Line2D] = []
    legend_labels: list[str] = []

    # Wind turbine section
    legend_entries.append(Line2D([], [], linestyle="", marker="", label=r"$\mathbf{Wind\ turbines}$"))
    legend_labels.append(r"$\mathbf{Wind\ turbines}$")
    legend_entries.extend(wind_handles)
    legend_labels.extend([h.get_label() for h in wind_handles])

    # Storage section (marker size)
    legend_entries.append(Line2D([], [], linestyle="", marker="", label=r"$\mathbf{Storage\ energy}$"))
    legend_labels.append(r"$\mathbf{Storage\ energy}$")
    size_handles: list[Line2D] = []
    if max_storage - min_storage > 0:
        storage_values = [min_storage, (min_storage + max_storage) / 2, max_storage]
        legend_marker_sizes = [40, 120, 240]
        for val, msize in zip(storage_values, legend_marker_sizes):
            size_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=(msize ** 0.5),
                    markerfacecolor="#ffffff",
                    markeredgecolor="#555555",
                    label=f"≈ {val:.0f} MWh",
                )
            )
        legend_entries.extend(size_handles)
        legend_labels.extend([h.get_label() for h in size_handles])

    # Highlights section
    legend_entries.append(Line2D([], [], linestyle="", marker="", label=r"$\mathbf{Highlights}$"))
    legend_labels.append(r"$\mathbf{Highlights}$")
    highlight_handles = [best_handle, target_handle]
    if cutoff_handle is not None:
        highlight_handles.append(cutoff_handle)
    legend_entries.extend(highlight_handles)
    legend_labels.extend([h.get_label() for h in highlight_handles])

    if infeasible_handle is not None:
        legend_entries.append(Line2D([], [], linestyle="", marker="", label=r"$\mathbf{Feasibility}$"))
        legend_labels.append(r"$\mathbf{Feasibility}$")
        legend_entries.append(infeasible_handle)
        legend_labels.append(infeasible_handle.get_label())

    ax.legend(legend_entries, legend_labels, loc="upper left", frameon=True, fontsize=9)

    output_path = Path("images/optimisation/design_variables_vs_cost.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_capacity_plane_plot(
    results: list[Dict[str, float]],
    best_result: Dict[str, float] | None,
) -> None:
    if not results or best_result is None:
        return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        return

    df = pd.DataFrame(results)
    required = {
        "pv_capacity_kw",
        "wind_turbine_count",
        "npc_total",
        "renewable_penetration",
    }
    if not required.issubset(df.columns):
        return

    df = df.dropna(subset=required)
    if df.empty:
        return

    df = df.copy()
    if "feasible" not in df.columns:
        df["feasible"] = True
    df["pv_mw"] = df["pv_capacity_kw"] / 1000
    df["npc_million"] = df["npc_total"] / 1e6
    df["penetration_pct"] = df["renewable_penetration"] * 100

    best_pv_mw = best_result.get("pv_capacity_kw", 0.0) / 1000
    best_wind = best_result.get("wind_turbine_count", 0.0)
    best_cost_million = best_result.get("npc_total", 0.0) / 1e6
    best_pen_pct = best_result.get("renewable_penetration", 0.0) * 100

    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=df["npc_million"].min(), vmax=df["npc_million"].max())

    fig, ax = plt.subplots(figsize=(9, 6.5))

    infeasible_df = df[~df["feasible"]]
    feasible_df = df[df["feasible"]]

    scatter = None
    if not infeasible_df.empty:
        ax.scatter(
            infeasible_df["pv_mw"],
            infeasible_df["wind_turbine_count"],
            c="#cfcfcf",
            s=80,
            alpha=0.5,
            edgecolor="#9a9a9a",
            linewidth=0.4,
            zorder=2,
        )

    if not feasible_df.empty:
        scatter = ax.scatter(
            feasible_df["pv_mw"],
            feasible_df["wind_turbine_count"],
            c=feasible_df["npc_million"],
            cmap=cmap,
            norm=norm,
            s=90,
            alpha=0.85,
            edgecolor="#ffffff",
            linewidth=0.5,
            zorder=3,
        )

    ax.scatter(
        best_pv_mw,
        best_wind,
        s=220,
        facecolors="none",
        edgecolors="red",
        linewidth=2.3,
        zorder=6,
    )

    # Place annotation above the point so it is not obscured by nearby markers
    ax.annotate(
        f"NPC {best_cost_million:.2f} M\nPen {best_pen_pct:.1f}%",
        xy=(best_pv_mw, best_wind),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": "#ffbf00"},
    )

    ax.set_xlabel("PV capacity (MW)")
    ax.set_ylabel("Wind turbines (count)")
    ax.set_title("Optimisation Candidates: PV vs Wind Fleet")
    ax.grid(True, linestyle="--", alpha=0.3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.018)
    cbar.set_label("Net present cost (million AUD)")

    handles: list[Line2D] = []
    labels: list[str] = []
    from matplotlib.lines import Line2D

    if not feasible_df.empty:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=cmap(0.7), markeredgecolor="#ffffff", label="Feasible candidate"))
    if not infeasible_df.empty:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#cfcfcf", markeredgecolor="#9a9a9a", label="Infeasible candidate"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="none", markeredgecolor="red", linewidth=2.3, label="Lowest-cost design"))
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=True)

    ax.text(
        0.03,
        0.94,
        "Colour ∝ NPC, red ring = lowest cost",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    output_path = Path("images/optimisation/pv_vs_wind.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_cashflow_plot(
    dispatch_summary: Dict[str, float],
    cost_summary: Dict[str, float],
    cost_config: CostModelConfig,
    capacities: Dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    lifetime = cost_config.lifetime_years
    rate = cost_config.discount_rate

    pv_capacity_kw = capacities["pv_capacity_kw"]
    wind_capacity_kw = capacities["wind_capacity_kw"]
    storage_energy_kwh = capacities["storage_energy_kwh"]
    storage_power_kw = capacities["storage_power_kw"]
    generator_capacity_kw = capacities["generator_capacity_kw"]

    annual_pv_om = pv_capacity_kw * cost_config.pv_costs.fixed_om_per_kw_year
    annual_wind_om = wind_capacity_kw * cost_config.wind_costs.fixed_om_per_kw_year
    annual_storage_om = storage_energy_kwh * cost_config.storage_fixed_om_per_kwh_year
    annual_generator_om = generator_capacity_kw * cost_config.generator_fixed_om_per_kw_year
    annual_fuel = dispatch_summary.get("fuel_cost", 0.0)
    annual_gen_var_om = dispatch_summary.get("generator_variable_om_cost", 0.0)

    storage_replacement_year = cost_config.storage_replacement_year
    storage_replacement_cost = (
        cost_summary["storage_capex"] * cost_config.storage_replacement_fraction
        if 0 < storage_replacement_year < lifetime
        else 0.0
    )

    categories = [
        "PV Capex",
        "Wind Capex",
        "Storage Capex",
        "Generator Capex",
        "PV O&M",
        "Wind O&M",
        "Storage O&M",
        "Generator O&M",
        "Fuel",
        "Generator Var O&M",
        "Storage Replacement",
    ]
    data = {cat: [0.0] * (lifetime + 1) for cat in categories}
    data["PV Capex"][0] = -cost_summary["pv_capex"]
    data["Wind Capex"][0] = -cost_summary["wind_capex"]
    data["Storage Capex"][0] = -cost_summary["storage_capex"]
    data["Generator Capex"][0] = -cost_summary["generator_capex"]

    for year in range(1, lifetime + 1):
        data["PV O&M"][year] = -annual_pv_om
        data["Wind O&M"][year] = -annual_wind_om
        data["Storage O&M"][year] = -annual_storage_om
        data["Generator O&M"][year] = -annual_generator_om
        data["Fuel"][year] = -annual_fuel
        data["Generator Var O&M"][year] = -annual_gen_var_om
        if year == storage_replacement_year:
            data["Storage Replacement"][year] = -storage_replacement_cost

    df = pd.DataFrame(data, index=range(0, lifetime + 1))
    df.index.name = "Year"
    df["Total"] = df.sum(axis=1)
    df["Discount Factor"] = [(1 + rate) ** (-year) for year in df.index]
    df["Discounted Total"] = df["Total"] * df["Discount Factor"]
    df["Cumulative NPV"] = df["Discounted Total"].cumsum()

    colours = {
        "PV Capex": "#1f77b4",
        "Wind Capex": "#ff7f0e",
        "Storage Capex": "#2ca02c",
        "Generator Capex": "#d62728",
        "PV O&M": "#6baed6",
        "Wind O&M": "#fd8d3c",
        "Storage O&M": "#74c476",
        "Generator O&M": "#e377c2",
        "Fuel": "#9467bd",
        "Generator Var O&M": "#bcbd22",
        "Storage Replacement": "#8c564b",
    }
    cat_list = [c for c in categories if df[c].abs().sum() > 0]
    values = -df[cat_list] / 1e6

    output_path = Path("images/finance/cashflow_breakdown.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    values.plot(kind="bar", stacked=True, color=[colours[c] for c in cat_list], ax=ax)
    ax.set_ylabel("Annual Costs (Million AUD)")
    ax.set_xlabel("Year")
    ax.set_title("Annual Cost Breakdown & Cumulative NPV")

    ax2 = ax.twinx()
    ax2.plot(df.index, df["Cumulative NPV"] / 1e6, color="black", marker="o", label="Cumulative NPV")
    ax2.set_ylabel("Cumulative NPV (Million AUD)")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_representative_day_plot(dispatch_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not dispatch_path.exists():
        return
    df = pd.read_csv(dispatch_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.empty:
        return

    hours_in_day = 24

    def _find_day_slice(months: tuple[int, ...]) -> pd.DataFrame | None:
        seasonal = df[df.index.month.isin(months)]
        if seasonal.empty:
            return None

        seasonal_sorted = seasonal.sort_index()
        # Select the second day of the month
        for date, group in seasonal_sorted.groupby(seasonal_sorted.index.normalize()):
            if date.day != 2:
                continue
            if len(group) < hours_in_day:
                continue
            window = group.iloc[:hours_in_day]
            diffs = window.index.to_series().diff().dropna()
            if (diffs == pd.Timedelta(hours=1)).all():
                return window
        return None

    season_defs = [
        ("Representative Summer Day", (12, 1, 2)),
        ("Representative Winter Day", (6, 7, 8)),
    ]

    day_slices: list[tuple[str, pd.DataFrame]] = []
    for title, months in season_defs:
        slice_df = _find_day_slice(months)
        if slice_df is not None:
            day_slices.append((title, slice_df))

    if not day_slices:
        return

    fig, axes = plt.subplots(
        nrows=len(day_slices),
        figsize=(12, 5 * len(day_slices)),
        sharey=True,
    )
    if len(day_slices) == 1:
        axes = [axes]

    for ax, (title, day_slice) in zip(axes, day_slices):
        x_index = day_slice.index
        load = day_slice["load_kw"] / 1000.0
        pv = day_slice["pv_kw"] / 1000.0
        wind = day_slice["wind_kw"] / 1000.0
        generator = day_slice["generator_kw"] / 1000.0
        # Compute BESS net flow if BESS columns exist; otherwise treat as unused
        has_bess_cols = {"bess_discharge_kw", "bess_charge_kw"}.issubset(day_slice.columns)
        if has_bess_cols:
            bess_net = (day_slice["bess_discharge_kw"] - day_slice["bess_charge_kw"]) / 1000.0
            bess_used = (day_slice["bess_discharge_kw"].abs().max() > 1e-6) or (
                day_slice["bess_charge_kw"].abs().max() > 1e-6
            )
        else:
            bess_net = None  # type: ignore[assignment]
            bess_used = False

        ax.plot(x_index, load.values, label="Load", linewidth=2, color="black")
        ax.plot(x_index, pv.values, label="PV", color="gold")
        ax.plot(x_index, wind.values, label="Wind", color="skyblue")
        ax.plot(x_index, generator.values, label="Generator", color="crimson")

        if bess_used and bess_net is not None:
            bess_discharge = bess_net.where(bess_net >= 0, other=np.nan)
            bess_charge = bess_net.where(bess_net < 0, other=np.nan)
            ax.fill_between(
                x_index,
                0,
                bess_discharge,
                color="green",
                alpha=0.3,
                label="BESS Discharging",
            )
            ax.fill_between(
                x_index,
                0,
                bess_charge,
                color="orange",
                alpha=0.3,
                label="BESS Charging",
            )

        ax.set_ylabel("Power (MW)")
        first_timestamp = x_index[0]
        ax.set_title(f"{title} ({first_timestamp.strftime('%d-%b')})")
        tick_positions = x_index[::2]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([ts.strftime("%H:%M") for ts in tick_positions], rotation=45)
        ax.grid(True, linestyle="--", alpha=0.4)

        daylight_mask = pv.values > 0.01
        band_handles = _draw_daylight_band(ax, x_index, daylight_mask)

        handles, labels = ax.get_legend_handles_labels()
        if band_handles:
            handles.extend(band_handles)
            labels.extend([handle.get_label() for handle in band_handles])
        ax.legend(handles, labels, loc="upper left")

    axes[-1].set_xlabel("Hour of Day")

    output_path = Path("images/finance/representative_day.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_week_profile_plot(dispatch_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not dispatch_path.exists():
        return

    df = pd.read_csv(dispatch_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.empty:
        return

    hours_in_week = 7 * 24

    def _find_week_slice(months: tuple[int, ...]) -> pd.DataFrame | None:
        seasonal = df[df.index.month.isin(months)]
        if seasonal.empty or len(seasonal) < hours_in_week:
            return None

        seasonal = seasonal.sort_index()
        # Find all possible week windows
        week_windows = []
        for start_idx in range(0, len(seasonal) - hours_in_week + 1):
            window = seasonal.iloc[start_idx : start_idx + hours_in_week]
            if window.empty:
                continue
            diffs = window.index.to_series().diff().dropna()
            if (diffs == pd.Timedelta(hours=1)).all():
                week_windows.append(window)
        # Select the second week window if available
        if len(week_windows) >= 2:
            return week_windows[1]
        elif week_windows:
            return week_windows[0]
        return None

    season_defs = [
        ("Representative Summer Week", (12, 1, 2)),
        ("Representative Winter Week", (6, 7, 8)),
    ]

    week_slices: list[tuple[str, pd.DataFrame]] = []
    for title, months in season_defs:
        slice_df = _find_week_slice(months)
        if slice_df is not None:
            week_slices.append((title, slice_df))

    if not week_slices:
        return

    fig, axes = plt.subplots(
        nrows=len(week_slices),
        figsize=(14, 6 * len(week_slices)),
        sharey=True,
    )
    if len(week_slices) == 1:
        axes = [axes]

    for ax, (title, week_slice) in zip(axes, week_slices):
        load = week_slice["load_kw"] / 1000.0
        pv = week_slice["pv_kw"] / 1000.0
        wind = week_slice["wind_kw"] / 1000.0
        generator = week_slice["generator_kw"] / 1000.0
        has_bess_cols = {"bess_discharge_kw", "bess_charge_kw"}.issubset(week_slice.columns)
        if has_bess_cols:
            bess_net = (week_slice["bess_discharge_kw"] - week_slice["bess_charge_kw"]) / 1000.0
            bess_used = (week_slice["bess_discharge_kw"].abs().max() > 1e-6) or (
                week_slice["bess_charge_kw"].abs().max() > 1e-6
            )
        else:
            bess_net = None  # type: ignore[assignment]
            bess_used = False

        ax.plot(load.index, load.values, label="Load", color="black", linewidth=2)
        ax.plot(pv.index, pv.values, label="PV", color="gold")
        ax.plot(wind.index, wind.values, label="Wind", color="skyblue")
        ax.plot(generator.index, generator.values, label="Generator", color="crimson")
        if bess_used and bess_net is not None:
            ax.fill_between(
                bess_net.index,
                0,
                bess_net.where(bess_net >= 0),
                color="green",
                alpha=0.3,
                label="BESS Discharging",
            )
            ax.fill_between(
                bess_net.index,
                0,
                bess_net.where(bess_net < 0),
                color="orange",
                alpha=0.3,
                label="BESS Charging",
            )
        ax.set_ylabel("Power (MW)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(week_slice.index[::24])
        ax.set_xticklabels([ts.strftime("%d-%b") for ts in week_slice.index[::24]], rotation=45)

        daylight_mask = pv.values > 0.01
        band_handles = _draw_daylight_band(ax, week_slice.index, daylight_mask)

        handles, labels = ax.get_legend_handles_labels()
        if band_handles:
            handles.extend(band_handles)
            labels.extend([handle.get_label() for handle in band_handles])
        ax.legend(handles, labels, loc="upper left", ncol=2)

    axes[-1].set_xlabel("Date")

    fig.tight_layout()
    output_path = Path("images/finance/week_generation_profile.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_daylight_band(
    ax: Axes,
    index: Index,
    daylight_mask: np.ndarray,
    height_frac: float = 0.05,
) -> list:
    import matplotlib.patches as mpatches

    if len(index) != len(daylight_mask):
        return []

    daylight_mask = np.asarray(daylight_mask, dtype=bool)
    if daylight_mask.size == 0:
        return []

    def _mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
        transitions = np.diff(mask.astype(int))
        starts = [0] if mask[0] else []
        starts += list(np.where(transitions == 1)[0] + 1)
        ends = list(np.where(transitions == -1)[0] + 1)
        if mask[-1]:
            ends.append(len(mask))
        return [(s, e) for s, e in zip(starts, ends) if s < e]

    day_intervals = _mask_to_intervals(daylight_mask)
    night_intervals = _mask_to_intervals(~daylight_mask)

    if not day_intervals and not night_intervals:
        return []

    ymin, ymax = ax.get_ylim()
    axis_height = ymax - ymin
    band_height = axis_height * height_frac
    band_bottom = ymin + axis_height * 0.02
    band_top = band_bottom + band_height

    day_color = "#ffcc23"
    night_color = "#002967"
    day_alpha = 0.35
    night_alpha = 0.28

    handles: list = []

    for start, end in day_intervals:
        ax.fill_between(index[start:end], band_bottom, band_top, color=day_color, alpha=day_alpha, step="mid")
    if day_intervals:
        handles.append(mpatches.Patch(facecolor=day_color, alpha=day_alpha, label="Daytime"))

    for start, end in night_intervals:
        ax.fill_between(index[start:end], band_bottom, band_top, color=night_color, alpha=night_alpha, step="mid")
    if night_intervals:
        handles.append(mpatches.Patch(facecolor=night_color, alpha=night_alpha, label="Nighttime"))

    label_y = band_bottom + band_height / 2
    text_style = {
        "va": "center",
        "fontsize": 9,
        "fontweight": "bold",
        "bbox": {"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    }

    if day_intervals:
        day_mid = day_intervals[0][0] + (day_intervals[0][1] - day_intervals[0][0]) // 2
        ax.text(index[day_mid], label_y, "Daytime", color="#b07902", ha="center", **text_style)
    if night_intervals:
        night_mid = night_intervals[0][0] + (night_intervals[0][1] - night_intervals[0][0]) // 2
        ax.text(index[night_mid], label_y, "Nighttime", color="#1f4b7a", ha="center", **text_style)

    ax.set_ylim(ymin, ymax)
    return handles


def _component_summary(
    best_result: Dict[str, float],
    best_cost_summary: Dict[str, float],
    config: OptimisationConfig,
) -> Dict[str, object]:
    pv_capacity_kw = float(best_result.get("pv_capacity_kw", 0.0))
    wind_turbines = int(best_result.get("wind_turbine_count", 0))
    storage_energy_kwh = float(best_result.get("storage_energy_kwh", 0.0))
    storage_power_kw = float(best_result.get("storage_power_kw", 0.0))
    generator_capacity_kw = float(best_result.get("generator_capacity_kw", 0.0))
    phes_energy_kwh = float(best_result.get("phes_energy_kwh", 0.0))
    phes_power_kw = float(best_result.get("phes_power_kw", 0.0))

    pv_params = config.base_cost_config.pv_costs
    panel_power_kw = pv_params.panel_power_kw if pv_params.panel_power_kw else 1.0
    panel_count = pv_capacity_kw / panel_power_kw
    panel_count_int = int(round(panel_count)) if panel_count else 0
    total_area_m2 = panel_count * pv_params.panel_area_m2
    pv_capex = float(best_cost_summary.get("pv_capex", 0.0))
    pv_opex = pv_capacity_kw * pv_params.fixed_om_per_kw_year

    wind_capacity_kw = wind_turbines * config.turbine_rated_kw
    wind_capex = float(best_cost_summary.get("wind_capex", 0.0))
    wind_opex = wind_capacity_kw * config.base_cost_config.wind_costs.fixed_om_per_kw_year

    bess_total_mwh = storage_energy_kwh / 1000.0
    bess_total_mw = storage_power_kw / 1000.0
    if config.bess_unit_energy_mwh > 0:
        units_required_energy = bess_total_mwh / config.bess_unit_energy_mwh
    else:
        units_required_energy = 0.0
    if config.bess_unit_power_mw > 0:
        units_required_power = bess_total_mw / config.bess_unit_power_mw
    else:
        units_required_power = 0.0
    bess_units = int(max(1, math.ceil(max(units_required_energy, units_required_power)))) if bess_total_mwh > 0 else 0
    storage_capex = float(best_cost_summary.get("storage_capex", 0.0))
    storage_opex = storage_energy_kwh * config.base_cost_config.storage_fixed_om_per_kwh_year

    if config.generator_unit_capacity_kw > 0:
        generator_units = int(max(1, math.ceil(generator_capacity_kw / config.generator_unit_capacity_kw))) if generator_capacity_kw > 0 else 0
    else:
        generator_units = 0
    generator_capex = float(best_cost_summary.get("generator_capex", 0.0))
    generator_opex = generator_capacity_kw * config.base_cost_config.generator_fixed_om_per_kw_year

    phes_capex = float(best_cost_summary.get("phes_capex", 0.0))
    phes_opex = phes_energy_kwh * config.base_cost_config.phes_fixed_om_per_kwh_year

    return {
        "pv": {
            "panel_count": panel_count_int,
            "total_area_m2": total_area_m2,
            "capex_aud": pv_capex,
            "annual_opex_aud": pv_opex,
        },
        "wind": {
            "turbine_count": wind_turbines,
            "total_capacity_kw": wind_capacity_kw,
            "capex_aud": wind_capex,
            "annual_opex_aud": wind_opex,
        },
        "battery_storage": {
            "unit_name": config.bess_unit_name,
            "unit_energy_mwh": config.bess_unit_energy_mwh,
            "unit_power_mw": config.bess_unit_power_mw,
            "unit_count": bess_units,
            "total_energy_mwh": bess_total_mwh,
            "total_power_mw": bess_total_mw,
            "capex_aud": storage_capex,
            "annual_opex_aud": storage_opex,
        },
        "pumped_hydro": {
            "total_energy_mwh": phes_energy_kwh / 1000.0,
            "total_power_mw": phes_power_kw / 1000.0,
            "capex_aud": phes_capex,
            "annual_opex_aud": phes_opex,
        },
        "generator": {
            "unit_name": config.generator_unit_name,
            "unit_capacity_kw": config.generator_unit_capacity_kw,
            "unit_count": generator_units,
            "total_capacity_kw": generator_capacity_kw,
            "capex_aud": generator_capex,
            "annual_opex_aud": generator_opex,
        },
    }
def run_optimisation(config: OptimisationConfig) -> Dict[str, float]:
    search = config.search_space
    constraints = config.constraints
    outputs = config.outputs

    results: list[Dict[str, float]] = []
    best_result: Dict[str, float] | None = None
    best_cost_summary: Dict[str, float] | None = None
    best_dispatch_summary: Dict[str, float] | None = None
    best_npc = float("inf")

    pv_values = list(_float_values(search.pv_capacity_kw))
    wind_values = list(_int_values(search.wind_turbine_count))
    storage_energy_values = list(_float_values(search.storage_energy_kwh))
    storage_power_values = list(_float_values(search.storage_power_kw))
    generator_values = list(_float_values(search.generator_capacity_kw))
    phes_energy_values = (
        list(_float_values(search.phes_energy_kwh)) if search.phes_energy_kwh is not None else [0.0]
    )
    phes_power_values = (
        list(_float_values(search.phes_power_kw)) if search.phes_power_kw is not None else [0.0]
    )

    total_runs = (
        len(pv_values)
        * len(wind_values)
        * len(storage_energy_values)
        * len(storage_power_values)
        * len(generator_values)
        * len(phes_energy_values)
        * len(phes_power_values)
    )
    progress_every = max(1, total_runs // 100)
    run_index = 0

    import time

    start_time = time.time()
    first_threshold = max(1, int(total_runs * 0.03))
    time_per_iteration: float | None = None

    # Build tasks for parallel evaluation
    tasks = []
    for pv_kw in pv_values:
        for wind_count in wind_values:
            for storage_energy in storage_energy_values:
                for storage_power in storage_power_values:
                    for generator_capacity in generator_values:
                        for phes_energy in phes_energy_values:
                            for phes_power in phes_power_values:
                                tasks.append(
                                    (
                                        pv_kw,
                                        wind_count,
                                        storage_energy,
                                        storage_power,
                                        generator_capacity,
                                        phes_energy,
                                        phes_power,
                                        config.base_simulation_config,
                                        config.base_cost_config,
                                        constraints,
                                        config.turbine_rated_kw,
                                        search.generator_min_power_fraction,
                                    )
                                )

    # Parallel pool: load inputs once per worker and evaluate
    workers = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(processes=workers, initializer=_init_pool, initargs=(config.base_simulation_config.inputs,)) as pool:
        for run_index, combined in enumerate(pool.imap_unordered(_evaluate_candidate, tasks), start=1):
            results.append(combined)

            if run_index % progress_every == 0 or run_index == 1:
                percent = (run_index / total_runs) * 100
                if run_index >= first_threshold and run_index > 0:
                    elapsed = time.time() - start_time
                    time_per_iteration = elapsed / run_index
                if time_per_iteration is not None:
                    remaining_eta = time_per_iteration * (total_runs - run_index)
                    eta_minutes = max(0.0, remaining_eta / 60)
                    print(
                        f"Optimisation progress: {percent:5.1f}% ({run_index}/{total_runs}) | ETA ~ {eta_minutes:5.1f} min"
                    )
                else:
                    print(f"Optimisation progress: {percent:5.1f}% ({run_index}/{total_runs})")

    # Choose best feasible from collected results
    for combined in results:
        if not combined.get("feasible", False):
            continue
        npc = combined.get("npc_total", float("inf"))
        if npc < best_npc:
            best_npc = npc
            best_result = combined.copy()

    outputs.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        # Use the union of keys from all result dicts to avoid missing-field errors
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        with outputs.results_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    if best_result is None:
        raise RuntimeError("No feasible solution met the renewable penetration and reliability constraints.")

    _create_penetration_cost_plot(results, best_result, constraints)
    _create_design_variable_plots(results, best_result, constraints)
    _create_capacity_plane_plot(results, best_result)

    outputs.best_result_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs.best_result_path.open("w", encoding="utf-8") as file:
        json.dump(best_result, file, indent=2)

    # Re-simulate best candidate to refresh dispatch outputs and timeseries
    base_dispatch = config.base_simulation_config.dispatch
    storage_cfg = replace(
        base_dispatch.storage,
        energy_capacity_kwh=best_result["storage_energy_kwh"],
        charge_power_kw=best_result["storage_power_kw"],
        discharge_power_kw=best_result["storage_power_kw"],
    )
    # PHES for best candidate
    base_phes = base_dispatch.pumped_hydro if base_dispatch.pumped_hydro is not None else base_dispatch.storage
    phes_cfg = None
    if float(best_result.get("phes_energy_kwh", 0.0)) > 0 and float(best_result.get("phes_power_kw", 0.0)) > 0:
        phes_cfg = replace(
            base_phes,
            energy_capacity_kwh=float(best_result.get("phes_energy_kwh", 0.0)),
            charge_power_kw=float(best_result.get("phes_power_kw", 0.0)),
            discharge_power_kw=float(best_result.get("phes_power_kw", 0.0)),
        )
    generator_cfg = replace(
        base_dispatch.generator,
        max_power_kw=best_result["generator_capacity_kw"],
        min_power_kw=best_result["generator_capacity_kw"] * search.generator_min_power_fraction,
    )
    dispatch_cfg = replace(
        base_dispatch,
        pv_capacity_kw=best_result["pv_capacity_kw"],
        wind_turbine_count=int(best_result["wind_turbine_count"]),
        storage=storage_cfg,
        generator=generator_cfg,
        pumped_hydro=phes_cfg,
    )
    sim_cfg = replace(config.base_simulation_config, dispatch=dispatch_cfg)
    refreshed_summary = run_system_simulation(sim_cfg)

    capacities = {
        "pv_capacity_kw": best_result["pv_capacity_kw"],
        "wind_capacity_kw": config.turbine_rated_kw * best_result["wind_turbine_count"],
        "storage_energy_kwh": best_result["storage_energy_kwh"],
        "storage_power_kw": best_result["storage_power_kw"],
        "phes_energy_kwh": float(best_result.get("phes_energy_kwh", 0.0)),
        "phes_power_kw": float(best_result.get("phes_power_kw", 0.0)),
        "generator_capacity_kw": best_result["generator_capacity_kw"],
    }

    best_inputs = CostModelInputs(
        pv_capacity_kw=capacities["pv_capacity_kw"],
        wind_capacity_kw=capacities["wind_capacity_kw"],
        storage_energy_kwh=capacities["storage_energy_kwh"],
        storage_power_kw=capacities["storage_power_kw"],
        phes_energy_kwh=capacities["phes_energy_kwh"],
        phes_power_kw=capacities["phes_power_kw"],
        generator_capacity_kw=capacities["generator_capacity_kw"],
        load_energy_mwh_per_year=refreshed_summary.get("load_energy_mwh", 0.0),
    )
    best_cost_config = replace(config.base_cost_config, write_output_path=outputs.best_cost_path)
    best_cost_summary = evaluate_costs(refreshed_summary, best_inputs, best_cost_config)

    component_summary = _component_summary(best_result, best_cost_summary, config)
    with config.outputs.component_summary_path.open("w", encoding="utf-8") as file:
        json.dump(component_summary, file, indent=2)

    _create_cashflow_plot(refreshed_summary, best_cost_summary, best_cost_config, capacities)
    dispatch_path = sim_cfg.inputs.output_timeseries_path
    _create_representative_day_plot(dispatch_path)
    _create_week_profile_plot(dispatch_path)

    # Persist best result JSON again to include any refreshed values
    with outputs.best_result_path.open("w", encoding="utf-8") as file:
        json.dump({**best_result, **best_cost_summary}, file, indent=2)

    return best_result


def regenerate_optimisation_plots(
    results_csv_path: Path,
    best_result_path: Path,
    constraints: OptimisationConstraints,
) -> None:
    import pandas as pd

    if not results_csv_path.exists() or not best_result_path.exists():
        return

    results_df = pd.read_csv(results_csv_path)
    if results_df.empty:
        return
    results = results_df.to_dict(orient="records")

    with best_result_path.open("r", encoding="utf-8") as file:
        best_result = json.load(file)

    _create_penetration_cost_plot(results, best_result, constraints)
    _create_design_variable_plots(results, best_result, constraints)
    _create_capacity_plane_plot(results, best_result)


__all__ = [
    "FloatRange",
    "IntRange",
    "OptimisationSearchSpace",
    "OptimisationConstraints",
    "OptimisationOutputs",
    "OptimisationConfig",
    "run_optimisation",
    "regenerate_optimisation_plots",
]
