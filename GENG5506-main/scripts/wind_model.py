"""Wind turbine performance modelling driven by external configuration."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindMonthlyEnergyPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title_template: str
    ylabel: str


@dataclass(frozen=True)
class WindDiurnalPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    title_template: str
    xlabel: str
    ylabel: str
    legend_loc: str


@dataclass(frozen=True)
class WindHeatmapPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    cmap: str
    dpi: int
    title_template: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class WindCapacityHistogramConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    bins: int
    color: str
    title_template: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class WindTurbineConfig:
    name: str
    rated_power_kw: float
    cut_in_ms: float
    rated_speed_ms: float
    cut_out_ms: float
    drivetrain_efficiency: float
    air_density_exponent: float
    power_curve_points: Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class WindPerformanceConfig:
    wind_resource_path: Path
    output_path: Path
    turbine: WindTurbineConfig
    reference_air_density: float
    float_format: str
    matplotlib_backend: str
    monthly_plot: WindMonthlyEnergyPlotConfig
    diurnal_plot: WindDiurnalPlotConfig
    heatmap_plot: WindHeatmapPlotConfig
    capacity_hist_plot: WindCapacityHistogramConfig


_matplotlib_backend_set = False


def _ensure_backend(config: WindPerformanceConfig) -> None:
    global _matplotlib_backend_set
    if not _matplotlib_backend_set:
        matplotlib.use(config.matplotlib_backend)
        _matplotlib_backend_set = True
        import matplotlib.pyplot as plt  # noqa: F401


def _load_wind_resource(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def _interpolate_power_curve(speeds: np.ndarray, turbine: WindTurbineConfig) -> np.ndarray:
    curve_speeds = np.array([pt[0] for pt in turbine.power_curve_points])
    curve_power = np.array([pt[1] for pt in turbine.power_curve_points])
    power = np.interp(speeds, curve_speeds, curve_power, left=0.0, right=0.0)
    below_cut_in = speeds < turbine.cut_in_ms
    above_cut_out = speeds >= turbine.cut_out_ms
    power = np.where(below_cut_in | above_cut_out, 0.0, power)
    power = np.minimum(power, turbine.rated_power_kw)
    return power


def _compute_turbine_output(df: pd.DataFrame, config: WindPerformanceConfig) -> pd.DataFrame:
    turbine = config.turbine
    speeds = df["wind_hub_ms"].to_numpy()
    densities = df["air_density_kgm3"].to_numpy()
    base_power = _interpolate_power_curve(speeds, turbine)
    density_ratio = np.power(np.clip(densities / config.reference_air_density, 0.0, None), turbine.air_density_exponent)
    adjusted_power = base_power * density_ratio * turbine.drivetrain_efficiency
    adjusted_power = np.clip(adjusted_power, 0.0, turbine.rated_power_kw)
    result = df.copy()
    result["turbine_ac_kw_per_unit"] = adjusted_power
    result["capacity_factor"] = adjusted_power / turbine.rated_power_kw
    return result


def _plot_monthly_energy(df: pd.DataFrame, config: WindPerformanceConfig) -> None:
    plot = config.monthly_plot
    if not plot.enabled:
        return
    monthly = df["turbine_ac_kw_per_unit"].resample("ME").sum() / 1000.0
    months = monthly.index.month.map(lambda m: datetime(2000, m, 1).strftime("%b"))
    plot.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot.figsize)
    title = plot.title_template.format(turbine_name=config.turbine.name)
    ax.bar(months, monthly.values, color="#1f77b4")
    ax.set_xlabel("Month")
    ax.set_ylabel(plot.ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot.path, dpi=plot.dpi)
    plt.close(fig)


def _plot_diurnal(df: pd.DataFrame, config: WindPerformanceConfig) -> None:
    plot = config.diurnal_plot
    if not plot.enabled:
        return
    grouped = (
        df["turbine_ac_kw_per_unit"]
        .groupby([df.index.month, df.index.hour])
        .mean()
        .unstack(level=0)
        .reindex(index=range(24))
        .dropna(axis=1, how="all")
    )
    plot.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot.figsize)
    title = plot.title_template.format(turbine_name=config.turbine.name)
    for month in grouped.columns:
        ax.plot(grouped.index, grouped[month], label=datetime(2000, month, 1).strftime("%b"))
    ax.set_xlabel(plot.xlabel)
    ax.set_ylabel(plot.ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc=plot.legend_loc, ncol=2)
    fig.tight_layout()
    fig.savefig(plot.path)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, config: WindPerformanceConfig) -> None:
    plot = config.heatmap_plot
    if not plot.enabled:
        return
    frame = df.copy()
    frame["day_of_year"] = frame.index.dayofyear
    frame["hour"] = frame.index.hour
    pivot = (
        frame.pivot_table(index="day_of_year", columns="hour", values="turbine_ac_kw_per_unit", aggfunc="mean")
        .reindex(index=range(1, frame.index.dayofyear.max() + 1))
        .reindex(columns=range(24))
    )
    plot.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot.figsize)
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap=plot.cmap,
        extent=[0, 24, 1, pivot.shape[0] + 1],
    )
    title = plot.title_template.format(turbine_name=config.turbine.name)
    ax.set_xlabel(plot.xlabel)
    ax.set_ylabel(plot.ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 25, 2))

    year = df.index[0].year
    month_starts = [datetime(year, m, 1).timetuple().tm_yday for m in range(1, 13)]
    month_lengths = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
    month_centers = [start + (length - 1) / 2 for start, length in zip(month_starts, month_lengths)]
    ax.set_yticks(month_centers)
    ax.set_yticklabels([datetime(2000, m, 1).strftime("%b") for m in range(1, 13)])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (kW per turbine)")
    fig.tight_layout()
    fig.savefig(plot.path, dpi=plot.dpi)
    plt.close(fig)


def _plot_capacity_histogram(df: pd.DataFrame, config: WindPerformanceConfig) -> None:
    plot = config.capacity_hist_plot
    if not plot.enabled:
        return
    plot.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot.figsize)
    title = plot.title_template.format(turbine_name=config.turbine.name)
    ax.hist(df["capacity_factor"].dropna(), bins=plot.bins, color=plot.color, edgecolor="black")
    ax.set_xlabel(plot.xlabel)
    ax.set_ylabel(plot.ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot.path)
    plt.close(fig)


def build_wind_performance(config: WindPerformanceConfig) -> pd.DataFrame:
    _ensure_backend(config)
    wind_df = _load_wind_resource(config.wind_resource_path)
    turbine_df = _compute_turbine_output(wind_df, config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    turbine_df.to_csv(config.output_path, float_format=config.float_format)

    _plot_monthly_energy(turbine_df, config)
    _plot_diurnal(turbine_df, config)
    _plot_heatmap(turbine_df, config)
    _plot_capacity_histogram(turbine_df, config)

    return turbine_df


__all__ = [
    "WindMonthlyEnergyPlotConfig",
    "WindDiurnalPlotConfig",
    "WindHeatmapPlotConfig",
    "WindCapacityHistogramConfig",
    "WindTurbineConfig",
    "WindPerformanceConfig",
    "build_wind_performance",
    "plot_turbine_power_curve",
]

def plot_turbine_power_curve(turbine: WindTurbineConfig, save_path: Path = None) -> None:
    """Plot the wind turbine power curve with cut-in and cut-out speed lines."""
    import matplotlib.pyplot as plt
    speeds = np.array([pt[0] for pt in turbine.power_curve_points])
    power = np.array([pt[1] for pt in turbine.power_curve_points])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, power, label="Power Curve", color="#1f77b4")
    # Cut-in speed line
    ax.axvline(x=3.0, color="red", linestyle="--", linewidth=2, label="Cut-in Speed (3 m/s)")
    # Cut-out speed line
    ax.axvline(x=turbine.cut_out_ms, color="orange", linestyle="--", linewidth=2, label=f"Cut-out Speed ({turbine.cut_out_ms} m/s)")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.set_title(f"Turbine Power Curve: {turbine.name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
