"""Wind resource processing driven by external configuration."""
from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

import matplotlib
import pandas as pd
import requests


@dataclass(frozen=True)
class MonthlyWindPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    bar_color: str
    line_color: str
    percentile: float
    percentile_label: str
    dpi: int
    title: str
    ylabel: str
    xlabel: str


@dataclass(frozen=True)
class WindDiurnalPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    title: str
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
    title: str
    xlabel: str
    ylabel: str
    month_ticks: bool


@dataclass(frozen=True)
class WindHistogramPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    bins: int
    color: str
    title: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class WindHistogramComparisonPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    bins: int
    color_reference: str
    color_hub: str
    alpha: float
    label_reference: str
    label_hub: str
    title: str
    xlabel: str
    ylabel: str
    legend_loc: str


@dataclass(frozen=True)
class WindResourceConfig:
    latitude: float
    longitude: float
    target_year: int
    hub_height_m: float
    reference_height_m: float
    shear_exponent: float
    air_gas_constant: float
    endpoint: str
    community: str
    user_agent: str
    parameters: Sequence[str]
    raw_response_path: Path
    processed_data_path: Path
    save_raw_response: bool
    use_cached_response: bool
    local_timezone: str
    expected_frequency: str
    fill_method: str
    column_renames: Mapping[str, str]
    matplotlib_backend: str
    matplotlib_rc: Mapping[str, float]
    float_format: str
    request_timeout: int
    monthly_plot: MonthlyWindPlotConfig
    diurnal_plot: WindDiurnalPlotConfig
    heatmap_plot: WindHeatmapPlotConfig
    histogram_plot: WindHistogramPlotConfig
    histogram_comparison_plot: WindHistogramComparisonPlotConfig


_matplotlib_backend_set = False


def _ensure_backend(config: WindResourceConfig) -> None:
    global _matplotlib_backend_set
    if not _matplotlib_backend_set:
        matplotlib.use(config.matplotlib_backend)
        if config.matplotlib_rc:
            matplotlib.rcParams.update(config.matplotlib_rc)
        _matplotlib_backend_set = True
        import matplotlib.pyplot as plt  # noqa: F401


def _build_request_params(config: WindResourceConfig) -> Dict[str, str]:
    year = config.target_year
    return {
        "latitude": f"{config.latitude:.4f}",
        "longitude": f"{config.longitude:.4f}",
        "start": f"{year}0101",
        "end": f"{year}1231",
        "parameters": ",".join(config.parameters),
        "community": config.community,
        "format": "JSON",
        "time-standard": "LST",
    }


def _download_payload(config: WindResourceConfig) -> Dict[str, object]:
    headers = {"User-Agent": config.user_agent}
    response = requests.get(
        config.endpoint,
        params=_build_request_params(config),
        timeout=config.request_timeout,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


def _load_or_download_raw(config: WindResourceConfig) -> Dict[str, object]:
    if config.use_cached_response and config.raw_response_path.exists():
        with config.raw_response_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    payload = _download_payload(config)

    if config.save_raw_response:
        config.raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        with config.raw_response_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)

    return payload


def _payload_to_dataframe(payload: Dict[str, object], column_renames: Mapping[str, str]) -> pd.DataFrame:
    properties = payload.get("properties", {})
    parameter_block = properties.get("parameter", {})
    if not parameter_block:
        raise ValueError("NASA POWER payload missing 'properties.parameter' block")

    sample_values = next(iter(parameter_block.values()))
    timestamps = sorted(sample_values.keys())

    records = []
    for ts in timestamps:
        row = {"timestamp": pd.to_datetime(ts, format="%Y%m%d%H")}
        for nasa_name, values in parameter_block.items():
            row[nasa_name] = values.get(ts)
        records.append(row)

    df = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
    df = df.rename(columns=column_renames)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _convert_to_local_time(df: pd.DataFrame, config: WindResourceConfig) -> pd.DataFrame:
    df_local = df.copy()
    df_local.index.name = "timestamp"
    expected_index = pd.date_range(
        start=f"{config.target_year}-01-01 00:00",
        end=f"{config.target_year}-12-31 23:00",
        freq=config.expected_frequency,
    )
    df_local = df_local.reindex(expected_index)

    if config.fill_method == "interpolate":
        df_local = df_local.interpolate(limit_direction="both")
    elif config.fill_method == "ffill":
        df_local = df_local.ffill().bfill()

    return df_local


def _derive_wind_metrics(df: pd.DataFrame, config: WindResourceConfig) -> pd.DataFrame:
    result = df.copy()
    hub_factor = (config.hub_height_m / config.reference_height_m) ** config.shear_exponent
    result["wind_hub_ms"] = result["wind_10m_ms"] * hub_factor
    temperature_k = result["temp_air_c"] + 273.15
    pressure_pa = result["surface_pressure_kpa"] * 1000.0
    result["air_density_kgm3"] = pressure_pa / (config.air_gas_constant * temperature_k)
    result["wind_power_density_wm2"] = 0.5 * result["air_density_kgm3"] * result["wind_hub_ms"] ** 3
    result.index.name = "timestamp"
    return result


def _plot_monthly_wind(df: pd.DataFrame, config: MonthlyWindPlotConfig) -> None:
    if not config.enabled:
        return
    monthly = (
        df["wind_hub_ms"]
        .resample("ME")
        .agg(["mean", lambda x: x.quantile(config.percentile)])
        .rename(columns={"<lambda>": "percentile"})
        .dropna()
    )
    months = monthly.index.month.map(lambda m: datetime(2000, m, 1).strftime("%b"))
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.bar(months, monthly["mean"], color=config.bar_color, label="Average Speed")
    ax.plot(
        months,
        monthly["percentile"],
        config.line_color,
        marker="o",
        label=config.percentile_label,
    )
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.path, dpi=config.dpi)
    plt.close(fig)


def _plot_diurnal(df: pd.DataFrame, config: WindDiurnalPlotConfig) -> None:
    if not config.enabled:
        return
    grouped = (
        df["wind_hub_ms"]
        .groupby([df.index.month, df.index.hour])
        .mean()
        .unstack(level=0)
        .reindex(index=range(24))
        .dropna(axis=1, how="all")
    )
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    for month in grouped.columns:
        ax.plot(grouped.index, grouped[month], label=datetime(2000, month, 1).strftime("%b"))
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc=config.legend_loc, ncol=2)
    fig.tight_layout()
    fig.savefig(config.path)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, config: WindHeatmapPlotConfig, year: int) -> None:
    if not config.enabled:
        return
    frame = df.copy()
    frame["day_of_year"] = frame.index.dayofyear
    frame["hour"] = frame.index.hour
    pivot = (
        frame.pivot_table(index="day_of_year", columns="hour", values="wind_hub_ms", aggfunc="mean")
        .reindex(index=range(1, frame.index.dayofyear.max() + 1))
        .reindex(columns=range(24))
    )
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap=config.cmap,
        extent=[0, 24, 1, pivot.shape[0] + 1],
    )
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.set_xticks(range(0, 25, 2))

    if config.month_ticks:
        month_starts = [datetime(year, m, 1).timetuple().tm_yday for m in range(1, 13)]
        month_lengths = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
        month_centers = [start + (length - 1) / 2 for start, length in zip(month_starts, month_lengths)]
        ax.set_yticks(month_centers)
        ax.set_yticklabels([datetime(2000, m, 1).strftime("%b") for m in range(1, 13)])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wind Speed (m/s)")
    fig.tight_layout()
    fig.savefig(config.path, dpi=config.dpi)
    plt.close(fig)


def _plot_histogram(df: pd.DataFrame, config: WindHistogramPlotConfig) -> None:
    if not config.enabled:
        return
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.hist(df["wind_hub_ms"].dropna(), bins=config.bins, color=config.color, edgecolor="black")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(config.path)
    plt.close(fig)


def _plot_histogram_comparison(
    df: pd.DataFrame,
    config: WindHistogramComparisonPlotConfig,
) -> None:
    if not config.enabled:
        return

    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    reference = df["wind_10m_ms"].dropna()
    hub = df["wind_hub_ms"].dropna()
    if reference.empty or hub.empty:
        return

    combined = pd.concat([reference, hub], axis=0)
    min_edge = max(0.0, combined.min() - 0.5)
    max_edge = combined.max() + 0.5
    bin_edges = np.linspace(min_edge, max_edge, config.bins + 1)

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.hist(
        reference,
        bins=bin_edges,
        alpha=config.alpha,
        color=config.color_reference,
        edgecolor="black",
        linewidth=0.6,
        label=f"{config.label_reference} (mean {reference.mean():.1f} m/s)",
    )
    ax.hist(
        hub,
        bins=bin_edges,
        alpha=config.alpha,
        color=config.color_hub,
        edgecolor="black",
        linewidth=0.6,
        label=f"{config.label_hub} (mean {hub.mean():.1f} m/s)",
    )

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc=config.legend_loc)

    fig.tight_layout()
    fig.savefig(config.path)
    plt.close(fig)


def build_wind_resource(config: WindResourceConfig) -> pd.DataFrame:
    _ensure_backend(config)
    payload = _load_or_download_raw(config)
    df_utc = _payload_to_dataframe(payload, config.column_renames)
    df_local = _convert_to_local_time(df_utc, config)
    df_local = _derive_wind_metrics(df_local, config)

    config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_local.to_csv(config.processed_data_path, float_format=config.float_format)

    _plot_monthly_wind(df_local, config.monthly_plot)
    _plot_diurnal(df_local, config.diurnal_plot)
    _plot_heatmap(df_local, config.heatmap_plot, config.target_year)
    _plot_histogram(df_local, config.histogram_plot)
    _plot_histogram_comparison(df_local, config.histogram_comparison_plot)

    return df_local


__all__ = [
    "MonthlyWindPlotConfig",
    "WindDiurnalPlotConfig",
    "WindHeatmapPlotConfig",
    "WindHistogramPlotConfig",
    "WindHistogramComparisonPlotConfig",
    "WindResourceConfig",
    "build_wind_resource",
]
