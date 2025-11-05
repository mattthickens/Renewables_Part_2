"""Solar resource processing driven by external configuration."""
from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple
import matplotlib
import pandas as pd
import requests


@dataclass(frozen=True)
class MonthlyGHIPlotConfig:
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
class MonthlyDiurnalPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    title: str
    xlabel: str
    ylabel: str
    legend_loc: str


@dataclass(frozen=True)
class GHIHeatmapPlotConfig:
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
class SolarResourceConfig:
    latitude: float
    longitude: float
    target_year: int
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
    monthly_plot: MonthlyGHIPlotConfig
    diurnal_plot: MonthlyDiurnalPlotConfig
    heatmap_plot: GHIHeatmapPlotConfig


_matplotlib_backend_set = False


def _ensure_backend(config: SolarResourceConfig) -> None:
    global _matplotlib_backend_set
    if not _matplotlib_backend_set:
        matplotlib.use(config.matplotlib_backend)
        if config.matplotlib_rc:
            matplotlib.rcParams.update(config.matplotlib_rc)
        _matplotlib_backend_set = True
        import matplotlib.pyplot as plt  # noqa: F401


def _build_request_params(config: SolarResourceConfig) -> Dict[str, str]:
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


def _download_payload(config: SolarResourceConfig) -> Dict[str, object]:
    headers = {"User-Agent": config.user_agent}
    response = requests.get(
        config.endpoint,
        params=_build_request_params(config),
        timeout=config.request_timeout,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


def _load_or_download_raw(config: SolarResourceConfig) -> Dict[str, object]:
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

    sample_series = next(iter(parameter_block.values()))
    timestamps = sorted(sample_series.keys())

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


def _convert_to_local_time(df: pd.DataFrame, config: SolarResourceConfig) -> pd.DataFrame:
    # NASA POWER timestamps are already in local time; treat as naive
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


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    df.index.name = "timestamp"
    return df


def _plot_monthly_ghi(df: pd.DataFrame, config: MonthlyGHIPlotConfig) -> None:
    if not config.enabled:
        return
    monthly = (
        df["ghi_wm2"]
        .resample("ME")
        .agg(["mean", lambda x: x.quantile(config.percentile)])
        .rename(columns={"<lambda>": "percentile"})
        .dropna()
    )
    months = monthly.index.month.map(lambda m: datetime(2000, m, 1).strftime("%b"))
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.bar(months, monthly["mean"], color=config.bar_color, label="Average GHI")
    ax.plot(
        months,
        monthly["percentile"],
        color=config.line_color,
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


def _plot_monthly_diurnal(df: pd.DataFrame, config: MonthlyDiurnalPlotConfig) -> None:
    if not config.enabled:
        return
    grouped = (
        df["ghi_wm2"]
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


def _plot_heatmap(df: pd.DataFrame, config: GHIHeatmapPlotConfig, year: int) -> None:
    if not config.enabled:
        return
    frame = df.copy()
    frame["day_of_year"] = frame.index.dayofyear
    frame["hour"] = frame.index.hour
    pivot = (
        frame.pivot_table(index="day_of_year", columns="hour", values="ghi_wm2", aggfunc="mean")
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
    cbar.set_label("GHI (W/m²)")
    fig.tight_layout()
    fig.savefig(config.path, dpi=config.dpi)
    plt.close(fig)


def build_solar_resource(config: SolarResourceConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_backend(config)
    payload = _load_or_download_raw(config)
    df_utc = _payload_to_dataframe(payload, config.column_renames)
    df_local = _convert_to_local_time(df_utc, config)
    df_local = _post_process(df_local)

    config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_local.to_csv(config.processed_data_path, float_format=config.float_format)

    _plot_monthly_ghi(df_local, config.monthly_plot)
    _plot_monthly_diurnal(df_local, config.diurnal_plot)
    _plot_heatmap(df_local, config.heatmap_plot, config.target_year)

    return df_utc, df_local


__all__ = [
    "MonthlyGHIPlotConfig",
    "MonthlyDiurnalPlotConfig",
    "GHIHeatmapPlotConfig",
    "SolarResourceConfig",
    "build_solar_resource",
]
