"""PV performance modelling driven by external configuration."""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class PVMonthlyEnergyPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title: str
    ylabel: str


@dataclass(frozen=True)
class PVDiurnalPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    title: str
    xlabel: str
    ylabel: str
    legend_loc: str


@dataclass(frozen=True)
class PVHeatmapPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    cmap: str
    dpi: int
    title: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class PVModelConfig:
    solar_resource_path: Path
    output_path: Path
    latitude: float
    longitude: float
    timezone: str
    module_tilt_deg: float
    module_azimuth_deg: float
    temp_coeff_pct_per_c: float
    temp_ref_c: float
    noct_c: float
    system_derate: float
    dc_ac_ratio: float
    inverter_efficiency: float
    ground_albedo: float
    float_format: str
    matplotlib_backend: str
    monthly_plot: PVMonthlyEnergyPlotConfig
    diurnal_plot: PVDiurnalPlotConfig
    heatmap_plot: PVHeatmapPlotConfig


_matplotlib_backend_set = False


def _ensure_backend(config: PVModelConfig) -> None:
    global _matplotlib_backend_set
    if not _matplotlib_backend_set:
        matplotlib.use(config.matplotlib_backend)
        _matplotlib_backend_set = True
        import matplotlib.pyplot as plt  # noqa: F401


def _load_solar_resource(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def _localize_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert(timezone)
    tz = ZoneInfo(timezone)
    aware = [
        datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=tz)
        for ts in index.to_pydatetime()
    ]
    return pd.DatetimeIndex(aware)


def _solar_position(times: pd.DatetimeIndex, latitude: float, longitude: float) -> Dict[str, np.ndarray]:
    tz_offsets = np.array([ts.utcoffset().total_seconds() / 3600 for ts in times])
    n = times.dayofyear.to_numpy()
    hour = times.hour + times.minute / 60 + times.second / 3600
    lstm = 15 * tz_offsets

    B = np.deg2rad((360 / 365.0) * (n - 81))
    equation_of_time = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    time_correction = equation_of_time + 4 * (longitude - lstm)
    solar_time = hour + time_correction / 60

    hour_angle = np.deg2rad(15 * (solar_time - 12))
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365.0 * (n + 284))))
    lat_rad = np.deg2rad(latitude)

    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)

    sin_zenith = np.sqrt(np.clip(1.0 - cos_zenith**2, 0.0, 1.0))
    sin_zenith = np.where(sin_zenith == 0, 1e-6, sin_zenith)

    sin_az = np.sin(hour_angle) * np.cos(decl) / sin_zenith
    cos_az = (np.sin(decl) - np.sin(lat_rad) * cos_zenith) / (np.cos(lat_rad) * sin_zenith)
    azimuth = np.arctan2(sin_az, cos_az)
    azimuth = (azimuth + 2 * np.pi) % (2 * np.pi)

    return {
        "zenith": zenith,
        "azimuth": azimuth,
    }


def _plane_of_array_irradiance(
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    tilt_rad: float,
    surface_azimuth_rad: float,
    ground_albedo: float,
) -> Dict[str, np.ndarray]:
    cos_zenith = np.clip(np.cos(zenith), 0.0, 1.0)
    sun_above_horizon = cos_zenith > 0

    s_x = np.sin(zenith) * np.sin(azimuth)
    s_y = np.sin(zenith) * np.cos(azimuth)
    s_z = np.cos(zenith)

    n_x = np.sin(tilt_rad) * np.sin(surface_azimuth_rad)
    n_y = np.sin(tilt_rad) * np.cos(surface_azimuth_rad)
    n_z = np.cos(tilt_rad)

    cos_incidence = np.clip(s_x * n_x + s_y * n_y + s_z * n_z, 0.0, None)
    cos_incidence = np.where(sun_above_horizon, cos_incidence, 0.0)

    multiplier = sun_above_horizon.astype(float)
    poa_beam = dni * cos_incidence * multiplier
    poa_sky = dhi * (1 + np.cos(tilt_rad)) / 2.0 * multiplier
    poa_ground = ghi * ground_albedo * (1 - np.cos(tilt_rad)) / 2.0 * multiplier
    poa_total = poa_beam + poa_sky + poa_ground

    return {
        "poa_total": poa_total,
        "poa_beam": poa_beam,
        "poa_sky": poa_sky,
        "poa_ground": poa_ground,
        "cos_incidence": cos_incidence,
    }


def _cell_temperature(poa: np.ndarray, ambient_c: np.ndarray, noct_c: float) -> np.ndarray:
    delta_t = (noct_c - 20.0) / 800.0 * poa
    return ambient_c + delta_t


def _pv_per_kw_dataframe(df: pd.DataFrame, config: PVModelConfig) -> pd.DataFrame:
    times_local = _localize_index(df.index, config.timezone)
    solar_geo = _solar_position(times_local, config.latitude, config.longitude)
    tilt_rad = math.radians(config.module_tilt_deg)
    surface_azimuth_rad = math.radians(config.module_azimuth_deg)

    poa = _plane_of_array_irradiance(
        df["ghi_wm2"].to_numpy(),
        df["dni_wm2"].to_numpy(),
        df["dhi_wm2"].to_numpy(),
        solar_geo["zenith"],
        solar_geo["azimuth"],
        tilt_rad,
        surface_azimuth_rad,
        config.ground_albedo,
    )

    cell_temp = _cell_temperature(poa["poa_total"], df["temp_air_c"].to_numpy(), config.noct_c)
    temp_multiplier = 1.0 + config.temp_coeff_pct_per_c * (cell_temp - config.temp_ref_c)
    temp_multiplier = np.maximum(temp_multiplier, 0.0)

    p_dc = np.maximum((poa["poa_total"] / 1000.0) * temp_multiplier * config.system_derate, 0.0)
    ac_limit = 1.0 / config.dc_ac_ratio
    p_ac = np.minimum(p_dc, ac_limit) * config.inverter_efficiency

    result = pd.DataFrame(
        {
            "ghi_wm2": df["ghi_wm2"],
            "dni_wm2": df["dni_wm2"],
            "dhi_wm2": df["dhi_wm2"],
            "poa_wm2": poa["poa_total"],
            "poa_beam_wm2": poa["poa_beam"],
            "poa_sky_wm2": poa["poa_sky"],
            "poa_ground_wm2": poa["poa_ground"],
            "cos_incidence": poa["cos_incidence"],
            "cell_temp_c": cell_temp,
            "pv_dc_kw_per_kw": p_dc,
            "pv_ac_kw_per_kw": p_ac,
        },
        index=df.index,
    )
    result.index.name = "timestamp"
    return result


def _plot_monthly_energy(df: pd.DataFrame, config: PVMonthlyEnergyPlotConfig) -> None:
    if not config.enabled:
        return
    monthly = df["pv_ac_kw_per_kw"].resample("ME").sum()
    months = monthly.index.month.map(lambda m: datetime(2000, m, 1).strftime("%b"))
    config.path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.bar(months, monthly.values, color="#1f77b4")
    ax.set_xlabel("Month")
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(config.path, dpi=config.dpi)
    plt.close(fig)


def _plot_diurnal(df: pd.DataFrame, config: PVDiurnalPlotConfig) -> None:
    if not config.enabled:
        return
    grouped = (
        df["pv_ac_kw_per_kw"]
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


def _plot_heatmap(df: pd.DataFrame, config: PVHeatmapPlotConfig, year: int) -> None:
    if not config.enabled:
        return
    frame = df.copy()
    frame["day_of_year"] = frame.index.dayofyear
    frame["hour"] = frame.index.hour
    pivot = (
        frame.pivot_table(index="day_of_year", columns="hour", values="pv_ac_kw_per_kw", aggfunc="mean")
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

    month_starts = [datetime(year, m, 1).timetuple().tm_yday for m in range(1, 13)]
    month_lengths = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
    month_centers = [start + (length - 1) / 2 for start, length in zip(month_starts, month_lengths)]
    ax.set_yticks(month_centers)
    ax.set_yticklabels([datetime(2000, m, 1).strftime("%b") for m in range(1, 13)])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AC Output (kW per kW DC)")
    fig.tight_layout()
    fig.savefig(config.path, dpi=config.dpi)
    plt.close(fig)


def build_pv_performance(config: PVModelConfig) -> pd.DataFrame:
    _ensure_backend(config)
    solar_df = _load_solar_resource(config.solar_resource_path)
    pv_df = _pv_per_kw_dataframe(solar_df, config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    pv_df.to_csv(config.output_path, float_format=config.float_format)

    _plot_monthly_energy(pv_df, config.monthly_plot)
    _plot_diurnal(pv_df, config.diurnal_plot)
    _plot_heatmap(pv_df, config.heatmap_plot, pv_df.index[0].year)

    return pv_df


__all__ = [
    "PVModelConfig",
    "PVMonthlyEnergyPlotConfig",
    "PVDiurnalPlotConfig",
    "PVHeatmapPlotConfig",
    "build_pv_performance",
]
