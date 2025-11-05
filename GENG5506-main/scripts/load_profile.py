"""Load profile generation utilities with external configuration."""
from __future__ import annotations

import bisect
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class MonthlyAveragePlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    bar_color: str
    peak_color: str
    peak_linestyle: str
    dpi: int
    title: str
    ylabel: str


@dataclass(frozen=True)
class HeatmapPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    cmap: str
    dpi: int
    title: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class SeasonalComparisonPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    summer_month: int
    winter_month: int
    weekday_type: str
    title: str
    ylabel: str
    xlabel: str


@dataclass(frozen=True)
class AnnualEnergyPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title: str
    xlabel: str
    ylabel: str
    bar_color: str


@dataclass(frozen=True)
class VariationComparisonPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title: str
    xlabel: str
    ylabel: str
    baseline_label: str
    varied_label: str


@dataclass(frozen=True)
class ExistingPVComparisonPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title: str
    xlabel: str
    ylabel: str
    load_label: str
    pv_label: str
    load_color: str
    pv_color: str


@dataclass(frozen=True)
class ExistingPVConfig:
    capacity_kw: float
    profile_path: Path
    comparison_plot: ExistingPVComparisonPlotConfig | None = None


@dataclass(frozen=True)
class TemplateInputPlotConfig:
    enabled: bool
    figsize: tuple[float, float]
    path: Path
    dpi: int
    title: str
    xlabel: str
    ylabel: str
    legend_loc: str


@dataclass(frozen=True)
class LoadProfileConfig:
    target_year: int
    load_profile_paths: Dict[str, Path]
    template_anchor_months: Dict[str, int]
    half_hourly_frequency: str
    output_half_hourly_path: Path
    output_hourly_path: Path
    save_half_hourly: bool
    half_hourly_timestamp_label: str
    hourly_timestamp_label: str
    load_column_name: str
    float_format: str
    matplotlib_backend: str
    project_lifespan_years: int
    annual_growth_rate: float
    monthly_plot: MonthlyAveragePlotConfig
    heatmap_plot: HeatmapPlotConfig
    seasonal_plot: SeasonalComparisonPlotConfig
    annual_energy_plot: AnnualEnergyPlotConfig | None = None
    existing_pv: ExistingPVConfig | None = None
    day_variation_fraction: float = 0.10
    variation_random_seed: int | None = None
    variation_plot: VariationComparisonPlotConfig | None = None
    template_plot: TemplateInputPlotConfig | None = None


_matplotlib_backend_set = False


def _ensure_backend(backend: str) -> None:
    global _matplotlib_backend_set
    if not _matplotlib_backend_set:
        matplotlib.use(backend)
        _matplotlib_backend_set = True
        import matplotlib.pyplot as plt  # noqa: F401


def _load_day_type_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Time"] = pd.to_timedelta(df["Time"].astype(str) + ":00")
    return df.set_index("Time")


def _resolve_day_type(timestamp: pd.Timestamp) -> str:
    weekday_index = timestamp.weekday()
    if weekday_index < 5:
        return "weekday"
    if weekday_index == 5:
        return "saturday"
    return "sunday"


def _blend_weights_for_month(
    month: int,
    anchor_months: Iterable[int],
    month_value_to_label: Dict[int, str],
) -> Dict[str, float]:
    anchor_list = sorted(set(anchor_months))
    if month in month_value_to_label:
        return {month_value_to_label[month]: 1.0}

    extended = anchor_list + [anchor + 12 for anchor in anchor_list]
    month_value = month
    if month < anchor_list[0]:
        month_value += 12

    idx = bisect.bisect_right(extended, month_value)
    prev_value = extended[idx - 1]
    next_value = extended[idx]
    span = next_value - prev_value
    alpha = (month_value - prev_value) / span

    return {
        month_value_to_label[prev_value]: 1.0 - alpha,
        month_value_to_label[next_value]: alpha,
    }


def build_load_profiles(config: LoadProfileConfig) -> pd.DataFrame:
    _ensure_backend(config.matplotlib_backend)
    import matplotlib.pyplot as plt

    templates = {name: _load_day_type_profile(path) for name, path in config.load_profile_paths.items()}
    sample_template = next(iter(templates.values()))
    half_hour_offsets = pd.timedelta_range(
        start="0h",
        periods=len(sample_template),
        freq=config.half_hourly_frequency,
    )

    if config.template_plot is not None and config.template_plot.enabled:
        config.template_plot.path.parent.mkdir(parents=True, exist_ok=True)
        ordered_day_types = [
            day_type
            for day_type in ("weekday", "saturday", "sunday")
            if day_type in templates
        ]
        if not ordered_day_types:
            ordered_day_types = sorted(templates.keys())

        fig, axes = plt.subplots(
            nrows=len(ordered_day_types),
            ncols=1,
            figsize=config.template_plot.figsize,
            sharex=True,
            squeeze=False,
        )
        cmap = plt.get_cmap("viridis")
        for ax, day_type in zip(axes.flat, ordered_day_types):
            df = templates[day_type]
            anchors = list(df.columns)
            colors = cmap(np.linspace(0.1, 0.9, len(anchors))) if anchors else []
            hours = df.index.total_seconds() / 3600
            for color, anchor in zip(colors, anchors):
                ax.plot(hours, df[anchor], label=anchor, color=color)
            ax.set_ylabel(config.template_plot.ylabel)
            ax.set_title(f"{config.template_plot.title}: {day_type.title()}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        axes.flat[-1].set_xlabel(config.template_plot.xlabel)
        axes.flat[0].legend(loc=config.template_plot.legend_loc)
        for ax in axes.flat:
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 4))
        fig.tight_layout()
        fig.savefig(config.template_plot.path, dpi=config.template_plot.dpi)
        plt.close(fig)

    anchor_months = list(config.template_anchor_months.values())
    month_value_to_label: Dict[int, str] = {}
    for label, value in config.template_anchor_months.items():
        month_value_to_label[value] = label
        month_value_to_label[value + 12] = label

    calendar_index = pd.date_range(
        start=f"{config.target_year}-01-01",
        end=f"{config.target_year}-12-31",
        freq="D",
    )

    half_hourly_segments = []
    representative_pre_variation: pd.Series | None = None
    representative_post_variation: pd.Series | None = None
    rng = np.random.default_rng(config.variation_random_seed)
    variation_sigma = config.day_variation_fraction / 3 if config.day_variation_fraction > 0 else 0.0

    for day in calendar_index:
        template_name = _resolve_day_type(day)
        weights = _blend_weights_for_month(day.month, anchor_months, month_value_to_label)
        blended_load = None
        for label, weight in weights.items():
            column_values = templates[template_name][label].to_numpy(copy=True)
            scaled = column_values * weight
            blended_load = scaled if blended_load is None else blended_load + scaled
        day_index = day + half_hour_offsets
        base_series = pd.Series(blended_load, index=day_index)

        if variation_sigma > 0:
            hour_index = base_series.index.floor("h")
            unique_hours = hour_index.unique()
            hour_multipliers = rng.normal(loc=1.0, scale=variation_sigma, size=len(unique_hours))
            while np.any(hour_multipliers <= 0):
                mask = hour_multipliers <= 0
                hour_multipliers[mask] = rng.normal(
                    loc=1.0,
                    scale=variation_sigma,
                    size=mask.sum(),
                )
            hour_multiplier_map = dict(zip(unique_hours, hour_multipliers))
            multipliers = hour_index.map(hour_multiplier_map).to_numpy()
            varied_values = base_series.to_numpy() * multipliers
            varied_series = pd.Series(varied_values, index=day_index)
        else:
            varied_series = base_series.copy()

        if (
            config.variation_plot is not None
            and config.variation_plot.enabled
            and representative_pre_variation is None
        ):
            representative_pre_variation = base_series
            representative_post_variation = varied_series

        half_hourly_segments.append(varied_series)

    base_half_hourly = pd.concat(half_hourly_segments).sort_index().to_frame(name=config.load_column_name)

    yearly_scaled_frames = []
    for offset in range(config.project_lifespan_years):
        scale = (1 + config.annual_growth_rate) ** offset
        start_year = config.target_year + offset
        scaled = base_half_hourly.copy()
        scaled.index = scaled.index.map(lambda ts: ts + pd.DateOffset(years=offset))
        scaled[config.load_column_name] *= scale
        yearly_scaled_frames.append(scaled)

    half_hourly = yearly_scaled_frames[0]

    config.output_half_hourly_path.parent.mkdir(parents=True, exist_ok=True)
    if config.save_half_hourly:
        half_hourly.to_csv(
            config.output_half_hourly_path,
            index_label=config.half_hourly_timestamp_label,
            float_format=config.float_format,
        )

    hourly = half_hourly.resample("1h").mean()

    existing_pv_series_mw: pd.Series | None = None
    if config.existing_pv is not None and config.existing_pv.capacity_kw > 0:
        pv_df = pd.read_csv(
            config.existing_pv.profile_path,
            parse_dates=[0],
            index_col=0,
        )
        pv_per_kw = pv_df.get("pv_ac_kw_per_kw")
        if pv_per_kw is None:
            raise ValueError(
                "Existing PV profile missing 'pv_ac_kw_per_kw' column"
            )
        existing_pv_series_kw = pv_per_kw * config.existing_pv.capacity_kw
        existing_pv_series_kw = existing_pv_series_kw.reindex(hourly.index).fillna(0.0)
        existing_pv_series_mw = existing_pv_series_kw / 1000.0
        hourly[config.load_column_name] = (hourly[config.load_column_name] - existing_pv_series_mw).clip(lower=0.0)
        hourly["existing_pv_mw"] = existing_pv_series_mw
    hourly.to_csv(
        config.output_hourly_path,
        index_label=config.hourly_timestamp_label,
        float_format=config.float_format,
    )

    month_abbr = {m: calendar.month_abbr[m] for m in range(1, 13)}

    if config.monthly_plot.enabled:
        monthly = hourly[config.load_column_name].resample("ME").agg(["mean", "max"])
        months = monthly.index.month.map(month_abbr.get)
        config.monthly_plot.path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=config.monthly_plot.figsize)
        ax.bar(months, monthly["mean"], color=config.monthly_plot.bar_color, label="Average Load")
        ax.plot(
            months,
            monthly["max"],
            config.monthly_plot.peak_linestyle,
            color=config.monthly_plot.peak_color,
            label="Peak Load",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel(config.monthly_plot.ylabel)
        ax.set_title(f"{config.monthly_plot.title} (Base Year)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.monthly_plot.path, dpi=config.monthly_plot.dpi)
        plt.close(fig)

    if (
        config.existing_pv is not None
        and config.existing_pv.comparison_plot is not None
        and config.existing_pv.comparison_plot.enabled
        and existing_pv_series_mw is not None
    ):
        plot_cfg = config.existing_pv.comparison_plot
        monthly_load_gwh = hourly[config.load_column_name].resample("ME").sum() / 1000.0
        monthly_pv_gwh = existing_pv_series_mw.resample("ME").sum() / 1000.0
        months = monthly_load_gwh.index.month.map(month_abbr.get)
        x = np.arange(len(months))
        width = 0.35
        plot_cfg.path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=plot_cfg.figsize)
        ax.bar(x - width / 2, monthly_load_gwh.to_numpy(), width, color=plot_cfg.load_color, label=plot_cfg.load_label)
        ax.bar(x + width / 2, monthly_pv_gwh.to_numpy(), width, color=plot_cfg.pv_color, label=plot_cfg.pv_label)
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.set_xlabel(plot_cfg.xlabel)
        ax.set_ylabel(plot_cfg.ylabel)
        ax.set_title(plot_cfg.title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_cfg.path, dpi=plot_cfg.dpi)
        plt.close(fig)

    if config.heatmap_plot.enabled:
        hourly_with_meta = hourly.copy()
        hourly_with_meta["month"] = hourly_with_meta.index.month
        hourly_with_meta["hour"] = hourly_with_meta.index.hour
        grouped = (
            hourly_with_meta.groupby(["month", "hour"], sort=True)[config.load_column_name]
            .mean()
            .unstack("hour")
            .reindex(index=range(1, 13), columns=range(24))
        )

        config.heatmap_plot.path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=config.heatmap_plot.figsize)
        heatmap = ax.imshow(
            grouped.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap=config.heatmap_plot.cmap,
        )
        ax.set_xlabel(config.heatmap_plot.xlabel)
        ax.set_ylabel(config.heatmap_plot.ylabel)
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(12))
        ax.set_yticklabels([month_abbr[m] for m in range(1, 13)])
        ax.set_title(f"{config.heatmap_plot.title} (Base Year)")
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label(config.load_column_name.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(config.heatmap_plot.path, dpi=config.heatmap_plot.dpi)
        plt.close(fig)

    if config.seasonal_plot.enabled:
        hourly_with_meta = hourly.copy()
        hourly_with_meta["month"] = hourly_with_meta.index.month
        hourly_with_meta["daytype"] = hourly_with_meta.index.map(_resolve_day_type)

        def _representative_day(month: int, daytype: str) -> pd.Series:
            subset = hourly_with_meta[
                (hourly_with_meta["month"] == month)
                & (hourly_with_meta["daytype"] == daytype)
            ]
            if subset.empty:
                return pd.Series(dtype=float)
            start = subset.index.floor("D").unique().min()
            return subset.loc[start : start + pd.Timedelta(hours=23), config.load_column_name]

        summer_series = _representative_day(config.seasonal_plot.summer_month, config.seasonal_plot.weekday_type)
        winter_series = _representative_day(config.seasonal_plot.winter_month, config.seasonal_plot.weekday_type)

        if not summer_series.empty and not winter_series.empty:
            config.seasonal_plot.path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=config.seasonal_plot.figsize)
            ax.plot(
                summer_series.index.hour,
                summer_series.values,
                label=f"{month_abbr[config.seasonal_plot.summer_month]} {config.seasonal_plot.weekday_type.title()}",
            )
            ax.plot(
                winter_series.index.hour,
                winter_series.values,
                label=f"{month_abbr[config.seasonal_plot.winter_month]} {config.seasonal_plot.weekday_type.title()}",
            )
            ax.set_xlim(0, 23)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel(config.seasonal_plot.xlabel)
            ax.set_ylabel(config.seasonal_plot.ylabel)
            ax.set_title(f"{config.seasonal_plot.title} (Base Year)")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(config.seasonal_plot.path)
            plt.close(fig)

    if (
        config.annual_energy_plot is not None
        and config.annual_energy_plot.enabled
    ):
        base_energy_mwh = hourly[config.load_column_name].sum()
        years = [config.target_year + offset for offset in range(config.project_lifespan_years)]
        energies = []
        for offset in range(config.project_lifespan_years):
            scale = (1 + config.annual_growth_rate) ** offset
            energies.append(base_energy_mwh * scale / 1000.0)

        plot_cfg = config.annual_energy_plot
        plot_cfg.path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=plot_cfg.figsize)
        ax.bar(years, energies, color=plot_cfg.bar_color)
        ax.set_xlabel(plot_cfg.xlabel)
        ax.set_ylabel(plot_cfg.ylabel)
        ax.set_title(plot_cfg.title)
        ax.set_xticks(years[:: max(1, len(years) // 10)])
        ax.set_ylim(0, max(energies) * 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for year, energy in zip(years, energies):
            ax.text(year, energy, f"{energy:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_cfg.path, dpi=plot_cfg.dpi)
        plt.close(fig)
    if (
        config.variation_plot is not None
        and config.variation_plot.enabled
        and representative_pre_variation is not None
        and representative_post_variation is not None
    ):
        config.variation_plot.path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=config.variation_plot.figsize)
        time_offsets = representative_pre_variation.index - representative_pre_variation.index[0]
        hours = (time_offsets.total_seconds() / 3600).to_numpy()
        ax.plot(hours, representative_pre_variation.values, label=config.variation_plot.baseline_label)
        ax.plot(hours, representative_post_variation.values, label=config.variation_plot.varied_label)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel(config.variation_plot.xlabel)
        ax.set_ylabel(config.variation_plot.ylabel)
        ax.set_title(config.variation_plot.title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.variation_plot.path, dpi=config.variation_plot.dpi)
        plt.close(fig)

    return hourly


__all__ = [
    "MonthlyAveragePlotConfig",
    "HeatmapPlotConfig",
    "SeasonalComparisonPlotConfig",
    "AnnualEnergyPlotConfig",
    "VariationComparisonPlotConfig",
    "ExistingPVComparisonPlotConfig",
    "ExistingPVConfig",
    "TemplateInputPlotConfig",
    "LoadProfileConfig",
    "build_load_profiles",
]
