"""Pipeline runner for Burnie microgrid resource preparation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.load_profile import (
    HeatmapPlotConfig,
    LoadProfileConfig,
    MonthlyAveragePlotConfig,
    SeasonalComparisonPlotConfig,
    AnnualEnergyPlotConfig,
    VariationComparisonPlotConfig,
    TemplateInputPlotConfig,
    ExistingPVComparisonPlotConfig,
    ExistingPVConfig,
    build_load_profiles,
)
from scripts.solar_resource import (
    GHIHeatmapPlotConfig,
    MonthlyDiurnalPlotConfig,
    MonthlyGHIPlotConfig as SolarMonthlyGHIPlotConfig,
    SolarResourceConfig,
    build_solar_resource,
)
from scripts.wind_resource import (
    MonthlyWindPlotConfig,
    WindDiurnalPlotConfig,
    WindHeatmapPlotConfig,
    WindHistogramPlotConfig,
    WindHistogramComparisonPlotConfig,
    WindResourceConfig,
    build_wind_resource,
)
from scripts.pv_model import (
    PVDiurnalPlotConfig,
    PVHeatmapPlotConfig,
    PVModelConfig,
    PVMonthlyEnergyPlotConfig,
    build_pv_performance,
)
from scripts.wind_model import (
    WindCapacityHistogramConfig,
    WindDiurnalPlotConfig as TurbineDiurnalPlotConfig,
    WindHeatmapPlotConfig as TurbineHeatmapPlotConfig,
    WindMonthlyEnergyPlotConfig,
    WindPerformanceConfig,
    WindTurbineConfig,
    build_wind_performance,
)
from scripts.cost_model import (
    CostModelConfig,
    CostModelInputs,
    PVCostParameters,
    WindCostParameters,
    evaluate_costs,
)
from scripts.system_simulation import (
    DispatchConfig,
    GeneratorConfig,
    SimulationInputsConfig,
    StorageConfig,
    SystemSimulationConfig,
    run_system_simulation,
)
from scripts.optimisation import (
    OptimisationConfig,
    OptimisationConstraints,
    OptimisationOutputs,
    OptimisationSearchSpace,
    FloatRange,
    IntRange,
    run_optimisation,
)

# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Shared site settings
SIMULATION_YEAR = 2023
SITE_LATITUDE = -41.05
SITE_LONGITUDE = 145.90
PROJECT_LIFESPAN_YEARS = 25
LOAD_ANNUAL_GROWTH_RATE = 0.0
BESS_UNIT_NAME = "Tesla Megapack 2XL"
BESS_UNIT_ENERGY_MWH = 3.9
BESS_UNIT_POWER_MW = 1.9
GENERATOR_UNIT_NAME = "Diesel Generator"
GENERATOR_UNIT_CAPACITY_KW = 5000.0
EXISTING_PV_CAPACITY_KW = 9231.0

# Load profile configuration
LOAD_TEMPLATE_PATHS = {
    "weekday": PROJECT_ROOT / "data" / "raw" / "weekday.csv",
    "saturday": PROJECT_ROOT / "data" / "raw" / "saturday.csv",
    "sunday": PROJECT_ROOT / "data" / "raw" / "sunday.csv",
}
LOAD_TEMPLATE_ANCHORS = {
    "Jan": 1,
    "Apr": 4,
    "Jul": 7,
    "Oct": 10,
}
LOAD_MONTHLY_PLOT = MonthlyAveragePlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "load" / "monthly_average_load.png",
    bar_color="#1f77b4",
    peak_color="#d62728",
    peak_linestyle="-o",
    dpi=150,
    title="Burnie Monthly Average and Peak Load",
    ylabel="Load (MW)",
)
LOAD_HEATMAP_PLOT = HeatmapPlotConfig(
    enabled=True,
    figsize=(10, 6),
    path=PROJECT_ROOT / "images" / "load" / "load_time_of_day_heatmap.png",
    cmap="viridis",
    dpi=150,
    title="Burnie Load Heatmap by Month and Hour",
    xlabel="Hour of Day",
    ylabel="Month",
)
LOAD_SEASONAL_PLOT = SeasonalComparisonPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "load" / "seasonal_daily_load.png",
    summer_month=1,
    winter_month=7,
    weekday_type="weekday",
    title="Representative Weekday Load: Summer vs Winter",
    ylabel="Load (MW)",
    xlabel="Hour of Day",
)
LOAD_ANNUAL_ENERGY_PLOT = AnnualEnergyPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "load" / "annual_energy_projection.png",
    dpi=150,
    title="Projected Annual Load Energy",
    xlabel="Year",
    ylabel="Energy (GWh)",
    bar_color="#1f77b4",
)
EXISTING_PV_COMPARISON_PLOT = ExistingPVComparisonPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "load" / "existing_pv_vs_load.png",
    dpi=150,
    title="Monthly Load vs Existing Rooftop PV",
    xlabel="Month",
    ylabel="Energy (GWh)",
    load_label="Net load",
    pv_label="Existing PV",
    load_color="#1f77b4",
    pv_color="#ff7f0e",
)
EXISTING_PV_CONFIG = ExistingPVConfig(
    capacity_kw=EXISTING_PV_CAPACITY_KW,
    profile_path=PROJECT_ROOT / "data" / "processed" / "pv" / "burnie_pv_per_kw_hourly.csv",
    comparison_plot=EXISTING_PV_COMPARISON_PLOT,
)
LOAD_VARIATION_PLOT = VariationComparisonPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "load" / "daily_variation_comparison.png",
    dpi=150,
    title="Representative Day Load: Baseline vs Randomised",
    xlabel="Hour of Day",
    ylabel="Load (MW)",
    baseline_label="Blended profile",
    varied_label="With day-to-day variation",
)
LOAD_TEMPLATE_PLOT = TemplateInputPlotConfig(
    enabled=True,
    figsize=(12, 9),
    path=PROJECT_ROOT / "images" / "load" / "template_anchor_profiles.png",
    dpi=150,
    title="Template Anchor Profiles",
    xlabel="Hour of Day",
    ylabel="Load (MW)",
    legend_loc="upper right",
)
LOAD_PROFILE_CONFIG = LoadProfileConfig(
    target_year=SIMULATION_YEAR,
    load_profile_paths=LOAD_TEMPLATE_PATHS,
    template_anchor_months=LOAD_TEMPLATE_ANCHORS,
    half_hourly_frequency="30min",
    output_half_hourly_path=PROJECT_ROOT / "data" / "processed" / "load" / "burnie_load_half_hourly.csv",
    output_hourly_path=PROJECT_ROOT / "data" / "processed" / "load" / "burnie_load_hourly.csv",
    save_half_hourly=True,
    half_hourly_timestamp_label="timestamp",
    hourly_timestamp_label="timestamp",
    load_column_name="load_mw",
    float_format="%.6f",
    matplotlib_backend="Agg",
    project_lifespan_years=PROJECT_LIFESPAN_YEARS,
    annual_growth_rate=LOAD_ANNUAL_GROWTH_RATE,
    monthly_plot=LOAD_MONTHLY_PLOT,
    heatmap_plot=LOAD_HEATMAP_PLOT,
    seasonal_plot=LOAD_SEASONAL_PLOT,
    annual_energy_plot=LOAD_ANNUAL_ENERGY_PLOT,
    existing_pv=EXISTING_PV_CONFIG,
    day_variation_fraction=0.15,
    variation_random_seed=1,
    variation_plot=LOAD_VARIATION_PLOT,
    template_plot=LOAD_TEMPLATE_PLOT,
)

# Solar resource configuration
SOLAR_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "T2M",
    "WS10M",
    "PS",
    "RH2M",
]
SOLAR_COLUMN_RENAMES = {
    "ALLSKY_SFC_SW_DWN": "ghi_wm2",
    "CLRSKY_SFC_SW_DWN": "ghi_clearsky_wm2",
    "ALLSKY_SFC_SW_DNI": "dni_wm2",
    "ALLSKY_SFC_SW_DIFF": "dhi_wm2",
    "T2M": "temp_air_c",
    "WS10M": "wind_10m_ms",
    "PS": "surface_pressure_kpa",
    "RH2M": "relative_humidity_pct",
}
SOLAR_MONTHLY_PLOT = SolarMonthlyGHIPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "solar" / "solar_monthly_ghi.png",
    bar_color="#ff7f0e",
    line_color="#1f77b4",
    percentile=0.9,
    percentile_label="P90 Hourly GHI",
    dpi=150,
    title="Burnie Monthly Solar Resource (GHI)",
    ylabel="Irradiance (W/m²)",
    xlabel="Month",
)
SOLAR_DIURNAL_PLOT = MonthlyDiurnalPlotConfig(
    enabled=True,
    figsize=(10, 6),
    path=PROJECT_ROOT / "images" / "solar" / "solar_monthly_diurnal.png",
    title="Average Hourly Solar Irradiance by Month",
    xlabel="Hour of Day",
    ylabel="GHI (W/m²)",
    legend_loc="upper right",
)
SOLAR_HEATMAP_PLOT = GHIHeatmapPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "solar" / "solar_ghi_heatmap.png",
    cmap="inferno",
    dpi=150,
    title="Solar Irradiance Heatmap (Day of Year × Hour)",
    xlabel="Hour of Day",
    ylabel="Day of Year",
    month_ticks=True,
)
SOLAR_RESOURCE_CONFIG = SolarResourceConfig(
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    target_year=SIMULATION_YEAR,
    endpoint="https://power.larc.nasa.gov/api/temporal/hourly/point",
    community="RE",
    user_agent="burnie-microgrid-analyst",
    parameters=SOLAR_PARAMETERS,
    raw_response_path=PROJECT_ROOT / "data" / "raw" / "solar" / f"nasa_power_hourly_{SIMULATION_YEAR}.json",
    processed_data_path=PROJECT_ROOT / "data" / "processed" / "solar" / "burnie_solar_resource_hourly.csv",
    save_raw_response=True,
    use_cached_response=True,
    local_timezone="Australia/Hobart",
    expected_frequency="1h",
    fill_method="interpolate",
    column_renames=SOLAR_COLUMN_RENAMES,
    matplotlib_backend="Agg",
    matplotlib_rc={},
    float_format="%.6f",
    request_timeout=60,
    monthly_plot=SOLAR_MONTHLY_PLOT,
    diurnal_plot=SOLAR_DIURNAL_PLOT,
    heatmap_plot=SOLAR_HEATMAP_PLOT,
)

# Wind resource configuration
WIND_PARAMETERS = [
    "WS10M",
    "WS50M",
    "T2M",
    "PS",
    "RH2M",
]
WIND_COLUMN_RENAMES = {
    "WS10M": "wind_10m_ms",
    "WS50M": "wind_50m_ms",
    "T2M": "temp_air_c",
    "PS": "surface_pressure_kpa",
    "RH2M": "relative_humidity_pct",
}
WIND_MONTHLY_PLOT = MonthlyWindPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "wind" / "wind_monthly_speed.png",
    bar_color="#1f77b4",
    line_color="#ff7f0e",
    percentile=0.9,
    percentile_label="P90 Hourly Speed",
    dpi=150,
    title="Burnie Monthly Wind Speeds at Hub Height",
    ylabel="Wind Speed (m/s)",
    xlabel="Month",
)
WIND_DIURNAL_PLOT = WindDiurnalPlotConfig(
    enabled=True,
    figsize=(10, 6),
    path=PROJECT_ROOT / "images" / "wind" / "wind_monthly_diurnal.png",
    title="Average Hourly Wind Speed by Month",
    xlabel="Hour of Day",
    ylabel="Wind Speed (m/s)",
    legend_loc="upper right",
)
WIND_HEATMAP_PLOT = WindHeatmapPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "wind" / "wind_speed_heatmap.png",
    cmap="viridis",
    dpi=150,
    title="Wind Speed Heatmap (Day of Year × Hour)",
    xlabel="Hour of Day",
    ylabel="Day of Year",
    month_ticks=True,
)
WIND_HISTOGRAM_PLOT = WindHistogramPlotConfig(
    enabled=True,
    figsize=(8, 5),
    path=PROJECT_ROOT / "images" / "wind" / "wind_speed_histogram.png",
    bins=30,
    color="#2ca02c",
    title="Wind Speed Distribution at Hub Height",
    xlabel="Wind Speed (m/s)",
    ylabel="Hours",
)
WIND_HUB_HEIGHT_M = 112

WIND_RESOURCE_CONFIG = WindResourceConfig(
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    target_year=SIMULATION_YEAR,
    hub_height_m=WIND_HUB_HEIGHT_M,
    reference_height_m=10.0,
    shear_exponent=0.25,
    air_gas_constant=287.05,
    endpoint="https://power.larc.nasa.gov/api/temporal/hourly/point",
    community="RE",
    user_agent="burnie-microgrid-analyst",
    parameters=WIND_PARAMETERS,
    raw_response_path=PROJECT_ROOT / "data" / "raw" / "wind" / f"nasa_power_wind_hourly_{SIMULATION_YEAR}.json",
    processed_data_path=PROJECT_ROOT / "data" / "processed" / "wind" / "burnie_wind_resource_hourly.csv",
    save_raw_response=True,
    use_cached_response=True,
    local_timezone="Australia/Hobart",
    expected_frequency="1h",
    fill_method="interpolate",
    column_renames=WIND_COLUMN_RENAMES,
    matplotlib_backend="Agg",
    matplotlib_rc={},
    float_format="%.6f",
    request_timeout=60,
    monthly_plot=WIND_MONTHLY_PLOT,
    diurnal_plot=WIND_DIURNAL_PLOT,
    heatmap_plot=WIND_HEATMAP_PLOT,
    histogram_plot=WIND_HISTOGRAM_PLOT,
    histogram_comparison_plot=WindHistogramComparisonPlotConfig(
        enabled=True,
        figsize=(9, 5.5),
        path=PROJECT_ROOT / "images" / "wind" / "wind_speed_histogram_comparison.png",
        bins=30,
        color_reference="#1f77b4",
        color_hub="#ff7f0e",
        alpha=0.55,
        label_reference="10 m reference height",
        label_hub=f"{int(WIND_HUB_HEIGHT_M)} m hub height",
        title="Wind Speed Distribution: 10 m vs Hub Height",
        xlabel="Wind Speed (m/s)",
        ylabel="Hours",
        legend_loc="upper right",
    ),
)

# PV performance configuration
PV_MONTHLY_PLOT = PVMonthlyEnergyPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "pv" / "pv_monthly_energy_per_kw.png",
    dpi=150,
    title="Monthly PV Energy Yield per kW DC",
    ylabel="Energy (kWh per kW)",
)
PV_DIURNAL_PLOT = PVDiurnalPlotConfig(
    enabled=True,
    figsize=(10, 6),
    path=PROJECT_ROOT / "images" / "pv" / "pv_monthly_diurnal_per_kw.png",
    title="Average Hourly PV Output by Month",
    xlabel="Hour of Day",
    ylabel="AC Output (kW per kW DC)",
    legend_loc="upper right",
)
PV_HEATMAP_PLOT = PVHeatmapPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "pv" / "pv_output_heatmap.png",
    cmap="magma",
    dpi=150,
    title="PV Output Heatmap (AC kW per kW DC)",
    xlabel="Hour of Day",
    ylabel="Day of Year",
)
PV_MODEL_CONFIG = PVModelConfig(
    solar_resource_path=SOLAR_RESOURCE_CONFIG.processed_data_path,
    output_path=PROJECT_ROOT / "data" / "processed" / "pv" / "burnie_pv_per_kw_hourly.csv",
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    timezone="Australia/Hobart",
    module_tilt_deg=41,
    module_azimuth_deg=0.0,
    temp_coeff_pct_per_c=-0.0045,
    temp_ref_c=25.0,
    noct_c=45.0,
    system_derate=0.92,
    dc_ac_ratio=1.15,
    inverter_efficiency=0.97,
    ground_albedo=0.3,
    float_format="%.6f",
    matplotlib_backend="Agg",
    monthly_plot=PV_MONTHLY_PLOT,
    diurnal_plot=PV_DIURNAL_PLOT,
    heatmap_plot=PV_HEATMAP_PLOT,
)

# Wind turbine performance configuration
WIND_TURBINE_RATED_POWER_KW = 3450
WIND_TURBINE_NAME = f"Generic_{int(WIND_TURBINE_RATED_POWER_KW / 1000)}MW"
WIND_TURBINE_POWER_CURVE = [
    (0.0, 0.0),
    (3.0, 50.0),
    (4.0, 250.0),
    (5.0, 500.0),
    (6.0, 850.0),
    (7.0, 1250.0),
    (8.0, 1750.0),
    (9.0, 2350.0),
    (10.0, 3100.0),
    (11.0, 3450.0),
    (12.0, 3450.0),
    (13.0, 3450.0),
    (14.0, 3450.0),
    (15.0, 3450.0),
    (16.0, 3450.0),
    (17.0, 3450.0),
    (18.0, 3450.0),
    (19.0, 3450.0),
    (20.0, 3450.0),
    (21.0, 3450.0),
    (22.5, 3450.0),
    (25.0, 3450.0)
]
WIND_TURBINE_CONFIG = WindTurbineConfig(
    name=WIND_TURBINE_NAME,
    rated_power_kw=WIND_TURBINE_RATED_POWER_KW,
    cut_in_ms=3.0,
    rated_speed_ms=10.5,
    cut_out_ms=25.0,
    drivetrain_efficiency=0.95,
    air_density_exponent=1.0,
    power_curve_points=WIND_TURBINE_POWER_CURVE,
)
TURBINE_MONTHLY_PLOT = WindMonthlyEnergyPlotConfig(
    enabled=True,
    figsize=(10, 5),
    path=PROJECT_ROOT / "images" / "wind" / "turbine_monthly_energy_per_unit.png",
    dpi=150,
    title_template="Monthly Energy Yield per {turbine_name}",
    ylabel="Energy (MWh per turbine)",
)
TURBINE_DIURNAL_PLOT = TurbineDiurnalPlotConfig(
    enabled=True,
    figsize=(10, 6),
    path=PROJECT_ROOT / "images" / "wind" / "turbine_monthly_diurnal_per_unit.png",
    title_template="Average Hourly Output by Month ({turbine_name})",
    xlabel="Hour of Day",
    ylabel="Power (kW per turbine)",
    legend_loc="upper right",
)
TURBINE_HEATMAP_PLOT = TurbineHeatmapPlotConfig(
    enabled=True,
    figsize=(12, 6),
    path=PROJECT_ROOT / "images" / "wind" / "turbine_output_heatmap.png",
    cmap="Blues",
    dpi=150,
    title_template="Turbine Output Heatmap ({turbine_name})",
    xlabel="Hour of Day",
    ylabel="Day of Year",
)
TURBINE_CAPACITY_PLOT = WindCapacityHistogramConfig(
    enabled=True,
    figsize=(8, 5),
    path=PROJECT_ROOT / "images" / "wind" / "turbine_capacity_factor_hist.png",
    bins=30,
    color="#ff7f0e",
    title_template="Capacity Factor Distribution ({turbine_name})",
    xlabel="Capacity Factor",
    ylabel="Hours",
)
WIND_PERFORMANCE_CONFIG = WindPerformanceConfig(
    wind_resource_path=WIND_RESOURCE_CONFIG.processed_data_path,
    output_path=PROJECT_ROOT / "data" / "processed" / "wind" / "burnie_wind_per_turbine_hourly.csv",
    turbine=WIND_TURBINE_CONFIG,
    reference_air_density=1.225,
    float_format="%.6f",
    matplotlib_backend="Agg",
    monthly_plot=TURBINE_MONTHLY_PLOT,
    diurnal_plot=TURBINE_DIURNAL_PLOT,
    heatmap_plot=TURBINE_HEATMAP_PLOT,
    capacity_hist_plot=TURBINE_CAPACITY_PLOT,
)

# System simulation configuration
STORAGE_CONFIG = StorageConfig(
    energy_capacity_kwh=180000.0,  # base energy size; optimiser overwrites per candidate
    charge_power_kw=30000.0,      # base charge limit; optimiser overwrites
    discharge_power_kw=30000.0,   # base discharge limit; optimiser overwrites
    charge_efficiency=0.97,       # retained (affects dispatch physics)
    discharge_efficiency=0.97,    # retained
    min_soc_fraction=0.1,         # retained (SOC floor)
    max_soc_fraction=1.0,         # retained (SOC ceiling)
    initial_soc_fraction=0.5,     # retained (starting SOC each run)
)
PHES_CONFIG = StorageConfig(
    energy_capacity_kwh=6_000_000.0,   # ~6.0 GWh delivered
    charge_power_kw=780_000.0,         # ~780 MW pump input (4 × 195 MW)
    discharge_power_kw=750_000.0,      # ~750 MW generation
    charge_efficiency=0.89,            # pump efficiency
    discharge_efficiency=0.92,         # turbine efficiency
    min_soc_fraction=0.05,
    max_soc_fraction=1.0,
    initial_soc_fraction=0.90,
)
GENERATOR_CONFIG = GeneratorConfig(
    max_power_kw=50000.0,         # base capacity; optimiser overwrites
    min_power_kw=5000,          # base min load; optimiser resets to fraction of candidate max of 
    fuel_litre_per_kwh=0.27,      # retained (fuel burn)
    fuel_cost_per_litre=1,      # retained (fuel price)
    variable_om_per_kwh=0.02,     # retained (variable O&M)
)
SIMULATION_INPUTS = SimulationInputsConfig(
    load_profile_path=LOAD_PROFILE_CONFIG.output_hourly_path,  # retained (data source)
    pv_profile_path=PV_MODEL_CONFIG.output_path,               # retained
    wind_profile_path=WIND_PERFORMANCE_CONFIG.output_path,     # retained
    output_timeseries_path=PROJECT_ROOT / "outputs" / "system" / "dispatch_timeseries.csv",
    output_summary_path=PROJECT_ROOT / "outputs" / "system" / "dispatch_summary.json",
)
DISPATCH_CONFIG = DispatchConfig(
    pv_capacity_kw=80000,       # base PV size; optimiser overwrites
    wind_turbine_count=14,         # base turbine count; optimiser overwrites
    storage=STORAGE_CONFIG,       # template; optimiser swaps energy/power
    generator=GENERATOR_CONFIG,   # template; optimiser swaps capacities
    timestep_hours=1.0,           # retained (dispatch resolution)
    pumped_hydro=PHES_CONFIG,     # pumped hydro storage (fixed for now)
)
SYSTEM_SIMULATION_CONFIG = SystemSimulationConfig(
    inputs=SIMULATION_INPUTS,
    dispatch=DISPATCH_CONFIG,
    float_format="%.6f",
)

# Cost model configuration
COST_PV_PARAMETERS = PVCostParameters(
    panel_power_kw=0.44,  # 440 W module
    panel_cost=120.0,
    panel_area_m2=2,
    land_cost_per_m2=25,
    bos_cost_per_kw=270,
    fixed_om_per_kw_year=20,
)
COST_WIND_PARAMETERS = WindCostParameters(
    turbine_rated_kw=WIND_TURBINE_RATED_POWER_KW,
    turbine_cost=7000000.0,
    balance_of_plant_per_kw=400.0,
    fixed_om_per_kw_year=50.0,
)
COST_MODEL_CONFIG = CostModelConfig(
    lifetime_years=PROJECT_LIFESPAN_YEARS,
    discount_rate=0.08,
    pv_costs=COST_PV_PARAMETERS,
    wind_costs=COST_WIND_PARAMETERS,
    storage_capex_per_kwh=400,
    storage_capex_per_kw=800,
    storage_fixed_om_per_kwh_year=10.0,
    storage_replacement_year=20,
    storage_replacement_fraction=0.6,
    generator_capex_per_kw=900.0,
    generator_fixed_om_per_kw_year=20.0,
    write_output_path=PROJECT_ROOT / "outputs" / "system" / "cost_summary.json",
    currency="AUD",
    curtailment_cost_per_mwh=50.0,
    # PHES cost model: 30,000 AUD/MWh capex and 400 AUD/MWh-yr O&M
    phes_capex_fixed=0.0,
    phes_capex_per_kwh=30.0,
    phes_fixed_om_per_kwh_year=0.4,
)

# Optimisation configuration
OPTIMISATION_SEARCH_SPACE = OptimisationSearchSpace(
    # Expanded grid ~20,480 runs (uses parallel sweep)
    pv_capacity_kw=FloatRange(minimum=80000, maximum=140000, step=20000),             # 80k..140k (4)
    wind_turbine_count=IntRange(minimum=12, maximum=18, step=2),                      # 12,14,16,18 (4)
    storage_energy_kwh=FloatRange(minimum=0, maximum=300000, step=100000),            # 0,100k,200k,300k (4)
    storage_power_kw=FloatRange(minimum=0, maximum=60000, step=20000),                # 0,20k,40k,60k (4)
    generator_capacity_kw=FloatRange(minimum=30000.0, maximum=90000.0, step=20000.0), # 30k,50k,70k,90k (4)
    generator_min_power_fraction=0.05,
    # PHES search space (finer: ~11 steps each)
    phes_energy_kwh=FloatRange(minimum=0.0, maximum=1600000.0, step=160000.0),        # 0..1.6M in 160k steps (~11 values)
    phes_power_kw=FloatRange(minimum=0.0, maximum=120000.0, step=12000.0),            # 0..120k in 12k steps (~11 values)
)
OPTIMISATION_CONSTRAINTS = OptimisationConstraints(
    min_renewable_penetration=0.8,
    max_unserved_energy_mwh=5.0,
)
OPTIMISATION_OUTPUTS = OptimisationOutputs(
    results_csv_path=PROJECT_ROOT / "outputs" / "optimisation" / "results.csv",
    best_result_path=PROJECT_ROOT / "outputs" / "optimisation" / "best_result.json",
    best_cost_path=PROJECT_ROOT / "outputs" / "optimisation" / "best_cost_summary.json",
    cost_candidate_path=PROJECT_ROOT / "outputs" / "optimisation" / "candidate_cost.json",
    component_summary_path=PROJECT_ROOT / "outputs" / "optimisation" / "component_summary.json",
)
OPTIMISATION_CONFIG = OptimisationConfig(
    base_simulation_config=SYSTEM_SIMULATION_CONFIG,
    base_cost_config=COST_MODEL_CONFIG,
    search_space=OPTIMISATION_SEARCH_SPACE,
    constraints=OPTIMISATION_CONSTRAINTS,
    turbine_rated_kw=WIND_TURBINE_RATED_POWER_KW,
    outputs=OPTIMISATION_OUTPUTS,
    bess_unit_energy_mwh=BESS_UNIT_ENERGY_MWH,
    bess_unit_power_mw=BESS_UNIT_POWER_MW,
    bess_unit_name=BESS_UNIT_NAME,
    generator_unit_capacity_kw=GENERATOR_UNIT_CAPACITY_KW,
    generator_unit_name=GENERATOR_UNIT_NAME,
)

# Pipeline toggles
RUN_LOAD_PROFILE = True
RUN_SOLAR_RESOURCE = True
RUN_WIND_RESOURCE = True
RUN_PV_PERFORMANCE = True
RUN_WIND_PERFORMANCE = True
RUN_SYSTEM_SIMULATION = True
RUN_OPTIMISATION = True
# -----------------------------


def calculate_annual_energy_outputs(timeseries_path: Path, timestep_hours: float) -> dict[str, float]:
    """Return annual energy outputs (MWh) for PV, wind, generator, and load."""

    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be positive")

    df = pd.read_csv(timeseries_path)
    required_columns = {"pv_kw", "wind_kw", "generator_kw", "load_kw"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in dispatch results: {missing}")

    scale = timestep_hours / 1000.0
    return {
        "pv_energy_mwh": df["pv_kw"].sum() * scale,
        "wind_energy_mwh": df["wind_kw"].sum() * scale,
        "generator_energy_mwh": df["generator_kw"].sum() * scale,
        "load_energy_mwh": df["load_kw"].sum() * scale,
    }


def calculate_asset_summary(
    pv_capacity_kw: float,
    wind_turbine_count: float,
    storage_energy_kwh: float,
    storage_power_kw: float,
    generator_capacity_kw: float,
    panel_power_kw: float,
    phes_energy_kwh: float = 0.0,
    phes_power_kw: float = 0.0,
) -> dict[str, float]:
    """Return rounded component counts and capacities for reporting."""

    if panel_power_kw <= 0:
        panel_count = 0
    else:
        panel_count = int(round(pv_capacity_kw / panel_power_kw))

    return {
        "wind_turbine_count": int(round(wind_turbine_count)),
        "pv_panel_count": panel_count,
        "generator_capacity_kw": generator_capacity_kw,
        "battery_energy_kwh": storage_energy_kwh,
        "battery_power_kw": storage_power_kw,
        "phes_energy_kwh": phes_energy_kwh,
        "phes_power_kw": phes_power_kw,
    }


def _format_energy_asset_line(label: str, energy_outputs: dict[str, float], assets: dict[str, float]) -> str:
    load = energy_outputs.get("load_energy_mwh", 0.0)
    pv = energy_outputs.get("pv_energy_mwh", 0.0)
    wind = energy_outputs.get("wind_energy_mwh", 0.0)
    generator = energy_outputs.get("generator_energy_mwh", 0.0)
    supply = pv + wind + generator
    balance = supply - load

    phes_e = assets.get("phes_energy_kwh", 0.0)
    phes_p = assets.get("phes_power_kw", 0.0)
    return (
        f"{label}: "
        f"Load={load:.1f} MWh | Supply PV={pv:.1f} MWh, Turbines={wind:.1f} MWh, "
        f"Generator={generator:.1f} MWh | Balance={balance:.1f} MWh | "
        f"Assets: Turbines={assets['wind_turbine_count']}, PV panels={assets['pv_panel_count']}, "
        f"Generator={assets['generator_capacity_kw']:.0f} kW, "
        f"Battery={assets['battery_energy_kwh']:.0f} kWh/{assets['battery_power_kw']:.0f} kW, "
        f"PHES={phes_e:.0f} kWh/{phes_p:.0f} kW"
    )


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _format_search_space_table(search_space: OptimisationSearchSpace) -> str:
    rows = [
        ("PV capacity (kW)", search_space.pv_capacity_kw),
        ("Wind turbines (count)", search_space.wind_turbine_count),
        ("Storage energy (kWh)", search_space.storage_energy_kwh),
        ("Storage power (kW)", search_space.storage_power_kw),
        ("Generator capacity (kW)", search_space.generator_capacity_kw),
    ]

    # Optional PHES ranges
    if getattr(search_space, "phes_energy_kwh", None) is not None:
        rows.append(("PHES energy (kWh)", search_space.phes_energy_kwh))
    if getattr(search_space, "phes_power_kw", None) is not None:
        rows.append(("PHES power (kW)", search_space.phes_power_kw))

    body_lines = []
    for label, range_cfg in rows:
        minimum = _format_number(range_cfg.minimum)
        maximum = _format_number(range_cfg.maximum)
        step = _format_number(range_cfg.step)
        body_lines.append(f"{label} & {minimum} & {maximum} & {step} \\")

    table_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Component & Min & Max & Step \\",
        "\\hline",
        *body_lines,
        "\\hline",
        "\\end{tabular}",
        "\\caption{Optimisation search space ranges.}",
        "\\end{table}",
    ]

    return "\n".join(table_lines)


def main() -> None:
    if RUN_LOAD_PROFILE:
        build_load_profiles(LOAD_PROFILE_CONFIG)

    if RUN_SOLAR_RESOURCE:
        build_solar_resource(SOLAR_RESOURCE_CONFIG)

    if RUN_WIND_RESOURCE:
        build_wind_resource(WIND_RESOURCE_CONFIG)

    if RUN_PV_PERFORMANCE:
        build_pv_performance(PV_MODEL_CONFIG)

    if RUN_WIND_PERFORMANCE:
        build_wind_performance(WIND_PERFORMANCE_CONFIG)

    dispatch_summary = None
    if RUN_SYSTEM_SIMULATION:
        dispatch_summary = run_system_simulation(SYSTEM_SIMULATION_CONFIG)

        cost_inputs = CostModelInputs(
            pv_capacity_kw=DISPATCH_CONFIG.pv_capacity_kw,
            wind_capacity_kw=WIND_TURBINE_RATED_POWER_KW * DISPATCH_CONFIG.wind_turbine_count,
            storage_energy_kwh=STORAGE_CONFIG.energy_capacity_kwh,
            storage_power_kw=STORAGE_CONFIG.discharge_power_kw,
            generator_capacity_kw=GENERATOR_CONFIG.max_power_kw,
            load_energy_mwh_per_year=dispatch_summary.get("load_energy_mwh", 0.0),
            phes_energy_kwh=(DISPATCH_CONFIG.pumped_hydro.energy_capacity_kwh if DISPATCH_CONFIG.pumped_hydro else 0.0),
            phes_power_kw=(DISPATCH_CONFIG.pumped_hydro.discharge_power_kw if DISPATCH_CONFIG.pumped_hydro else 0.0),
        )
        evaluate_costs(dispatch_summary, cost_inputs, COST_MODEL_CONFIG)

        energy_outputs = calculate_annual_energy_outputs(
            SIMULATION_INPUTS.output_timeseries_path,
            DISPATCH_CONFIG.timestep_hours,
        )
        asset_summary = calculate_asset_summary(
            pv_capacity_kw=dispatch_summary.get("pv_capacity_kw", DISPATCH_CONFIG.pv_capacity_kw),
            wind_turbine_count=dispatch_summary.get("wind_turbine_count", DISPATCH_CONFIG.wind_turbine_count),
            storage_energy_kwh=dispatch_summary.get(
                "storage_energy_capacity_kwh", DISPATCH_CONFIG.storage.energy_capacity_kwh
            ),
            storage_power_kw=dispatch_summary.get(
                "storage_power_discharge_kw", DISPATCH_CONFIG.storage.discharge_power_kw
            ),
            generator_capacity_kw=dispatch_summary.get("generator_capacity_kw", GENERATOR_CONFIG.max_power_kw),
            panel_power_kw=COST_PV_PARAMETERS.panel_power_kw,
            phes_energy_kwh=dispatch_summary.get("phes_energy_capacity_kwh", 0.0),
            phes_power_kw=dispatch_summary.get("phes_power_discharge_kw", 0.0),
        )
        dispatch_summary.update(energy_outputs)
        dispatch_summary.update(asset_summary)

        summary_path = SIMULATION_INPUTS.output_summary_path
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(dispatch_summary, file, indent=2)

        print(_format_energy_asset_line("Dispatch", energy_outputs, asset_summary))

    if RUN_OPTIMISATION:
        print(_format_search_space_table(OPTIMISATION_SEARCH_SPACE))
        best_result = run_optimisation(OPTIMISATION_CONFIG)

        energy_outputs = calculate_annual_energy_outputs(
            SIMULATION_INPUTS.output_timeseries_path,
            DISPATCH_CONFIG.timestep_hours,
        )
        asset_summary = calculate_asset_summary(
            pv_capacity_kw=best_result.get("pv_capacity_kw", DISPATCH_CONFIG.pv_capacity_kw),
            wind_turbine_count=best_result.get("wind_turbine_count", DISPATCH_CONFIG.wind_turbine_count),
            storage_energy_kwh=best_result.get("storage_energy_kwh", DISPATCH_CONFIG.storage.energy_capacity_kwh),
            storage_power_kw=best_result.get("storage_power_kw", DISPATCH_CONFIG.storage.discharge_power_kw),
            generator_capacity_kw=best_result.get("generator_capacity_kw", GENERATOR_CONFIG.max_power_kw),
            panel_power_kw=COST_PV_PARAMETERS.panel_power_kw,
            phes_energy_kwh=best_result.get("phes_energy_kwh", 0.0),
            phes_power_kw=best_result.get("phes_power_kw", 0.0),
        )

        summary_path = SIMULATION_INPUTS.output_summary_path
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file:
                summary_data = json.load(file)
        else:
            summary_data = {}
        summary_data.update(energy_outputs)
        summary_data.update(asset_summary)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary_data, file, indent=2)

        print(_format_energy_asset_line("Optimised design", energy_outputs, asset_summary))


if __name__ == "__main__":
    main()
