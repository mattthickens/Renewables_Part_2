import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_npc_over_time(best_cost_path, output_path, discount_rate=0.06):
    with open(best_cost_path, "r") as f:
        cost_summary = json.load(f)
    # Extract component costs
    lifetime = cost_summary.get("lifetime_years", 25)
    pv_capex = cost_summary.get("pv_capex", 0.0)
    wind_capex = cost_summary.get("wind_capex", 0.0)
    storage_capex = cost_summary.get("storage_capex", 0.0)
    generator_capex = cost_summary.get("generator_capex", 0.0)
    annual_pv_om = cost_summary.get("annual_pv_om", 0.0)
    annual_wind_om = cost_summary.get("annual_wind_om", 0.0)
    annual_storage_om = cost_summary.get("annual_storage_om", 0.0)
    annual_generator_om = cost_summary.get("annual_generator_om", 0.0)
    annual_fuel = cost_summary.get("fuel_cost", 0.0)
    annual_gen_var_om = cost_summary.get("generator_variable_om_cost", 0.0)
    storage_replacement_year = cost_summary.get("storage_replacement_year", 20)
    storage_replacement_cost = cost_summary.get("storage_capex", 0.0) * cost_summary.get("storage_replacement_fraction", 0.6)
    # Build discounted cash flows
    years = list(range(lifetime + 1))
    pv = [pv_capex] + [annual_pv_om for _ in range(lifetime)]
    wind = [wind_capex] + [annual_wind_om for _ in range(lifetime)]
    bess = [storage_capex] + [annual_storage_om for _ in range(lifetime)]
    generator = [generator_capex] + [annual_generator_om + annual_fuel + annual_gen_var_om for _ in range(lifetime)]
    # Add storage replacement
    if 0 < storage_replacement_year < lifetime:
        bess[storage_replacement_year] += storage_replacement_cost
    # Discount each year
    discount_factors = [(1 + discount_rate) ** (-y) for y in years]
    pv_disc = [v * d for v, d in zip(pv, discount_factors)]
    wind_disc = [v * d for v, d in zip(wind, discount_factors)]
    bess_disc = [v * d for v, d in zip(bess, discount_factors)]
    generator_disc = [v * d for v, d in zip(generator, discount_factors)]
    # Cumulative sums
    pv_cum = pd.Series(pv_disc).cumsum()
    wind_cum = pd.Series(wind_disc).cumsum()
    bess_cum = pd.Series(bess_disc).cumsum()
    generator_cum = pd.Series(generator_disc).cumsum()
    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(years, pv_cum/1e6, label="PV", color="#1f77b4", linewidth=2)
    plt.plot(years, wind_cum/1e6, label="WIND", color="#ff7f0e", linewidth=2)
    plt.plot(years, bess_cum/1e6, label="BESS", color="#2ca02c", linewidth=2)
    plt.plot(years, generator_cum/1e6, label="GENERATOR", color="#d62728", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Cumulative NPC (Million AUD)")
    plt.title("Cumulative NPC Over Time by Component (Discount Rate 0.06)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    best_cost_path = "outputs/optimisation/best_cost_summary.json"
    output_path = "images/finance/npc_over_time.png"
    plot_cumulative_npc_over_time(best_cost_path, output_path, discount_rate=0.06)
