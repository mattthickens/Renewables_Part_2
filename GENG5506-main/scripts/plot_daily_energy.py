"""Daily energy overview plots from dispatch timeseries.

Reads `outputs/system/dispatch_timeseries.csv` written by the system
simulation and produces a two-panel figure showing:
  1) Daily energy (MWh): stacked PV/Wind/Generator vs Load line
  2) Daily storage: net BESS/PHES energy (MWh, discharge positive) and
     end-of-day state-of-charge (SOC, MWh) on a secondary axis.

Usage:
  - Default paths: `python -m scripts.plot_daily_energy`
  - Custom CSV:    `python -m scripts.plot_daily_energy path/to/dispatch_timeseries.csv`

Output:
  - images/finance/daily_overview.png (daily view)
  - images/finance/typical_day.png (typical 24-hour view)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


def _ensure_backend(backend: str = "Agg") -> None:
    try:
        matplotlib.use(backend)
        import matplotlib.pyplot as _  # noqa: F401  (ensure backend loads)
    except Exception:
        pass


def _load_timeseries(csv_path: Path):
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"Dispatch timeseries not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if df.empty:
        raise ValueError("Dispatch timeseries CSV is empty.")

    required = {"load_kw", "pv_kw", "wind_kw", "generator_kw", "bess_charge_kw", "bess_discharge_kw", "bess_soc_kwh"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Allow PHES to be optional
        optional_ok = {"bess_charge_kw", "bess_discharge_kw", "bess_soc_kwh"}
        still_missing = [c for c in missing if c not in optional_ok]
        if still_missing:
            missing_str = ", ".join(still_missing)
            raise ValueError(f"Missing required columns in dispatch results: {missing_str}")

    return df


def _aggregate_daily(df, timestep_hours: float = 1.0):
    import pandas as pd

    scale = timestep_hours / 1000.0  # kW * h -> MWh
    daily = pd.DataFrame(index=df.resample("D").sum().index)

    # Energy components (MWh/day)
    for col in ("load_kw", "pv_kw", "wind_kw", "generator_kw"):
        if col in df.columns:
            daily[col.replace("_kw", "_mwh")] = df[col].resample("D").sum() * scale

    # Storage net energy (discharge positive, charge negative), MWh/day
    if {"bess_charge_kw", "bess_discharge_kw"}.issubset(df.columns):
        daily["bess_net_mwh"] = (df["bess_discharge_kw"].resample("D").sum() - df["bess_charge_kw"].resample("D").sum()) * scale
    if {"phes_charge_kw", "phes_discharge_kw"}.issubset(df.columns):
        daily["phes_net_mwh"] = (df["phes_discharge_kw"].resample("D").sum() - df["phes_charge_kw"].resample("D").sum()) * scale

    # End-of-day SOC (MWh)
    if "bess_soc_kwh" in df.columns:
        daily["bess_soc_mwh_eod"] = df["bess_soc_kwh"].resample("D").last() / 1000.0
    if "phes_soc_kwh" in df.columns:
        daily["phes_soc_mwh_eod"] = df["phes_soc_kwh"].resample("D").last() / 1000.0

    return daily


def _aggregate_typical_day(df, timestep_hours: float = 1.0):
    import pandas as pd

    # Ensure hourly frequency; compute average per hour of day
    df = df.sort_index()
    grp = df.groupby(df.index.hour)

    # Power (kW) -> MW for plotting typical profile
    def to_mw(series):
        return grp[series].mean() / 1000.0 if series in df.columns else None

    out = pd.DataFrame({
        "hour": range(24),
    }).set_index("hour")

    for col in ("load_kw", "pv_kw", "wind_kw", "generator_kw"):
        s = to_mw(col)
        if s is not None:
            out[col.replace("_kw", "_mw")] = s

    # Storage net power (MW): discharge positive, charge negative
    if {"bess_charge_kw", "bess_discharge_kw"}.issubset(df.columns):
        net_kw = grp["bess_discharge_kw"].mean() - grp["bess_charge_kw"].mean()
        out["bess_net_mw"] = net_kw / 1000.0
    if {"phes_charge_kw", "phes_discharge_kw"}.issubset(df.columns):
        net_kw = grp["phes_discharge_kw"].mean() - grp["phes_charge_kw"].mean()
        out["phes_net_mw"] = net_kw / 1000.0

    # Average SOC per hour (MWh)
    if "bess_soc_kwh" in df.columns:
        out["bess_soc_mwh"] = grp["bess_soc_kwh"].mean() / 1000.0
    if "phes_soc_kwh" in df.columns:
        out["phes_soc_mwh"] = grp["phes_soc_kwh"].mean() / 1000.0

    return out


def _plot_daily_overview(daily, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)

    # Panel 1: Daily generation vs load
    pv = daily.get("pv_mwh")
    wind = daily.get("wind_mwh")
    gen = daily.get("generator_mwh")
    load = daily.get("load_mwh")

    stacks = []
    labels = []
    colors = []
    if pv is not None:
        stacks.append(pv)
        labels.append("PV")
        colors.append("#f7c948")
    if wind is not None:
        stacks.append(wind)
        labels.append("Wind")
        colors.append("#76b7e5")
    if gen is not None:
        stacks.append(gen)
        labels.append("Generator")
        colors.append("#d62728")

    if stacks:
        ax1.stackplot(daily.index, stacks, labels=labels, colors=colors, alpha=0.8)
    if load is not None:
        ax1.plot(daily.index, load, color="black", linewidth=1.8, label="Load")

    ax1.set_title("Daily Energy: Supply vs Load")
    ax1.set_ylabel("Energy (MWh/day)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", ncol=4)

    # Panel 2: Storage daily net flow + end-of-day SOC
    bess_net = daily.get("bess_net_mwh")
    phes_net = daily.get("phes_net_mwh")
    bess_soc = daily.get("bess_soc_mwh_eod")
    phes_soc = daily.get("phes_soc_mwh_eod")

    # Bars for net energy
    width = 0.9
    if bess_net is not None:
        ax2.bar(daily.index, bess_net, width=width, color="#2ca02c", alpha=0.5, label="BESS net (MWh)")
    if phes_net is not None:
        ax2.bar(daily.index, phes_net, width=width, color="#ff7f0e", alpha=0.4, label="PHES net (MWh)")

    ax2.set_ylabel("Net Storage Energy (MWh/day)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for SOC
    ax2b = ax2.twinx()
    soc_handles = []
    if bess_soc is not None:
        h1 = ax2b.plot(daily.index, bess_soc, color="#1b7f3a", linewidth=1.5, label="BESS SOC EoD (MWh)")[0]
        soc_handles.append(h1)
    if phes_soc is not None:
        h2 = ax2b.plot(daily.index, phes_soc, color="#c65102", linewidth=1.5, label="PHES SOC EoD (MWh)")[0]
        soc_handles.append(h2)
    ax2b.set_ylabel("SOC End of Day (MWh)")

    # Combined legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + soc_handles, l1 + [h.get_label() for h in soc_handles], loc="upper left", ncol=3)

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_typical_day(
    typical,
    output_path: Path,
    *,
    bess_capacity_mwh: float | None = None,
    phes_capacity_mwh: float | None = None,
    soc_as_percent: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    hours = typical.index
    fig, ax1 = plt.subplots(figsize=(12, 6))

    load = typical.get("load_mw")
    pv = typical.get("pv_mw")
    wind = typical.get("wind_mw")
    gen = typical.get("generator_mw")

    if load is not None:
        ax1.plot(hours, load, color="black", linewidth=2, label="Load")
    if pv is not None:
        ax1.plot(hours, pv, color="#f7c948", label="PV")
    if wind is not None:
        ax1.plot(hours, wind, color="#76b7e5", label="Wind")
    if gen is not None:
        ax1.plot(hours, gen, color="#d62728", label="Generator")

    ax1.set_title("Typical 24-hour Profile (Average by Hour)")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Power (MW)")
    ax1.set_xticks(list(range(0, 24, 2)))
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Storage on secondary axis
    ax2 = ax1.twinx()
    bess_net = typical.get("bess_net_mw")
    phes_net = typical.get("phes_net_mw")
    bess_soc = typical.get("bess_soc_mwh")
    phes_soc = typical.get("phes_soc_mwh")

    # Optionally convert SOC to percent of capacity for readability
    if soc_as_percent:
        if bess_soc is not None and (bess_capacity_mwh or bess_soc.max() > 0):
            cap = bess_capacity_mwh if bess_capacity_mwh and bess_capacity_mwh > 0 else float(bess_soc.max())
            bess_soc = (bess_soc / cap) * 100.0
        if phes_soc is not None and (phes_capacity_mwh or phes_soc.max() > 0):
            cap = phes_capacity_mwh if phes_capacity_mwh and phes_capacity_mwh > 0 else float(phes_soc.max())
            phes_soc = (phes_soc / cap) * 100.0

    # Bars for net storage power
    width = 0.8
    if bess_net is not None:
        ax2.bar(hours, bess_net, width=width, color="#2ca02c", alpha=0.35, label="BESS net (MW)")
    if phes_net is not None:
        ax2.bar(hours, phes_net, width=width, color="#ff7f0e", alpha=0.25, label="PHES net (MW)")

    # Lines for SOC
    soc_handles = []
    if bess_soc is not None:
        label = "BESS SOC (%)" if soc_as_percent else "BESS SOC (MWh)"
        h1 = ax2.plot(hours, bess_soc, color="#1b7f3a", linewidth=1.6, label=label)[0]
        soc_handles.append(h1)
    if phes_soc is not None:
        label = "PHES SOC (%)" if soc_as_percent else "PHES SOC (MWh)"
        h2 = ax2.plot(hours, phes_soc, color="#c65102", linewidth=1.6, label=label)[0]
        soc_handles.append(h2)
    ax2.set_ylabel("Storage: Net (MW) / SOC (%" + (")" if soc_as_percent else ") (MWh)"))

    # Combined legend
    h1_, l1_ = ax1.get_legend_handles_labels()
    h2_, l2_ = ax2.get_legend_handles_labels()
    ax1.legend(h1_ + h2_, l1_ + l2_, loc="upper left", ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse

    argv = sys.argv[1:] if argv is None else argv

    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "outputs" / "system" / "dispatch_timeseries.csv"
    parser = argparse.ArgumentParser(description="Plot daily or typical-day energy views from dispatch timeseries.")
    parser.add_argument("csv", nargs="?", default=str(default_csv), help="Path to dispatch_timeseries.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--daily", action="store_true", help="Show multi-day overview (daily totals)")
    mode.add_argument("--typical", action="store_true", help="Show typical 24-hour profile (average by hour)")
    parser.add_argument("--soc-absolute", action="store_true", help="Plot SOC in MWh instead of % of capacity")

    args = parser.parse_args(argv)
    csv_path = Path(args.csv).resolve()

    _ensure_backend("Agg")

    df = _load_timeseries(csv_path)
    if args.daily and not args.typical:
        daily = _aggregate_daily(df, timestep_hours=1.0)
        out_path = project_root / "images" / "finance" / "daily_overview.png"
        _plot_daily_overview(daily, out_path)
        print(f"Saved daily overview plot to: {out_path}")
    else:
        typical = _aggregate_typical_day(df, timestep_hours=1.0)
        # Try to read capacities from dispatch summary for SOC normalization
        summary_path = project_root / "outputs" / "system" / "dispatch_summary.json"
        bess_cap_mwh = None
        phes_cap_mwh = None
        try:
            import json
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    s = json.load(f)
                bess_cap_mwh = float(s.get("storage_energy_capacity_kwh", 0.0)) / 1000.0 or None
                phes_cap_mwh = float(s.get("phes_energy_capacity_kwh", 0.0)) / 1000.0 or None
        except Exception:
            pass
        out_path = project_root / "images" / "finance" / "typical_day.png"
        _plot_typical_day(
            typical,
            out_path,
            bess_capacity_mwh=bess_cap_mwh,
            phes_capacity_mwh=phes_cap_mwh,
            soc_as_percent=(not args.soc_absolute),
        )
        print(f"Saved typical 24-hour plot to: {out_path}")


if __name__ == "__main__":
    main()
