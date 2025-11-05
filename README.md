# GENG5506 – Burnie Microgrid Modelling and Optimisation

A reproducible Python pipeline for modelling and optimising a hybrid microgrid system in Burnie, Tasmania. The project includes load/resource preparation, PV and wind performance modelling, dispatch simulation with storage and diesel generation, cost evaluation (NPC/LCOE), and a parallel brute-force optimisation to identify low-cost system designs meeting renewable penetration and reliability targets.

## Project Details
- **Site:** Burnie, Tasmania (lat −41.05, lon 145.90)  
- **Year modelled:** 2023  
- **Project lifetime:** 25 years  
- **Time step:** Hourly  

## Repository Layout
- **main.py** – Full pipeline configuration and runner.  
- **scripts/** – Modular components:  
  - `load_profile.py`, `solar_resource.py`, `wind_resource.py`: Build load and weather inputs.  
  - `pv_model.py`, `wind_model.py`: Convert resources to generation profiles.  
  - `system_simulation.py`: Simulate dispatch (PV, wind, BESS, PHES, generator).  
  - `cost_model.py`: Compute CAPEX, OPEX, NPC, and LCOE.  
  - `optimisation.py`: Run brute-force optimisation and plot results.  
- **data/** –  
  - `raw/`: Templates and cached NASA POWER data.  
  - `processed/`: Generated hourly/half-hourly profiles.  
- **outputs/** – Results, costs, optimisation data, and figures.  
- **images/** – Automatically generated plots (load, solar, wind, PV, finance, optimisation).  

## Requirements
- **Python:** 3.10+ (3.11 recommended)
- **Packages:** `pandas`, `numpy`, `matplotlib`, `requests`
- *(Optional)* `seaborn` for nicer plot defaults.

### Quick Setup
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
pip install pandas numpy matplotlib requests
