import numpy as np
import matplotlib.pyplot as plt

# --- User inputs from your results ---
initial_capex = 310829090.91  # AUD
annual_opex = 22746402        # AUD (includes fuel, variable O&M, fixed O&M for all components)
discount_rate = 0.06
years = 25

# --- Calculate discounted annual OPEX ---
discount_factors = [(1 + discount_rate) ** -year for year in range(1, years + 1)]
discounted_opex = [annual_opex * df for df in discount_factors]

# --- Cumulative NPC calculation ---
npc_yearly = [initial_capex]
for i in range(years):
    npc_yearly.append(npc_yearly[-1] + discounted_opex[i])

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(0, years + 1), npc_yearly, marker='o', color='navy')
plt.title('Cumulative Net Present Cost (NPC) Over 25 Years')
plt.xlabel('Year')
plt.ylabel('Cumulative NPC (AUD)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('npc_over_time.png', dpi=150)
plt.show()