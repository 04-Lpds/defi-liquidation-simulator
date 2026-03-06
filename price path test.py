
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import Config

# Instantiate config and fetch historical prices once
config = Config()

# Extract base prices for all three assets
base_prices = config.price_path[['WETH', 'WBTC', 'SOL']]  # DataFrame with the three columns

# Print quick info about the base data
print(f"Base path length: {len(base_prices)} minutes")
for asset in ['WETH', 'WBTC', 'SOL']:
    series = base_prices[asset]
    open_p = series.iloc[0]
    min_p = series.min()
    drop_pct = (open_p - min_p) / open_p * 100
    print(f"{asset}: Open = {open_p:.2f} | Min = {min_p:.2f} | Max drop = {drop_pct:.1f}%")

# Define test scales
test_scales = [1.0, 1.5, 2.0, 3.0, 4.0]


# Generate derived paths for ALL assets at once
derived_multi = {}
for scale in test_scales:
    paths = Config.derive_price_path(
        historical_prices=base_prices,  # pass the full 3-column DataFrame
        scale_factor=scale,  # same scale for all (or use dict for per-asset)
        max_drop_fraction=0.75,
        noise_std=0.015,  # same noise for all (or dict)
        front_load_fraction=0.6 if scale >= 2.0 else 0.0,  # more aggressive crash for higher scales
        random_seed=42 + int(scale * 10)  # different seed per scale
    )
    derived_multi[scale] = paths  # dict of {asset: array}

# ────────────────────────────────────────────────
# Plotting: one figure with subplots for each asset
# ────────────────────────────────────────────────
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True)
fig.suptitle(f"Price Path Stress Testing: Original vs Scaled Variants\nBase Date: {config.crisis_date}", fontsize=16)

minutes = np.arange(len(base_prices))

assets = ['WETH', 'WBTC', 'SOL']
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(test_scales)))

for i, asset in enumerate(assets):
    ax = axes[i]

    # Original path
    ax.plot(minutes, base_prices[asset].values,
            label=f'Original (1×)', linewidth=2, color='black')

    # Derived paths
    for j, scale in enumerate(test_scales):
        ax.plot(minutes, derived_multi[scale][asset],
                label=f'{scale:.1f}×',
                color=colors[j], alpha=0.9)

    ax.set_title(f"{asset}")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Minutes since start of day")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()  # or just plt.show()