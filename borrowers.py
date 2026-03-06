# borrowers.py
from config import Config
import numpy as np
import matplotlib.pyplot as plt

"""Note: generate_borrowers() moved to config.py so the distribution of borrowers is only generated once during monte carlo and to avoid circular dependencies"""

def update_health_factors(state: dict, config: Config):
    """
    Vectorized health factor (HF) calculation with LTV weighting.

    Purpose:
    - Recalculates HF for every borrower based on current oracle prices.
    - Applies per-asset LTV haircuts to collateral to get "effective" collateral.
    - Sets liquidatable_mask for borrowers where HF drops below threshold.

    Key logic:
    - Weighted collateral = collateral_amount × oracle_price × LTV (per asset, per borrower)
    - Total weighted collateral = sum across all assets
    - Total debt = sum(debt_amount × oracle_price) across assets
    - HF = total_weighted / total_debt (inf if no debt)
    - Liquidatable if HF < config.liquidation_threshold (usually 1.0 in this sim)

    Args:
        state (dict): Simulation state with:
            - "borrower_data": dict of "collateral" and "debt" arrays (n_borrowers × n_assets)
            - "oracle_prices": current price array (n_assets)
            - "ltv": LTV array (n_assets)
        config (Config): Contains liquidation_threshold

    Returns:
        None — mutates state in-place:
            - state["borrower_data"]["health_factor"]
            - state["liquidatable_mask"] (boolean array)

    Notes:
    - Simplified model: LTV is applied directly in HF numerator.
    - Real Aave V3 uses separate LTV (borrow limit) and liquidation_threshold (HF trigger).
    - Called frequently (before/after liqs, rebalancing) to keep HF/mask current.
    """
    bd = state["borrower_data"]
    prices = state["oracle_prices"]  # array, same order as assets
    ltv = state["ltv"]  # array

    # Weighted collateral value per borrower (matrix * vector)
    weighted_collateral = bd["collateral"] * prices * ltv
    total_weighted = np.sum(weighted_collateral, axis=1)

    # Total debt value (USDC = 1.0)
    total_debt = np.sum(bd["debt"] * prices, axis=1)

    # Health factor
    bd["health_factor"] = np.where(
        total_debt > 0,
        total_weighted / total_debt,
        np.inf
    )

    # Liquidatable mask
    state["liquidatable_mask"] = bd["health_factor"] < config.liquidation_threshold

def plot_borrower_distributions(state: dict, config):
    """
    Plot initial/final borrower distributions (collateral, debt, leverage, HF)
    using data from state["borrower_data"].
    """
    if not config.plot_borrower_distributions:
        return

    # Extract filtered borrower data
    borrower_data = state["borrower_data"]
    collateral = borrower_data["collateral"]           # (n_borrowers, n_assets)
    debt = borrower_data["debt"]                       # (n_borrowers, n_assets)

    # Recompute necessary arrays
    prices = np.array([config.initial_prices[a] for a in config.assets])  # shape (n_assets,)
    ltv_values = np.array([config.ltv[a] for a in config.assets])         # shape (n_assets,)

    collateral_usd_per_asset = collateral * prices                         # (n_borrowers, n_assets)
    total_debt_usd = np.sum(debt * prices, axis=1)                         # (n_borrowers,)

    # Leverage
    total_collateral_usd = np.sum(collateral_usd_per_asset, axis=1)
    leverage = total_collateral_usd / total_debt_usd
    leverage = np.clip(leverage, 1.1, 8.0)  # match generation clip

    # Health factor (recompute for consistency)
    weighted_collateral = collateral * prices * ltv_values
    total_weighted = np.sum(weighted_collateral, axis=1)
    hf = np.where(total_debt_usd > 0, total_weighted / total_debt_usd, np.inf)

    n_borrowers = len(hf)

    # === Plotting ===
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Borrower Distributions", fontsize=16)

    # Row 0: Collateral per asset + Total Debt
    for i, asset in enumerate(config.assets[:-1]):  # exclude USDC
        ax = axs[0, i]
        data = collateral_usd_per_asset[:, i]
        ax.hist(data, bins=80, color='skyblue', alpha=0.8, log=True)
        ax.set_title(f"{asset} Collateral (USD)")
        ax.set_xlabel("USD Value")

    # Total Debt – consistent salmon/pink
    axs[0, 2].hist(
        total_debt_usd,
        bins=80,
        color='salmon',
        alpha=1.0,  # full opacity — no transparency to let edges show
        log=True,
        edgecolor='none',  # removes all outlines → no blue overlay
        linewidth=0  # extra zero-width edges
    )
    axs[0, 2].set_title("Total Debt (USDC) – All Borrowers")
    axs[0, 2].set_xlabel("USD")

    # Row 1: Leverage + HF (zoomed + full log)
    axs[1, 0].hist(leverage, bins=80, color='orange', alpha=0.8)
    axs[1, 0].set_title("Leverage Distribution")
    axs[1, 0].set_xlabel("Leverage")

    axs[1, 1].hist(hf, bins=100, range=(0.5, 4), color='lightgreen', alpha=0.8)
    axs[1, 1].axvline(config.liquidation_threshold, color='red', linestyle='--', linewidth=2,
                       label=f"Threshold ({config.liquidation_threshold})")
    axs[1, 1].set_title("Health Factor (Zoomed)")
    axs[1, 1].set_xlabel("Health Factor")
    axs[1, 1].legend()

    axs[1, 2].hist(hf, bins=100, color='purple', alpha=0.8, log=True)
    axs[1, 2].axvline(config.liquidation_threshold, color='red', linestyle='--', linewidth=2)
    axs[1, 2].set_title("Health Factor (Full, log)")
    axs[1, 2].set_xlabel("Health Factor")

    plt.tight_layout()
    plt.savefig("borrower_distributions.png", dpi=200, bbox_inches='tight')
    plt.show()