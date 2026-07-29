# state.py
import numpy as np
import copy
from config import Config
from borrowers import update_health_factors


def initialize_state(config: Config, borrower_data=None):
    state = {}

    state["assets"] = np.array(config.assets)
    state["oracle_prices"] = np.array([config.initial_prices[a] for a in state["assets"]])
    state["ltv"] = np.array([config.ltv[a] for a in state["assets"]])
    state["amm_reserves"] = copy.deepcopy(config.amm_pools)
    state["initial_amm_ratios"] = {}

    for pool_key, reserves in state["amm_reserves"].items():
        token_in, token_out = pool_key.split('_')
        if token_out in reserves and token_in in reserves and reserves[token_in] > 0:
            state["initial_amm_ratios"][pool_key] = reserves[token_out] / reserves[token_in]

    # ==================== BORROWER DATA ====================
    if borrower_data is None:
        state["borrower_data"] = copy.deepcopy(config.borrowers)
    else:
        state["borrower_data"] = copy.deepcopy(borrower_data)   # Important: deep copy here too

    # ==================== HISTORY DICT ====================
    state["history"] = {
        "steps": [],
        "liquidations_per_step": [],
        "percent_liquidatable": [],
        "pending_bad_debt_per_step": [],
        "economic_shortfall_per_step": [],
        "total_bad_debt_per_step": [],
        "seized_usd_cumulative": [0.0],
        "debt_closed_cumulative": [0.0],

        # Price tracking
        "price_WETH": [], "price_WBTC": [], "price_SOL": [],
        "api_price_WETH": [], "api_price_WBTC": [], "api_price_SOL": [],
        "amm_spot_WETH": [], "amm_spot_WBTC": [], "amm_spot_SOL": [],

        # Research scalars
        "peak_liquidatable_pct": 0.0,
        "peak_pending_debt": 0.0,
        "peak_economic_shortfall": 0.0,
        "cumulative_liquidations": 0,
    }

    update_health_factors(state, config)

    state["cumulative_bad_debt"] = 0.0
    state["liquidation_volume_this_step"] = 0.0

    return state


if __name__ == "__main__":
    config = Config()
    state = initialize_state(config)
    print("State initialization test successful!")
    print(f"Total borrowers: {len(state['borrower_data']['health_factor'])}")
    print(f"Median HF: {np.median(state['borrower_data']['health_factor']):.4f}")