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

    if borrower_data is None:
        state["borrower_data"] = copy.deepcopy(config.borrowers)
    else:
        state["borrower_data"] = borrower_data

    state["history"] = {
        "steps": [],
        "timestamp": [],
        "liquidations_per_step": [],
        "percent_liquidatable": [],
        "cumulative_realized_bad_debt": [],
        "pending_bad_debt_per_step": [],
        "cumulative_pending_bad_debt": [],
        "total_bad_debt_per_step": [],
        "economic_shortfall_per_step": [],
        "cumulative_total_bad_debt": [],
        "seized_usd_cumulative": [],
        "debt_closed_cumulative": [],
        "price_main_asset": [],
    }

    update_health_factors(state, config)

    state["cumulative_bad_debt"] = 0.0
    state["liquidation_volume_this_step"] = 0.0

    return state


if __name__ == "__main__":
    config = Config()
    state = initialize_state(config)

    bd = state["borrower_data"]
    hf = bd["health_factor"]
    hf_finite = hf[np.isfinite(hf)]

    print(f"Initialization complete!")
    print(f"Price path length: {len(config.price_path)} minutes")   # i.e., should be 1440 (mins/day)
    print(f"Initial prices: {config.initial_prices}")
    print(f"\nBorrower stats:")
    print(f"  Total borrowers: {config.num_borrowers}")
    print(f"  Median HF: {np.median(hf_finite):.3f}")
    print(f"  Min HF: {np.min(hf_finite):.3f}")
    print(f"  % liquidatable at start: {state['liquidatable_mask'].mean() * 100:.2f}%")
    print(f"  Average total debt USD: {np.mean(np.sum(bd['debt'] * state['oracle_prices'], axis=1)):.0f}")