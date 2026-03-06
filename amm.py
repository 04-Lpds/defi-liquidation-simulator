# amm.py
# Constant Product AMM (Uniswap v2 style) functions
# Assumes pairwise pools (e.g., WETH/USDC, WBTC/USDC)
# ****Rebalances pools towards CEX api price each step. Rebalance rate and threshold configurable in config.py
import numpy as np
from typing import Dict

def swap(
    reserves: Dict[str, float],
    amount_in: float,
    token_in: str,
    token_out: str,
    fee: float = 0.003,
    *,
    execute: bool = True
) -> dict:
    """
    Unified constant-product swap function for any token pair pool.

    Args:
        reserves: Dict of current reserves in the specific pool (e.g., {"WETH": 6000.0, "USDC": 18e6})
        amount_in: Amount of token_in to sell
        token_in/out: Names of tokens (must match keys in reserves)
        fee: Pool fee (0.003 = 0.3%)
        execute: If True, mutate reserves in-place; if False, dry-run (preview only)

    Returns:
        dict with:
            amount_out, spot_price_before/after, effective_price, price_impact_bps
    """
    if amount_in <= 0:
        spot_price = reserves.get(token_out, 0.0) / reserves.get(token_in, 1.0)
        return {
            "amount_out": 0.0,
            "spot_price_before": spot_price,
            "effective_price": 0.0,
            "price_impact_bps": 0.0,
            "spot_price_after": spot_price
        }

    reserve_in = reserves[token_in]
    reserve_out = reserves[token_out]

    if reserve_in <= 0 or reserve_out <= 0:
        raise ValueError(f"Insufficient liquidity in pool {token_in}/{token_out}")

    spot_price_before = reserve_out / reserve_in

    amount_in_after_fee = amount_in * (1 - fee)
    amount_out = (reserve_out * amount_in_after_fee) / (reserve_in + amount_in_after_fee)

    effective_price = amount_out / amount_in if amount_in > 0 else 0.0

    # Projected prices after trade
    new_reserve_in = reserve_in + amount_in_after_fee
    new_reserve_out = reserve_out - amount_out
    spot_price_after = new_reserve_out / new_reserve_in

    # Price impact in basis points (standard in DeFi)
    price_impact_bps = (spot_price_before - effective_price) / spot_price_before * 10_000

    result = {
        "amount_out": amount_out,
        "spot_price_before": spot_price_before,
        "effective_price": effective_price,
        "price_impact_bps": price_impact_bps,
        "spot_price_after": spot_price_after
    }

    if execute:
        reserves[token_in] += amount_in_after_fee
        reserves[token_out] -= amount_out

    return result



def rebalance_amm_pools(state: dict, config):
    """
    Bidirectional rebalancing towards API price target each step.
    - Calculates exact amount to reach target ratio
    - Applies fractional correction (rate) with caps to prevent over-correction
    - If USDC depleted (asset-heavy), add USDC.
    - If asset depleted (USDC-heavy), add asset.
    """
    if not config.use_rebalancing:
        return

    for pool_key in state["amm_reserves"]:
        reserves = state["amm_reserves"][pool_key]
        token_asset, token_usdc = pool_key.split('_')

        # Get current raw API price (pure external feed, delayed if applicable)
        asset_idx = config.assets.index(token_asset)
        delayed_step = max(0, state.get("current_step", 0) - config.oracle_delay)
        api_price = config.price_path.iloc[delayed_step][token_asset]

        target_ratio = api_price  # dynamic target: current API price

        # Current ratio
        current_ratio = reserves[token_usdc] / (reserves[token_asset] + 1e-8)

        # Imbalance sign: positive = USDC depleted, negative = asset depleted
        imbalance_sign = np.sign(target_ratio - current_ratio)

        if abs(target_ratio - current_ratio) > config.rebalance_threshold:
            # Exact amount needed to reach target
            if imbalance_sign > 0:  # USDC depleted → add USDC
                exact_add_usdc = (target_ratio * reserves[token_asset]) - reserves[token_usdc]
                correction = exact_add_usdc * config.rebalance_rate
                # Cap to prevent over-correction (e.g., max 10% of current USDC)
                correction = min(correction, 0.1 * reserves[token_usdc])
                reserves[token_usdc] += correction
                #print(f"Rebalanced {pool_key}: Added ${correction:,.0f} USDC (asset-heavy)")
            else:  # Asset depleted → add asset
                exact_add_asset = (reserves[token_usdc] / target_ratio) - reserves[token_asset]
                correction = exact_add_asset * config.rebalance_rate
                # Cap to 10% of current asset
                correction = min(correction, 0.1 * reserves[token_asset])
                reserves[token_asset] += correction
                #print(f"Rebalanced {pool_key}: Added {correction:,.2f} {token_asset} (USDC-heavy)")

        # Safety floor
        reserves[token_asset] = max(reserves[token_asset], 1e-6)
        reserves[token_usdc] = max(reserves[token_usdc], 1e-6)

# Optional local test block
if __name__ == "__main__":
    # Example: test one pool
    test_pool = {
        "WETH": 6000.0,
        "USDC": 18_000_000.0
    }

    print("Initial spot price (USDC per WETH):", test_pool["USDC"] / test_pool["WETH"])

    # Preview a large trade
    preview = swap(
        reserves=test_pool,
        amount_in=300.0,
        token_in="WETH",
        token_out="USDC",
        execute=False
    )
    print("\nPreview (300 WETH sell):")
    for k, v in preview.items():
        print(f"  {k}: {v:.4f}")

    # Execute smaller trade
    result = swap(
        reserves=test_pool,
        amount_in=50.0,
        token_in="WETH",
        token_out="USDC",
        execute=True
    )
    print("\nAfter executing 50 WETH sell:")
    print(f"  Received: {result['amount_out']:.2f} USDC")
    print(f"  Price impact: {result['price_impact_bps']:.2f} bps")
    print(f"  New spot price: {result['spot_price_after']:.2f} USDC/WETH")