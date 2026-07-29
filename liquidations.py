import numpy as np
import random  # for np.random.shuffle
from amm import swap
from borrowers import update_health_factors


def process_liquidations(state: dict, config):
    """Simplified liquidation processing:
    - Force liquidates positions when profitable (>= config.default_min_profit_bps)
    - Random order within block (np.random.shuffle)
    - Up to config.max_liqs_per_block per step
    - Tracks unique liquidated borrowers across the entire simulation
    - Accumulates stats: liquidated_count, seized_usd, debt_closed, bad_debt_added, unique_liquidated
    - Mutates state in-place (borrower_data, amm_reserves, cumulative_bad_debt, etc.)
    - Calls update_health_factors before and after to refresh mask/HF

    Returns dict with:
    - liquidated_count (this step)
    - seized_usd (total USD value of collateral seized this step)
    - debt_closed (total debt covered this step)
    - bad_debt_added (shortfall/shortfall this step)
    - unique_liquidated (cumulative liq. borrowers across whole sim)
        """


    if "liquidated_borrowers_set" not in state:
        state["liquidated_borrowers_set"] = set()

    update_health_factors(state, config)

    mask = state["liquidatable_mask"]  # array of liquidatable borrowers' indices this step
    indices = np.nonzero(mask)[0]  # Note; [0] returns array, instead of tuple of (1) array
    if len(indices) == 0:
        return {
            "liquidated_count": 0,
            "seized_usd": 0.0,
            "debt_closed": 0.0,
            "bad_debt_added": 0.0,
            "unique_liquidated": len(state["liquidated_borrowers_set"])
        }

    np.random.shuffle(indices)  # simulate non-deterministic order

    liquidated_count = 0
    bad_debt_added = 0.0
    seized_usd = 0.0
    debt_closed = 0.0

    bd = state["borrower_data"]
    prices = state["oracle_prices"]

    for idx in indices[:config.max_liqs_per_block]:
        pre_debt = np.sum(bd["debt"][idx])
        if pre_debt <= 1e-3:
            #debug: print(f"Step | SKIP candidate {idx}: pre_debt too low ({pre_debt:.6f})")
            continue

        # Single uniform profitability check
        profit_bps = estimate_profitability(state, idx, config)
        # print(f"Step  | Candidate {idx}: pre_debt = ${pre_debt:,.2f} | profit_bps = {profit_bps:.2f} | "
        #       f"threshold = {config.default_min_profit_bps} | "
        #       f"would_liq = {profit_bps >= config.default_min_profit_bps}")
        if profit_bps < config.default_min_profit_bps:
            # print(f"Step  | SKIP candidate {idx}: unprofitable (bps {profit_bps:.2f})")
            continue  # i.e., skip unprofitable liq and proceed to next borrowers' index

        # Execution
        #print(f"Step | EXECUTING liq for {idx}")
        seized_per_asset = bd["collateral"][idx] * config.close_factor
        seized_usd_this = np.sum(seized_per_asset * prices)
        seized_usd += seized_usd_this

        bad_debt_this = execute_liquidation(state, idx, config)  # See note for execute_liquidations()

        debt_to_cover = pre_debt * config.close_factor
        debt_closed += debt_to_cover
        bad_debt_added += bad_debt_this
        liquidated_count += 1

        state["liquidated_borrowers_set"].add(idx)

    state["cumulative_bad_debt"] += bad_debt_added
    update_health_factors(state, config)

    return {
        "liquidated_count": liquidated_count,
        "seized_usd": seized_usd,
        "debt_closed": debt_closed,
        "bad_debt_added": bad_debt_added,
        "unique_liquidated": len(state["liquidated_borrowers_set"])
    }


def estimate_profitability(state: dict, borrower_idx: int, config) -> float:
    """Estimates profit in basis points (bps) for liquidating a single borrower.
    Uses preview swap (no state mutation) to calculate proceeds vs required amount.
    Returns float >= 0 if profitable, can be negative."""

    bd = state["borrower_data"]
    total_debt = np.sum(bd["debt"][borrower_idx])
    if total_debt <= 1e-8:
        return 0.0

    total_proceeds = 0.0
    seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
    # print(f"Step {step} | Estimating for borrower {borrower_idx}: total_debt = ${total_debt:,.2f} | "
    #       f"debt_to_cover = ${total_debt:,.2f} | required_proceeds = ${required_proceeds:,.2f}")
    for asset_idx, asset in enumerate(state["assets"]):
        seized = seized_per_asset[asset_idx]
        if seized <= 1e-8:
            continue

        pool_key = f"{asset}_USDC"
        if pool_key not in state["amm_reserves"]:
            continue

        # === DEBUG PRINTS START HERE ===
        # pool = state["amm_reserves"][pool_key]
        # fair_value = seized * state["oracle_prices"][asset_idx]
        # print(f"Step {state.get('current_step', 'unknown')} | "
        #       f"Asset {asset}: seized = {seized:,.2f} | "
        #       f"fair_value = ${fair_value:,.2f} | "
        #       f"Pool {pool_key} reserves: {asset}: {pool[asset]:,.0f} | USDC: {pool['USDC']:,.0f}")

        preview = swap(
            reserves=state["amm_reserves"][pool_key],
            amount_in=seized,
            token_in=asset,
            token_out="USDC",
            execute=False
        )

        # === DEBUG PRINTS  ===
        # actual_out = preview["amount_out"]
        # slippage_pct = (actual_out / fair_value - 1) * 100 if fair_value > 0 else -100
        # print(f"  → amount_out = ${actual_out:,.2f} | slippage = {slippage_pct:.2f}%")

        total_proceeds += preview["amount_out"]

        # DEBUG:
        # fair_value = seized * state["oracle_prices"][asset_idx]
        # actual_out = preview["amount_out"]
        # slippage_pct = (actual_out / fair_value - 1) * 100 if fair_value > 0 else -100
        #
        # print(f"  Asset {asset}: seized = {seized:,.2f} | fair_value = ${fair_value:,.2f} | "
        #       f"amount_out = ${actual_out:,.2f} | slippage = {slippage_pct:.2f}%")
        # print(f"  Pool reserves: {asset}: {state['amm_reserves'][pool_key][asset]:,.0f} | "
        #       f"USDC: {state['amm_reserves'][pool_key]['USDC']:,.0f}")

    debt_to_cover = total_debt * config.close_factor
    required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)  # i.e., to check if col. sold covers debt+bonus

    if required_proceeds <= 1e-8:
        return 0.0

    profit_bps = (total_proceeds - required_proceeds) / required_proceeds * 10_000
    return profit_bps


def execute_liquidation(state: dict, borrower_idx: int, config) -> float:
    """
    Executes liquidation for one borrower:
    - Performs real AMM swaps (mutates reserves in-place)
    - Reduces borrower collateral and debt
    - Returns bad debt/shortfall absorbed by protocol (if proceeds < required)*
    - Mutates state in-place (amm_reserves, borrower_data["collateral"], ["debt"])

    *see note at bottom of function
    """

    bd = state["borrower_data"]
    total_debt = np.sum(bd["debt"][borrower_idx])
    if total_debt <= 1e-8:
        return 0.0

    seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
    total_proceeds = 0.0

    for asset_idx, asset in enumerate(state["assets"]):
        seized = seized_per_asset[asset_idx]
        if seized <= 1e-8:
            continue

        pool_key = f"{asset}_USDC"
        if pool_key not in state["amm_reserves"]:
            continue

        result = swap(
            reserves=state["amm_reserves"][pool_key],
            amount_in=seized,
            token_in=asset,
            token_out="USDC",
            execute=True
        )
        total_proceeds += result["amount_out"]
        bd["collateral"][borrower_idx, asset_idx] -= seized

    debt_to_cover = total_debt * config.close_factor
    required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)

    bad_debt = max(0.0, required_proceeds - total_proceeds)

    # Reduce debt (protocol covers shortfall)
    bd["debt"][borrower_idx] -= debt_to_cover
    bd["debt"][borrower_idx] = np.maximum(bd["debt"][borrower_idx], 0.0)

    """
    Liquidation profitability & execution logic notes:
    - Profitability is checked via preview swap (execute=False) before any state mutation.
    - Execution only proceeds if preview shows profit >= threshold → ensures atomicity and no shortfall in current model.
    - This is an optimistic simplification: in reality, slippage can worsen between preview and execution (mempool delay, concurrent liqs, MEV).
    - Current design prevents bad debt from individual liqs (proceeds always >= required) → may underestimate tail risk in high-volatility/concurrent scenarios.
    - Potential basic improvement: add small random slippage noise after preview to simulate execution risk.
    """

    return bad_debt