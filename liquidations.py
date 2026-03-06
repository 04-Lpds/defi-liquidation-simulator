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












# import numpy as np
# from amm import swap
# from borrowers import update_health_factors  # for feedback after liquidations
#
#
#
# # liquidations.py
# # Complete liquidation processing module
# # Handles profitability checks, execution, and optional liquidator heterogeneity
# # Called from sim.py every step
# # Random execution order - because in reality, searchers compete in a race; order is not deterministic
#
# # liquidations.py
# # Complete liquidation processing module with debugging checks
# # Tracks unique borrowers and debt reduction to diagnose excessive liquidation events
#
#
# # Global / sim-level setup (add this near the top of your file)
# import random
#
# LIQUIDATOR_STRATEGIES = [
#     {"name": "aggressive", "min_profit_bps": 30},
#     {"name": "conservative", "min_profit_bps": 150},
#     {"name": "mev_searcher", "min_profit_bps": 20},
# ]
#
# NUM_LIQUIDATORS = 8  # Small fixed pool — real ecosystems have a handful of dominant keepers
#
# liquidators = []
# for i in range(NUM_LIQUIDATORS):
#     strat = random.choice(LIQUIDATOR_STRATEGIES)
#     liquidators.append({
#         "id": i,
#         "strategy": strat["name"],
#         "min_profit_bps": strat["min_profit_bps"],
#         "active": True  # could deactivate later if losses mount
#     })
#
#
# def process_liquidations(state: dict, config):
#     """
#     Main liquidation function with optional heterogeneous liquidator strategies.
#     """
#     if "liquidated_borrowers_set" not in state:
#         state["liquidated_borrowers_set"] = set()
#
#     update_health_factors(state, config)
#
#     mask = state["liquidatable_mask"]
#     indices = np.nonzero(mask)[0]
#     if len(indices) == 0:
#         return {
#             "liquidated_count": 0,
#             "seized_usd": 0.0,
#             "debt_closed": 0.0,
#             "bad_debt_added": 0.0,
#             "unique_liquidated": len(state["liquidated_borrowers_set"])
#         }
#
#     np.random.shuffle(indices)
#
#     liquidated_count = 0
#     bad_debt_added = 0.0
#     seized_usd = 0.0
#     debt_closed = 0.0
#
#     bd = state["borrower_data"]
#     prices = state["oracle_prices"]
#
#     for idx in indices[:config.max_liqs_per_block]:
#         pre_debt = np.sum(bd["debt"][idx])
#         if pre_debt <= 1e-3:
#             continue
#
#         # === Toggle logic starts here ===
#         if config.use_heterogeneous_liquidators:
#             # Heterogeneous branch: check against all liquidators
#             profit_bps = estimate_profitability(state, idx, config)
#             eligible = [liq for liq in liquidators if profit_bps >= liq["min_profit_bps"]]
#             if not eligible:
#                 continue  # No liquidator wants it → skip
#             winner = min(eligible, key=lambda x: x["min_profit_bps"])
#             # Optional: log who won
#             # print(f"Liq by {winner['strategy']} (profit {profit_bps:.1f} bps)")
#         else:
#             # Fallback: use uniform/default threshold
#             if not estimate_profitability(state, idx, config.default_min_profit_bps, config):
#                 continue
#
#         # === Execution (same for both branches) ===
#         seized = bd["collateral"][idx] * config.close_factor
#         seized_usd_this = np.sum(seized * prices)
#         seized_usd += seized_usd_this
#
#         bad_debt_this = execute_liquidation(state, idx, config)
#
#         debt_to_cover = pre_debt * config.close_factor
#         debt_closed += debt_to_cover
#         bad_debt_added += bad_debt_this
#         liquidated_count += 1
#
#         state["liquidated_borrowers_set"].add(idx)
#
#     state["cumulative_bad_debt"] += bad_debt_added
#     update_health_factors(state, config)
#
#     return {
#         "liquidated_count": liquidated_count,
#         "seized_usd": seized_usd,
#         "debt_closed": debt_closed,
#         "bad_debt_added": bad_debt_added,
#         "unique_liquidated": len(state["liquidated_borrowers_set"])
#     }
#
#
# def estimate_profitability(state: dict, borrower_idx, config) -> float:
#     """
#     Same logic as estimate_profitability but returns actual bps (not bool).
#     """
#     bd = state["borrower_data"]
#     total_debt = np.sum(bd["debt"][borrower_idx])
#     if total_debt <= 1e-8:
#         return 0.0
#
#     total_proceeds = 0.0
#     seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
#     for asset_idx, asset in enumerate(state["assets"]):
#         seized = seized_per_asset[asset_idx]
#         if seized <= 1e-8:
#             continue
#
#         pool_key = f"{asset}_USDC"
#         if pool_key not in state["amm_reserves"]:
#             continue
#
#         preview = swap(
#             reserves=state["amm_reserves"][pool_key],
#             amount_in=seized,
#             token_in=asset,
#             token_out="USDC",
#             execute=False
#         )
#         total_proceeds += preview["amount_out"]
#
#     debt_to_cover = total_debt * config.close_factor
#     required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)
#
#     if required_proceeds <= 1e-8:
#         return 0.0
#
#     profit_bps = (total_proceeds - required_proceeds) / required_proceeds * 10_000
#     return profit_bps
#
# def execute_liquidation(state: dict, borrower_idx, config) -> float:
#     """
#     Execute liquidation for one borrower.
#     Returns bad debt from this single liquidation.
#     """
#     bd = state["borrower_data"]
#     total_debt = np.sum(bd["debt"][borrower_idx])
#     if total_debt <= 1e-8:
#         return 0.0
#
#     seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
#     total_proceeds = 0.0
#
#     for asset_idx, asset in enumerate(state["assets"]):
#         seized = seized_per_asset[asset_idx]
#         if seized <= 1e-8:
#             continue
#
#         pool_key = f"{asset}_USDC"
#         if pool_key not in state["amm_reserves"]:
#             continue
#
#         result = swap(
#             reserves=state["amm_reserves"][pool_key],
#             amount_in=seized,
#             token_in=asset,
#             token_out="USDC",
#             execute=True
#         )
#         total_proceeds += result["amount_out"]
#         bd["collateral"][borrower_idx, asset_idx] -= seized
#
#     debt_to_cover = total_debt * config.close_factor
#     required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)
#
#     bad_debt = max(0.0, required_proceeds - total_proceeds)
#
#     # Reduce debt by the intended cover amount (protocol absorbs shortfall)
#     bd["debt"][borrower_idx] -= debt_to_cover
#     bd["debt"][borrower_idx] = np.maximum(bd["debt"][borrower_idx], 0.0)
#
#     return bad_debt

#
# def process_liquidations(state: dict, config):
#     """
#     Main liquidating function:
#     - Tracks unique borrowers liquidated across the whole sim
#     - Prints pre/post debt for each liquidation
#     """
#     # Initialize unique tracking set if not present
#     if "liquidated_borrowers_set" not in state:
#         state["liquidated_borrowers_set"] = set()
#
#     update_health_factors(state, config)
#
#     mask = state["liquidatable_mask"]
#     indices = np.nonzero(mask)[0]   # i.e., obtain array of all liquidatable indices
#     if len(indices) == 0:
#         return {
#             "liquidated_count": 0,
#             "seized_usd": 0.0,
#             "debt_closed": 0.0,
#             "bad_debt_added": 0.0,
#             "unique_liquidated": len(state["liquidated_borrowers_set"])
#         }
#
#     np.random.shuffle(indices)  # i.e., to randomize liquidation order
#
#     liquidated_count = 0
#     bad_debt_added = 0.0
#     seized_usd = 0.0
#     debt_closed = 0.0
#
#     bd = state["borrower_data"]
#     prices = state["oracle_prices"]
#
#     for idx in indices[:config.max_liqs_per_block]:
#         pre_debt = np.sum(bd["debt"][idx])
#         if pre_debt <= 1e-3:  # already essentially zero — skip
#             continue
#
#         seized = bd["collateral"][idx] * config.close_factor
#         seized_usd_this = np.sum(seized * prices)
#         seized_usd += seized_usd_this
#
#         bad_debt_this = execute_liquidation(state, idx, config)
#
#         debt_to_cover = pre_debt * config.close_factor  # use pre_debt for accuracy
#         debt_closed += debt_to_cover
#         bad_debt_added += bad_debt_this
#         liquidated_count += 1
#
#         # Track unique borrower
#         state["liquidated_borrowers_set"].add(idx)
#
#         post_debt = np.sum(bd["debt"][idx])
#         # print(f"Liq borrower {idx}: pre-debt ${pre_debt:,.0f} → post-debt ${post_debt:,.0f} | "
#         #       f"bad debt this liq ${bad_debt_this:,.0f}")
#
#     state["cumulative_bad_debt"] += bad_debt_added
#
#     update_health_factors(state, config)
#
#     unique_liquidated = len(state["liquidated_borrowers_set"])
#
#     # print(f"Liquidations this step: {liquidated_count} | "
#     #       f"Seized ~${seized_usd:,.0f} | "
#     #       f"Debt closed ~${debt_closed:,.0f} | "
#     #       f"Bad debt added ${bad_debt_added:,.0f} | "
#     #       f"Unique borrowers liquidated so far: {unique_liquidated}/{config.num_borrowers}")
#
#     return {
#         "liquidated_count": liquidated_count,
#         "seized_usd": seized_usd,
#         "debt_closed": debt_closed,
#         "bad_debt_added": bad_debt_added,
#         "unique_liquidated": unique_liquidated
#     }
#

#
#
#
#
#
# #
# # # ==================
# # # Define liquidators (small pool)
# # # ==================
# # import random
# # import numpy as np
# #
# # LIQUIDATOR_STRATEGIES = [
# #     {"name": "aggressive", "min_profit_bps": 30},
# #     {"name": "conservative", "min_profit_bps": 150},
# #     {"name": "mev_searcher", "min_profit_bps": 20},
# # ]
# #
# # NUM_LIQUIDATORS = 8  # small and fixed
# #
# # liquidators = []
# # for i in range(NUM_LIQUIDATORS):
# #     strat = random.choice(LIQUIDATOR_STRATEGIES)
# #     liquidators.append({
# #         "id": i,
# #         "strategy": strat["name"],
# #         "min_profit_bps": strat["min_profit_bps"]
# #     })
# #
#
# # ==================
# # In your per-timestep liquidation logic
# # ==================
# def process_het_liquidations(positions, current_price, amm_state, liquidators):
#     undercollateralized = [p for p in positions if p["health_factor"] < 1.0 and not p.get("liquidated", False)]
#     if not undercollateralized:
#         return  # nothing to do
#
#     # Vectorized profit calc for all undercollateralized positions
#     debts = np.array([p["debt"] for p in undercollateralized])
#     collateral_to_sell = debts / current_price
#     bonuses = collateral_to_sell * 0.08  # example 8% bonus
#     gas_costs = np.full(len(debts), 10.0)  # fixed gas USD
#     slippage_costs = collateral_to_sell * amm_state["slippage_factor"]
#     net_profits_usd = bonuses - gas_costs - slippage_costs
#     profits_bps = (net_profits_usd / debts) * 10000
#
#     # For each position, find if any liquidator would take it
#     for idx, pos in enumerate(undercollateralized):
#         profit_bps = profits_bps[idx]
#
#         eligible = [liq for liq in liquidators if profit_bps >= liq["min_profit_bps"]]
#         if not eligible:
#             continue  # no one wants it → position lingers
#
#         # Winner: lowest threshold (most aggressive) or random
#         winner = min(eligible, key=lambda x: x["min_profit_bps"])
#
#         # Execute
#         pos["liquidated"] = True
#         # Update AMM price impact, bad debt if shortfall, etc.
#         # (your existing liquidation code here)




#
# import numpy as np
# import random
#
# # Liquidator Strategies (Heterogeneity)
# LIQUIDATOR_STRATEGIES = [
#     {"name": "aggressive", "min_profit_bps": 40},
#     {"name": "most_conservative", "min_profit_bps": 200},
#     {"name": "conservative", "min_profit_bps": 140},
#     {"name": "mev_1", "min_profit_bps": 10},
#     {"name": "mev_2", "min_profit_bps": 20},
#     {"name": "less_aggressive", "min_profit_bps": 80}
# ]
#
# NUM_LIQUIDATORS = 10
#
# liquidators = []
# for i in range(NUM_LIQUIDATORS):
#     strat = random.choice(LIQUIDATOR_STRATEGIES)
#     liquidators.append({
#         "id": i,
#         "strategy": strat["name"],
#         "min_profit_bps": strat["min_profit_bps"],
#         "active": True
#     })
#
#
# def get_profit_bps(state: dict, borrower_idx, config) -> float:
#     """
#     Calculates profit in bps for a liquidation.
#     Used for both bool checks and value returns (merged from estimate_profitability and calculate_profit_bps).
#     Returns profit_bps (float). Caller can compare to threshold for bool.
#     """
#     bd = state["borrower_data"]
#     total_debt = np.sum(bd["debt"][borrower_idx])
#     if total_debt <= 1e-8:
#         return 0.0
#
#     total_proceeds = 0.0
#     seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
#     for asset_idx, asset in enumerate(state["assets"]):
#         seized = seized_per_asset[asset_idx]
#         if seized <= 1e-8:
#             continue
#
#         pool_key = f"{asset}_USDC"
#         if pool_key not in state["amm_reserves"]:
#             continue
#
#         preview = swap(
#             reserves=state["amm_reserves"][pool_key],
#             amount_in=seized,
#             token_in=asset,
#             token_out="USDC",
#             execute=False
#         )
#         total_proceeds += preview["amount_out"]
#
#     debt_to_cover = total_debt * config.close_factor
#     required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)
#
#     if required_proceeds <= 1e-8:
#         return 0.0
#
#     profit_bps = (total_proceeds - required_proceeds) / required_proceeds * 10_000
#     return profit_bps
#
#
# def execute_liquidation(state: dict, borrower_idx, config) -> float:
#     # Your original code — unchanged
#     bd = state["borrower_data"]
#     total_debt = np.sum(bd["debt"][borrower_idx])
#     if total_debt <= 1e-8:
#         return 0.0
#
#     seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
#     total_proceeds = 0.0
#
#     for asset_idx, asset in enumerate(state["assets"]):
#         seized = seized_per_asset[asset_idx]
#         if seized <= 1e-8:
#             continue
#
#         pool_key = f"{asset}_USDC"
#         if pool_key not in state["amm_reserves"]:
#             continue
#
#         result = swap(
#             reserves=state["amm_reserves"][pool_key],
#             amount_in=seized,
#             token_in=asset,
#             token_out="USDC",
#             execute=True
#         )
#         total_proceeds += result["amount_out"]
#         bd["collateral"][borrower_idx, asset_idx] -= seized
#
#     debt_to_cover = total_debt * config.close_factor
#     required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)
#
#     bad_debt = max(0.0, required_proceeds - total_proceeds)
#
#     # Reduce debt by the intended cover amount
#     bd["debt"][borrower_idx] -= debt_to_cover
#     bd["debt"][borrower_idx] = np.maximum(bd["debt"][borrower_idx], 0.0)
#
#     return bad_debt
#
#
# def process_liquidations(state: dict, config):
#     """
#     Main liquidation function with optional heterogeneous liquidator strategies.
#     """
#     if "liquidated_borrowers_set" not in state:
#         state["liquidated_borrowers_set"] = set()
#
#     update_health_factors(state, config)
#
#     mask = state["liquidatable_mask"]
#     indices = np.nonzero(mask)[0]
#     if len(indices) == 0:
#         return {
#             "liquidated_count": 0,
#             "seized_usd": 0.0,
#             "debt_closed": 0.0,
#             "bad_debt_added": 0.0,
#             "unique_liquidated": len(state["liquidated_borrowers_set"])
#         }
#
#     np.random.shuffle(indices)
#
#     liquidated_count = 0
#     bad_debt_added = 0.0
#     seized_usd = 0.0
#     debt_closed = 0.0
#
#     bd = state["borrower_data"]
#     prices = state["oracle_prices"]
#
#     for idx in indices[:config.max_liqs_per_block]:
#         pre_debt = np.sum(bd["debt"][idx])
#         if pre_debt <= 1e-3:
#             continue
#
#         # === Heterogeneous liquidator logic ===
#         if config.use_heterogeneous_liquidators:
#             profit_bps = get_profit_bps(state, idx, config)
#             eligible = [liq for liq in liquidators if profit_bps >= liq["min_profit_bps"]]
#             if not eligible:
#                 continue  # Skip
#             winner = min(eligible, key=lambda x: x["min_profit_bps"])
#         else:
#             # Fallback: uniform check with default threshold
#             profit_bps = get_profit_bps(state, idx, config)
#             if profit_bps < config.default_min_profit_bps:
#                 continue
#
#         # === Execute liquidation ===
#         seized = bd["collateral"][idx] * config.close_factor
#         seized_usd_this = np.sum(seized * prices)
#         seized_usd += seized_usd_this
#
#         bad_debt_this = execute_liquidation(state, idx, config)
#
#         debt_to_cover = pre_debt * config.close_factor
#         debt_closed += debt_to_cover
#         bad_debt_added += bad_debt_this
#         liquidated_count += 1
#
#         state["liquidated_borrowers_set"].add(idx)
#
#         post_debt = np.sum(bd["debt"][idx])
#
#
#     update_health_factors(state, config)
#     pending_bad_debt_this_step = 0.0
#
#     # Recompute mask (in case last liqs changed HFs)
#     update_health_factors(state, config)  # already called, but safe to call again if needed
#     mask = state["liquidatable_mask"]
#     indices = np.nonzero(mask)[0]
#
#     bd = state["borrower_data"]
#     prices = state["oracle_prices"]
#
#     for idx in indices:
#         total_debt = np.sum(bd["debt"][idx])
#         total_collateral_value = np.sum(bd["collateral"][idx] * prices)
#         shortfall = max(0.0, total_debt - total_collateral_value)
#         pending_bad_debt_this_step += shortfall
#
#     # Add to cumulative (or keep separate if you want to distinguish realized vs pending)
#     state["cumulative_bad_debt"] += pending_bad_debt_this_step
#
#     # Add to return dict for metrics
#     return {
#         "liquidated_count": liquidated_count,
#         "seized_usd": seized_usd,
#         "debt_closed": debt_closed,
#         "bad_debt_added": bad_debt_added,  # realized from executed liqs
#         "pending_bad_debt_this_step": pending_bad_debt_this_step,  # new: unliquidated underwater
#         "total_bad_debt_this_step": bad_debt_added + pending_bad_debt_this_step,
#         "unique_liquidated": len(state["liquidated_borrowers_set"])
#     }


# Define liquidator strategies for heterogeneity
# LIQUIDATOR_STRATEGIES = [
#     {"name": "aggressive", "min_profit_bps": 30},
#     {"name": "conservative", "min_profit_bps": 150},
#     {"name": "mev_searcher", "min_profit_bps": 20},
# ]
#
#
# def estimate_profitability(state: dict, borrower_idx, min_profit_bps: float, config) -> bool:
#     bd = state["borrower_data"]
#     total_debt = np.sum(bd["debt"][borrower_idx])
#     if total_debt <= 1e-8:
#         return False
#
#     total_proceeds = 0.0
#     seized_per_asset = bd["collateral"][borrower_idx] * config.close_factor
#     for asset_idx, asset in enumerate(state["assets"]):
#         seized = seized_per_asset[asset_idx]
#         if seized <= 1e-8:
#             continue
#
#         pool_key = f"{asset}_USDC"
#         if pool_key not in state["amm_reserves"]:
#             continue
#
#         preview = swap(
#             reserves=state["amm_reserves"][pool_key],
#             amount_in=seized,
#             token_in=asset,
#             token_out="USDC",
#             execute=False
#         )
#         total_proceeds += preview["amount_out"]
#
#     debt_to_cover = total_debt * config.close_factor
#     required_proceeds = debt_to_cover * (1 + config.liquidation_bonus)
#
#     if required_proceeds <= 1e-8:
#         return False
#
#     profit_bps = (total_proceeds - required_proceeds) / required_proceeds * 10_000
#     return profit_bps >= min_profit_bps
#
