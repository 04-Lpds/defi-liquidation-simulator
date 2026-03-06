# metrics.py
# Collection, summarization, and plotting of simulation metrics
# Centralized time-series history with cumulative tracking

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
from pathlib import Path
from borrowers import update_health_factors



RESULTS_FILE = Path("results/liquidation_runs.csv")

def calculate_pending_bad_debt(state, config):
    """
    Computes two values:
    - pending_debt: total outstanding debt on underwater (HF < 1) positions
    - economic_shortfall: sum of (debt - collateral_value) for underwater positions
    Returns tuple (pending_debt: float, economic_shortfall: float)
    """
    update_health_factors(state, config)

    bd = state["borrower_data"]
    health = bd["health_factor"]

    underwater_mask = health < 1.0

    # Pending debt: full outstanding debt on underwater borrowers
    pending_debt = np.sum(bd["debt"][underwater_mask])

    # Economic shortfall: how much is actually unrecoverable if liquidated now
    prices = state["oracle_prices"]
    collateral_usd = np.sum(bd["collateral"][underwater_mask] * prices, axis=1)
    debt_usd = np.sum(bd["debt"][underwater_mask] * prices, axis=1)  # adjust if debt priced differently
    economic_shortfall = np.sum(np.maximum(0.0, debt_usd - collateral_usd))

    return pending_debt, economic_shortfall


def write_results(row: dict, totals):
    df = pd.DataFrame([row])
    if RESULTS_FILE.exists():
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

def record_step_metrics(state: dict, config, step: int, price_row: pd.Series, liq_data: dict) -> dict:
    """
    Record key metrics for one simulation step.
    Central place for all metric calculations and history appends.
    """
    bd = state["borrower_data"]
    hf = bd["health_factor"]
    hf_finite = hf[np.isfinite(hf)]

    def safe_get(key: str, default=0.0):
        if isinstance(liq_data, dict):
            return liq_data.get(key, default)
        return default

    # === Bad debt calculations ===
    pending_debt, economic_shortfall = calculate_pending_bad_debt(state, config)

    realized_bad_debt_this_step = safe_get("bad_debt_added", 0.0)
    total_bad_debt_this_step = realized_bad_debt_this_step + economic_shortfall

    cumulative_realized = state["cumulative_bad_debt"]
    cumulative_total_approx = cumulative_realized + economic_shortfall  # running approx

    metrics = {
        "step": step,
        "timestamp": price_row.name if hasattr(price_row, 'name') else step,
        "cumulative_realized_bad_debt": cumulative_realized,
        "pending_debt": pending_debt,
        "economic_shortfall_this_step": economic_shortfall,
        "realized_bad_debt_this_step": realized_bad_debt_this_step,
        "total_bad_debt_this_step": total_bad_debt_this_step,
        "cumulative_total_bad_debt_approx": cumulative_total_approx,
        "liquidatable_count": np.sum(state["liquidatable_mask"]),
        "liquidatable_pct": np.mean(state["liquidatable_mask"]) * 100,
        "median_hf": np.median(hf_finite) if len(hf_finite) > 0 else np.nan,
        "mean_hf": np.mean(hf_finite) if len(hf_finite) > 0 else np.nan,
        "min_hf": np.min(hf_finite) if len(hf_finite) > 0 else np.nan,
        "liquidations_this_step": safe_get("liquidated_count", 0),
        "seized_usd_this_step": safe_get("seized_usd", 0.0),
        "debt_closed_this_step": safe_get("debt_closed", 0.0),
    }

    # === Append all time-series data to history ===
    history = state["history"]

    # Core time-series
    history["steps"].append(step)
    history["liquidations_per_step"].append(metrics["liquidations_this_step"])
    history["percent_liquidatable"].append(metrics["liquidatable_pct"])
    history["pending_bad_debt_per_step"].append(metrics["pending_debt"])
    history["economic_shortfall_per_step"].append(metrics["economic_shortfall_this_step"])
    history["total_bad_debt_per_step"].append(metrics["total_bad_debt_this_step"])

    # Cumulative seized / debt closed
    prev_seized = history["seized_usd_cumulative"][-1] if history["seized_usd_cumulative"] else 0.0
    prev_debt_closed = history["debt_closed_cumulative"][-1] if history["debt_closed_cumulative"] else 0.0
    history["seized_usd_cumulative"].append(prev_seized + metrics["seized_usd_this_step"])
    history["debt_closed_cumulative"].append(prev_debt_closed + metrics["debt_closed_this_step"])

    # Oracle prices, API prices, AMM spots, and deviation for each asset
    for asset in config.assets[:-1]:  # exclude USDC
        idx = config.assets.index(asset)

        # Oracle price (blended/hybrid)
        oracle_price = state["oracle_prices"][idx]
        price_key = f"price_{asset}"
        if price_key not in history:
            history[price_key] = []
        history[price_key].append(oracle_price)

        # Raw API price (pure external feed, delayed if applicable)
        delayed_step = max(0, step - config.oracle_delay)
        api_price = config.price_path.iloc[delayed_step][asset]
        api_key = f"api_price_{asset}"
        if api_key not in history:
            history[api_key] = []
        history[api_key].append(api_price)

        # AMM spot price
        pool_key = f"{asset}_USDC"
        amm_spot = np.nan
        if pool_key in state["amm_reserves"]:
            pool = state["amm_reserves"][pool_key]
            amm_spot = pool['USDC'] / pool[asset] if pool[asset] > 0 else np.nan

        spot_key = f"amm_spot_{asset}"
        if spot_key not in history:
            history[spot_key] = []
        history[spot_key].append(amm_spot)

        # Deviation (oracle - AMM spot)
        deviation = oracle_price - amm_spot if not np.isnan(amm_spot) else np.nan
        dev_key = f"price_deviation_{asset}"
        if dev_key not in history:
            history[dev_key] = []
        history[dev_key].append(deviation)

    # Progress print
    if config.plot_sim_metrics and (step % config.print_steps_size == 0 or step == len(config.price_path) - 1):
        print(f"Step {step} | Liqs: {metrics['liquidations_this_step']} | "
              f"% Liq: {metrics['liquidatable_pct']:.1f}% | "
              f"Pending Debt: ${metrics['pending_debt']:,.0f} | "
              f"Econ Shortfall: ${metrics['economic_shortfall_this_step']:,.0f}")

    return metrics



def plot_key_metrics(state: dict, config, title: str = "Liquidity Cascade Simulation"):
    """
    Plot time-series cascade dynamics.
    - Normalized oracle prices as top chart
    - Normalized API vs AMM spot prices in second chart
    - % Liquidatable + cumulative liqs on twin axis
    - Pending Bad Debt linear
    - Economic Shortfall & Total Bad Debt at bottom
    """
    history = state["history"]
    if not history["steps"]:
        print("No history data to plot.")
        return

    steps = np.array(history["steps"])

    fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # 0. Normalized Oracle Prices (Start = 100)
    assets_to_plot = ["WETH", "WBTC", "SOL"]
    colors = {"WETH": "blue", "WBTC": "orange", "SOL": "green"}

    plotted_oracle_lines = []
    for asset in assets_to_plot:
        price_key = f"price_{asset}"
        if price_key in history and history[price_key]:
            prices = np.array(history[price_key])
            if len(prices) > 0 and prices[0] > 0:
                normalized = 100 * prices / prices[0]
                normalized[0] = 100.0
                color = colors.get(asset, 'gray')
                line, = axes[0].plot(steps, normalized, label=f"{asset} Oracle", color=color)
                plotted_oracle_lines.append(line)

    axes[0].set_ylabel("Normalized Price")
    axes[0].set_title("Normalized Oracle Prices per Asset")
    if plotted_oracle_lines:
        axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3)

    # 1. API vs AMM Spot Prices (Normalized)
    plotted_api_lines = []
    plotted_amm_lines = []
    for asset in assets_to_plot:
        # API price
        api_key = f"api_price_{asset}"
        if api_key in history and history[api_key]:
            api_prices = np.array(history[api_key])
            if len(api_prices) > 0 and api_prices[0] > 0:
                normalized_api = 100 * api_prices / api_prices[0]
                normalized_api[0] = 100.0
                color = colors.get(asset, 'gray')
                line, = axes[1].plot(steps, normalized_api, label=f"{asset} API", linestyle='-', color=color, alpha=0.9)
                plotted_api_lines.append(line)

        # AMM spot price
        spot_key = f"amm_spot_{asset}"
        if spot_key in history and history[spot_key]:
            spots = np.array(history[spot_key])
            if len(spots) > 0 and not np.isnan(spots[0]) and spots[0] > 0:
                normalized_spot = 100 * spots / spots[0]
                normalized_spot[0] = 100.0
                line, = axes[1].plot(steps, normalized_spot, label=f"{asset} AMM Spot", linestyle='--', color=color, alpha=0.7)
                plotted_amm_lines.append(line)

    axes[1].set_ylabel("Normalized Price")
    axes[1].set_title("API Price vs AMM Spot Price (Normalized)")
    if plotted_api_lines or plotted_amm_lines:
        axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(alpha=0.3)

    # 2. % Liquidatable + Cumulative Liquidations (twin axis)
    axes[2].plot(steps, history["percent_liquidatable"], color='orange', lw=2, label="% Liquidatable")
    axes[2].set_ylabel("% Liquidatable", color='orange')
    axes[2].tick_params(axis='y', labelcolor='orange')
    axes[2].set_title("Underwater Borrowers & Cumulative Liquidations")
    axes[2].grid(alpha=0.3)

    cum_liqs = np.cumsum(history["liquidations_per_step"])
    ax_cum = axes[2].twinx()
    ax_cum.plot(steps, cum_liqs, color='darkred', lw=2, label="Cumulative Liqs")
    ax_cum.set_ylabel("Cumulative Liquidations", color='darkred')
    ax_cum.tick_params(axis='y', labelcolor='darkred')

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax_cum.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # 3. Liquidation Waves
    axes[3].plot(steps, history["liquidations_per_step"], color='red', lw=1.5)
    axes[3].set_ylabel("Liqs / step")
    axes[3].set_title("Liquidation Waves")
    axes[3].grid(alpha=0.3)

    # 4. Pending Bad Debt (Linear Scale)
    axes[4].plot(steps, history["pending_bad_debt_per_step"], color='purple', lw=2)
    axes[4].set_ylabel("Pending Bad Debt (USD)")
    axes[4].set_title("Pending Bad Debt (Unliquidated Underwater Debt)")
    axes[4].grid(alpha=0.3)

    # 5. Economic Shortfall & Total Bad Debt
    axes[5].plot(steps, history["economic_shortfall_per_step"], color='magenta', lw=2,
                 label="Economic Shortfall")
    axes[5].plot(steps, history["total_bad_debt_per_step"], color='black', lw=2,
                 label="Total Bad Debt This Step")
    axes[5].set_ylabel("USD")
    axes[5].set_title("Economic Shortfall & Total Bad Debt")
    axes[5].legend()
    axes[5].set_xlabel("Simulation Time (Minutes)")
    axes[5].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"cascade_dynamics_{config.crisis_date}_amm{config.oracle_amm_weight}_ema{config.oracle_ema_alpha}.png", dpi=250, bbox_inches='tight')
    plt.show()


def plot_final_hf_distribution(state: dict, config):
    """
    Plot the final health factor distribution as a histogram.
    Call this at the end of the simulation if config.plot_final_hf_dist is True.

    Args:
        state: The final simulation state
        config: Config object (for toggles, crisis_date, etc.)
    """
    if not config.plot_sim_metrics or not config.plot_final_hf_dist:
        return  # Skip if toggled off

    # Ensure health factors are up to date
    update_health_factors(state, config)

    hf = state["borrower_data"]["health_factor"]
    hf_finite = hf[np.isfinite(hf)]

    if len(hf_finite) == 0:
        print("No finite health factors to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(hf_finite, bins=config.hf_hist_bins or 100,
             range=config.hf_hist_range or (0, 3),
             color='lightgreen', edgecolor='black', alpha=0.7)

    # Threshold line
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label=f"HF < 1 = Liquidatable (Threshold {config.liquidation_threshold})")

    # Optional: highlight danger zones
    plt.axvspan(0, 0.8, color='red', alpha=0.1, label="Deep Underwater (<0.8)")
    plt.axvspan(0.8, 1.0, color='yellow', alpha=0.1, label="Marginal (0.8–1.0)")

    plt.title(f"Final Health Factor Distribution\n{config.crisis_date} - {len(hf_finite)} Borrowers")
    plt.xlabel("Health Factor")
    plt.ylabel("Number of Borrowers")
    plt.legend()
    plt.grid(alpha=0.3)

    # Save with date
    plt.savefig(f"final_hf_distribution_{config.crisis_date}.png", dpi=200, bbox_inches='tight')
    plt.show()

def summarize_simulation(metrics_history: List[dict]) -> pd.DataFrame:
    if not metrics_history:
        return pd.DataFrame()
    df = pd.DataFrame(metrics_history)
    df.set_index("step", inplace=True)
    return df


def print_final_summary(metrics_history: List[dict], state: dict, config):
    if not metrics_history:
        print("No metrics recorded.")
        return

    df = summarize_simulation(metrics_history)
    final = df.iloc[-1]
    peak_liq = df["liquidatable_pct"].max()

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total steps (minutes): {len(df)}")
    print(f"Final cumulative realized bad debt: ${final['cumulative_realized_bad_debt']:,.0f}")
    print(f"Final pending debt: ${final['pending_debt']:,.0f}")
    print(f"Final economic shortfall (last step): ${final['economic_shortfall_this_step']:,.0f}")
    print(f"Final total bad debt: ${final['total_bad_debt_this_step']:,.0f}")
    print(f"Peak % liquidatable borrowers: {peak_liq:.2f}%")
    print(f"Total # liquidations: {df['liquidations_this_step'].sum():,.0f}")
    print(f"Total seized collateral (USD): ${df['seized_usd_this_step'].sum():,.0f}")
    print(f"Total debt closed (USD): ${df['debt_closed_this_step'].sum():,.0f}")
    print(f"Final median HF: {final['median_hf']:.3f}")
    print(f"Final mean HF: {final['mean_hf']:.3f}")
    print("Final AMM pools reserves:")
    for pool_key, pool in state["amm_reserves"].items():
        asset = pool_key.split("_")[0]
        if asset:
            print(f"{pool_key}: {pool[asset]:.0f} / {pool['USDC']:.0f} USDC")
    print("="*60)

    totals = {
        "total_liq_volume": float(df['liquidations_this_step'].sum()),
        "total_seized_col": float(df['seized_usd_this_step'].sum()),
        "total_debt_closed": float(df['debt_closed_this_step'].sum()),
        "final_realized_bad_debt": float(final['cumulative_realized_bad_debt']),
        "final_pending_debt": float(final['pending_debt']),
        "final_total_bad_debt": float(final['total_bad_debt_this_step']),
    }
    write_results(final.to_dict(), totals)

