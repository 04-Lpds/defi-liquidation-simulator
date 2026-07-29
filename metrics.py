# Collection, summarization, and plotting of simulation metrics
# Centralized time-series history with cumulative tracking

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path
from borrowers import update_health_factors
from pathlib import Path
from datetime import datetime

# Global charts directory
CHARTS_DIR = Path("results/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_pending_bad_debt(state, config):
    """
    Computes two values:
    - pending_debt: total outstanding debt on underwater (HF < 1) positions
    - economic_shortfall: sum of (debt - collateral_value) for underwater positions
    """
    update_health_factors(state, config)

    bd = state["borrower_data"]
    health = bd["health_factor"]

    underwater_mask = health < 1.0

    # Pending debt: full outstanding debt on underwater borrowers
    pending_debt = np.sum(bd["debt"][underwater_mask])

    # Economic shortfall
    prices = state["oracle_prices"]
    collateral_usd = np.sum(bd["collateral"][underwater_mask] * prices, axis=1)
    debt_usd = np.sum(bd["debt"][underwater_mask] * prices, axis=1)
    economic_shortfall = np.sum(np.maximum(0.0, debt_usd - collateral_usd))

    return pending_debt, economic_shortfall


def record_step_metrics(state: dict, config, step: int, price_row: pd.Series, liq_data: dict) -> dict:
    """
    Record key metrics for one simulation step.
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

    metrics = {
        "step": step,
        "timestamp": price_row.name if hasattr(price_row, 'name') else step,
        "cumulative_realized_bad_debt": state["cumulative_bad_debt"],
        "pending_debt": pending_debt,
        "economic_shortfall_this_step": economic_shortfall,
        "realized_bad_debt_this_step": realized_bad_debt_this_step,
        "total_bad_debt_this_step": total_bad_debt_this_step,
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

    # Oracle prices, API prices, AMM spots
    for asset in config.assets[:-1]:
        idx = config.assets.index(asset)
        oracle_price = state["oracle_prices"][idx]
        history.setdefault(f"price_{asset}", []).append(oracle_price)

        delayed_step = max(0, step - config.oracle_delay)
        api_price = config.price_path.iloc[delayed_step][asset]
        history.setdefault(f"api_price_{asset}", []).append(api_price)

        pool_key = f"{asset}_USDC"
        amm_spot = np.nan
        if pool_key in state.get("amm_reserves", {}):
            pool = state["amm_reserves"][pool_key]
            amm_spot = pool['USDC'] / pool[asset] if pool[asset] > 0 else np.nan
        history.setdefault(f"amm_spot_{asset}", []).append(amm_spot)

    # ==================== NEW RESEARCH TRACKING ====================
    history["peak_liquidatable_pct"] = max(history.get("peak_liquidatable_pct", 0.0), metrics["liquidatable_pct"])
    history["peak_pending_debt"] = max(history.get("peak_pending_debt", 0.0), metrics["pending_debt"])
    history["peak_economic_shortfall"] = max(history.get("peak_economic_shortfall", 0.0), metrics["economic_shortfall_this_step"])
    history["cumulative_liquidations"] = sum(history["liquidations_per_step"])

    # Progress print
    if config.plot_sim_metrics and (step % config.print_steps_size == 0 or step == len(config.price_path) - 1):
        print(f"Step {step} | Liqs: {metrics['liquidations_this_step']} | "
              f"% Liq: {metrics['liquidatable_pct']:.1f}% | "
              f"Pending Debt: ${metrics['pending_debt']:,.0f} | "
              f"Econ Shortfall: ${metrics['economic_shortfall_this_step']:,.0f}")

    return metrics


def plot_key_metrics(state: dict, config, title: str = "Liquidity Cascade Simulation ", save_dir=None):
    """
    Plot cascade dynamics and ALWAYS save the chart.
    - Accepts save_dir from param_sweep (recommended)
    - Creates new folder only if no save_dir is passed (manual runs)
    """
    history = state["history"]
    if not history["steps"]:
        print("No history data to plot.")
        return

    # === Determine save directory ===
    if save_dir is None:
        # Manual run → create new folder
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = CHARTS_DIR / f"unknown_run_{run_timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
    # else: use the folder passed from param_sweep

    # === Create filename ===
    alpha = getattr(config, 'oracle_ema_alpha', 0.3)
    beta = getattr(config, 'oracle_amm_weight', 0.1)
    date_str = getattr(config, 'crisis_date', 'unknown').replace("-", "")

    filename = f"cascade_{date_str}_alpha{alpha}_beta{beta}.png"
    save_path = save_dir / filename

    # ==================== PLOTTING CODE (your original) ====================
    steps = np.array(history["steps"])

    fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(title+config.crisis_date, fontsize=18, fontweight='bold')

    # 0. Normalized Oracle Prices
    assets_to_plot = ["WETH", "WBTC", "SOL"]
    colors = {"WETH": "blue", "WBTC": "orange", "SOL": "green"}

    for asset in assets_to_plot:
        price_key = f"price_{asset}"
        if price_key in history and history[price_key]:
            prices = np.array(history[price_key])
            if len(prices) > 0 and prices[0] > 0:
                normalized = 100 * prices / prices[0]
                normalized[0] = 100.0
                axes[0].plot(steps, normalized, label=f"{asset} Oracle", color=colors.get(asset, 'gray'))

    axes[0].set_ylabel("Normalized Price")
    axes[0].set_title("Normalized Oracle Prices per Asset")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 1. API vs AMM Spot Prices
    for asset in assets_to_plot:
        api_key = f"api_price_{asset}"
        if api_key in history and history[api_key]:
            api_prices = np.array(history[api_key])
            if len(api_prices) > 0 and api_prices[0] > 0:
                normalized = 100 * api_prices / api_prices[0]
                normalized[0] = 100.0
                axes[1].plot(steps, normalized, label=f"{asset} API", linestyle='-', alpha=0.9)

        spot_key = f"amm_spot_{asset}"
        if spot_key in history and history[spot_key]:
            spots = np.array(history[spot_key])
            if len(spots) > 0 and not np.isnan(spots[0]) and spots[0] > 0:
                normalized = 100 * spots / spots[0]
                normalized[0] = 100.0
                axes[1].plot(steps, normalized, label=f"{asset} AMM Spot", linestyle='--', alpha=0.7)

    axes[1].set_ylabel("Normalized Price")
    axes[1].set_title("API Price vs AMM Spot Price (Normalized)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 2. % Liquidatable + Cumulative Liquidations
    axes[2].plot(steps, history["percent_liquidatable"], color='orange', lw=2, label="% Liquidatable")
    axes[2].set_ylabel("% Liquidatable", color='orange')
    ax_cum = axes[2].twinx()
    cum_liqs = np.cumsum(history["liquidations_per_step"])
    ax_cum.plot(steps, cum_liqs, color='darkred', lw=2, label="Cumulative Liquidations")
    ax_cum.set_ylabel("Cumulative Liquidations", color='darkred')
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax_cum.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    axes[2].grid(alpha=0.3)

    # 3. Liquidation Waves
    axes[3].plot(steps, history["liquidations_per_step"], color='red', lw=1.5)
    axes[3].set_ylabel("Liqs / step")
    axes[3].set_title("Liquidation Waves")
    axes[3].grid(alpha=0.3)

    # 4. Pending Bad Debt
    axes[4].plot(steps, history["pending_bad_debt_per_step"], color='purple', lw=2)
    axes[4].set_ylabel("Pending Liquidations")
    axes[4].set_title("Unliquidated Underwater Debt")
    axes[4].grid(alpha=0.3)

    # 5. Economic Shortfall & Total Bad Debt
    axes[5].plot(steps, history["economic_shortfall_per_step"], color='magenta', lw=2, label="Economic Shortfall")
    axes[5].plot(steps, history["total_bad_debt_per_step"], color='black', lw=2, label="Total Bad Debt This Step")
    axes[5].set_ylabel("USD")
    axes[5].set_title("Economic Shortfall & Total Bad Debt")
    axes[5].legend()
    axes[5].set_xlabel("Simulation Time (Minutes)")
    axes[5].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save the chart
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    #print(f"Chart saved → {save_path}")

    # Show GUI only if requested
    if getattr(config, 'plot_sim_metrics', True):
        plt.show()
    else:
        plt.close()

    return


def plot_final_hf_distribution(state: dict, config, save_dir=None):
    if not config.plot_final_hf_dist:
        return
    filename = f"final_hf_distribution_{config.crisis_date}.png"
    save_path = save_dir / filename
    update_health_factors(state, config)
    hf = state["borrower_data"]["health_factor"]
    hf_finite = hf[np.isfinite(hf)]

    if len(hf_finite) == 0:
        print("No finite health factors to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(hf_finite, bins=config.hf_hist_bins or 100, range=config.hf_hist_range or (0, 3),
             color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label="HF < 1 = Liquidatable")
    plt.title(f"Final Health Factor Distribution\n{config.crisis_date}")
    plt.xlabel("Health Factor")
    plt.ylabel("Number of Borrowers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"chart saved to {save_path}")
    if getattr(config, 'plot_sim_metrics', True):
        plt.show()
    else:
        plt.close()

    return


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
    print(f"Peak % liquidatable: {peak_liq:.2f}%")
    print(f"Total liquidations: {df['liquidations_this_step'].sum():,.0f}")
    print("="*60)


def get_research_summary(state: dict, config) -> dict:
    """Summary of liquidations data"""
    h = state["history"]

    total_steps = len(h["steps"])
    total_liqs = int(h.get("cumulative_liquidations", 0))

    # === Calculate actual cascade duration ===
    liqs_per_step = np.array(h["liquidations_per_step"])

    if total_liqs == 0:
        cascade_duration = 0
        cascade_start = 0
        cascade_end = 0
    else:
        # First step with any liquidation
        cascade_start = np.where(liqs_per_step > 0)[0][0]

        # Find the last significant liquidation wave
        # (last step where liqs > 5, or end of sim)
        significant_liqs = np.where(liqs_per_step > 5)[0]
        if len(significant_liqs) > 0:
            cascade_end = significant_liqs[-1]
        else:
            cascade_end = cascade_start

        cascade_duration = cascade_end - cascade_start + 1

    summary = {
        "crisis_date": config.crisis_date,
        "oracle": "Hybrid" if getattr(config, 'use_hybrid_oracle', False) else "Pure CEX",
        "amm_weight": getattr(config, 'oracle_amm_weight', 0.0),
        "ema_alpha": getattr(config, 'oracle_ema_alpha', 0.0),

        "total_liquidations": total_liqs,
        "peak_liquidatable_pct": round(float(h.get("peak_liquidatable_pct", 0)), 2),
        "peak_pending_debt_usd": round(float(h.get("peak_pending_debt", 0)), 0),
        "peak_economic_shortfall_usd": round(float(h.get("peak_economic_shortfall", 0)), 0),
        "final_pending_debt_usd": round(
            float(h["pending_bad_debt_per_step"][-1] if h["pending_bad_debt_per_step"] else 0), 0),

        "cascade_duration_minutes": int(cascade_duration),
        "cascade_start_minute": int(cascade_start),
        "full_sim_minutes": total_steps,
    }
    return summary