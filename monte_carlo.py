# monte_carlo.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from pathlib import Path
from datetime import datetime

from sim import run_simulation
from state import initialize_state
from config import Config
from metrics import plot_key_metrics, plot_final_hf_distribution

def run_monte_carlo(
    base_config: Config = None,
    n_per_bucket: int = 500,
    save_path: str = "monte_carlo_results.csv"
) -> pd.DataFrame:

    # Instantiate config one time
    if base_config is None:
        base_config = Config()

    np.random.seed(base_config.seed)  # Set random seed for reproducibility

    print(f"Monte Carlo on base date: {base_config.crisis_date}")
    print(f"Random seed: {base_config.seed}")

    # === ONE-TIME INITIALIZATION ===
    print("Initializing base data once...")
    base_config.reload_for_new_date(base_config.crisis_date)


    # Copy initial simulation state
    base_borrowers = copy.deepcopy(base_config.borrowers)
    base_price_path = base_config.price_path.copy()
    base_amm_pools = copy.deepcopy(base_config.amm_pools)

    # Create unique folder for this Monte Carlo run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_base_dir = Path("results/charts") / f"monte_carlo_{timestamp}"
    charts_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Charts shall be saved to: {charts_base_dir}\n")

    # Define severity buckets
    buckets = [
        (1.0, 1.5, "1.0–1.5× (Mild)"),
        (1.6, 2.2, "1.6–2.2× (Moderate)"),
        (2.3, 3.0, "2.3–3.0× (Severe)"),
        (3.1, 4.0, "3.1–4.0× (Extreme)"),
        (4.1, 5.0, "4.1–5.0× (Tail/Capped)")
    ]

    total_runs = n_per_bucket * len(buckets)
    print(f"Starting Monte Carlo: {total_runs} runs\n")

    all_scales = []
    for min_s, max_s, _ in buckets:
        all_scales.extend(np.random.uniform(min_s, max_s, n_per_bucket))
    np.random.shuffle(all_scales)

    results = []

    for i in tqdm(range(total_runs), desc="Monte Carlo runs"):
        scale = all_scales[i]
        bucket_label = next(label for min_s, max_s, label in buckets if min_s <= scale <= max_s)  # “Go through list of buckets, find first one where current scale fits inside range, give name/label.”



        # Fresh config
        #config = Config()
        config.crisis_date = base_config.crisis_date

        config.price_path = base_price_path.copy()
        config.initial_prices = base_config.initial_prices.copy()
        config.amm_pools = copy.deepcopy(base_amm_pools)
        config.borrowers = copy.deepcopy(base_borrowers)

        # Stochastic path
        derived_dict = Config.derive_new_price_path(
            historical_prices=base_price_path[['WETH', 'WBTC', 'SOL']],
            scale_factor=scale,
            max_drop_fraction=0.85,
            noise_std=0.015,
            front_load_fraction=0.6,
            random_seed=base_config.seed + i * 10
        )

        derived_df = pd.DataFrame(derived_dict)
        derived_df['USDC'] = 1.0
        derived_df = derived_df[config.assets]

        config.initial_prices = derived_df.iloc[0].to_dict()

        # Fresh state
        borrower_data = copy.deepcopy(config.borrowers)
        state = initialize_state(config, borrower_data)
        state["amm_reserves"] = copy.deepcopy(base_amm_pools)

        # === CHART SETTINGS + PER-RUN FOLDER ===
        config.plot_sim_metrics = False
        config.save_charts = True
        config.plot_final_hf_dist = True
        config.plot_borrower_distributions = False

        # Optional: tell plot_key_metrics to use a specific folder
        config.current_chart_dir = charts_base_dir / f"run_{i:03d}_scale{scale:.3f}"
        config.current_chart_dir.mkdir(exist_ok=True)

        # Run
        config.use_hybrid_oracle = False
        state = run_simulation(config, state, custom_price_path=derived_df)

        # Results
        history = state.get("history", {})
        outcome = {
            "run_id": i,
            "scale_factor": round(scale, 3),
            "bucket": bucket_label,
            "final_bad_debt": float(state.get("cumulative_bad_debt", 0.0)),
            "peak_liquidatable_pct": float(max(history.get("percent_liquidatable", [0.0]))),
            "total_liquidations": int(sum(history.get("liquidations_per_step", [0]))),
            "final_median_hf": float(history.get("median_hf", [np.nan])[-1]),
        }

        results.append(outcome)

        # Optional save plots:
        if getattr(config, 'save_charts', True):
            plot_key_metrics(state, config, save_dir=config.current_chart_dir)

            if getattr(config, 'plot_final_hf_dist', True):
                plot_final_hf_distribution(state, config, save_dir=config.current_chart_dir)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    print(f"\nMonte Carlo completed! Results saved to: {save_path}")
    print(f"Charts saved in: {config.current_chart_dir}")
    return df_results


if __name__ == "__main__":
    config = Config()
    results = run_monte_carlo(config, n_per_bucket=80)