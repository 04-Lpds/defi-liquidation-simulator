# monte_carlo.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from sim import run_simulation
from state import initialize_state   # make sure this import is correct
from config import Config


def run_monte_carlo(
    base_config: Config,
    n_per_bucket: int = 800,
    save_path: str = "monte_carlo_price_stress_results.csv"
) -> pd.DataFrame:

    random_seed_base = base_config.seed
    np.random.seed(random_seed_base)

    #Generate borrower dist. once
    base_borrowers = base_config.generate_borrowers()

    buckets = [
        (1.0, 1.5, "1.0–1.5× (Mild)"),
        (1.6, 2.2, "1.6–2.2× (Moderate)"),
        (2.3, 3.0, "2.3–3.0× (Severe)"),
        (3.1, 4.0, "3.1–4.0× (Extreme)"),
        (4.1, 5.0, "4.1–5.0× (Tail/Capped)")
    ]

    total_runs = n_per_bucket * len(buckets)
    print(f"Starting Monte Carlo: {total_runs} runs ({n_per_bucket} per bucket)")

    # Generate scale factors
    all_scales = []
    for min_s, max_s, _ in buckets:
        bucket_scales = np.random.uniform(min_s, max_s, n_per_bucket)
        all_scales.extend(bucket_scales)
    np.random.shuffle(all_scales)

    results = []
    # DEBUG print:
    print("Initial base min HF:",
          np.min(base_borrowers["health_factor"]))

    for i in tqdm(range(total_runs), desc="Monte Carlo runs"):
        # DEBUG PRINT:
        print(f"\nIteration {i}")
        print("Base min HF:",
              np.min(base_borrowers["health_factor"]))

        scale = all_scales[i]

        # Identify bucket label
        bucket_label = next(
            label for min_s, max_s, label in buckets
            if min_s <= scale <= max_s
        )

        config = base_config

        # Fresh borrower state for this iteration
        borrower_data = copy.deepcopy(base_borrowers)

        # Derive new stress price path
        derived_dict = Config.derive_new_price_path(
            historical_prices=config.price_path[['WETH', 'WBTC', 'SOL']],
            scale_factor=scale,
            max_drop_fraction=0.85,
            noise_std=0.015,
            front_load_fraction=0.6,
            random_seed=random_seed_base + i * 10
        )

        derived_df = pd.DataFrame(derived_dict)
        derived_df['USDC'] = 1.0
        derived_df = derived_df[config.assets]

        # state initialization (borrowers injected here)
        state = initialize_state(config, borrower_data)

        # Run simulation
        state = run_simulation(
            config,
            state=state,
            custom_price_path=derived_df
        )

        history = state.get("history", {})

        outcome = {
            "run_id": i,
            "scale_factor": round(scale, 3),
            "bucket": bucket_label,
            "final_bad_debt": state.get("cumulative_bad_debt", 0),
            "peak_liquidatable_pct": max(history.get("percent_liquidatable", [0])),
            "total_liquidations": sum(history.get("liquidations_per_step", [0])),
            "max_liquidations_per_step": max(history.get("liquidations_per_step", [0])),
            "final_median_hf": history.get("median_hf", [np.nan])[-1],
        }

        results.append(outcome)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    print(f"\nSaved to: {save_path}")

    return df_results


if __name__ == "__main__":
    config = Config()
    results = run_monte_carlo(config, n_per_bucket=2)

    # Quick visualization
    import seaborn as sns
    import matplotlib.pyplot as plt

    bucket_order = [
        "1.0–1.5× (Mild)",
        "1.6–2.2× (Moderate)",
        "2.3–3.0× (Severe)",
        "3.1–4.0× (Extreme)",
        "4.1–5.0× (Tail/Capped)"
    ]

    results['bucket'] = pd.Categorical(
        results['bucket'],
        categories=bucket_order,
        ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='bucket',
        y='final_bad_debt',
        data=results,
        palette='viridis'
    )
    sns.pointplot(
        x='bucket',
        y='final_bad_debt',
        data=results,
        color='red',
        markers='o',
        linestyles='--',
        errorbar=None
    )

    plt.title(
        f"Final Bad Debt Distribution by Scaled Price Shock Severity "
        f"from {config.crisis_date}"
    )
    plt.xlabel("Severity Bucket (Scale Factor)")
    plt.ylabel("Final Bad Debt ($)")
    plt.xticks(rotation=70)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bad_debt_by_bucket.png", dpi=300, bbox_inches='tight')
    print("Chart saved as bad_debt_by_bucket.png")
    plt.show()