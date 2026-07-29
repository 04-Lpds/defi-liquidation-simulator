from config import Config
from state import initialize_state
from borrowers import update_health_factors, plot_borrower_distributions
from liquidations import process_liquidations
from metrics import record_step_metrics, print_final_summary, plot_key_metrics, plot_final_hf_distribution, \
    get_research_summary
from oracle import Oracle, HybridOracle
from amm import rebalance_amm_pools


def run_simulation(config: Config, state: dict, custom_price_path=None):
    if config.plot_borrower_distributions:
        plot_borrower_distributions(state, config)

    price_path = custom_price_path if custom_price_path is not None else config.price_path

    if config.use_hybrid_oracle:
        oracle = HybridOracle(
            price_path=price_path,
            amm_reserves=state["amm_reserves"],
            delay_minutes=config.oracle_delay,
            amm_weight=config.oracle_amm_weight,
            ema_alpha=config.oracle_ema_alpha
        )
    else:
        oracle = Oracle(price_path, config.oracle_delay)

    metrics_history = []

    for step in range(len(price_path)):
        state["current_step"] = step
        state["oracle_prices"] = oracle.get_current_prices()

        update_health_factors(state, config)
        liq_data = process_liquidations(state, config)
        rebalance_amm_pools(state, config)

        row = price_path.iloc[step]
        step_metrics = record_step_metrics(state, config, step, row, liq_data)
        metrics_history.append(step_metrics)

        oracle.advance_step()

    state["metrics_history"] = metrics_history
    if config.plot_sim_metrics:
        print_final_summary(metrics_history, state, config)
        plot_key_metrics(state, config)
        plot_final_hf_distribution(state, config)

    # ==================== RESEARCH OUTPUT ====================
    research_summary = get_research_summary(state, config)
    print_research_summary = False
    if print_research_summary:
        print("\n" + "=" * 70)
        print("RESEARCH SUMMARY")
        print("=" * 70)
        for k, v in research_summary.items():
            print(f"  {k:30}: {v}")
        print("=" * 70)

    state["research_summary"] = research_summary
    return state


if __name__ == "__main__":
    config = Config()
    state = initialize_state(config)

    """Set toggleable config settings for single simulation"""
    config.plot_borrower_distributions = False
    print_config = True
    config.plot_sim_metrics = True
    config.print_steps_size = 100  # bad for multiple iterations
    config.save_charts = True

    # Run simulation
    run_simulation(config, state)