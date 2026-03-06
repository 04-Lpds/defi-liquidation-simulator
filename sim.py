# sim.py
from config import Config
from state import initialize_state
from borrowers import update_health_factors, plot_borrower_distributions
from liquidations import process_liquidations
from metrics import record_step_metrics, summarize_simulation, plot_final_hf_distribution, print_final_summary, \
    calculate_pending_bad_debt, plot_key_metrics
from oracle import Oracle, HybridOracle
from amm import rebalance_amm_pools


def run_simulation(config: Config, state: dict, custom_price_path=None):

    if config.plot_borrower_distributions:
        plot_borrower_distributions(state, config)

    if custom_price_path is None:
        print("Using unscaled historical data...")
    price_path = custom_price_path if custom_price_path is not None else config.price_path

    if config.use_hybrid_oracle:
        oracle = HybridOracle(
            price_path=price_path,
            amm_reserves=state["amm_reserves"],
            delay_minutes=config.oracle_delay,
            amm_weight=config.oracle_amm_weight,  # e.g. 0.3
            ema_alpha=config.oracle_ema_alpha  # e.g. 0.1
        )
    else:
        oracle = Oracle(price_path, config.oracle_delay)

    metrics_history = []

    for step in range(len(price_path)):
        state["current_step"] = step  # needed for delayed API price lookup
        state["oracle_prices"] = oracle.get_current_prices()

        # === DEBUG: Oracle price check checks ===
        # print(f"\nStep {step} - oracle_prices type: {type(state['oracle_prices'])}")
        # print(
        #     f"oracle_prices shape/len: {state['oracle_prices'].shape if hasattr(state['oracle_prices'], 'shape') else len(state['oracle_prices'])}")
        # print(
        #     f"oracle_prices sample: {state['oracle_prices'][:5] if hasattr(state['oracle_prices'], '__getitem__') else state['oracle_prices']}")
        # print(
        #     f"oracle_prices dtype: {state['oracle_prices'].dtype if hasattr(state['oracle_prices'], 'dtype') else type(state['oracle_prices'][0]) if state['oracle_prices'] else 'empty'}")

        # # Compare to expected
        # expected_sample = config.price_path.iloc[
        #     step - config.oracle_delay if step >= config.oracle_delay else 0].values[:5]
        # print(f"Expected API sample (no delay): {expected_sample}")

        # update_health_factors(state, config)
        # print(f"After health update | min HF: {np.min(state['borrower_data']['health_factor']):.4f}")
        # print(f"Num liquidatable: {np.sum(state['liquidatable_mask'])}")
        # print("-" * 60)

        update_health_factors(state, config)
        liq_data = process_liquidations(state, config)
        # REMOVED: calculate_pending_bad_debt(state, config)  # moved to record_step_metrics
        rebalance_amm_pools(state, config)

        row = price_path.iloc[step]
        step_metrics = record_step_metrics(state, config, step, row, liq_data)
        metrics_history.append(step_metrics)

        oracle.advance_step()

    # Store metrics history in state for plotting
    state["metrics_history"] = metrics_history

    if config.plot_sim_metrics:
        print_final_summary(metrics_history, state, config)
        plot_key_metrics(state, config, title=f"Liquidation Cascade Simulation for {config.crisis_date}")
        if config.plot_sim_metrics:
            plot_final_hf_distribution(state, config)

    return state


if __name__ == "__main__":
    config = Config()
    state = initialize_state(config)
    run_simulation(config, state)
