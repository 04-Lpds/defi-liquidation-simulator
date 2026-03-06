# config.py
import numpy as np
import pandas as pd
import requests
from datetime import datetime


class Config:
    # --------------------- General ---------------------
    seed = 43
    num_borrowers = 5500
    removal_buffer = 0.02  # optional buffer to remove preemptive large liqs
    #num_runs = 1
    use_heterogeneous_liquidators = False  # Liquidators with different profit targets
    remove_unhealthy_borrowers = True  # Remove all initially liquidatable borrowers before sim proceeds - recommended
    plot_borrower_distributions = False  # Plot borrower distributions after generating them
    print_config = True  # Print configurable sim values at start

    # --------------------- Final HF distributions ---------------------
    plot_final_hf_dist = False
    hf_hist_bins = 100
    hf_hist_range = (0, 3)  # zoom on 0–3 for readability

    """Important: Toggle plot_sim_metrics on/off for PER SIMULATION info & time series; i.e., plots of bad debt, liquidations, etc., per sim run. 
    Also prints bad debt and liquidations to console every print_steps_size steps."""
    plot_sim_metrics = True
    print_steps_size = 100  # Print sim info every print_step_size intervals when plot_sim_metrics = True

    # --------------------- Assets ---------------------
    # NOTE: FIX SO ASSETS CAN BE CHOSEN/CHANGED JUST FROM HERE
    assets = ["WETH", "WBTC", "SOL", "USDC"]  # order matters

    ltv = {
        "WETH": 0.83,
        "WBTC": 0.76,
        "SOL": 0.70,
        "USDC": 1.0
    }

    # --------------------- Oracle ---------------------
    oracle_delay: int = 0  # units: step intervals; i.e., mins (unverified / WIP)
    use_hybrid_oracle: bool = True
    oracle_amm_weight: float = 0.3
    oracle_ema_alpha: float = 0.1
    print("ORACLE CONFIGURATION:")
    if use_hybrid_oracle:
        print(f"Oracle AMM weight: {oracle_amm_weight}")
        print(f"Oracle EMA alpha: {oracle_ema_alpha}\n")
    else:
        print("Basic Oracle\n")

    # --------------------- Liquidations ---------------------
    default_min_profit_bps = 100  # Default min profit when use_heterogeneous_liquidators = False
    print(f"Minimum profit bps: {default_min_profit_bps}\n")
    liquidation_threshold = 1.0
    close_factor = 0.5  # plan to make dynamic / per asset
    liquidation_bonus = 0.09  # plan to make dynamic / per asset
    max_liqs_per_block = 70

    # --------------------- AMM ---------------------
    use_rebalancing = True
    rebalance_rate = 0.03  # defend in assumptions
    rebalance_threshold = 0.03  # rebalance if imbalance > threshold %
    amm_liquidity_usd_per_pool = 2e8  # per side — tune higher for less slippage (1e8–1e9)

    print("AMM CONFIGURATION:")
    if use_rebalancing:
        print(f"AMM rebalance rate: {rebalance_rate}")
        print(f"Rebalance threshold: {rebalance_threshold}")
    else:
        print("Rebalancing off.")
    print(f"AMM liquidity per pool side: ${amm_liquidity_usd_per_pool:,.0f}")
    print("=" * 60 + "\n")

    # --------------------- Historical Price Data ---------------------
    # Choose a day:
    """YYYY-MM-DD : 
    2022-11-08 / 2022-11-11 / 2022-06-12 / 2022-05-07 / 2025-10-10 / 2025-10-09 / 2026-02-03 / 2026-02-05 / 2026-01-10
    """
    crisis_date = "2022-11-08"
    binance_symbols = {"WETH": "ETHUSDT", "WBTC": "BTCUSDT", "SOL": "SOLUSDT"}  # mapping

    # When config is instantiated, fetch prices and initialize balanced pools
    def __init__(self):
        np.random.seed(self.seed)
        #print(f"Seed used: {np.random.get_state()[1][0]}")  # shows actual seed state
        print(f"Fetching 1-minute prices for {self.crisis_date} from Binance API...")
        self.price_path = self.fetch_minute_prices(self.crisis_date)
        print(f"Fetched {len(self.price_path)} minutes of data.")

        # Set initial prices from first row of historical data
        self.initial_prices = self.price_path.iloc[0].to_dict()

        # Initialize AMM pools balanced at initial oracle prices
        self.amm_pools = {}
        print("Initializing AMM pools balanced at initial oracle prices:")
        for asset in self.assets:
            if asset == 'USDC':
                continue
            pool_key = f"{asset}_USDC"
            initial_price = self.initial_prices[asset]

            # Balanced: same USD value on both sides
            usdc_amount = self.amm_liquidity_usd_per_pool
            asset_amount = usdc_amount / initial_price if initial_price > 0 else 0

            self.amm_pools[pool_key] = {
                asset: asset_amount,
                'USDC': usdc_amount,
                'fee': 0.003
            }

            spot = usdc_amount / asset_amount if asset_amount > 0 else np.nan
            print(f"  {pool_key}: {asset}: {asset_amount:,.2f} | USDC: {usdc_amount:,.0f} | "
                  f"spot: ${spot:.2f} (oracle: ${initial_price:.2f})")

        print(f"Generating borrowers data...")
        self.borrowers = self.generate_borrowers()
        # Example: if you have np.random calls




    @staticmethod
    def fetch_minute_prices(date_str: str) -> pd.DataFrame:
        """
        Fetch 1-minute historical closing prices from Binance public API (no key needed).
        """
        prices_dict = {}

        for asset, symbol in Config.binance_symbols.items():
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": int(datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp() * 1000),
                "endTime": int(datetime.strptime(f"{date_str} 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() * 1000),
                "limit": 1000
            }

            all_klines = []
            while True:
                response = requests.get(url, params=params)
                data = response.json()
                if not data:
                    break
                all_klines.extend(data)
                if len(data) < 1000:
                    break
                params["startTime"] = data[-1][0] + 1

            if all_klines:
                df = pd.DataFrame(all_klines,
                                  columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
                                           'ignore'])
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close'] = df['close'].astype(float)
                prices_dict[asset] = df.set_index('timestamp')['close']
        price_df = pd.concat(prices_dict.values(), axis=1, keys=prices_dict.keys())
        price_df = price_df.ffill()
        price_df["USDC"] = 1.0
        return price_df[Config.assets]

    @staticmethod
    def derive_new_price_path(
            historical_prices: pd.DataFrame,
            scale_factor: float,
            max_drop_fraction: float = 0.85,
            noise_std: float = 0.015,
            front_load_fraction: float = 0.0,
            random_seed: int = None
    ) -> dict:
        """
        Scale historical prices and add Gaussian noise for MC stress testing.
        Supports front-loading for realistic crash shapes.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Scale base path
        scaled = historical_prices * scale_factor

        # Prevent extreme drops below realistic floor
        floor = historical_prices.min() * (1 - max_drop_fraction)
        scaled = scaled.clip(lower=floor, axis=1)

        # Multiplicative Gaussian noise
        noise = np.random.normal(0, noise_std, scaled.shape)
        shocked = scaled * (1 + noise)

        # Optional front-loading: steeper initial drop
        if front_load_fraction > 0:
            n = len(scaled)
            front_steps = int(n * front_load_fraction)
            front_factor = np.linspace(1.0, 0.7, front_steps)  # 30% steeper drop
            shocked.iloc[:front_steps] *= front_factor[:, np.newaxis]

        return shocked.to_dict(orient='list')

    def generate_borrowers(self) -> dict:
        """
        Generate realistic multi-asset borrower positions.
        """
        n_borrowers = self.num_borrowers
        assets = np.array(self.assets)
        n_assets = len(assets)
        collateral_assets = assets[:-1]
        n_collateral = len(collateral_assets)

        prices = np.array([self.initial_prices[a] for a in assets])
        ltv = np.array([self.ltv[a] for a in assets])

        collateral_prices = prices[:-1]
        collateral_ltv = ltv[:-1]

        # Leverage distribution
        leverage_mean = 0.55
        leverage_sigma = 0.25

        # Collateral in USD per asset
        collateral_usd_means = np.array([10.2, 11.1, 10.6])  # WETH, WBTC, SOL
        collateral_usd_sigmas = np.array([1.7, 1.5, 1.8])

        collateral_usd_per_asset = np.random.lognormal(
            mean=collateral_usd_means,
            sigma=collateral_usd_sigmas,
            size=(n_borrowers, n_collateral)
        )

        total_collateral_usd = np.sum(collateral_usd_per_asset, axis=1)

        leverage = np.random.lognormal(
            mean=leverage_mean,
            sigma=leverage_sigma,
            size=n_borrowers
        )
        leverage = np.clip(leverage, 1.1, 8.0)

        total_debt_usd = total_collateral_usd / leverage

        collateral_tokens = collateral_usd_per_asset / collateral_prices

        collateral = np.zeros((n_borrowers, n_assets))
        collateral[:, :n_collateral] = collateral_tokens

        debt = np.zeros((n_borrowers, n_assets))
        usdc_idx = np.where(assets == "USDC")[0][0]
        debt[:, usdc_idx] = total_debt_usd

        # --- Compute TRUE initial health factors ---
        weighted_collateral = collateral * prices * ltv
        total_weighted = np.sum(weighted_collateral, axis=1)
        total_debt = np.sum(debt * prices, axis=1)

        hf = np.where(
            total_debt > 0,
            total_weighted / total_debt,
            np.inf
        )

        initial_liquidatable = np.sum(hf < self.liquidation_threshold)

        if self.remove_unhealthy_borrowers:
            mask = hf >= self.liquidation_threshold + self.removal_buffer  # optional buffer to avoid early liqs (e.g. 0.01 )
            initial_removed = n_borrowers - np.sum(mask)

            print(
                f"Removed {initial_removed} / {n_borrowers} "
                f"initially liquidatable borrowers "
                f"({initial_removed / n_borrowers:.1%})"
            )
            print(f"Remaining borrowers: {np.sum(mask)}. All start healthy")
        else:
            mask = np.ones(n_borrowers, dtype=bool)

        borrower_data = {
            "collateral": collateral[mask],
            "debt": debt[mask],
            "health_factor": hf[mask].copy(),
            "liquidated": np.zeros(np.sum(mask), dtype=bool),
        }

        if self.plot_borrower_distributions:
            print("\n=== Borrower Generation Summary ===")
            print(f"Total borrowers: {np.sum(mask)}")
            print(f"Median total collateral: ${np.median(total_collateral_usd[mask]):,.0f}")
            print(f"Median total debt:       ${np.median(total_debt_usd[mask]):,.0f}")
            print(f"Median leverage:         {np.median(leverage[mask]):.2f}x")
            print(f"Median Health Factor:    {np.median(hf[mask]):.3f}")
            print(f"Mean Health Factor:      {np.mean(hf[mask]):.3f}")
            print(
                f"Initial liquidatable (<{self.liquidation_threshold}): "
                f"{initial_liquidatable} / {n_borrowers} "
                f"({initial_liquidatable / n_borrowers:.1%})"
            )
            print("====================================\n")

        return borrower_data


if __name__ == "__main__":
    config = Config()