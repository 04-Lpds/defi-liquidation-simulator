# oracle.py

import numpy as np
import pandas as pd

class Oracle:
    def __init__(self, price_path: pd.DataFrame, delay_minutes: int = 0):
        """
        Oracle that provides asset prices with optional delay.

        Args:
            price_path: DataFrame with datetime index and asset columns
            delay_minutes: Number of steps to delay price feed (0 = real-time)
        """
        self.set_price_path(price_path)
        self.delay = max(0, delay_minutes)  # prevent negative delay
        self.current_step = 0

    def set_price_path(self, new_price_path: pd.DataFrame):
        """Update the price path (used for Monte Carlo or different scenarios)."""
        if not isinstance(new_price_path, pd.DataFrame):
            raise TypeError("price_path must be a pandas DataFrame")
        if new_price_path.empty:
            raise ValueError("price_path cannot be empty")

        self.price_path = new_price_path
        self.current_step = 0  # reset to beginning of new path

    def get_current_prices(self) -> np.ndarray:
        """Return current prices as numpy array (one value per asset)."""
        if self.current_step >= len(self.price_path):
            # Safety: if sim overruns path length, repeat last price
            row = self.price_path.iloc[-1]
        else:
            delayed_step = max(0, self.current_step - self.delay)
            row = self.price_path.iloc[delayed_step]

        return row.values.astype(float)  # ensure float dtype

    def advance_step(self):
        """Move to next time step."""
        self.current_step += 1


class HybridOracle(Oracle):
    def __init__(self, price_path: pd.DataFrame, amm_reserves: dict,
                 delay_minutes: int = 0,
                 amm_weight: float = 0.3,
                 ema_alpha: float = 0.1):
        """
        Hybrid Oracle Formula (Blended + EMA smoothing)

        For each asset (except USDC):

        1. Get (potentially delayed) API price: api_price_t = price_path.iloc[current_step - delay]

        2. Get current AMM spot price: amm_price_t = reserves['USDC'] / reserves[asset]

        3. Blend the two prices:
           blended_t = (1 - amm_weight) × api_price_t + amm_weight × amm_price_t

        4. Apply EMA smoothing recursively:
           EMA_t = ema_alpha × blended_t + (1 - ema_alpha) × EMA_{t-1}
           (with EMA_0 initialized to first blended value)

        5. Return EMA_t as the hybrid oracle price for that asset

        This creates a weighted feedback loop where AMM spot influences the oracle (reflexivity),
        but EMA smoothing adds lag to reduce sensitivity.

        Key parameters:
        - amm_weight (0.0 = pure API, 1.0 = pure AMM)
        - ema_alpha (smoothing factor, 0.0 = no smoothing, 1.0 = no memory, depends entirely on most recent input90)
        """
        super().__init__(price_path, delay_minutes)
        self.amm_reserves = amm_reserves  # live reference to state["amm_reserves"]
        self.amm_weight = max(0.0, min(1.0, amm_weight))
        self.ema_alpha = max(0.0, min(1.0, ema_alpha))

        # Cache last blended prices
        self.last_blended = None

    def get_current_prices(self) -> np.ndarray:
        api_prices = super().get_current_prices()  # get delayed API prices

        # If first call, initialize cache
        if self.last_blended is None:
            self.last_blended = api_prices.copy()

        hybrid_prices = np.zeros_like(api_prices)

        # Assume price_path columns match amm_reserves keys (e.g., 'WETH', 'USDC')
        for i, asset in enumerate(self.price_path.columns):
            if asset == 'USDC':
                hybrid_prices[i] = 1.0
                continue

            # Get current AMM spot price
            pool_key = f"{asset}_USDC"
            if pool_key not in self.amm_reserves:
                # Fallback: use last blended if pool missing
                hybrid_prices[i] = self.last_blended[i]
                continue

            pool = self.amm_reserves[pool_key]
            if pool.get(asset, 0) <= 0 or pool.get('USDC', 0) <= 0:
                hybrid_prices[i] = self.last_blended[i]
                continue

            # Constant-product spot price (USDC per asset)
            amm_price = pool['USDC'] / pool[asset]

            # Blend
            blended = (1 - self.amm_weight) * api_prices[i] + self.amm_weight * amm_price

            # EMA smoothing
            self.last_blended[i] = self.last_blended[i] * (1 - self.ema_alpha) + blended * self.ema_alpha
            hybrid_prices[i] = self.last_blended[i]

        return hybrid_prices