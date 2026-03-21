# DeFi Liquidation Cascade Simulator

Agent-based simulation of a lending protocol under price shocks.

## Features
- Multi-asset borrowers (WETH/WBTC/SOL collateral, USDC debt)
- Profitability-gated liquidations with realistic AMM slippage
- Optional hybrid oracle (delayed API prices + live AMM spot weighting + EMA smoothing)
- Tracks key risk metrics: pending/stuck debt, liquidation waves, health factor distributions, shortfall
- Monte Carlo stress testing with scaled price shocks and noise (WIP)
