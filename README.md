# DeFi Liquidation Cascade Simulator

**Monte Carlo simulation engine for modeling liquidation cascades and tail risk in DeFi lending protocols.**

Agent-based simulation of a lending protocol under price shocks.

## Features

- Multi-asset borrowers with collateral (WETH, WBTC, SOL) and USDC debt
- Profitability-gated liquidations with realistic AMM slippage
- Optional hybrid oracle (delayed API prices + live AMM spot weighting + EMA smoothing)
- Comprehensive risk metrics: pending/stuck debt, liquidation waves, health factor distributions, shortfall
- Monte Carlo stress testing with scaled price shocks and noise (WIP)

## Project Structure

defi-liquidation-simulator/
├── amm.py                    # AMM model with slippage
├── borrowers.py              # Borrower agent logic
├── config.py                 # Simulation parameters
├── liquidations.py           # Liquidation mechanics
├── metrics.py                # Risk metrics calculation
├── monte_carlo.py            # Main Monte Carlo engine
├── oracle.py                 # Price oracle (hybrid option)
├── sim.py                    # Core simulation runner
├── state.py                  # Simulation state management
├── analyze_monte_carlo.py    # Post-simulation analysis
├── price_path_test.py        # Price path testing utility
├── results/                  # Output directory for simulation results
└── README.md


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/04-Lpds/defi-liquidation-simulator.git
   cd defi-liquidation-simulator

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn requests

## Usage

Run a basic single simulation: python sim.py

Run full Monte Carlo stress tests: python monte_carlo.py

Analyze previous Monte Carlo results: python analyze_monte_carlo.py

Test price paths (useful for debugging): python price_path_test.py

## Configuration:

All simulation parameters (number of borrowers, collateral ratios, shock magnitude, oracle settings, etc.) are defined in config.py.
Edit config.py to customize your runs.

## Roadmap (WIP)

- Fix state issues in monte_carlo.py 
- Improve visualization & data for liquidation cascades
- Double-check and refine bad debt modeling

## Contributing

This is an early-stage research project focused on DeFi risk modeling.  
Feel free to open issues or submit pull requests!

## License

MIT License

---

Built for DeFi tail-risk research and liquidation modeling.

