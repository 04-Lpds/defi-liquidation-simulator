# Agent-Based DeFi Liquidation Risk Simulator

A modular quantitative research framework for studying liquidation cascades, liquidity dynamics, oracle design, and systemic risk in decentralized lending protocols.

---

## Overview

This project provides a configurable simulation environment for investigating how market conditions, liquidity, participant behavior, and protocol design interact to produce liquidation cascades and bad debt.

The framework combines historical market replay, stochastic simulation, and agent-based modeling to analyze protocol resilience under stressed market conditions.

Current research includes:

- Liquidation cascade dynamics
- Oracle architecture design
- AMM liquidity and price impact
- Tail-risk analysis (VaR / Expected Shortfall)
- Sensitivity analysis through parameter sweeps

---

## Key Features

- Agent-based simulation with 12,500+ heterogeneous borrower positions
- Historical replay using 1-minute cryptocurrency market data
- Monte Carlo stress testing with stochastic price generation
- Configurable lending protocol parameters
- Partial liquidation mechanics
- Constant-product AMM with price impact and configurable liquidity
- Configurable oracle architectures (spot, delayed, EMA smoothing, AMM-weighted)
- Parameter sweep infrastructure for sensitivity analysis
- Comprehensive visualization and analytics pipeline

---

## Architecture

The simulator is organized into modular components:

### Market Layer

- Historical market replay
- Stochastic price generation
- AMM liquidity model
- Oracle system

### Protocol Layer

- Lending parameters
- Health factor calculation
- Liquidation engine
- Bad debt accounting

### Agent Layer

- Heterogeneous borrower populations
- Liquidation processing
- Market interaction

### Analytics Layer

- Monte Carlo experiments
- Parameter sweeps
- VaR / Expected Shortfall
- Visualization and reporting

---

## Research Questions

Examples of questions explored with this framework include:

- How do different oracle designs influence liquidation cascades?
- How does liquidity depth affect systemic risk?
- Which protocol parameters minimize bad debt?
- How sensitive are outcomes to borrower distributions?
- Under what market conditions do cascading liquidations emerge?

---

## Research Outputs

The framework produces quantitative analysis including:

- Liquidation volumes
- Bad debt formation
- Health factor distributions
- Tail-risk metrics
- Cascade dynamics
- Oracle comparisons
- Parameter sensitivity analysis

---

## Current Status

The core simulation framework, liquidation engine, AMM interactions, oracle models, and historical replay infrastructure are implemented.

Current research focuses on:

- Monte Carlo validation and calibration
- Additional liquidator behavior models
- Historical validation against on-chain events
- Expanded stress-testing capabilities

---

## Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Repository Structure

```
amm.py                 # Constant-product AMM and price impact model
borrowers.py           # Borrower generation and position management
config.py              # Global simulation configuration
liquidations.py        # Liquidation engine and execution logic
metrics.py             # Metrics, analytics, and visualization
monte_carlo.py         # Monte Carlo stress-testing framework
oracle.py              # Oracle models (spot, delay, EMA, hybrid)
param_sweep.py         # Parameter sweep and sensitivity analysis
sim.py                 # Main simulation loop
state.py               # Global simulation state management
```

Supporting files:

```
README.md
.gitignore

```

The codebase is designed as a modular research framework, allowing individual market components, protocol parameters, and simulation assumptions to be modified independently for systematic experimentation.

---

## Related Research

Technical write-ups based on this simulator are available on Substack:

- The Effects of Oracle Design on Liquidation Cascade Dynamics
- How AMM Reflexivity Changes Liquidation Cascade Dynamics