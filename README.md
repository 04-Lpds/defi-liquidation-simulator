# Agent-Based DeFi Liquidation Risk Simulator

A quantitative simulation framework for studying liquidation cascades, liquidity dynamics, oracle design, and systemic risk in decentralized lending markets.

## Overview

This project models the interaction between borrowers, liquidators, automated market makers, and oracle mechanisms during stressed market conditions.

The framework is designed to analyze how market structure, liquidity conditions, and protocol parameters influence liquidation events and bad debt formation.

## Key Features

- 12,500+ heterogeneous borrower agents
- Historical market replay using crypto price data
- Monte Carlo stress testing
- Agent-based liquidation modeling
- AMM liquidity and price impact simulation
- Configurable oracle architectures
- Tail risk analysis using VaR and Expected Shortfall

## Architecture

Borrowers
   |
   v
Health Factor Monitoring
   |
   v
Liquidation Engine
   |
   +----> AMM Execution
   |
   +----> Price Impact
   |
   v
Protocol Risk Metrics

## Research Questions

- How do oracle designs influence liquidation cascades?
- How does liquidity depth affect bad debt formation?
- How do liquidation parameters influence protocol resilience?
- What market conditions create systemic failure?

## Research Outputs

The simulator generates analysis of:

- Liquidation volumes
- Bad debt formation
- Health factor distributions
- Tail risk metrics
- Liquidity stress scenarios

## Current Status

This project is an active research prototype. The core simulation framework, liquidation mechanics, AMM interactions, and oracle models are implemented.

Current development areas include:
- Monte Carlo validation and calibration
- Historical event comparison
- More realistic liquidator behavior
- Additional market participant modeling

## Technical Implementation

Built in Python using modular components for quantitative experimentation.

Technologies:
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn