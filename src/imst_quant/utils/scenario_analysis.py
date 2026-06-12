"""Scenario analysis for portfolio stress testing and what-if simulations.

Provides tools to define custom market scenarios (crashes, rate shocks, sector
rotations) and evaluate their impact on portfolio returns and risk metrics.

Functions:
    define_scenario: Create a named market scenario with asset shocks
    apply_scenario: Apply a scenario to portfolio holdings
    run_scenario_analysis: Run multiple scenarios and compare results
    historical_scenario_lookup: Retrieve shocks from historical crisis events
    scenario_sensitivity: Measure portfolio sensitivity to scenario parameters

Example:
    >>> from imst_quant.utils.scenario_analysis import run_scenario_analysis
    >>> scenarios = [
    ...     define_scenario("crash", {"SPY": -0.20, "TLT": 0.05}),
    ...     define_scenario("rate_hike", {"SPY": -0.05, "TLT": -0.10}),
    ... ]
    >>> weights = {"SPY": 0.6, "TLT": 0.4}
    >>> results = run_scenario_analysis(weights, scenarios)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Scenario:
    """A market scenario defining asset-level shocks.

    Attributes:
        name: Descriptive name for the scenario.
        shocks: Mapping of asset symbol to return shock (e.g., -0.20 = -20%).
        description: Optional longer description.
        probability: Estimated probability of occurrence (0 to 1).
    """

    name: str
    shocks: Dict[str, float]
    description: str = ""
    probability: float = 0.0


@dataclass
class ScenarioResult:
    """Result of applying a scenario to a portfolio.

    Attributes:
        scenario_name: Name of the applied scenario.
        portfolio_impact: Total portfolio return under the scenario.
        asset_contributions: Per-asset contribution to portfolio impact.
        worst_asset: Asset with the largest negative impact.
        best_asset: Asset with the largest positive impact.
        max_loss: Maximum single-asset loss in the portfolio.
    """

    scenario_name: str
    portfolio_impact: float
    asset_contributions: Dict[str, float]
    worst_asset: str
    best_asset: str
    max_loss: float


@dataclass
class ScenarioAnalysisReport:
    """Aggregated report across multiple scenarios.

    Attributes:
        results: List of individual scenario results.
        worst_case: The scenario with the largest portfolio loss.
        expected_loss: Probability-weighted expected loss across scenarios.
        portfolio_weights: The portfolio weights used.
    """

    results: List[ScenarioResult]
    worst_case: ScenarioResult
    expected_loss: float
    portfolio_weights: Dict[str, float]


# --- Historical scenario templates ---

HISTORICAL_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008_financial_crisis": {
        "SPY": -0.37,
        "QQQ": -0.42,
        "TLT": 0.20,
        "GLD": 0.05,
        "IWM": -0.34,
        "EEM": -0.49,
    },
    "2020_covid_crash": {
        "SPY": -0.34,
        "QQQ": -0.28,
        "TLT": 0.15,
        "GLD": 0.02,
        "IWM": -0.41,
        "EEM": -0.31,
    },
    "2022_rate_hike": {
        "SPY": -0.19,
        "QQQ": -0.33,
        "TLT": -0.31,
        "GLD": 0.00,
        "IWM": -0.21,
        "EEM": -0.22,
    },
    "dot_com_burst": {
        "SPY": -0.23,
        "QQQ": -0.36,
        "TLT": 0.12,
        "GLD": -0.06,
        "IWM": -0.04,
    },
    "flash_crash_2010": {
        "SPY": -0.10,
        "QQQ": -0.12,
        "TLT": 0.03,
        "GLD": 0.01,
    },
}


def define_scenario(
    name: str,
    shocks: Dict[str, float],
    description: str = "",
    probability: float = 0.0,
) -> Scenario:
    """Create a named market scenario.

    Args:
        name: Scenario name (e.g., "market_crash").
        shocks: Dict mapping asset symbols to return shocks.
        description: Optional description.
        probability: Estimated probability (0 to 1).

    Returns:
        Scenario instance.
    """
    return Scenario(
        name=name,
        shocks=shocks,
        description=description,
        probability=max(0.0, min(1.0, probability)),
    )


def apply_scenario(
    weights: Dict[str, float],
    scenario: Scenario,
) -> ScenarioResult:
    """Apply a single scenario to portfolio holdings.

    Calculates the portfolio-level impact by multiplying each asset's weight
    by its scenario shock.

    Args:
        weights: Portfolio weights mapping asset symbols to weight fractions.
        scenario: The scenario to apply.

    Returns:
        ScenarioResult with portfolio impact and per-asset contributions.
    """
    asset_contributions: Dict[str, float] = {}
    portfolio_impact = 0.0

    for asset, weight in weights.items():
        shock = scenario.shocks.get(asset, 0.0)
        contribution = weight * shock
        asset_contributions[asset] = contribution
        portfolio_impact += contribution

    worst_asset = min(asset_contributions, key=asset_contributions.get)  # type: ignore[arg-type]
    best_asset = max(asset_contributions, key=asset_contributions.get)  # type: ignore[arg-type]
    max_loss = min(asset_contributions.values())

    return ScenarioResult(
        scenario_name=scenario.name,
        portfolio_impact=portfolio_impact,
        asset_contributions=asset_contributions,
        worst_asset=worst_asset,
        best_asset=best_asset,
        max_loss=max_loss,
    )


def run_scenario_analysis(
    weights: Dict[str, float],
    scenarios: List[Scenario],
) -> ScenarioAnalysisReport:
    """Run multiple scenarios against a portfolio and produce a report.

    Args:
        weights: Portfolio weights mapping asset symbols to weight fractions.
        scenarios: List of scenarios to evaluate.

    Returns:
        ScenarioAnalysisReport with all results, worst case, and expected loss.
    """
    results = [apply_scenario(weights, s) for s in scenarios]

    worst_case = min(results, key=lambda r: r.portfolio_impact)

    # Expected loss: probability-weighted sum of impacts
    total_prob = sum(s.probability for s in scenarios)
    if total_prob > 0:
        expected_loss = sum(
            r.portfolio_impact * s.probability
            for r, s in zip(results, scenarios)
        ) / total_prob
    else:
        # If no probabilities set, use simple average
        expected_loss = np.mean([r.portfolio_impact for r in results]).item()

    return ScenarioAnalysisReport(
        results=results,
        worst_case=worst_case,
        expected_loss=expected_loss,
        portfolio_weights=weights,
    )


def historical_scenario_lookup(
    event_name: str,
) -> Optional[Scenario]:
    """Retrieve a pre-defined historical crisis scenario.

    Available scenarios:
    - 2008_financial_crisis
    - 2020_covid_crash
    - 2022_rate_hike
    - dot_com_burst
    - flash_crash_2010

    Args:
        event_name: Name of the historical event.

    Returns:
        Scenario if found, None otherwise.
    """
    shocks = HISTORICAL_SCENARIOS.get(event_name)
    if shocks is None:
        return None

    return Scenario(
        name=event_name,
        shocks=shocks,
        description=f"Historical scenario: {event_name.replace('_', ' ')}",
        probability=0.0,
    )


def list_historical_scenarios() -> List[str]:
    """List all available historical scenario names.

    Returns:
        List of scenario name strings.
    """
    return list(HISTORICAL_SCENARIOS.keys())


def scenario_sensitivity(
    weights: Dict[str, float],
    base_scenario: Scenario,
    shock_asset: str,
    shock_range: Tuple[float, float] = (-0.50, 0.10),
    steps: int = 20,
) -> List[Tuple[float, float]]:
    """Measure portfolio sensitivity to a single asset's scenario shock.

    Varies one asset's shock across a range while holding others constant,
    showing how portfolio impact changes.

    Args:
        weights: Portfolio weights.
        base_scenario: Starting scenario.
        shock_asset: Asset symbol to vary.
        shock_range: (min_shock, max_shock) range to sweep.
        steps: Number of steps in the sweep.

    Returns:
        List of (shock_value, portfolio_impact) tuples.
    """
    results: List[Tuple[float, float]] = []
    shock_values = np.linspace(shock_range[0], shock_range[1], steps)

    for shock_val in shock_values:
        modified_shocks = dict(base_scenario.shocks)
        modified_shocks[shock_asset] = float(shock_val)
        modified_scenario = Scenario(
            name=f"{base_scenario.name}_sensitivity",
            shocks=modified_shocks,
        )
        result = apply_scenario(weights, modified_scenario)
        results.append((float(shock_val), result.portfolio_impact))

    return results


def custom_stress_test(
    weights: Dict[str, float],
    base_shocks: Dict[str, float],
    correlation_multiplier: float = 1.5,
) -> ScenarioResult:
    """Run a stress test with amplified correlated moves.

    Takes base shocks and amplifies them by a correlation multiplier to
    simulate contagion effects where correlated assets move together
    more than expected.

    Args:
        weights: Portfolio weights.
        base_shocks: Initial shock values per asset.
        correlation_multiplier: Factor to amplify negative shocks (>1 = worse).

    Returns:
        ScenarioResult under the stressed conditions.
    """
    stressed_shocks = {}
    for asset, shock in base_shocks.items():
        if shock < 0:
            stressed_shocks[asset] = shock * correlation_multiplier
        else:
            # Positive shocks (hedges) may be reduced under stress
            stressed_shocks[asset] = shock / correlation_multiplier

    scenario = Scenario(
        name="custom_stress_test",
        shocks=stressed_shocks,
        description=f"Stress test with {correlation_multiplier}x correlation amplification",
    )

    return apply_scenario(weights, scenario)
