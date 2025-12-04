"""
Stress Testing Framework

Simulates historical and hypothetical market scenarios
to assess portfolio resilience.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd


class ScenarioType(Enum):
    """Types of stress scenarios."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    scenario_type: ScenarioType
    description: str
    
    # Market shocks (symbol -> return shock)
    shocks: Dict[str, float] = field(default_factory=dict)
    
    # Global market shock
    market_shock: float = 0.0
    
    # Volatility multiplier
    volatility_multiplier: float = 1.0
    
    # Correlation shock (increase correlations)
    correlation_shock: float = 0.0
    
    # Duration in days
    duration_days: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.scenario_type.value,
            "description": self.description,
            "shocks": self.shocks,
            "market_shock": self.market_shock,
            "volatility_multiplier": self.volatility_multiplier,
            "correlation_shock": self.correlation_shock,
            "duration_days": self.duration_days,
        }


@dataclass
class StressResult:
    """Stress test result."""
    scenario: StressScenario
    
    # P&L impact
    portfolio_pnl: float
    portfolio_pnl_pct: float
    
    # Position-level impacts
    position_pnls: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics after shock
    new_var: float = 0.0
    new_volatility: float = 0.0
    
    # Liquidity impact
    liquidity_haircut: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario.name,
            "portfolio_pnl": self.portfolio_pnl,
            "portfolio_pnl_pct": self.portfolio_pnl_pct,
            "position_pnls": self.position_pnls,
            "new_var": self.new_var,
            "new_volatility": self.new_volatility,
            "liquidity_haircut": self.liquidity_haircut,
            "timestamp": self.timestamp.isoformat(),
        }


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="September-October 2008 market crash",
        market_shock=-0.40,
        volatility_multiplier=3.0,
        correlation_shock=0.3,
        duration_days=30,
    ),
    "2020_covid_crash": StressScenario(
        name="2020 COVID Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="March 2020 pandemic market crash",
        market_shock=-0.35,
        volatility_multiplier=4.0,
        correlation_shock=0.4,
        duration_days=20,
    ),
    "2022_rate_shock": StressScenario(
        name="2022 Rate Shock",
        scenario_type=ScenarioType.HISTORICAL,
        description="2022 interest rate hiking cycle",
        market_shock=-0.25,
        shocks={"TLT": -0.30, "QQQ": -0.35},  # Bonds and tech hit harder
        volatility_multiplier=2.0,
        duration_days=60,
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="May 2010 flash crash scenario",
        market_shock=-0.10,
        volatility_multiplier=5.0,
        duration_days=1,
    ),
    "black_monday_1987": StressScenario(
        name="Black Monday 1987",
        scenario_type=ScenarioType.HISTORICAL,
        description="October 19, 1987 market crash",
        market_shock=-0.22,
        volatility_multiplier=4.0,
        correlation_shock=0.5,
        duration_days=1,
    ),
}

# Pre-defined hypothetical scenarios
HYPOTHETICAL_SCENARIOS = {
    "moderate_correction": StressScenario(
        name="Moderate Correction",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="10% market correction",
        market_shock=-0.10,
        volatility_multiplier=1.5,
        duration_days=5,
    ),
    "severe_bear_market": StressScenario(
        name="Severe Bear Market",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="50% market decline",
        market_shock=-0.50,
        volatility_multiplier=3.0,
        correlation_shock=0.4,
        duration_days=90,
    ),
    "volatility_spike": StressScenario(
        name="Volatility Spike",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="VIX spike without major price decline",
        market_shock=-0.05,
        volatility_multiplier=4.0,
        duration_days=3,
    ),
    "sector_rotation": StressScenario(
        name="Sector Rotation",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Tech selloff, value rally",
        shocks={"QQQ": -0.20, "XLF": 0.10, "XLE": 0.15},
        volatility_multiplier=1.5,
        duration_days=10,
    ),
    "liquidity_crisis": StressScenario(
        name="Liquidity Crisis",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Market liquidity dries up",
        market_shock=-0.15,
        volatility_multiplier=3.0,
        correlation_shock=0.5,
        duration_days=5,
    ),
}


class StressTester:
    """
    Portfolio Stress Tester.
    
    Runs various stress scenarios against a portfolio to assess risk.
    """
    
    def __init__(self):
        """Initialize Stress Tester."""
        self._scenarios: Dict[str, StressScenario] = {}
        self._results: List[StressResult] = []
        
        # Load pre-defined scenarios
        self._scenarios.update(HISTORICAL_SCENARIOS)
        self._scenarios.update(HYPOTHETICAL_SCENARIOS)
    
    def add_scenario(self, scenario: StressScenario) -> None:
        """Add a custom scenario."""
        key = scenario.name.lower().replace(" ", "_")
        self._scenarios[key] = scenario
    
    def get_scenario(self, name: str) -> Optional[StressScenario]:
        """Get a scenario by name."""
        return self._scenarios.get(name.lower().replace(" ", "_"))
    
    def list_scenarios(self) -> List[str]:
        """List available scenarios."""
        return list(self._scenarios.keys())
    
    def run_scenario(
        self,
        scenario_name: str,
        positions: Dict[str, float],  # symbol -> market value
        betas: Optional[Dict[str, float]] = None,  # symbol -> beta
    ) -> StressResult:
        """
        Run a stress scenario against a portfolio.
        
        Args:
            scenario_name: Name of scenario to run
            positions: Dictionary of symbol to market value
            betas: Optional dictionary of symbol to beta (default 1.0)
            
        Returns:
            StressResult with impact analysis
        """
        scenario = self.get_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        if betas is None:
            betas = {s: 1.0 for s in positions}
        
        portfolio_value = sum(positions.values())
        position_pnls = {}
        total_pnl = 0.0
        
        for symbol, value in positions.items():
            # Get specific shock or use market shock with beta
            if symbol in scenario.shocks:
                shock = scenario.shocks[symbol]
            else:
                beta = betas.get(symbol, 1.0)
                shock = scenario.market_shock * beta
            
            # Apply volatility multiplier to shock magnitude
            adjusted_shock = shock * np.sqrt(scenario.volatility_multiplier)
            
            # Calculate P&L
            pnl = value * adjusted_shock
            position_pnls[symbol] = pnl
            total_pnl += pnl
        
        # Calculate portfolio-level metrics
        pnl_pct = (total_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Estimate new volatility
        base_vol = 0.15  # Assume 15% base volatility
        new_vol = base_vol * scenario.volatility_multiplier
        
        # Estimate liquidity haircut
        liquidity_haircut = abs(scenario.market_shock) * 0.1  # 10% of shock as haircut
        
        result = StressResult(
            scenario=scenario,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=pnl_pct,
            position_pnls=position_pnls,
            new_volatility=new_vol,
            liquidity_haircut=liquidity_haircut,
        )
        
        self._results.append(result)
        return result
    
    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        betas: Optional[Dict[str, float]] = None,
        scenario_type: Optional[ScenarioType] = None,
    ) -> Dict[str, StressResult]:
        """
        Run all available scenarios.
        
        Args:
            positions: Dictionary of symbol to market value
            betas: Optional dictionary of symbol to beta
            scenario_type: Optional filter by scenario type
            
        Returns:
            Dictionary of scenario name to result
        """
        results = {}
        
        for name, scenario in self._scenarios.items():
            if scenario_type and scenario.scenario_type != scenario_type:
                continue
            
            try:
                result = self.run_scenario(name, positions, betas)
                results[name] = result
            except Exception as e:
                print(f"Error running scenario {name}: {e}")
        
        return results
    
    def run_monte_carlo_stress(
        self,
        positions: Dict[str, float],
        returns_df: pd.DataFrame,
        num_simulations: int = 1000,
        horizon_days: int = 10,
        confidence_level: float = 0.99,
    ) -> StressResult:
        """
        Run Monte Carlo stress test.
        
        Simulates many possible market scenarios based on historical data.
        
        Args:
            positions: Dictionary of symbol to market value
            returns_df: DataFrame of historical returns
            num_simulations: Number of simulations
            horizon_days: Simulation horizon in days
            confidence_level: Confidence level for worst case
            
        Returns:
            StressResult with worst-case scenario
        """
        portfolio_value = sum(positions.values())
        symbols = list(positions.keys())
        
        # Get available symbols from returns
        available_symbols = [s for s in symbols if s in returns_df.columns]
        if not available_symbols:
            raise ValueError("No matching symbols in returns data")
        
        # Calculate weights
        weights = np.array([
            positions.get(s, 0) / portfolio_value for s in available_symbols
        ])
        
        # Estimate return parameters
        returns = returns_df[available_symbols]
        mu = returns.mean().values * horizon_days
        cov = returns.cov().values * horizon_days
        
        # Run simulations
        np.random.seed(42)
        simulated_returns = np.random.multivariate_normal(mu, cov, num_simulations)
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # Find worst case at confidence level
        worst_case_idx = int((1 - confidence_level) * num_simulations)
        sorted_returns = np.sort(portfolio_returns)
        worst_return = sorted_returns[worst_case_idx]
        
        # Calculate P&L
        worst_pnl = portfolio_value * worst_return
        
        # Create scenario
        mc_scenario = StressScenario(
            name=f"Monte Carlo {confidence_level*100:.0f}% Worst Case",
            scenario_type=ScenarioType.MONTE_CARLO,
            description=f"{num_simulations} simulations, {horizon_days} day horizon",
            market_shock=worst_return,
            duration_days=horizon_days,
        )
        
        # Position-level worst case (approximation)
        position_pnls = {}
        worst_sim_idx = np.argmin(portfolio_returns)
        worst_sim_returns = simulated_returns[worst_sim_idx]
        
        for i, symbol in enumerate(available_symbols):
            position_pnls[symbol] = positions[symbol] * worst_sim_returns[i]
        
        result = StressResult(
            scenario=mc_scenario,
            portfolio_pnl=worst_pnl,
            portfolio_pnl_pct=worst_return * 100,
            position_pnls=position_pnls,
        )
        
        self._results.append(result)
        return result
    
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        shocks: Dict[str, float],
        market_shock: float = 0.0,
        volatility_multiplier: float = 1.0,
    ) -> StressScenario:
        """
        Create a custom stress scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            shocks: Dictionary of symbol-specific shocks
            market_shock: General market shock
            volatility_multiplier: Volatility multiplier
            
        Returns:
            Created StressScenario
        """
        scenario = StressScenario(
            name=name,
            scenario_type=ScenarioType.HYPOTHETICAL,
            description=description,
            shocks=shocks,
            market_shock=market_shock,
            volatility_multiplier=volatility_multiplier,
        )
        
        self.add_scenario(scenario)
        return scenario
    
    def get_worst_scenarios(
        self,
        n: int = 5,
    ) -> List[StressResult]:
        """Get the n worst scenario results by P&L."""
        sorted_results = sorted(self._results, key=lambda r: r.portfolio_pnl)
        return sorted_results[:n]
    
    def get_summary_report(
        self,
        positions: Dict[str, float],
        betas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive stress test report.
        
        Args:
            positions: Current portfolio positions
            betas: Optional betas
            
        Returns:
            Summary report dictionary
        """
        # Run all scenarios
        results = self.run_all_scenarios(positions, betas)
        
        portfolio_value = sum(positions.values())
        
        # Find worst case
        worst = min(results.values(), key=lambda r: r.portfolio_pnl)
        
        # Calculate statistics
        pnls = [r.portfolio_pnl for r in results.values()]
        
        return {
            "portfolio_value": portfolio_value,
            "num_scenarios": len(results),
            "worst_case": {
                "scenario": worst.scenario.name,
                "pnl": worst.portfolio_pnl,
                "pnl_pct": worst.portfolio_pnl_pct,
            },
            "average_pnl": np.mean(pnls),
            "median_pnl": np.median(pnls),
            "std_pnl": np.std(pnls),
            "scenarios": {
                name: {
                    "pnl": r.portfolio_pnl,
                    "pnl_pct": r.portfolio_pnl_pct,
                }
                for name, r in results.items()
            },
        }
    
    def get_results(self, limit: int = 100) -> List[StressResult]:
        """Get recent stress test results."""
        return self._results[-limit:]

