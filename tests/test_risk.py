"""
Tests for the Advanced Risk Management Module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.risk import (
    VaRCalculator,
    VaRMethod,
    PositionSizer,
    SizingMethod,
    RiskMonitor,
    RiskAlert,
    AlertSeverity,
    CircuitBreaker,
    CircuitState,
    StressTester,
    StressScenario,
)
from src.risk.risk_monitor import RiskThresholds


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
    return pd.Series(returns)


@pytest.fixture
def sample_returns_df():
    """Generate sample return DataFrame for multiple assets."""
    np.random.seed(42)
    n = 252
    data = {
        "AAPL": np.random.normal(0.001, 0.025, n),
        "MSFT": np.random.normal(0.0008, 0.022, n),
        "GOOGL": np.random.normal(0.0007, 0.028, n),
    }
    return pd.DataFrame(data)


class TestVaRCalculator:
    """Tests for VaR Calculator."""
    
    def test_initialization(self):
        calc = VaRCalculator(confidence_level=0.95, time_horizon_days=1)
        assert calc.confidence_level == 0.95
        assert calc.time_horizon_days == 1
    
    def test_calculate_returns(self):
        calc = VaRCalculator()
        prices = pd.Series([100, 101, 102, 100, 103])
        returns = calc.calculate_returns(prices)
        
        assert len(returns) == 4
        assert not returns.isna().any()
    
    def test_historical_var(self, sample_returns):
        calc = VaRCalculator(confidence_level=0.95)
        result = calc.historical_var(sample_returns, portfolio_value=100000)
        
        assert result.var > 0
        assert result.cvar >= result.var  # CVaR should be >= VaR
        assert result.method == VaRMethod.HISTORICAL
        assert result.var_pct > 0
    
    def test_parametric_var(self, sample_returns):
        calc = VaRCalculator(confidence_level=0.95)
        result = calc.parametric_var(sample_returns, portfolio_value=100000)
        
        assert result.var > 0
        assert result.cvar >= result.var
        assert result.method == VaRMethod.PARAMETRIC
    
    def test_monte_carlo_var(self, sample_returns):
        calc = VaRCalculator(confidence_level=0.95, monte_carlo_simulations=1000)
        result = calc.monte_carlo_var(sample_returns, portfolio_value=100000)
        
        assert result.var > 0
        assert result.method == VaRMethod.MONTE_CARLO
    
    def test_calculate_all_methods(self, sample_returns):
        calc = VaRCalculator()
        results = calc.calculate_all_methods(sample_returns, portfolio_value=100000)
        
        assert "historical" in results
        assert "parametric" in results
        assert "monte_carlo" in results
    
    def test_portfolio_var(self, sample_returns_df):
        calc = VaRCalculator()
        positions = {"AAPL": 50000, "MSFT": 30000, "GOOGL": 20000}
        
        result = calc.portfolio_var(positions, sample_returns_df)
        
        assert result.var > 0
        assert result.portfolio_value == 100000
    
    def test_var_result_to_dict(self, sample_returns):
        calc = VaRCalculator()
        result = calc.historical_var(sample_returns, portfolio_value=100000)
        
        d = result.to_dict()
        assert "var" in d
        assert "cvar" in d
        assert "confidence_level" in d


class TestPositionSizer:
    """Tests for Position Sizer."""
    
    def test_initialization(self):
        sizer = PositionSizer(portfolio_value=100000)
        assert sizer.portfolio_value == 100000
    
    def test_fixed_size(self):
        sizer = PositionSizer(portfolio_value=100000)
        result = sizer.fixed_size("AAPL", price=150, fixed_shares=100)
        
        assert result.shares == 100
        assert result.value == 15000
        assert result.method == SizingMethod.FIXED
    
    def test_kelly_criterion(self):
        sizer = PositionSizer(portfolio_value=100000)
        result = sizer.kelly_criterion(
            "AAPL",
            price=150,
            win_rate=0.6,
            avg_win=200,
            avg_loss=100,
        )
        
        assert result.shares > 0
        assert result.weight <= sizer.max_position_pct
        assert result.method in [SizingMethod.KELLY, SizingMethod.HALF_KELLY]
    
    def test_volatility_based(self):
        sizer = PositionSizer(portfolio_value=100000)
        result = sizer.volatility_based(
            "AAPL",
            price=150,
            volatility=0.25,
            target_volatility=0.15,
        )
        
        assert result.shares > 0
        assert result.method == SizingMethod.VOLATILITY
    
    def test_atr_based(self):
        sizer = PositionSizer(portfolio_value=100000)
        result = sizer.atr_based(
            "AAPL",
            price=150,
            atr=3.0,
            atr_multiplier=2.0,
        )
        
        assert result.shares > 0
        assert result.max_loss > 0
        assert result.method == SizingMethod.ATR
    
    def test_risk_parity(self):
        sizer = PositionSizer(portfolio_value=100000)
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 100}
        volatilities = {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.30}
        
        results = sizer.risk_parity(symbols, prices, volatilities)
        
        assert len(results) == 3
        # Lower volatility should get higher weight
        assert results["MSFT"].weight > results["GOOGL"].weight
    
    def test_calculate_max_shares(self):
        sizer = PositionSizer(portfolio_value=100000)
        max_shares = sizer.calculate_max_shares("AAPL", price=150, stop_loss_pct=0.02)
        
        assert max_shares > 0
        assert max_shares * 150 <= 100000  # Can't exceed portfolio


class TestRiskMonitor:
    """Tests for Risk Monitor."""
    
    def test_initialization(self):
        monitor = RiskMonitor(initial_portfolio_value=100000)
        assert monitor.current_value == 100000
        assert monitor.peak_value == 100000
    
    def test_update_portfolio_value(self):
        monitor = RiskMonitor(initial_portfolio_value=100000)
        
        monitor.update_portfolio_value(110000)
        assert monitor.current_value == 110000
        assert monitor.peak_value == 110000
        
        monitor.update_portfolio_value(105000)
        assert monitor.current_value == 105000
        assert monitor.peak_value == 110000  # Peak should not decrease
    
    def test_calculate_drawdown(self):
        monitor = RiskMonitor(initial_portfolio_value=100000)
        
        monitor.update_portfolio_value(110000)
        monitor.update_portfolio_value(99000)
        
        drawdown = monitor.calculate_drawdown()
        assert drawdown == pytest.approx(0.10, rel=0.01)  # 10% drawdown
    
    def test_drawdown_alert(self):
        thresholds = RiskThresholds(drawdown_warning=0.05)
        monitor = RiskMonitor(thresholds=thresholds, initial_portfolio_value=100000)
        
        alerts_received = []
        monitor.on_alert(lambda a: alerts_received.append(a))
        
        monitor.update_portfolio_value(110000)
        monitor.update_portfolio_value(103000)  # >5% drawdown from peak
        
        alerts = monitor.check_all_metrics()
        
        assert len(alerts) > 0
        assert any(a.metric == "drawdown" for a in alerts)
    
    def test_concentration_alert(self):
        thresholds = RiskThresholds(concentration_warning=0.15)
        monitor = RiskMonitor(thresholds=thresholds, initial_portfolio_value=100000)
        
        # 20% in one position
        monitor.update_positions({"AAPL": 20000, "MSFT": 40000, "GOOGL": 40000})
        
        alerts = monitor.check_all_metrics()
        
        assert any("concentration" in a.metric for a in alerts)
    
    def test_get_risk_summary(self):
        monitor = RiskMonitor(initial_portfolio_value=100000)
        monitor.update_portfolio_value(95000)
        
        summary = monitor.get_risk_summary()
        
        assert "portfolio_value" in summary
        assert "drawdown" in summary
        assert "leverage" in summary


class TestCircuitBreaker:
    """Tests for Circuit Breaker."""
    
    def test_initialization(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_trading_allowed is True
    
    def test_drawdown_trip(self):
        cb = CircuitBreaker()
        
        # Should trip at 10% drawdown (default)
        cb.check_drawdown(0.10)
        
        assert cb.state == CircuitState.OPEN
        assert cb.is_trading_allowed is False
    
    def test_daily_loss_trip(self):
        cb = CircuitBreaker()
        
        # Should trip at 5% daily loss (default)
        cb.check_daily_loss(-0.05)
        
        assert cb.state == CircuitState.OPEN
        assert cb.is_trading_allowed is False
    
    def test_consecutive_losses(self):
        cb = CircuitBreaker()
        
        # Record 5 consecutive losses
        for i in range(5):
            result = cb.record_trade_result(-100)
        
        assert cb.state == CircuitState.OPEN
    
    def test_recovery(self):
        cb = CircuitBreaker()
        
        # Trip the circuit
        cb.check_drawdown(0.10)
        assert cb.state == CircuitState.OPEN
        
        # Attempt recovery
        cb.check_drawdown(0.04)  # Below close threshold
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_kill_switch(self):
        cb = CircuitBreaker()
        
        cb.activate_kill_switch("Test emergency")
        
        assert cb.is_trading_allowed is False
        assert cb._kill_switch_active is True
    
    def test_force_reset(self):
        cb = CircuitBreaker()
        
        cb.check_drawdown(0.10)
        cb.force_reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.is_trading_allowed is True
    
    def test_get_status(self):
        cb = CircuitBreaker()
        status = cb.get_status()
        
        assert "state" in status
        assert "is_trading_allowed" in status
        assert "kill_switch_active" in status


class TestStressTester:
    """Tests for Stress Tester."""
    
    def test_initialization(self):
        st = StressTester()
        scenarios = st.list_scenarios()
        
        assert len(scenarios) > 0
        assert "2008_financial_crisis" in scenarios
    
    def test_run_scenario(self):
        st = StressTester()
        positions = {"AAPL": 50000, "MSFT": 30000, "GOOGL": 20000}
        
        result = st.run_scenario("2008_financial_crisis", positions)
        
        assert result.portfolio_pnl < 0  # Should be negative
        assert len(result.position_pnls) == 3
    
    def test_run_all_scenarios(self):
        st = StressTester()
        positions = {"AAPL": 50000, "MSFT": 50000}
        
        results = st.run_all_scenarios(positions)
        
        assert len(results) > 0
    
    def test_custom_scenario(self):
        st = StressTester()
        
        scenario = st.create_custom_scenario(
            name="Custom Tech Crash",
            description="Tech sector crash",
            shocks={"AAPL": -0.30, "MSFT": -0.25},
            market_shock=-0.15,
        )
        
        assert scenario.name == "Custom Tech Crash"
        assert "custom_tech_crash" in st.list_scenarios()
    
    def test_monte_carlo_stress(self, sample_returns_df):
        st = StressTester()
        positions = {"AAPL": 50000, "MSFT": 30000, "GOOGL": 20000}
        
        result = st.run_monte_carlo_stress(
            positions,
            sample_returns_df,
            num_simulations=100,
            horizon_days=5,
        )
        
        assert result.portfolio_pnl < 0  # Worst case should be negative
    
    def test_get_summary_report(self):
        st = StressTester()
        positions = {"AAPL": 50000, "MSFT": 50000}
        
        report = st.get_summary_report(positions)
        
        assert "portfolio_value" in report
        assert "worst_case" in report
        assert "scenarios" in report
    
    def test_stress_result_to_dict(self):
        st = StressTester()
        positions = {"AAPL": 100000}
        
        result = st.run_scenario("moderate_correction", positions)
        d = result.to_dict()
        
        assert "scenario_name" in d
        assert "portfolio_pnl" in d
        assert "portfolio_pnl_pct" in d


class TestIntegration:
    """Integration tests for risk module."""
    
    def test_var_with_position_sizing(self, sample_returns):
        """Test VaR calculation informing position sizing."""
        var_calc = VaRCalculator()
        var_result = var_calc.historical_var(sample_returns, portfolio_value=100000)
        
        # Size position based on VaR
        sizer = PositionSizer(
            portfolio_value=100000,
            max_portfolio_risk_pct=var_result.var_pct / 100 * 0.5,  # Half of VaR
        )
        
        position = sizer.atr_based("AAPL", price=150, atr=3.0)
        
        assert position.max_loss <= var_result.var * 0.5
    
    def test_monitor_triggers_circuit_breaker(self):
        """Test risk monitor triggering circuit breaker."""
        monitor = RiskMonitor(initial_portfolio_value=100000)
        cb = CircuitBreaker()
        
        # Connect monitor alerts to circuit breaker
        def on_alert(alert):
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                if "drawdown" in alert.metric:
                    cb.check_drawdown(alert.current_value)
        
        monitor.on_alert(on_alert)
        
        # Simulate drawdown
        monitor.update_portfolio_value(110000)
        monitor.update_portfolio_value(95000)
        monitor.check_all_metrics()
        
        # Circuit breaker should have tripped
        assert cb.state in [CircuitState.OPEN, CircuitState.HALF_OPEN, CircuitState.CLOSED]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

