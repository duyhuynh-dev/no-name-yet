"""
Risk Monitor

Real-time monitoring of portfolio risk metrics with alerting.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert."""
    alert_id: str
    severity: AlertSeverity
    metric: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "metric": self.metric,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class RiskThresholds:
    """Risk monitoring thresholds."""
    # Drawdown thresholds
    drawdown_warning: float = 0.05      # 5% warning
    drawdown_critical: float = 0.10     # 10% critical
    drawdown_emergency: float = 0.15    # 15% emergency
    
    # VaR thresholds (as % of portfolio)
    var_warning: float = 0.03           # 3% VaR warning
    var_critical: float = 0.05          # 5% VaR critical
    
    # P&L thresholds
    daily_loss_warning: float = 0.02    # 2% daily loss warning
    daily_loss_critical: float = 0.03   # 3% daily loss critical
    daily_loss_emergency: float = 0.05  # 5% daily loss emergency
    
    # Position concentration
    concentration_warning: float = 0.15  # 15% single position warning
    concentration_critical: float = 0.25 # 25% single position critical
    
    # Volatility
    volatility_warning: float = 0.30     # 30% annualized vol warning
    volatility_critical: float = 0.50    # 50% annualized vol critical
    
    # Leverage
    leverage_warning: float = 1.5        # 1.5x leverage warning
    leverage_critical: float = 2.0       # 2x leverage critical


class RiskMonitor:
    """
    Portfolio Risk Monitor.
    
    Continuously monitors risk metrics and generates alerts
    when thresholds are breached.
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        initial_portfolio_value: float = 100000.0,
    ):
        """
        Initialize Risk Monitor.
        
        Args:
            thresholds: Risk thresholds for alerting
            initial_portfolio_value: Starting portfolio value
        """
        self.thresholds = thresholds or RiskThresholds()
        self.initial_value = initial_portfolio_value
        self.peak_value = initial_portfolio_value
        self.current_value = initial_portfolio_value
        
        # Tracking
        self._daily_pnl = 0.0
        self._daily_start_value = initial_portfolio_value
        self._position_values: Dict[str, float] = {}
        self._portfolio_returns: List[float] = []
        
        # Alerts
        self._alerts: List[RiskAlert] = []
        self._active_alerts: Dict[str, RiskAlert] = {}
        self._alert_counter = 0
        
        # Callbacks
        self._alert_callbacks: List[Callable[[RiskAlert], None]] = []
    
    def on_alert(self, callback: Callable[[RiskAlert], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)
    
    def update_portfolio_value(self, value: float) -> None:
        """
        Update current portfolio value.
        
        Args:
            value: Current portfolio value
        """
        # Update peak value
        if value > self.peak_value:
            self.peak_value = value
        
        # Calculate daily P&L
        self._daily_pnl = value - self._daily_start_value
        
        # Store return
        if self.current_value > 0:
            ret = (value - self.current_value) / self.current_value
            self._portfolio_returns.append(ret)
            # Keep last 252 days
            if len(self._portfolio_returns) > 252:
                self._portfolio_returns = self._portfolio_returns[-252:]
        
        self.current_value = value
    
    def update_positions(self, positions: Dict[str, float]) -> None:
        """
        Update position values.
        
        Args:
            positions: Dictionary of symbol to market value
        """
        self._position_values = positions.copy()
    
    def reset_daily_stats(self) -> None:
        """Reset daily tracking (call at start of trading day)."""
        self._daily_pnl = 0.0
        self._daily_start_value = self.current_value
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.current_value) / self.peak_value
    
    def calculate_daily_return(self) -> float:
        """Calculate today's return."""
        if self._daily_start_value <= 0:
            return 0.0
        return self._daily_pnl / self._daily_start_value
    
    def calculate_volatility(self, annualize: bool = True) -> float:
        """Calculate portfolio volatility from returns."""
        if len(self._portfolio_returns) < 2:
            return 0.0
        
        vol = np.std(self._portfolio_returns)
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def calculate_concentration(self) -> Dict[str, float]:
        """Calculate position concentration."""
        if not self._position_values or self.current_value <= 0:
            return {}
        
        return {
            symbol: abs(value) / self.current_value
            for symbol, value in self._position_values.items()
        }
    
    def calculate_leverage(self) -> float:
        """Calculate portfolio leverage."""
        if self.current_value <= 0:
            return 0.0
        
        total_exposure = sum(abs(v) for v in self._position_values.values())
        return total_exposure / self.current_value
    
    def check_all_metrics(self) -> List[RiskAlert]:
        """
        Check all risk metrics and generate alerts.
        
        Returns:
            List of new alerts generated
        """
        new_alerts = []
        
        # Check drawdown
        drawdown = self.calculate_drawdown()
        alert = self._check_drawdown(drawdown)
        if alert:
            new_alerts.append(alert)
        
        # Check daily P&L
        daily_return = self.calculate_daily_return()
        alert = self._check_daily_pnl(daily_return)
        if alert:
            new_alerts.append(alert)
        
        # Check volatility
        volatility = self.calculate_volatility()
        alert = self._check_volatility(volatility)
        if alert:
            new_alerts.append(alert)
        
        # Check concentration
        concentrations = self.calculate_concentration()
        for symbol, conc in concentrations.items():
            alert = self._check_concentration(symbol, conc)
            if alert:
                new_alerts.append(alert)
        
        # Check leverage
        leverage = self.calculate_leverage()
        alert = self._check_leverage(leverage)
        if alert:
            new_alerts.append(alert)
        
        return new_alerts
    
    def _check_drawdown(self, drawdown: float) -> Optional[RiskAlert]:
        """Check drawdown against thresholds."""
        metric = "drawdown"
        
        if drawdown >= self.thresholds.drawdown_emergency:
            return self._create_alert(
                AlertSeverity.EMERGENCY, metric,
                f"EMERGENCY: Drawdown at {drawdown*100:.1f}%",
                drawdown, self.thresholds.drawdown_emergency
            )
        elif drawdown >= self.thresholds.drawdown_critical:
            return self._create_alert(
                AlertSeverity.CRITICAL, metric,
                f"Critical drawdown: {drawdown*100:.1f}%",
                drawdown, self.thresholds.drawdown_critical
            )
        elif drawdown >= self.thresholds.drawdown_warning:
            return self._create_alert(
                AlertSeverity.WARNING, metric,
                f"Drawdown warning: {drawdown*100:.1f}%",
                drawdown, self.thresholds.drawdown_warning
            )
        else:
            self._clear_alert(metric)
            return None
    
    def _check_daily_pnl(self, daily_return: float) -> Optional[RiskAlert]:
        """Check daily P&L against thresholds."""
        metric = "daily_pnl"
        
        # Only check losses (negative returns)
        if daily_return >= 0:
            self._clear_alert(metric)
            return None
        
        loss = abs(daily_return)
        
        if loss >= self.thresholds.daily_loss_emergency:
            return self._create_alert(
                AlertSeverity.EMERGENCY, metric,
                f"EMERGENCY: Daily loss {loss*100:.1f}%",
                loss, self.thresholds.daily_loss_emergency
            )
        elif loss >= self.thresholds.daily_loss_critical:
            return self._create_alert(
                AlertSeverity.CRITICAL, metric,
                f"Critical daily loss: {loss*100:.1f}%",
                loss, self.thresholds.daily_loss_critical
            )
        elif loss >= self.thresholds.daily_loss_warning:
            return self._create_alert(
                AlertSeverity.WARNING, metric,
                f"Daily loss warning: {loss*100:.1f}%",
                loss, self.thresholds.daily_loss_warning
            )
        else:
            self._clear_alert(metric)
            return None
    
    def _check_volatility(self, volatility: float) -> Optional[RiskAlert]:
        """Check volatility against thresholds."""
        metric = "volatility"
        
        if volatility >= self.thresholds.volatility_critical:
            return self._create_alert(
                AlertSeverity.CRITICAL, metric,
                f"Critical volatility: {volatility*100:.1f}%",
                volatility, self.thresholds.volatility_critical
            )
        elif volatility >= self.thresholds.volatility_warning:
            return self._create_alert(
                AlertSeverity.WARNING, metric,
                f"Volatility warning: {volatility*100:.1f}%",
                volatility, self.thresholds.volatility_warning
            )
        else:
            self._clear_alert(metric)
            return None
    
    def _check_concentration(self, symbol: str, concentration: float) -> Optional[RiskAlert]:
        """Check position concentration."""
        metric = f"concentration_{symbol}"
        
        if concentration >= self.thresholds.concentration_critical:
            return self._create_alert(
                AlertSeverity.CRITICAL, metric,
                f"Critical concentration in {symbol}: {concentration*100:.1f}%",
                concentration, self.thresholds.concentration_critical
            )
        elif concentration >= self.thresholds.concentration_warning:
            return self._create_alert(
                AlertSeverity.WARNING, metric,
                f"Concentration warning for {symbol}: {concentration*100:.1f}%",
                concentration, self.thresholds.concentration_warning
            )
        else:
            self._clear_alert(metric)
            return None
    
    def _check_leverage(self, leverage: float) -> Optional[RiskAlert]:
        """Check leverage against thresholds."""
        metric = "leverage"
        
        if leverage >= self.thresholds.leverage_critical:
            return self._create_alert(
                AlertSeverity.CRITICAL, metric,
                f"Critical leverage: {leverage:.2f}x",
                leverage, self.thresholds.leverage_critical
            )
        elif leverage >= self.thresholds.leverage_warning:
            return self._create_alert(
                AlertSeverity.WARNING, metric,
                f"Leverage warning: {leverage:.2f}x",
                leverage, self.thresholds.leverage_warning
            )
        else:
            self._clear_alert(metric)
            return None
    
    def _create_alert(
        self,
        severity: AlertSeverity,
        metric: str,
        message: str,
        value: float,
        threshold: float,
    ) -> Optional[RiskAlert]:
        """Create a new alert if not already active."""
        # Check if alert already active
        if metric in self._active_alerts:
            existing = self._active_alerts[metric]
            # Only create new alert if severity increased
            if severity.value <= existing.severity.value:
                return None
        
        self._alert_counter += 1
        alert = RiskAlert(
            alert_id=f"alert_{self._alert_counter}",
            severity=severity,
            metric=metric,
            message=message,
            current_value=value,
            threshold=threshold,
        )
        
        self._alerts.append(alert)
        self._active_alerts[metric] = alert
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Risk Alert: {message}")
        
        return alert
    
    def _clear_alert(self, metric: str) -> None:
        """Clear an active alert for a metric."""
        self._active_alerts.pop(metric, None)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active (unacknowledged) alerts."""
        return [a for a in self._active_alerts.values() if not a.acknowledged]
    
    def get_all_alerts(self, limit: int = 100) -> List[RiskAlert]:
        """Get all alerts."""
        return self._alerts[-limit:]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk metrics."""
        return {
            "portfolio_value": self.current_value,
            "peak_value": self.peak_value,
            "drawdown": self.calculate_drawdown(),
            "daily_pnl": self._daily_pnl,
            "daily_return": self.calculate_daily_return(),
            "volatility": self.calculate_volatility(),
            "leverage": self.calculate_leverage(),
            "max_concentration": max(self.calculate_concentration().values()) if self._position_values else 0,
            "active_alerts": len(self._active_alerts),
            "total_alerts": len(self._alerts),
        }

