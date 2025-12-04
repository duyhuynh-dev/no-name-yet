"""
Circuit Breaker

Automated risk controls that halt trading when thresholds are breached.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open" # Limited trading (recovery)


@dataclass
class CircuitEvent:
    """Circuit breaker event."""
    timestamp: datetime
    event_type: str
    old_state: CircuitState
    new_state: CircuitState
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    # Drawdown triggers
    drawdown_open_threshold: float = 0.10      # 10% drawdown opens circuit
    drawdown_close_threshold: float = 0.05     # 5% drawdown to close circuit
    
    # Daily loss triggers
    daily_loss_open_threshold: float = 0.05    # 5% daily loss opens circuit
    daily_loss_close_threshold: float = 0.02   # 2% daily loss to close
    
    # Consecutive losses
    consecutive_loss_threshold: int = 5        # 5 consecutive losses
    
    # VaR breach
    var_breach_threshold: float = 1.5          # VaR breached by 1.5x
    
    # Recovery
    recovery_period_minutes: int = 30          # Minutes in half-open before closing
    max_trades_half_open: int = 3              # Max trades in half-open state
    
    # Cool-off
    cool_off_period_minutes: int = 60          # Minutes after emergency stop


class CircuitBreaker:
    """
    Trading Circuit Breaker.
    
    Automatically halts trading when risk thresholds are breached
    and manages the recovery process.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize Circuit Breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        
        # State
        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        self._trades_in_half_open = 0
        
        # Tracking
        self._consecutive_losses = 0
        self._last_trade_profitable = True
        
        # History
        self._events: List[CircuitEvent] = []
        
        # Callbacks
        self._on_state_change: List[Callable[[CircuitState, CircuitState, str], None]] = []
        
        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_at: Optional[datetime] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        if self._kill_switch_active:
            return False
        
        if self._state == CircuitState.OPEN:
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            return self._trades_in_half_open < self.config.max_trades_half_open
        
        return True
    
    def on_state_change(
        self,
        callback: Callable[[CircuitState, CircuitState, str], None]
    ) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)
    
    def check_drawdown(self, drawdown: float) -> bool:
        """
        Check drawdown and potentially trip circuit.
        
        Args:
            drawdown: Current drawdown (0.1 = 10%)
            
        Returns:
            True if trading should continue, False if halted
        """
        if self._state == CircuitState.CLOSED:
            if drawdown >= self.config.drawdown_open_threshold:
                self._trip_circuit(
                    "drawdown",
                    f"Drawdown {drawdown*100:.1f}% exceeded threshold"
                )
                return False
        
        elif self._state == CircuitState.OPEN:
            if drawdown < self.config.drawdown_close_threshold:
                self._attempt_recovery("Drawdown recovered")
        
        elif self._state == CircuitState.HALF_OPEN:
            if drawdown >= self.config.drawdown_open_threshold:
                self._trip_circuit(
                    "drawdown",
                    "Drawdown increased during recovery"
                )
                return False
        
        return self.is_trading_allowed
    
    def check_daily_loss(self, daily_return: float) -> bool:
        """
        Check daily loss and potentially trip circuit.
        
        Args:
            daily_return: Today's return (negative = loss)
            
        Returns:
            True if trading should continue, False if halted
        """
        if daily_return >= 0:
            return self.is_trading_allowed
        
        daily_loss = abs(daily_return)
        
        if self._state == CircuitState.CLOSED:
            if daily_loss >= self.config.daily_loss_open_threshold:
                self._trip_circuit(
                    "daily_loss",
                    f"Daily loss {daily_loss*100:.1f}% exceeded threshold"
                )
                return False
        
        elif self._state == CircuitState.OPEN:
            if daily_loss < self.config.daily_loss_close_threshold:
                self._attempt_recovery("Daily loss recovered")
        
        return self.is_trading_allowed
    
    def record_trade_result(self, pnl: float) -> bool:
        """
        Record trade result for consecutive loss tracking.
        
        Args:
            pnl: Trade P&L
            
        Returns:
            True if trading should continue, False if halted
        """
        if pnl < 0:
            self._consecutive_losses += 1
            self._last_trade_profitable = False
            
            if self._consecutive_losses >= self.config.consecutive_loss_threshold:
                self._trip_circuit(
                    "consecutive_losses",
                    f"{self._consecutive_losses} consecutive losing trades"
                )
                return False
        else:
            self._consecutive_losses = 0
            self._last_trade_profitable = True
        
        # Track trades in half-open state
        if self._state == CircuitState.HALF_OPEN:
            self._trades_in_half_open += 1
            
            if pnl >= 0:
                # Successful trade in recovery
                self._check_recovery_complete()
            else:
                # Failed trade in recovery
                self._trip_circuit(
                    "recovery_failure",
                    "Loss during recovery period"
                )
                return False
        
        return self.is_trading_allowed
    
    def check_var_breach(self, actual_loss: float, var: float) -> bool:
        """
        Check if actual loss breached VaR.
        
        Args:
            actual_loss: Actual loss experienced
            var: Expected VaR
            
        Returns:
            True if trading should continue, False if halted
        """
        if var <= 0:
            return self.is_trading_allowed
        
        breach_ratio = actual_loss / var
        
        if breach_ratio >= self.config.var_breach_threshold:
            self._trip_circuit(
                "var_breach",
                f"Loss exceeded VaR by {breach_ratio:.1f}x"
            )
            return False
        
        return self.is_trading_allowed
    
    def _trip_circuit(self, trigger: str, reason: str) -> None:
        """Trip the circuit breaker."""
        old_state = self._state
        new_state = CircuitState.OPEN
        
        self._state = new_state
        self._state_changed_at = datetime.now()
        self._trades_in_half_open = 0
        
        self._log_event(old_state, new_state, f"TRIPPED ({trigger}): {reason}")
        
        logger.warning(f"Circuit breaker TRIPPED: {reason}")
        
        # Notify callbacks
        for callback in self._on_state_change:
            try:
                callback(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def _attempt_recovery(self, reason: str) -> None:
        """Attempt to enter recovery (half-open) state."""
        old_state = self._state
        new_state = CircuitState.HALF_OPEN
        
        self._state = new_state
        self._state_changed_at = datetime.now()
        self._trades_in_half_open = 0
        
        self._log_event(old_state, new_state, f"RECOVERY: {reason}")
        
        logger.info(f"Circuit breaker entering recovery: {reason}")
    
    def _check_recovery_complete(self) -> None:
        """Check if recovery period is complete."""
        if self._state != CircuitState.HALF_OPEN:
            return
        
        time_in_recovery = datetime.now() - self._state_changed_at
        recovery_minutes = time_in_recovery.total_seconds() / 60
        
        if (recovery_minutes >= self.config.recovery_period_minutes and
            self._trades_in_half_open >= self.config.max_trades_half_open):
            self._close_circuit("Recovery period successful")
    
    def _close_circuit(self, reason: str) -> None:
        """Close the circuit (resume normal trading)."""
        old_state = self._state
        new_state = CircuitState.CLOSED
        
        self._state = new_state
        self._state_changed_at = datetime.now()
        self._consecutive_losses = 0
        
        self._log_event(old_state, new_state, f"CLOSED: {reason}")
        
        logger.info(f"Circuit breaker CLOSED: {reason}")
        
        for callback in self._on_state_change:
            try:
                callback(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """
        Activate kill switch - immediately halt all trading.
        
        This is a manual emergency stop that overrides all other states.
        """
        self._kill_switch_active = True
        self._kill_switch_at = datetime.now()
        
        old_state = self._state
        self._state = CircuitState.OPEN
        
        self._log_event(
            old_state, CircuitState.OPEN,
            f"KILL SWITCH: {reason}"
        )
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        for callback in self._on_state_change:
            try:
                callback(old_state, CircuitState.OPEN, f"KILL SWITCH: {reason}")
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def deactivate_kill_switch(self) -> bool:
        """
        Deactivate kill switch.
        
        Returns:
            True if deactivated, False if cool-off period not elapsed
        """
        if not self._kill_switch_active:
            return True
        
        if self._kill_switch_at:
            elapsed = datetime.now() - self._kill_switch_at
            if elapsed.total_seconds() < self.config.cool_off_period_minutes * 60:
                remaining = self.config.cool_off_period_minutes - elapsed.total_seconds() / 60
                logger.warning(f"Kill switch cool-off: {remaining:.1f} minutes remaining")
                return False
        
        self._kill_switch_active = False
        self._kill_switch_at = None
        self._state = CircuitState.HALF_OPEN  # Enter recovery mode
        self._state_changed_at = datetime.now()
        
        logger.info("Kill switch deactivated, entering recovery mode")
        
        return True
    
    def force_reset(self) -> None:
        """Force reset circuit breaker (use with caution)."""
        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        self._kill_switch_active = False
        self._kill_switch_at = None
        self._consecutive_losses = 0
        self._trades_in_half_open = 0
        
        logger.warning("Circuit breaker FORCE RESET")
    
    def _log_event(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
        reason: str,
    ) -> None:
        """Log a circuit breaker event."""
        event = CircuitEvent(
            timestamp=datetime.now(),
            event_type="state_change",
            old_state=old_state,
            new_state=new_state,
            reason=reason,
        )
        self._events.append(event)
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self._state.value,
            "is_trading_allowed": self.is_trading_allowed,
            "kill_switch_active": self._kill_switch_active,
            "state_changed_at": self._state_changed_at.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "trades_in_half_open": self._trades_in_half_open,
            "events_count": len(self._events),
        }
    
    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "old_state": e.old_state.value,
                "new_state": e.new_state.value,
                "reason": e.reason,
            }
            for e in self._events[-limit:]
        ]

