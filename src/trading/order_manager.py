"""
Order Management System (OMS)

Manages order lifecycle, risk checks, and order state tracking.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

from .base import (
    BaseExchange,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_order_value: float = 100000.0
    max_position_value: float = 500000.0
    max_daily_loss: float = 10000.0
    max_orders_per_minute: int = 60
    max_position_pct: float = 0.2  # Max 20% of portfolio in one position
    allowed_symbols: Optional[List[str]] = None


class OrderManager:
    """
    Order Management System.
    
    Features:
    - Pre-trade risk checks
    - Order throttling
    - Order state management
    - Order modification and cancellation
    - Audit trail
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize the Order Manager.
        
        Args:
            exchange: Exchange to execute orders on
            risk_limits: Risk limits configuration
        """
        self.exchange = exchange
        self.risk_limits = risk_limits or RiskLimits()
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._pending_orders: Dict[str, Order] = {}
        self._order_history: List[Dict[str, Any]] = []
        
        # Risk tracking
        self._daily_pnl = 0.0
        self._orders_this_minute = 0
        self._last_minute = datetime.now().minute
        
        # Callbacks
        self._on_order_fill: List[Callable[[Order], None]] = []
        self._on_order_reject: List[Callable[[Order, str], None]] = []
    
    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order fills."""
        self._on_order_fill.append(callback)
    
    def on_reject(self, callback: Callable[[Order, str], None]) -> None:
        """Register callback for order rejections."""
        self._on_order_reject.append(callback)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_minute = datetime.now().minute
        if current_minute != self._last_minute:
            self._orders_this_minute = 0
            self._last_minute = current_minute
        
        if self._orders_this_minute >= self.risk_limits.max_orders_per_minute:
            return False
        
        return True
    
    async def _check_risk(self, order: Order) -> tuple[bool, str]:
        """
        Perform pre-trade risk checks.
        
        Returns:
            Tuple of (passed, reason)
        """
        # Check rate limit
        if not self._check_rate_limit():
            return False, "Rate limit exceeded"
        
        # Check allowed symbols
        if self.risk_limits.allowed_symbols:
            if order.symbol not in self.risk_limits.allowed_symbols:
                return False, f"Symbol {order.symbol} not allowed"
        
        # Get current quote for value calculation
        try:
            quote = await self.exchange.get_quote(order.symbol)
            price = quote.get("last", 0) or order.limit_price or 0
        except Exception:
            price = order.limit_price or 0
        
        if price == 0:
            return False, "Unable to determine order price"
        
        order_value = order.quantity * price
        
        # Check max order value
        if order_value > self.risk_limits.max_order_value:
            return False, f"Order value ${order_value:.2f} exceeds limit ${self.risk_limits.max_order_value:.2f}"
        
        # Check daily loss limit
        if self._daily_pnl < -self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded (${abs(self._daily_pnl):.2f})"
        
        # Check position size
        try:
            account = await self.exchange.get_account()
            max_position = account.portfolio_value * self.risk_limits.max_position_pct
            
            current_position = await self.exchange.get_position(order.symbol)
            current_value = current_position.market_value if current_position else 0
            
            if order.side == OrderSide.BUY:
                new_value = current_value + order_value
            else:
                new_value = current_value - order_value
            
            if abs(new_value) > max_position:
                return False, f"Position would exceed {self.risk_limits.max_position_pct*100:.0f}% limit"
        except Exception as e:
            logger.warning(f"Could not check position limits: {e}")
        
        return True, "OK"
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        skip_risk_check: bool = False,
    ) -> Order:
        """
        Submit a new order with risk checks.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of shares/units
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order duration
            skip_risk_check: Skip risk checks (use carefully)
            
        Returns:
            Submitted order
        """
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )
        
        # Risk check
        if not skip_risk_check:
            passed, reason = await self._check_risk(order)
            if not passed:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = reason
                self._log_order_event(order, "rejected", reason)
                
                for callback in self._on_order_reject:
                    callback(order, reason)
                
                return order
        
        # Submit to exchange
        try:
            order = await self.exchange.submit_order(order)
            self._orders[order.order_id] = order
            self._orders_this_minute += 1
            
            if order.is_active:
                self._pending_orders[order.order_id] = order
            
            self._log_order_event(order, "submitted")
            
            # Check if filled immediately (market orders)
            if order.is_filled:
                self._handle_fill(order)
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = str(e)
            self._log_order_event(order, "error", str(e))
            logger.error(f"Order submission failed: {e}")
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self._orders[order_id]
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active (status: {order.status})")
            return False
        
        success = await self.exchange.cancel_order(order_id)
        
        if success:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()
            self._pending_orders.pop(order_id, None)
            self._log_order_event(order, "cancelled")
        
        return success
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all pending orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        orders_to_cancel = list(self._pending_orders.values())
        
        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol]
        
        for order in orders_to_cancel:
            if await self.cancel_order(order.order_id):
                cancelled += 1
        
        return cancelled
    
    def _handle_fill(self, order: Order) -> None:
        """Handle order fill."""
        self._pending_orders.pop(order.order_id, None)
        self._log_order_event(order, "filled")
        
        for callback in self._on_order_fill:
            callback(order)
    
    def _log_order_event(
        self,
        order: Order,
        event: str,
        details: str = ""
    ) -> None:
        """Log an order event."""
        self._order_history.append({
            "timestamp": datetime.now().isoformat(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "event": event,
            "status": order.status.value,
            "details": details,
        })
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L for risk tracking."""
        self._daily_pnl += pnl
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of each day)."""
        self._daily_pnl = 0.0
        self._orders_this_minute = 0
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        """Get orders with filtering."""
        orders = list(self._orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if side:
            orders = [o for o in orders if o.side == side]
        
        return orders
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order event history."""
        return self._order_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        total = len(self._orders)
        filled = len([o for o in self._orders.values() if o.is_filled])
        cancelled = len([o for o in self._orders.values() if o.status == OrderStatus.CANCELLED])
        rejected = len([o for o in self._orders.values() if o.status == OrderStatus.REJECTED])
        
        return {
            "total_orders": total,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "rejected_orders": rejected,
            "pending_orders": len(self._pending_orders),
            "fill_rate": filled / total if total > 0 else 0,
            "daily_pnl": self._daily_pnl,
        }

