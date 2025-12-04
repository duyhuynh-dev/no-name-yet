"""
Real-Time Trading Infrastructure

This module provides exchange connectivity, order management,
and execution capabilities for live and paper trading.
"""

from .base import (
    BaseExchange,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Order,
    Position,
    Trade,
    AccountInfo,
)
from .alpaca_exchange import AlpacaExchange
from .paper_exchange import PaperExchange
from .order_manager import OrderManager
from .execution_engine import ExecutionEngine
from .trading_session import TradingSession

__all__ = [
    # Base classes
    "BaseExchange",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "Position",
    "Trade",
    "AccountInfo",
    # Exchanges
    "AlpacaExchange",
    "PaperExchange",
    # Management
    "OrderManager",
    "ExecutionEngine",
    "TradingSession",
]

