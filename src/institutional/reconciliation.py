"""
Trade Reconciliation System

Provides automated reconciliation between internal records
and external sources (brokers, exchanges, custodians).
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd


class ReconciliationStatus(Enum):
    """Reconciliation status."""
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    PARTIAL_MATCH = "partial_match"
    BREAK = "break"
    PENDING = "pending"


class BreakType(Enum):
    """Types of reconciliation breaks."""
    MISSING_INTERNAL = "missing_internal"
    MISSING_EXTERNAL = "missing_external"
    QUANTITY_MISMATCH = "quantity_mismatch"
    PRICE_MISMATCH = "price_mismatch"
    SETTLEMENT_MISMATCH = "settlement_mismatch"
    DATE_MISMATCH = "date_mismatch"
    SYMBOL_MISMATCH = "symbol_mismatch"
    OTHER = "other"


@dataclass
class TradeRecord:
    """Trade record for reconciliation."""
    trade_id: str
    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    
    trade_date: date
    settlement_date: Optional[date] = None
    
    broker: Optional[str] = None
    exchange: Optional[str] = None
    counterparty: Optional[str] = None
    
    fees: float = 0.0
    currency: str = "USD"
    
    source: str = "internal"  # internal/external
    
    @property
    def notional(self) -> float:
        return self.quantity * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "trade_date": self.trade_date.isoformat(),
            "settlement_date": self.settlement_date.isoformat() if self.settlement_date else None,
            "broker": self.broker,
            "exchange": self.exchange,
            "fees": self.fees,
            "notional": self.notional,
            "source": self.source,
        }


@dataclass
class ReconciliationBreak:
    """Reconciliation break/discrepancy."""
    break_id: str
    break_type: BreakType
    
    internal_trade: Optional[TradeRecord]
    external_trade: Optional[TradeRecord]
    
    description: str
    expected_value: Any
    actual_value: Any
    difference: Optional[float] = None
    
    status: ReconciliationStatus = ReconciliationStatus.BREAK
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "break_id": self.break_id,
            "break_type": self.break_type.value,
            "internal_trade_id": self.internal_trade.trade_id if self.internal_trade else None,
            "external_trade_id": self.external_trade.trade_id if self.external_trade else None,
            "description": self.description,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "difference": self.difference,
            "status": self.status.value,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""
    reconciliation_id: str
    run_date: datetime
    
    # Statistics
    total_internal: int = 0
    total_external: int = 0
    matched: int = 0
    unmatched: int = 0
    breaks: int = 0
    
    # Breaks detail
    break_list: List[ReconciliationBreak] = field(default_factory=list)
    
    # Value reconciliation
    internal_notional: float = 0.0
    external_notional: float = 0.0
    notional_difference: float = 0.0
    
    # Status
    status: ReconciliationStatus = ReconciliationStatus.PENDING
    
    @property
    def match_rate(self) -> float:
        total = self.total_internal + self.total_external
        if total == 0:
            return 1.0
        return self.matched * 2 / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "run_date": self.run_date.isoformat(),
            "total_internal": self.total_internal,
            "total_external": self.total_external,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "breaks": self.breaks,
            "match_rate": self.match_rate,
            "internal_notional": self.internal_notional,
            "external_notional": self.external_notional,
            "notional_difference": self.notional_difference,
            "status": self.status.value,
        }


class Reconciler:
    """
    Trade Reconciliation Engine.
    
    Matches internal trades against external records
    and identifies discrepancies.
    """
    
    def __init__(
        self,
        price_tolerance: float = 0.01,  # 1% tolerance
        quantity_tolerance: float = 0.001,  # 0.1% tolerance
    ):
        """
        Initialize Reconciler.
        
        Args:
            price_tolerance: Acceptable price difference (percentage)
            quantity_tolerance: Acceptable quantity difference (percentage)
        """
        self.price_tolerance = price_tolerance
        self.quantity_tolerance = quantity_tolerance
        
        self._results: List[ReconciliationResult] = []
        self._break_counter = 0
    
    def _generate_break_id(self) -> str:
        """Generate unique break ID."""
        self._break_counter += 1
        return f"BRK-{datetime.now().strftime('%Y%m%d')}-{self._break_counter:06d}"
    
    def _match_key(self, trade: TradeRecord) -> Tuple:
        """Generate matching key for trade."""
        return (
            trade.symbol,
            trade.side,
            trade.trade_date,
            trade.broker or "",
        )
    
    def _compare_trades(
        self,
        internal: TradeRecord,
        external: TradeRecord,
    ) -> List[ReconciliationBreak]:
        """Compare two trades and return any breaks."""
        breaks = []
        
        # Check quantity
        qty_diff = abs(internal.quantity - external.quantity) / max(internal.quantity, 1)
        if qty_diff > self.quantity_tolerance:
            breaks.append(ReconciliationBreak(
                break_id=self._generate_break_id(),
                break_type=BreakType.QUANTITY_MISMATCH,
                internal_trade=internal,
                external_trade=external,
                description=f"Quantity mismatch for {internal.symbol}",
                expected_value=internal.quantity,
                actual_value=external.quantity,
                difference=internal.quantity - external.quantity,
            ))
        
        # Check price
        price_diff = abs(internal.price - external.price) / max(internal.price, 0.01)
        if price_diff > self.price_tolerance:
            breaks.append(ReconciliationBreak(
                break_id=self._generate_break_id(),
                break_type=BreakType.PRICE_MISMATCH,
                internal_trade=internal,
                external_trade=external,
                description=f"Price mismatch for {internal.symbol}",
                expected_value=internal.price,
                actual_value=external.price,
                difference=internal.price - external.price,
            ))
        
        # Check settlement date
        if internal.settlement_date and external.settlement_date:
            if internal.settlement_date != external.settlement_date:
                breaks.append(ReconciliationBreak(
                    break_id=self._generate_break_id(),
                    break_type=BreakType.SETTLEMENT_MISMATCH,
                    internal_trade=internal,
                    external_trade=external,
                    description=f"Settlement date mismatch for {internal.symbol}",
                    expected_value=str(internal.settlement_date),
                    actual_value=str(external.settlement_date),
                ))
        
        return breaks
    
    def reconcile(
        self,
        internal_trades: List[TradeRecord],
        external_trades: List[TradeRecord],
        reconciliation_id: Optional[str] = None,
    ) -> ReconciliationResult:
        """
        Run reconciliation between internal and external trades.
        
        Args:
            internal_trades: Internal trade records
            external_trades: External trade records (from broker/custodian)
            reconciliation_id: Optional ID for the reconciliation run
            
        Returns:
            ReconciliationResult with matches and breaks
        """
        if reconciliation_id is None:
            reconciliation_id = f"REC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        result = ReconciliationResult(
            reconciliation_id=reconciliation_id,
            run_date=datetime.now(),
            total_internal=len(internal_trades),
            total_external=len(external_trades),
        )
        
        # Index external trades by matching key
        external_index: Dict[Tuple, List[TradeRecord]] = {}
        for trade in external_trades:
            key = self._match_key(trade)
            if key not in external_index:
                external_index[key] = []
            external_index[key].append(trade)
        
        # Track matched external trades
        matched_external = set()
        
        # Match internal trades
        for internal in internal_trades:
            key = self._match_key(internal)
            result.internal_notional += internal.notional
            
            if key in external_index and external_index[key]:
                # Find best match by quantity
                best_match = None
                best_diff = float('inf')
                
                for ext in external_index[key]:
                    if ext.trade_id in matched_external:
                        continue
                    diff = abs(internal.quantity - ext.quantity)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = ext
                
                if best_match:
                    matched_external.add(best_match.trade_id)
                    
                    # Compare and check for breaks
                    trade_breaks = self._compare_trades(internal, best_match)
                    
                    if trade_breaks:
                        result.break_list.extend(trade_breaks)
                        result.breaks += len(trade_breaks)
                    else:
                        result.matched += 1
                else:
                    # No available match
                    result.break_list.append(ReconciliationBreak(
                        break_id=self._generate_break_id(),
                        break_type=BreakType.MISSING_EXTERNAL,
                        internal_trade=internal,
                        external_trade=None,
                        description=f"No external match for internal trade {internal.trade_id}",
                        expected_value=internal.trade_id,
                        actual_value=None,
                    ))
                    result.unmatched += 1
            else:
                # No match found
                result.break_list.append(ReconciliationBreak(
                    break_id=self._generate_break_id(),
                    break_type=BreakType.MISSING_EXTERNAL,
                    internal_trade=internal,
                    external_trade=None,
                    description=f"No external match for internal trade {internal.trade_id}",
                    expected_value=internal.trade_id,
                    actual_value=None,
                ))
                result.unmatched += 1
        
        # Check for unmatched external trades
        for external in external_trades:
            result.external_notional += external.notional
            
            if external.trade_id not in matched_external:
                result.break_list.append(ReconciliationBreak(
                    break_id=self._generate_break_id(),
                    break_type=BreakType.MISSING_INTERNAL,
                    internal_trade=None,
                    external_trade=external,
                    description=f"External trade {external.trade_id} not in internal records",
                    expected_value=None,
                    actual_value=external.trade_id,
                ))
                result.unmatched += 1
        
        # Calculate notional difference
        result.notional_difference = result.internal_notional - result.external_notional
        
        # Set status
        if result.breaks == 0 and result.unmatched == 0:
            result.status = ReconciliationStatus.MATCHED
        elif result.breaks > 0 or result.unmatched > 0:
            result.status = ReconciliationStatus.BREAK
        
        self._results.append(result)
        
        return result
    
    def reconcile_positions(
        self,
        internal_positions: Dict[str, float],
        external_positions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Reconcile position holdings.
        
        Args:
            internal_positions: Internal positions (symbol -> quantity)
            external_positions: External positions (symbol -> quantity)
            
        Returns:
            Position reconciliation summary
        """
        all_symbols = set(internal_positions.keys()) | set(external_positions.keys())
        
        matched = []
        breaks = []
        
        for symbol in all_symbols:
            internal_qty = internal_positions.get(symbol, 0)
            external_qty = external_positions.get(symbol, 0)
            
            diff = internal_qty - external_qty
            diff_pct = abs(diff) / max(abs(internal_qty), abs(external_qty), 1)
            
            if diff_pct <= self.quantity_tolerance:
                matched.append(symbol)
            else:
                breaks.append({
                    "symbol": symbol,
                    "internal_quantity": internal_qty,
                    "external_quantity": external_qty,
                    "difference": diff,
                    "difference_pct": diff_pct * 100,
                })
        
        return {
            "total_symbols": len(all_symbols),
            "matched": len(matched),
            "breaks": len(breaks),
            "match_rate": len(matched) / len(all_symbols) if all_symbols else 1.0,
            "break_details": breaks,
        }
    
    def resolve_break(
        self,
        break_id: str,
        resolution: str,
        resolved_by: str,
    ) -> bool:
        """
        Resolve a reconciliation break.
        
        Args:
            break_id: Break ID
            resolution: Resolution description
            resolved_by: User who resolved
            
        Returns:
            True if resolved, False if not found
        """
        for result in self._results:
            for brk in result.break_list:
                if brk.break_id == break_id:
                    brk.resolution = resolution
                    brk.resolved_by = resolved_by
                    brk.resolved_at = datetime.now()
                    brk.status = ReconciliationStatus.MATCHED
                    return True
        return False
    
    def get_open_breaks(self) -> List[ReconciliationBreak]:
        """Get all unresolved breaks."""
        breaks = []
        for result in self._results:
            for brk in result.break_list:
                if brk.status == ReconciliationStatus.BREAK:
                    breaks.append(brk)
        return breaks
    
    def get_break_summary(self) -> Dict[str, Any]:
        """Get summary of all breaks by type."""
        summary: Dict[BreakType, int] = {bt: 0 for bt in BreakType}
        
        for result in self._results:
            for brk in result.break_list:
                summary[brk.break_type] += 1
        
        return {
            bt.value: count
            for bt, count in summary.items()
            if count > 0
        }
    
    def get_results(self, limit: int = 50) -> List[ReconciliationResult]:
        """Get recent reconciliation results."""
        return self._results[-limit:]

