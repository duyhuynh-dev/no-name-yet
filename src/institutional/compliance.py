"""
Compliance Checking System

Provides pre-trade and post-trade compliance checks
for regulatory requirements.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, date, time
from abc import ABC, abstractmethod


class ComplianceCategory(Enum):
    """Compliance rule categories."""
    POSITION_LIMITS = "position_limits"
    CONCENTRATION = "concentration"
    ASSET_ELIGIBILITY = "asset_eligibility"
    TRADING_RESTRICTIONS = "trading_restrictions"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


class ComplianceSeverity(Enum):
    """Compliance violation severity."""
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    HARD_BLOCK = "hard_block"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    name: str
    description: str
    category: ComplianceCategory
    severity: ComplianceSeverity
    
    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_active: bool = True
    
    # Effective dates
    effective_from: Optional[date] = None
    effective_until: Optional[date] = None
    
    # Metadata
    regulatory_reference: Optional[str] = None  # e.g., "MiFID II Article 25"
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_effective(self, check_date: Optional[date] = None) -> bool:
        """Check if rule is effective."""
        check_date = check_date or date.today()
        
        if not self.is_active:
            return False
        
        if self.effective_from and check_date < self.effective_from:
            return False
        
        if self.effective_until and check_date > self.effective_until:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "parameters": self.parameters,
            "is_active": self.is_active,
            "regulatory_reference": self.regulatory_reference,
        }


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    rule: ComplianceRule
    passed: bool
    
    # Details
    message: str
    current_value: Any = None
    limit_value: Any = None
    
    # Context
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule.rule_id,
            "rule_name": self.rule.name,
            "passed": self.passed,
            "severity": self.rule.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Order:
    """Order for compliance checking."""
    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: Optional[float]
    order_type: str  # market/limit
    
    account_id: Optional[str] = None
    user_id: Optional[str] = None


class ComplianceChecker:
    """
    Compliance Checking Engine.
    
    Performs pre-trade and post-trade compliance checks.
    """
    
    def __init__(self):
        """Initialize Compliance Checker."""
        self._rules: Dict[str, ComplianceRule] = {}
        self._check_history: List[ComplianceResult] = []
        
        # Add default rules
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default compliance rules."""
        default_rules = [
            ComplianceRule(
                rule_id="POS-001",
                name="Single Position Limit",
                description="Maximum position size for any single security",
                category=ComplianceCategory.POSITION_LIMITS,
                severity=ComplianceSeverity.HARD_BLOCK,
                parameters={"max_position_pct": 0.10},  # 10% of portfolio
            ),
            ComplianceRule(
                rule_id="POS-002",
                name="Sector Concentration",
                description="Maximum exposure to a single sector",
                category=ComplianceCategory.CONCENTRATION,
                severity=ComplianceSeverity.WARNING,
                parameters={"max_sector_pct": 0.30},  # 30%
            ),
            ComplianceRule(
                rule_id="LEV-001",
                name="Leverage Limit",
                description="Maximum portfolio leverage",
                category=ComplianceCategory.LEVERAGE,
                severity=ComplianceSeverity.HARD_BLOCK,
                parameters={"max_leverage": 2.0},  # 2x
            ),
            ComplianceRule(
                rule_id="TRD-001",
                name="Trading Hours",
                description="Trading allowed only during market hours",
                category=ComplianceCategory.TRADING_RESTRICTIONS,
                severity=ComplianceSeverity.HARD_BLOCK,
                parameters={
                    "market_open": "09:30",
                    "market_close": "16:00",
                    "timezone": "US/Eastern",
                },
            ),
            ComplianceRule(
                rule_id="TRD-002",
                name="Wash Sale Prevention",
                description="Prevent wash sales within 30 days",
                category=ComplianceCategory.REGULATORY,
                severity=ComplianceSeverity.WARNING,
                parameters={"washsale_days": 30},
                regulatory_reference="IRS Wash Sale Rule",
            ),
            ComplianceRule(
                rule_id="AST-001",
                name="Restricted Securities",
                description="Block trading of restricted securities",
                category=ComplianceCategory.ASSET_ELIGIBILITY,
                severity=ComplianceSeverity.HARD_BLOCK,
                parameters={"restricted_list": []},
            ),
            ComplianceRule(
                rule_id="LIQ-001",
                name="Minimum Liquidity",
                description="Ensure minimum liquidity for trade execution",
                category=ComplianceCategory.LIQUIDITY,
                severity=ComplianceSeverity.WARNING,
                parameters={"min_avg_volume": 100000},
            ),
        ]
        
        for rule in default_rules:
            self._rules[rule.rule_id] = rule
    
    def add_rule(self, rule: ComplianceRule) -> None:
        """Add a compliance rule."""
        self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a compliance rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def list_rules(
        self,
        category: Optional[ComplianceCategory] = None,
        active_only: bool = True,
    ) -> List[ComplianceRule]:
        """List compliance rules."""
        rules = list(self._rules.values())
        
        if category:
            rules = [r for r in rules if r.category == category]
        
        if active_only:
            rules = [r for r in rules if r.is_effective()]
        
        return rules
    
    def check_position_limit(
        self,
        order: Order,
        current_position: float,
        portfolio_value: float,
    ) -> ComplianceResult:
        """Check position limit compliance."""
        rule = self._rules.get("POS-001")
        if not rule or not rule.is_effective():
            return ComplianceResult(
                rule=rule or ComplianceRule(
                    rule_id="POS-001", name="Position Limit",
                    description="", category=ComplianceCategory.POSITION_LIMITS,
                    severity=ComplianceSeverity.INFO
                ),
                passed=True,
                message="Rule not active",
            )
        
        max_pct = rule.parameters.get("max_position_pct", 0.10)
        
        # Calculate new position
        if order.side == "buy":
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        position_value = abs(new_position * (order.price or 0))
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        passed = position_pct <= max_pct
        
        result = ComplianceResult(
            rule=rule,
            passed=passed,
            message=f"Position {'within' if passed else 'exceeds'} limit",
            current_value=position_pct,
            limit_value=max_pct,
            symbol=order.symbol,
            order_id=order.order_id,
        )
        
        self._check_history.append(result)
        return result
    
    def check_leverage(
        self,
        total_exposure: float,
        portfolio_value: float,
    ) -> ComplianceResult:
        """Check leverage compliance."""
        rule = self._rules.get("LEV-001")
        if not rule or not rule.is_effective():
            return ComplianceResult(
                rule=rule or ComplianceRule(
                    rule_id="LEV-001", name="Leverage Limit",
                    description="", category=ComplianceCategory.LEVERAGE,
                    severity=ComplianceSeverity.INFO
                ),
                passed=True,
                message="Rule not active",
            )
        
        max_leverage = rule.parameters.get("max_leverage", 2.0)
        current_leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        passed = current_leverage <= max_leverage
        
        result = ComplianceResult(
            rule=rule,
            passed=passed,
            message=f"Leverage {'within' if passed else 'exceeds'} limit",
            current_value=current_leverage,
            limit_value=max_leverage,
        )
        
        self._check_history.append(result)
        return result
    
    def check_restricted_list(
        self,
        order: Order,
        restricted_symbols: Optional[List[str]] = None,
    ) -> ComplianceResult:
        """Check if symbol is on restricted list."""
        rule = self._rules.get("AST-001")
        if not rule or not rule.is_effective():
            return ComplianceResult(
                rule=rule or ComplianceRule(
                    rule_id="AST-001", name="Restricted Securities",
                    description="", category=ComplianceCategory.ASSET_ELIGIBILITY,
                    severity=ComplianceSeverity.INFO
                ),
                passed=True,
                message="Rule not active",
            )
        
        restricted = restricted_symbols or rule.parameters.get("restricted_list", [])
        is_restricted = order.symbol in restricted
        
        result = ComplianceResult(
            rule=rule,
            passed=not is_restricted,
            message="Trading restricted" if is_restricted else "Symbol allowed",
            current_value=order.symbol,
            symbol=order.symbol,
            order_id=order.order_id,
        )
        
        self._check_history.append(result)
        return result
    
    def check_trading_hours(
        self,
        check_time: Optional[datetime] = None,
    ) -> ComplianceResult:
        """Check if trading is allowed at current time."""
        rule = self._rules.get("TRD-001")
        if not rule or not rule.is_effective():
            return ComplianceResult(
                rule=rule or ComplianceRule(
                    rule_id="TRD-001", name="Trading Hours",
                    description="", category=ComplianceCategory.TRADING_RESTRICTIONS,
                    severity=ComplianceSeverity.INFO
                ),
                passed=True,
                message="Rule not active",
            )
        
        check_time = check_time or datetime.now()
        
        market_open = datetime.strptime(
            rule.parameters.get("market_open", "09:30"), "%H:%M"
        ).time()
        market_close = datetime.strptime(
            rule.parameters.get("market_close", "16:00"), "%H:%M"
        ).time()
        
        current_time = check_time.time()
        within_hours = market_open <= current_time <= market_close
        
        # Check if weekend
        is_weekend = check_time.weekday() >= 5
        
        passed = within_hours and not is_weekend
        
        result = ComplianceResult(
            rule=rule,
            passed=passed,
            message="Within trading hours" if passed else "Outside trading hours",
            current_value=str(current_time),
            limit_value=f"{market_open} - {market_close}",
        )
        
        self._check_history.append(result)
        return result
    
    def check_liquidity(
        self,
        order: Order,
        avg_volume: float,
    ) -> ComplianceResult:
        """Check liquidity compliance."""
        rule = self._rules.get("LIQ-001")
        if not rule or not rule.is_effective():
            return ComplianceResult(
                rule=rule or ComplianceRule(
                    rule_id="LIQ-001", name="Minimum Liquidity",
                    description="", category=ComplianceCategory.LIQUIDITY,
                    severity=ComplianceSeverity.INFO
                ),
                passed=True,
                message="Rule not active",
            )
        
        min_volume = rule.parameters.get("min_avg_volume", 100000)
        
        passed = avg_volume >= min_volume
        
        # Also check order size vs volume
        order_pct = order.quantity / avg_volume if avg_volume > 0 else float('inf')
        size_warning = order_pct > 0.1  # Order is >10% of daily volume
        
        result = ComplianceResult(
            rule=rule,
            passed=passed and not size_warning,
            message="Liquidity check passed" if passed else "Insufficient liquidity",
            current_value=avg_volume,
            limit_value=min_volume,
            symbol=order.symbol,
            order_id=order.order_id,
        )
        
        self._check_history.append(result)
        return result
    
    def run_pre_trade_checks(
        self,
        order: Order,
        current_position: float,
        portfolio_value: float,
        total_exposure: float,
        avg_volume: float,
        restricted_symbols: Optional[List[str]] = None,
    ) -> List[ComplianceResult]:
        """
        Run all pre-trade compliance checks.
        
        Args:
            order: Order to check
            current_position: Current position in symbol
            portfolio_value: Total portfolio value
            total_exposure: Total market exposure
            avg_volume: Average daily volume
            restricted_symbols: List of restricted symbols
            
        Returns:
            List of compliance check results
        """
        results = []
        
        # Position limit
        results.append(self.check_position_limit(order, current_position, portfolio_value))
        
        # Leverage
        results.append(self.check_leverage(total_exposure, portfolio_value))
        
        # Restricted list
        results.append(self.check_restricted_list(order, restricted_symbols))
        
        # Trading hours
        results.append(self.check_trading_hours())
        
        # Liquidity
        results.append(self.check_liquidity(order, avg_volume))
        
        return results
    
    def is_order_allowed(self, results: List[ComplianceResult]) -> bool:
        """Check if order is allowed based on compliance results."""
        for result in results:
            if not result.passed and result.rule.severity == ComplianceSeverity.HARD_BLOCK:
                return False
        return True
    
    def get_violations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ComplianceResult]:
        """Get compliance violations."""
        violations = [r for r in self._check_history if not r.passed]
        
        if start_time:
            violations = [v for v in violations if v.timestamp >= start_time]
        if end_time:
            violations = [v for v in violations if v.timestamp <= end_time]
        
        return violations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get compliance summary."""
        total = len(self._check_history)
        violations = len([r for r in self._check_history if not r.passed])
        
        by_category = {}
        for result in self._check_history:
            cat = result.rule.category.value
            if cat not in by_category:
                by_category[cat] = {"total": 0, "violations": 0}
            by_category[cat]["total"] += 1
            if not result.passed:
                by_category[cat]["violations"] += 1
        
        return {
            "total_checks": total,
            "total_violations": violations,
            "violation_rate": violations / total if total > 0 else 0,
            "by_category": by_category,
            "active_rules": len([r for r in self._rules.values() if r.is_effective()]),
        }

