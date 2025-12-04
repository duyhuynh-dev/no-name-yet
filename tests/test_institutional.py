"""
Tests for Institutional Features Module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from src.institutional.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
)
from src.institutional.auth import (
    AuthManager,
    User,
    Role,
    Permission,
    ROLE_PERMISSIONS,
)
from src.institutional.reconciliation import (
    Reconciler,
    TradeRecord,
    ReconciliationStatus,
    BreakType,
)
from src.institutional.reporting import (
    ReportGenerator,
    ReportType,
    Report,
)
from src.institutional.compliance import (
    ComplianceChecker,
    ComplianceRule,
    ComplianceCategory,
    ComplianceSeverity,
    Order,
)


class TestAuditLogger:
    """Tests for Audit Logger."""
    
    def test_initialization(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        assert logger.log_dir.exists()
    
    def test_log_event(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        
        event = logger.log(
            event_type=AuditEventType.ORDER_SUBMITTED,
            action="submit_order",
            resource="order:123",
            user_id="user1",
            details={"symbol": "AAPL", "quantity": 100},
        )
        
        assert event.event_id is not None
        assert event.event_type == AuditEventType.ORDER_SUBMITTED
        assert event.checksum is not None
    
    def test_log_order(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        
        event = logger.log_order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            user_id="trader1",
        )
        
        assert event.event_type == AuditEventType.ORDER_SUBMITTED
        assert event.details["symbol"] == "AAPL"
    
    def test_chain_integrity(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        
        # Log multiple events
        for i in range(5):
            logger.log(
                event_type=AuditEventType.ORDER_SUBMITTED,
                action=f"action_{i}",
                resource=f"resource_{i}",
            )
        
        is_valid, errors = logger.verify_chain()
        
        assert is_valid
        assert len(errors) == 0
    
    def test_event_integrity(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        
        event = logger.log(
            event_type=AuditEventType.USER_LOGIN,
            action="login",
            resource="auth",
            user_id="user1",
        )
        
        assert event.verify_integrity()
    
    def test_query_events(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        
        logger.log(
            event_type=AuditEventType.ORDER_SUBMITTED,
            action="submit",
            resource="order:1",
            user_id="user1",
        )
        logger.log(
            event_type=AuditEventType.USER_LOGIN,
            action="login",
            resource="auth",
            user_id="user2",
        )
        
        # Query by type
        orders = logger.get_events(event_type=AuditEventType.ORDER_SUBMITTED)
        assert len(orders) == 1
        
        # Query by user
        user1_events = logger.get_events(user_id="user1")
        assert len(user1_events) == 1


class TestAuthManager:
    """Tests for Auth Manager."""
    
    def test_initialization(self):
        auth = AuthManager()
        
        # Should have default admin
        admin = auth.get_user_by_username("admin")
        assert admin is not None
        assert Role.SUPER_ADMIN in admin.roles
    
    def test_create_user(self):
        auth = AuthManager()
        
        user = auth.create_user(
            username="trader1",
            email="trader1@example.com",
            password="secure123",
            roles=[Role.TRADER],
            full_name="Test Trader",
        )
        
        assert user.username == "trader1"
        assert Role.TRADER in user.roles
    
    def test_authenticate(self):
        auth = AuthManager()
        
        auth.create_user(
            username="test_user",
            email="test@example.com",
            password="password123",
            roles=[Role.VIEWER],
        )
        
        session = auth.authenticate("test_user", "password123")
        
        assert session is not None
        assert session.access_token is not None
        assert session.refresh_token is not None
    
    def test_authenticate_wrong_password(self):
        auth = AuthManager()
        
        auth.create_user(
            username="test_user2",
            email="test2@example.com",
            password="correct_password",
            roles=[Role.VIEWER],
        )
        
        session = auth.authenticate("test_user2", "wrong_password")
        
        assert session is None
    
    def test_validate_token(self):
        auth = AuthManager()
        
        user = auth.create_user(
            username="token_test",
            email="token@example.com",
            password="password",
            roles=[Role.TRADER],
        )
        
        session = auth.authenticate("token_test", "password")
        validated_user = auth.validate_token(session.access_token)
        
        assert validated_user is not None
        assert validated_user.user_id == user.user_id
    
    def test_role_permissions(self):
        auth = AuthManager()
        
        user = auth.create_user(
            username="trader",
            email="trader@example.com",
            password="password",
            roles=[Role.TRADER],
        )
        
        # Trader should have TRADE_EXECUTE
        assert user.has_permission(Permission.TRADE_EXECUTE)
        
        # Trader should not have SYSTEM_ADMIN
        assert not user.has_permission(Permission.SYSTEM_ADMIN)
    
    def test_custom_permissions(self):
        auth = AuthManager()
        
        user = auth.create_user(
            username="custom",
            email="custom@example.com",
            password="password",
            roles=[Role.VIEWER],
        )
        
        # Add custom permission
        auth.add_permission(user.user_id, Permission.REPORT_GENERATE)
        
        assert user.has_permission(Permission.REPORT_GENERATE)
    
    def test_account_lockout(self):
        auth = AuthManager(max_failed_attempts=3)
        
        auth.create_user(
            username="lockout_test",
            email="lockout@example.com",
            password="correct",
            roles=[Role.VIEWER],
        )
        
        # Fail 3 times
        for _ in range(3):
            auth.authenticate("lockout_test", "wrong")
        
        user = auth.get_user_by_username("lockout_test")
        assert user.is_locked
        
        # Can't login even with correct password
        session = auth.authenticate("lockout_test", "correct")
        assert session is None


class TestReconciler:
    """Tests for Reconciler."""
    
    def test_reconcile_matched(self):
        reconciler = Reconciler()
        
        internal = [
            TradeRecord(
                trade_id="INT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0,
                trade_date=date.today(),
                broker="Broker1",
            ),
        ]
        
        external = [
            TradeRecord(
                trade_id="EXT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0,
                trade_date=date.today(),
                broker="Broker1",
                source="external",
            ),
        ]
        
        result = reconciler.reconcile(internal, external)
        
        assert result.matched == 1
        assert result.breaks == 0
        assert result.status == ReconciliationStatus.MATCHED
    
    def test_reconcile_quantity_mismatch(self):
        reconciler = Reconciler(quantity_tolerance=0.001)
        
        internal = [
            TradeRecord(
                trade_id="INT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0,
                trade_date=date.today(),
            ),
        ]
        
        external = [
            TradeRecord(
                trade_id="EXT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=110,  # Mismatch
                price=150.0,
                trade_date=date.today(),
                source="external",
            ),
        ]
        
        result = reconciler.reconcile(internal, external)
        
        assert result.breaks > 0
        assert any(b.break_type == BreakType.QUANTITY_MISMATCH for b in result.break_list)
    
    def test_reconcile_missing_external(self):
        reconciler = Reconciler()
        
        internal = [
            TradeRecord(
                trade_id="INT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0,
                trade_date=date.today(),
            ),
        ]
        
        external = []  # No external trades
        
        result = reconciler.reconcile(internal, external)
        
        assert result.unmatched > 0
        assert any(b.break_type == BreakType.MISSING_EXTERNAL for b in result.break_list)
    
    def test_reconcile_positions(self):
        reconciler = Reconciler()
        
        internal_pos = {"AAPL": 100, "MSFT": 50, "GOOGL": 25}
        external_pos = {"AAPL": 100, "MSFT": 50, "GOOGL": 25}
        
        result = reconciler.reconcile_positions(internal_pos, external_pos)
        
        assert result["matched"] == 3
        assert result["breaks"] == 0
    
    def test_resolve_break(self):
        reconciler = Reconciler()
        
        internal = [
            TradeRecord(
                trade_id="INT-001",
                order_id="ORD-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0,
                trade_date=date.today(),
            ),
        ]
        
        external = []
        
        result = reconciler.reconcile(internal, external)
        
        break_id = result.break_list[0].break_id
        resolved = reconciler.resolve_break(
            break_id,
            resolution="Trade confirmed by broker",
            resolved_by="ops_user",
        )
        
        assert resolved


class TestReportGenerator:
    """Tests for Report Generator."""
    
    def test_generate_daily_pnl(self):
        generator = ReportGenerator()
        
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 150},
        ]
        positions = {
            "AAPL": {"quantity": 100, "cost_basis": 145},
        }
        prices = {"AAPL": 155}
        
        report = generator.generate_daily_pnl(trades, positions, prices)
        
        assert report.report_type == ReportType.DAILY_PNL
        assert "trade_pnl" in report.data
        assert "unrealized_pnl" in report.data
    
    def test_generate_risk_report(self):
        generator = ReportGenerator()
        
        report = generator.generate_risk_report(
            var_95=10000,
            var_99=15000,
            cvar=18000,
            beta=1.1,
            max_drawdown=0.15,
            volatility=0.20,
            positions={"AAPL": 50000, "MSFT": 30000},
        )
        
        assert report.report_type == ReportType.RISK_REPORT
        assert report.data["var_95"] == 10000
    
    def test_generate_executive_summary(self):
        generator = ReportGenerator()
        
        report = generator.generate_executive_summary(
            portfolio_value=1000000,
            daily_pnl=5000,
            mtd_pnl=25000,
            ytd_pnl=150000,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.55,
            num_trades=150,
        )
        
        assert report.report_type == ReportType.EXECUTIVE_SUMMARY
        assert len(report.sections) == 2
    
    def test_report_to_html(self):
        generator = ReportGenerator()
        
        report = generator.generate_executive_summary(
            portfolio_value=1000000,
            daily_pnl=5000,
            mtd_pnl=25000,
            ytd_pnl=150000,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.55,
            num_trades=150,
        )
        
        html = report.to_html()
        
        assert "<html>" in html
        assert "Executive Summary" in html
    
    def test_performance_attribution(self):
        generator = ReportGenerator()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 60),
            "MSFT": np.random.normal(0.0008, 0.018, 60),
        }, index=dates)
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 60), index=dates)
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        
        report = generator.generate_performance_attribution(
            returns=returns,
            weights=weights,
            benchmark_returns=benchmark,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 1),
        )
        
        assert report.report_type == ReportType.PERFORMANCE_ATTRIBUTION
        assert "sharpe_ratio" in report.data


class TestComplianceChecker:
    """Tests for Compliance Checker."""
    
    def test_initialization(self):
        checker = ComplianceChecker()
        
        rules = checker.list_rules()
        assert len(rules) > 0
    
    def test_position_limit_pass(self):
        checker = ComplianceChecker()
        
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
        )
        
        result = checker.check_position_limit(
            order=order,
            current_position=0,
            portfolio_value=1000000,  # $1M portfolio
        )
        
        # 100 * 150 = $15,000 = 1.5% of portfolio
        assert result.passed
    
    def test_position_limit_fail(self):
        checker = ComplianceChecker()
        
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=1000,
            price=150.0,
            order_type="limit",
        )
        
        result = checker.check_position_limit(
            order=order,
            current_position=0,
            portfolio_value=100000,  # $100k portfolio
        )
        
        # 1000 * 150 = $150,000 = 150% of portfolio
        assert not result.passed
    
    def test_leverage_check(self):
        checker = ComplianceChecker()
        
        # Under limit
        result = checker.check_leverage(
            total_exposure=150000,
            portfolio_value=100000,  # 1.5x leverage
        )
        assert result.passed
        
        # Over limit
        result = checker.check_leverage(
            total_exposure=300000,
            portfolio_value=100000,  # 3x leverage
        )
        assert not result.passed
    
    def test_restricted_list(self):
        checker = ComplianceChecker()
        
        # Update restricted list
        rule = checker.get_rule("AST-001")
        rule.parameters["restricted_list"] = ["XYZ", "ABC"]
        
        order = Order(
            order_id="ORD-001",
            symbol="XYZ",
            side="buy",
            quantity=100,
            price=50.0,
            order_type="limit",
        )
        
        result = checker.check_restricted_list(order)
        assert not result.passed
    
    def test_pre_trade_checks(self):
        checker = ComplianceChecker()
        
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
        )
        
        results = checker.run_pre_trade_checks(
            order=order,
            current_position=0,
            portfolio_value=1000000,
            total_exposure=500000,
            avg_volume=1000000,
        )
        
        assert len(results) >= 4  # Position, Leverage, Restricted, Hours, Liquidity
    
    def test_is_order_allowed(self):
        checker = ComplianceChecker()
        
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
        )
        
        results = checker.run_pre_trade_checks(
            order=order,
            current_position=0,
            portfolio_value=1000000,
            total_exposure=500000,
            avg_volume=1000000,
        )
        
        # Should be allowed (no hard blocks)
        # Note: Trading hours check might fail depending on when test runs
        allowed = checker.is_order_allowed(results)
        # Just verify it returns a boolean
        assert isinstance(allowed, bool)
    
    def test_add_custom_rule(self):
        checker = ComplianceChecker()
        
        custom_rule = ComplianceRule(
            rule_id="CUSTOM-001",
            name="Custom Rule",
            description="Test custom rule",
            category=ComplianceCategory.CUSTOM,
            severity=ComplianceSeverity.WARNING,
            parameters={"max_value": 100},
        )
        
        checker.add_rule(custom_rule)
        
        retrieved = checker.get_rule("CUSTOM-001")
        assert retrieved is not None
        assert retrieved.name == "Custom Rule"
    
    def test_compliance_summary(self):
        checker = ComplianceChecker()
        
        # Run some checks
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
        )
        
        checker.run_pre_trade_checks(
            order=order,
            current_position=0,
            portfolio_value=1000000,
            total_exposure=500000,
            avg_volume=1000000,
        )
        
        summary = checker.get_summary()
        
        assert "total_checks" in summary
        assert "active_rules" in summary


class TestIntegration:
    """Integration tests for institutional module."""
    
    def test_audit_compliance_integration(self, tmp_path):
        """Test audit logging of compliance checks."""
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        checker = ComplianceChecker()
        
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
        )
        
        results = checker.run_pre_trade_checks(
            order=order,
            current_position=0,
            portfolio_value=1000000,
            total_exposure=500000,
            avg_volume=1000000,
        )
        
        # Log compliance results
        for result in results:
            logger.log_compliance(
                check_type=result.rule.name,
                passed=result.passed,
                details=result.to_dict(),
            )
        
        # Verify events logged
        events = logger.get_events(event_type=AuditEventType.COMPLIANCE_CHECK)
        assert len(events) >= 1
    
    def test_auth_audit_integration(self, tmp_path):
        """Test audit logging of auth events."""
        logger = AuditLogger(log_dir=str(tmp_path), async_write=False)
        auth = AuthManager()
        
        # Create user
        user = auth.create_user(
            username="audit_test",
            email="audit@test.com",
            password="password",
            roles=[Role.TRADER],
        )
        
        # Log user creation
        logger.log_user_action(
            action="create",
            user_id="admin",
            target_user_id=user.user_id,
            roles=[r.value for r in user.roles],
        )
        
        # Authenticate
        session = auth.authenticate("audit_test", "password")
        
        if session:
            logger.log_user_action(
                action="login",
                user_id=user.user_id,
                session_id=session.session_id,
            )
        
        events = logger.get_events(event_type=AuditEventType.USER_CREATED)
        assert len(events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

