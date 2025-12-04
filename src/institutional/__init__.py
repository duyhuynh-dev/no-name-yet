"""
Institutional Features Module

Provides enterprise-grade features for institutional trading:
- Audit trail and compliance logging
- User authentication and authorization
- Trade reconciliation
- Reporting and analytics
"""

from .audit import AuditLogger, AuditEvent, AuditEventType
from .auth import AuthManager, User, Role, Permission
from .reconciliation import Reconciler, ReconciliationResult
from .reporting import ReportGenerator, ReportType
from .compliance import ComplianceChecker, ComplianceRule, ComplianceResult

__all__ = [
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuthManager",
    "User",
    "Role",
    "Permission",
    "Reconciler",
    "ReconciliationResult",
    "ReportGenerator",
    "ReportType",
    "ComplianceChecker",
    "ComplianceRule",
    "ComplianceResult",
]

