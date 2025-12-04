"""
Audit Trail Logging System

Provides comprehensive audit logging for regulatory compliance
and operational transparency.
"""

import json
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue


class AuditEventType(Enum):
    """Types of audit events."""
    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    
    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    MARGIN_CALL = "margin_call"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    MODEL_LOADED = "model_loaded"
    MODEL_UPDATED = "model_updated"
    
    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_PERMISSION_CHANGE = "user_permission_change"
    
    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    RECONCILIATION_RUN = "reconciliation_run"
    RECONCILIATION_BREAK = "reconciliation_break"
    
    # Data events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    REPORT_GENERATED = "report_generated"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    
    # Event details
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Outcome
    success: bool = True
    error_message: Optional[str] = None
    
    # Integrity
    checksum: Optional[str] = None
    previous_checksum: Optional[str] = None
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for integrity."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "previous_checksum": self.previous_checksum,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        return self.checksum == self._calculate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "success": self.success,
            "error_message": self.error_message,
            "checksum": self.checksum,
            "previous_checksum": self.previous_checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            action=data["action"],
            resource=data["resource"],
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            checksum=data.get("checksum"),
            previous_checksum=data.get("previous_checksum"),
        )


class AuditLogger:
    """
    Audit Trail Logger.
    
    Provides tamper-evident logging with blockchain-like
    integrity verification.
    """
    
    def __init__(
        self,
        log_dir: str = "audit_logs",
        retention_days: int = 2555,  # ~7 years for regulatory compliance
        async_write: bool = True,
    ):
        """
        Initialize Audit Logger.
        
        Args:
            log_dir: Directory for audit logs
            retention_days: Days to retain logs
            async_write: Whether to write logs asynchronously
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.async_write = async_write
        
        self._events: List[AuditEvent] = []
        self._last_checksum: Optional[str] = None
        self._event_counter = 0
        self._lock = threading.Lock()
        
        # Async writing
        if async_write:
            self._write_queue: Queue = Queue()
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
        
        # Setup Python logging
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        
        # Log to file
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self._logger.addHandler(handler)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        with self._lock:
            self._event_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            return f"AE-{timestamp}-{self._event_counter:06d}"
    
    def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            action: Action performed
            resource: Resource affected
            user_id: User who performed action
            details: Additional details
            ip_address: Client IP address
            session_id: Session identifier
            correlation_id: For tracing related events
            success: Whether action succeeded
            error_message: Error message if failed
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id,
            correlation_id=correlation_id,
            success=success,
            error_message=error_message,
            previous_checksum=self._last_checksum,
        )
        
        # Update chain
        self._last_checksum = event.checksum
        
        # Store event
        with self._lock:
            self._events.append(event)
        
        # Write to log
        if self.async_write:
            self._write_queue.put(event)
        else:
            self._write_event(event)
        
        return event
    
    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        self._logger.info(json.dumps(event.to_dict()))
    
    def _writer_loop(self) -> None:
        """Background writer thread."""
        while True:
            event = self._write_queue.get()
            if event is None:
                break
            self._write_event(event)
    
    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_type: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log order submission."""
        return self.log(
            event_type=AuditEventType.ORDER_SUBMITTED,
            action="submit_order",
            resource=f"order:{order_id}",
            user_id=user_id,
            details={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
                **kwargs,
            },
        )
    
    def log_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log order fill."""
        return self.log(
            event_type=AuditEventType.ORDER_FILLED,
            action="fill_order",
            resource=f"order:{order_id}",
            user_id=user_id,
            details={
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_quantity": fill_quantity,
                **kwargs,
            },
        )
    
    def log_risk_breach(
        self,
        breach_type: str,
        current_value: float,
        limit_value: float,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log risk limit breach."""
        return self.log(
            event_type=AuditEventType.RISK_LIMIT_BREACH,
            action="risk_breach",
            resource="risk_management",
            user_id=user_id,
            details={
                "breach_type": breach_type,
                "current_value": current_value,
                "limit_value": limit_value,
                **kwargs,
            },
            success=False,
        )
    
    def log_user_action(
        self,
        action: str,
        user_id: str,
        target_user_id: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log user-related action."""
        event_type = {
            "login": AuditEventType.USER_LOGIN,
            "logout": AuditEventType.USER_LOGOUT,
            "create": AuditEventType.USER_CREATED,
            "modify": AuditEventType.USER_MODIFIED,
            "permission_change": AuditEventType.USER_PERMISSION_CHANGE,
        }.get(action, AuditEventType.USER_MODIFIED)
        
        return self.log(
            event_type=event_type,
            action=action,
            resource=f"user:{target_user_id or user_id}",
            user_id=user_id,
            details=kwargs,
        )
    
    def log_compliance(
        self,
        check_type: str,
        passed: bool,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log compliance check."""
        event_type = AuditEventType.COMPLIANCE_CHECK if passed else AuditEventType.COMPLIANCE_VIOLATION
        
        return self.log(
            event_type=event_type,
            action="compliance_check",
            resource=f"compliance:{check_type}",
            user_id=user_id,
            details=details,
            success=passed,
        )
    
    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            event_type: Filter by event type
            user_id: Filter by user
            resource: Filter by resource
            limit: Maximum results
            
        Returns:
            List of matching events
        """
        results = []
        
        for event in reversed(self._events):
            if len(results) >= limit:
                break
            
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource and event.resource != resource:
                continue
            
            results.append(event)
        
        return results
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify audit log integrity.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        for i, event in enumerate(self._events):
            # Verify checksum
            if not event.verify_integrity():
                errors.append(f"Event {event.event_id}: Checksum mismatch")
            
            # Verify chain
            if i > 0:
                expected_prev = self._events[i-1].checksum
                if event.previous_checksum != expected_prev:
                    errors.append(f"Event {event.event_id}: Chain broken")
        
        return len(errors) == 0, errors
    
    def export_logs(
        self,
        output_file: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Export audit logs to file.
        
        Args:
            output_file: Output file path
            start_time: Filter start
            end_time: Filter end
            
        Returns:
            Number of events exported
        """
        events = self.get_events(start_time=start_time, end_time=end_time, limit=100000)
        
        with open(output_file, 'w') as f:
            for event in events:
                f.write(json.dumps(event.to_dict()) + "\n")
        
        # Log the export
        self.log(
            event_type=AuditEventType.DATA_EXPORT,
            action="export_audit_logs",
            resource=output_file,
            details={"event_count": len(events)},
        )
        
        return len(events)
    
    def close(self) -> None:
        """Shutdown the audit logger."""
        if self.async_write:
            self._write_queue.put(None)
            self._writer_thread.join(timeout=5)

