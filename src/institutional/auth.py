"""
Authentication and Authorization System

Provides user management, JWT authentication, and
role-based access control (RBAC).
"""

import hashlib
import secrets
import jwt
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta


class Permission(Enum):
    """System permissions."""
    # Trading permissions
    TRADE_READ = auto()
    TRADE_EXECUTE = auto()
    TRADE_MANAGE = auto()
    
    # Portfolio permissions
    PORTFOLIO_READ = auto()
    PORTFOLIO_MANAGE = auto()
    
    # Risk permissions
    RISK_READ = auto()
    RISK_MANAGE = auto()
    RISK_OVERRIDE = auto()
    
    # User permissions
    USER_READ = auto()
    USER_MANAGE = auto()
    USER_ADMIN = auto()
    
    # System permissions
    SYSTEM_READ = auto()
    SYSTEM_CONFIG = auto()
    SYSTEM_ADMIN = auto()
    
    # Data permissions
    DATA_READ = auto()
    DATA_EXPORT = auto()
    DATA_IMPORT = auto()
    
    # Report permissions
    REPORT_READ = auto()
    REPORT_GENERATE = auto()
    REPORT_EXPORT = auto()
    
    # Audit permissions
    AUDIT_READ = auto()
    AUDIT_EXPORT = auto()


class Role(Enum):
    """User roles with predefined permission sets."""
    VIEWER = "viewer"
    TRADER = "trader"
    ANALYST = "analyst"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Default permissions for each role
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.REPORT_READ,
    },
    Role.TRADER: {
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.REPORT_READ,
    },
    Role.ANALYST: {
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.DATA_READ,
        Permission.DATA_EXPORT,
        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
    },
    Role.RISK_MANAGER: {
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.RISK_MANAGE,
        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
    },
    Role.PORTFOLIO_MANAGER: {
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_MANAGE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_MANAGE,
        Permission.RISK_READ,
        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
    },
    Role.COMPLIANCE_OFFICER: {
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,
    },
    Role.ADMIN: {
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_MANAGE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_MANAGE,
        Permission.RISK_READ,
        Permission.RISK_MANAGE,
        Permission.USER_READ,
        Permission.USER_MANAGE,
        Permission.SYSTEM_READ,
        Permission.SYSTEM_CONFIG,
        Permission.DATA_READ,
        Permission.DATA_EXPORT,
        Permission.DATA_IMPORT,
        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,
        Permission.AUDIT_READ,
    },
    Role.SUPER_ADMIN: set(Permission),  # All permissions
}


@dataclass
class User:
    """User account."""
    user_id: str
    username: str
    email: str
    password_hash: str
    
    # Role and permissions
    roles: List[Role] = field(default_factory=list)
    custom_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    
    # Status
    is_active: bool = True
    is_locked: bool = False
    failed_login_attempts: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    
    # MFA
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    
    # Metadata
    full_name: Optional[str] = None
    department: Optional[str] = None
    
    def get_permissions(self) -> Set[Permission]:
        """Get effective permissions for user."""
        permissions = set()
        
        # Add role permissions
        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))
        
        # Add custom permissions
        permissions.update(self.custom_permissions)
        
        # Remove denied permissions
        permissions -= self.denied_permissions
        
        return permissions
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        if not self.is_active or self.is_locked:
            return False
        return permission in self.get_permissions()
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the permissions."""
        user_perms = self.get_permissions()
        return any(p in user_perms for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the permissions."""
        user_perms = self.get_permissions()
        return all(p in user_perms for p in permissions)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        result = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [r.value for r in self.roles],
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "mfa_enabled": self.mfa_enabled,
            "full_name": self.full_name,
            "department": self.department,
        }
        
        if include_sensitive:
            result["permissions"] = [p.name for p in self.get_permissions()]
        
        return result


@dataclass
class Session:
    """User session."""
    session_id: str
    user_id: str
    access_token: str
    refresh_token: str
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    refresh_expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at
    
    def is_refresh_expired(self) -> bool:
        """Check if refresh token is expired."""
        return datetime.now() > self.refresh_expires_at


class AuthManager:
    """
    Authentication Manager.
    
    Handles user authentication, token management, and access control.
    """
    
    def __init__(
        self,
        secret_key: str = "your-secret-key-change-in-production",
        access_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 604800,  # 7 days
        max_failed_attempts: int = 5,
        lockout_duration: int = 1800,  # 30 minutes
    ):
        """
        Initialize Auth Manager.
        
        Args:
            secret_key: Secret key for JWT signing
            access_token_expiry: Access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
            max_failed_attempts: Max failed login attempts before lockout
            lockout_duration: Lockout duration in seconds
        """
        self.secret_key = secret_key
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._username_to_id: Dict[str, str] = {}
        self._email_to_id: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin = User(
            user_id="admin-001",
            username="admin",
            email="admin@system.local",
            password_hash=self._hash_password("admin123"),  # Change in production!
            roles=[Role.SUPER_ADMIN],
            full_name="System Administrator",
        )
        self._users[admin.user_id] = admin
        self._username_to_id[admin.username] = admin.user_id
        self._email_to_id[admin.email] = admin.user_id
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        
        return f"{salt}:{hashed}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hashed = password_hash.split(":")
            new_hash = self._hash_password(password, salt)
            return new_hash == password_hash
        except Exception:
            return False
    
    def _generate_token(self, user: User, token_type: str = "access") -> str:
        """Generate JWT token."""
        now = datetime.utcnow()
        
        if token_type == "access":
            expiry = now + timedelta(seconds=self.access_token_expiry)
        else:
            expiry = now + timedelta(seconds=self.refresh_token_expiry)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "type": token_type,
            "iat": now,
            "exp": expiry,
            "jti": secrets.token_hex(16),
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[Role],
        full_name: Optional[str] = None,
        department: Optional[str] = None,
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: Email address
            password: Password
            roles: User roles
            full_name: Full name
            department: Department
            
        Returns:
            Created user
        """
        # Check uniqueness
        if username in self._username_to_id:
            raise ValueError(f"Username {username} already exists")
        if email in self._email_to_id:
            raise ValueError(f"Email {email} already exists")
        
        user_id = f"user-{secrets.token_hex(8)}"
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            roles=roles,
            full_name=full_name,
            department=department,
            password_changed_at=datetime.now(),
        )
        
        self._users[user_id] = user
        self._username_to_id[username] = user_id
        self._email_to_id[email] = user_id
        
        return user
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[Session]:
        """
        Authenticate user and create session.
        
        Args:
            username: Username or email
            password: Password
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            Session if authenticated, None otherwise
        """
        # Find user
        user_id = self._username_to_id.get(username) or self._email_to_id.get(username)
        if not user_id:
            return None
        
        user = self._users.get(user_id)
        if not user:
            return None
        
        # Check if locked
        if user.is_locked:
            return None
        
        # Check if active
        if not user.is_active:
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.is_locked = True
            
            return None
        
        # Reset failed attempts
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create session
        session = Session(
            session_id=secrets.token_hex(16),
            user_id=user.user_id,
            access_token=self._generate_token(user, "access"),
            refresh_token=self._generate_token(user, "refresh"),
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        self._sessions[session.session_id] = session
        
        return session
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate access token and return user.
        
        Args:
            token: JWT access token
            
        Returns:
            User if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            if payload.get("type") != "access":
                return None
            
            user_id = payload.get("user_id")
            user = self._users.get(user_id)
            
            if not user or not user.is_active or user.is_locked:
                return None
            
            return user
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_session(self, refresh_token: str) -> Optional[Session]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New session if valid, None otherwise
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=["HS256"])
            
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("user_id")
            user = self._users.get(user_id)
            
            if not user or not user.is_active or user.is_locked:
                return None
            
            # Create new session
            session = Session(
                session_id=secrets.token_hex(16),
                user_id=user.user_id,
                access_token=self._generate_token(user, "access"),
                refresh_token=self._generate_token(user, "refresh"),
            )
            
            self._sessions[session.session_id] = session
            
            return session
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def logout(self, session_id: str) -> bool:
        """Invalidate session."""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False
            return True
        return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_id = self._username_to_id.get(username)
        return self._users.get(user_id) if user_id else None
    
    def update_user_roles(self, user_id: str, roles: List[Role]) -> bool:
        """Update user roles."""
        user = self._users.get(user_id)
        if user:
            user.roles = roles
            user.updated_at = datetime.now()
            return True
        return False
    
    def add_permission(self, user_id: str, permission: Permission) -> bool:
        """Add custom permission to user."""
        user = self._users.get(user_id)
        if user:
            user.custom_permissions.add(permission)
            user.updated_at = datetime.now()
            return True
        return False
    
    def remove_permission(self, user_id: str, permission: Permission) -> bool:
        """Remove/deny permission from user."""
        user = self._users.get(user_id)
        if user:
            user.denied_permissions.add(permission)
            user.updated_at = datetime.now()
            return True
        return False
    
    def unlock_user(self, user_id: str) -> bool:
        """Unlock a locked user account."""
        user = self._users.get(user_id)
        if user:
            user.is_locked = False
            user.failed_login_attempts = 0
            user.updated_at = datetime.now()
            return True
        return False
    
    def change_password(self, user_id: str, new_password: str) -> bool:
        """Change user password."""
        user = self._users.get(user_id)
        if user:
            user.password_hash = self._hash_password(new_password)
            user.password_changed_at = datetime.now()
            user.updated_at = datetime.now()
            return True
        return False
    
    def list_users(
        self,
        role: Optional[Role] = None,
        department: Optional[str] = None,
        active_only: bool = True,
    ) -> List[User]:
        """List users with optional filtering."""
        users = list(self._users.values())
        
        if role:
            users = [u for u in users if role in u.roles]
        
        if department:
            users = [u for u in users if u.department == department]
        
        if active_only:
            users = [u for u in users if u.is_active and not u.is_locked]
        
        return users


def require_permission(permission: Permission):
    """Decorator to require permission for function access."""
    def decorator(func):
        def wrapper(self, *args, user: User = None, **kwargs):
            if user is None:
                raise PermissionError("Authentication required")
            if not user.has_permission(permission):
                raise PermissionError(f"Permission denied: {permission.name}")
            return func(self, *args, user=user, **kwargs)
        return wrapper
    return decorator

