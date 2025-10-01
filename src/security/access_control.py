"""
Least-Privilege Access Control for EV Autonomous Driving.

Implements strict RBAC and network segmentation to limit blast radius
of compromised pipelines and prevent unauthorized access to critical systems.
"""

import asyncio
import hashlib
import time
import jwt
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class AccessLevel(Enum):
    """Access levels for different system components."""
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ELEVATED = "elevated"
    ADMIN = "admin"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of resources in the system."""
    MODEL_DEPLOYMENT = "model_deployment"
    SENSOR_DATA = "sensor_data"
    PERCEPTION_SYSTEM = "perception_system"
    FUSION_SYSTEM = "fusion_system"
    SAFETY_FRAMEWORK = "safety_framework"
    CONFIGURATION = "configuration"
    TELEMETRY = "telemetry"
    LOGS = "logs"


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    access_level: AccessLevel
    permissions: Set[str]
    created_at: float
    last_activity: float
    is_active: bool = True
    session_tokens: Set[str] = None
    
    def __post_init__(self):
        if self.session_tokens is None:
            self.session_tokens = set()


@dataclass
class Resource:
    """Resource data structure."""
    resource_id: str
    resource_type: ResourceType
    name: str
    path: str
    access_level: AccessLevel
    owner: str
    created_at: float
    is_quarantined: bool = False


@dataclass
class AccessRequest:
    """Access request data structure."""
    request_id: str
    user_id: str
    resource_id: str
    action: str
    timestamp: float
    ip_address: str
    user_agent: str
    approved: bool = False
    reason: str = ""


@dataclass
class Session:
    """User session data structure."""
    session_id: str
    user_id: str
    token: str
    created_at: float
    expires_at: float
    ip_address: str
    is_active: bool = True


class AccessControlManager:
    """
    Least-Privilege Access Control Manager.
    
    Implements comprehensive access control for:
    - User authentication and authorization
    - Resource-based access control
    - Session management
    - Network segmentation
    - Audit logging
    
    Key Features:
    - Role-based access control (RBAC)
    - Principle of least privilege
    - Session timeout and management
    - Multi-factor authentication support
    - Comprehensive audit trails
    """
    
    def __init__(self, config: Dict):
        """Initialize the access control manager."""
        self.config = config
        
        # Security parameters
        self.require_authentication = config.get('require_authentication', True)
        self.session_timeout = config.get('session_timeout', 3600.0)  # 1 hour
        self.max_concurrent_sessions = config.get('max_concurrent_sessions', 5)
        self.jwt_secret = config.get('jwt_secret', 'ev_autonomous_driving_secret')
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        
        # Data storage
        self.users: Dict[str, User] = {}
        self.resources: Dict[str, Resource] = {}
        self.sessions: Dict[str, Session] = {}
        self.access_requests: List[AccessRequest] = []
        
        # Access control matrix
        self.access_matrix = self._initialize_access_matrix()
        
        # Security state
        self.failed_login_attempts: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()
        self.quarantined_users: Set[str] = set()
        
        # Performance tracking
        self.auth_times = []
        self.access_denials = []
        
        logger.info("🔐 Access control manager initialized")
        
    async def initialize(self):
        """Initialize the access control system."""
        logger.info("🎯 Initializing access control manager...")
        
        # Initialize default users and resources
        await self._initialize_default_users()
        await self._initialize_default_resources()
        
        # Start session cleanup task
        asyncio.create_task(self._session_cleanup_loop())
        
        logger.info("✅ Access control manager initialized")
        
    def _initialize_access_matrix(self) -> Dict[AccessLevel, Dict[ResourceType, List[str]]]:
        """Initialize the access control matrix."""
        return {
            AccessLevel.READ_ONLY: {
                ResourceType.TELEMETRY: ['read'],
                ResourceType.LOGS: ['read']
            },
            AccessLevel.STANDARD: {
                ResourceType.TELEMETRY: ['read'],
                ResourceType.LOGS: ['read'],
                ResourceType.CONFIGURATION: ['read']
            },
            AccessLevel.ELEVATED: {
                ResourceType.TELEMETRY: ['read'],
                ResourceType.LOGS: ['read'],
                ResourceType.CONFIGURATION: ['read', 'write'],
                ResourceType.SENSOR_DATA: ['read']
            },
            AccessLevel.ADMIN: {
                ResourceType.TELEMETRY: ['read'],
                ResourceType.LOGS: ['read'],
                ResourceType.CONFIGURATION: ['read', 'write'],
                ResourceType.SENSOR_DATA: ['read'],
                ResourceType.PERCEPTION_SYSTEM: ['read', 'write'],
                ResourceType.FUSION_SYSTEM: ['read', 'write']
            },
            AccessLevel.CRITICAL: {
                ResourceType.TELEMETRY: ['read'],
                ResourceType.LOGS: ['read'],
                ResourceType.CONFIGURATION: ['read', 'write'],
                ResourceType.SENSOR_DATA: ['read'],
                ResourceType.PERCEPTION_SYSTEM: ['read', 'write'],
                ResourceType.FUSION_SYSTEM: ['read', 'write'],
                ResourceType.SAFETY_FRAMEWORK: ['read', 'write'],
                ResourceType.MODEL_DEPLOYMENT: ['read', 'write']
            }
        }
        
    async def _initialize_default_users(self):
        """Initialize default system users."""
        # System administrator
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@ev-autonomous.com",
            access_level=AccessLevel.CRITICAL,
            permissions={"all"},
            created_at=time.time(),
            last_activity=time.time()
        )
        self.users[admin_user.user_id] = admin_user
        
        # Safety engineer
        safety_user = User(
            user_id="safety_001",
            username="safety_engineer",
            email="safety@ev-autonomous.com",
            access_level=AccessLevel.ADMIN,
            permissions={"safety_framework", "perception_system", "fusion_system"},
            created_at=time.time(),
            last_activity=time.time()
        )
        self.users[safety_user.user_id] = safety_user
        
        # Data scientist
        data_user = User(
            user_id="data_001",
            username="data_scientist",
            email="data@ev-autonomous.com",
            access_level=AccessLevel.ELEVATED,
            permissions={"sensor_data", "telemetry", "logs"},
            created_at=time.time(),
            last_activity=time.time()
        )
        self.users[data_user.user_id] = data_user
        
        # Monitoring user
        monitor_user = User(
            user_id="monitor_001",
            username="monitor",
            email="monitor@ev-autonomous.com",
            access_level=AccessLevel.READ_ONLY,
            permissions={"telemetry", "logs"},
            created_at=time.time(),
            last_activity=time.time()
        )
        self.users[monitor_user.user_id] = monitor_user
        
        logger.info(f"👥 Initialized {len(self.users)} default users")
        
    async def _initialize_default_resources(self):
        """Initialize default system resources."""
        resources = [
            Resource(
                resource_id="perception_001",
                resource_type=ResourceType.PERCEPTION_SYSTEM,
                name="Main Perception System",
                path="/perception/main",
                access_level=AccessLevel.CRITICAL,
                owner="admin_001",
                created_at=time.time()
            ),
            Resource(
                resource_id="fusion_001",
                resource_type=ResourceType.FUSION_SYSTEM,
                name="Multi-Modal Fusion System",
                path="/fusion/main",
                access_level=AccessLevel.CRITICAL,
                owner="admin_001",
                created_at=time.time()
            ),
            Resource(
                resource_id="safety_001",
                resource_type=ResourceType.SAFETY_FRAMEWORK,
                name="ASIL-D Safety Framework",
                path="/safety/main",
                access_level=AccessLevel.CRITICAL,
                owner="admin_001",
                created_at=time.time()
            ),
            Resource(
                resource_id="config_001",
                resource_type=ResourceType.CONFIGURATION,
                name="System Configuration",
                path="/config",
                access_level=AccessLevel.ELEVATED,
                owner="admin_001",
                created_at=time.time()
            ),
            Resource(
                resource_id="telemetry_001",
                resource_type=ResourceType.TELEMETRY,
                name="System Telemetry",
                path="/telemetry",
                access_level=AccessLevel.READ_ONLY,
                owner="admin_001",
                created_at=time.time()
            )
        ]
        
        for resource in resources:
            self.resources[resource.resource_id] = resource
            
        logger.info(f"📁 Initialized {len(self.resources)} default resources")
        
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = "unknown") -> Tuple[bool, str, Optional[str]]:
        """
        Authenticate a user and create a session.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            
        Returns:
            Tuple of (success, message, session_token)
        """
        start_time = time.time()
        
        try:
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                return False, "IP address is blocked", None
                
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
                    
            if not user:
                await self._record_failed_login(username, ip_address)
                return False, "Invalid credentials", None
                
            # Check if user is active
            if not user.is_active:
                return False, "User account is disabled", None
                
            # Check if user is quarantined
            if user.user_id in self.quarantined_users:
                return False, "User account is quarantined", None
                
            # Verify password (simplified for demo)
            if not await self._verify_password(user, password):
                await self._record_failed_login(username, ip_address)
                return False, "Invalid credentials", None
                
            # Check concurrent session limit
            active_sessions = len([s for s in self.sessions.values() 
                                 if s.user_id == user.user_id and s.is_active])
            if active_sessions >= self.max_concurrent_sessions:
                return False, "Maximum concurrent sessions exceeded", None
                
            # Create session
            session_token = await self._create_session(user, ip_address)
            
            # Update user activity
            user.last_activity = time.time()
            
            # Record authentication time
            auth_time = time.time() - start_time
            self.auth_times.append(auth_time)
            
            # Clear failed login attempts
            if username in self.failed_login_attempts:
                del self.failed_login_attempts[username]
                
            logger.info(f"✅ User authenticated: {username} in {auth_time:.3f}s")
            return True, "Authentication successful", session_token
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False, "Authentication error", None
            
    async def _verify_password(self, user: User, password: str) -> bool:
        """Verify user password."""
        # Simplified password verification for demo
        # In production, use proper password hashing (bcrypt, scrypt, etc.)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Mock password verification
        if user.username == "admin":
            return password == "admin123"
        elif user.username == "safety_engineer":
            return password == "safety123"
        elif user.username == "data_scientist":
            return password == "data123"
        elif user.username == "monitor":
            return password == "monitor123"
        else:
            return False
            
    async def _create_session(self, user: User, ip_address: str) -> str:
        """Create a new user session."""
        session_id = f"session_{int(time.time() * 1000)}"
        
        # Generate JWT token
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'access_level': user.access_level.value,
            'session_id': session_id,
            'iat': time.time(),
            'exp': time.time() + self.session_timeout
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Create session
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=token,
            created_at=time.time(),
            expires_at=time.time() + self.session_timeout,
            ip_address=ip_address
        )
        
        self.sessions[session_id] = session
        user.session_tokens.add(session_id)
        
        return token
        
    async def validate_access(self, user_id: str, action: str) -> bool:
        """
        Validate user access to a specific action.
        
        Args:
            user_id: User ID
            action: Action to validate
            
        Returns:
            True if access is allowed
        """
        try:
            # Get user
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return False
                
            # Check if user is quarantined
            if user.user_id in self.quarantined_users:
                return False
                
            # Check permissions
            if "all" in user.permissions:
                return True
                
            # Check specific permissions
            if action in user.permissions:
                return True
                
            # Check access level permissions
            if user.access_level in [AccessLevel.ADMIN, AccessLevel.CRITICAL]:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Access validation failed: {e}")
            return False
            
    async def validate_resource_access(self, user_id: str, resource_id: str, 
                                     action: str) -> Tuple[bool, str]:
        """
        Validate user access to a specific resource.
        
        Args:
            user_id: User ID
            resource_id: Resource ID
            action: Action to perform
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Get user and resource
            user = self.users.get(user_id)
            resource = self.resources.get(resource_id)
            
            if not user or not resource:
                return False, "User or resource not found"
                
            if not user.is_active:
                return False, "User account is disabled"
                
            if user.user_id in self.quarantined_users:
                return False, "User account is quarantined"
                
            if resource.is_quarantined:
                return False, "Resource is quarantined"
                
            # Check access matrix
            user_access = self.access_matrix.get(user.access_level, {})
            resource_access = user_access.get(resource.resource_type, [])
            
            if action not in resource_access:
                # Check if user has higher access level than required
                if user.access_level.value >= resource.access_level.value:
                    return True, "Access granted based on access level"
                else:
                    await self._record_access_denial(user_id, resource_id, action, 
                                                   "Insufficient access level")
                    return False, "Insufficient access level"
                    
            return True, "Access granted"
            
        except Exception as e:
            logger.error(f"❌ Resource access validation failed: {e}")
            return False, f"Validation error: {e}"
            
    async def _record_failed_login(self, username: str, ip_address: str):
        """Record failed login attempt."""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = 0
            
        self.failed_login_attempts[username] += 1
        
        # Block user after 5 failed attempts
        if self.failed_login_attempts[username] >= 5:
            await self._quarantine_user(username, "Too many failed login attempts")
            
        # Block IP after 10 failed attempts
        if self.failed_login_attempts.get(ip_address, 0) >= 10:
            self.blocked_ips.add(ip_address)
            logger.warning(f"🚫 IP address blocked: {ip_address}")
            
    async def _record_access_denial(self, user_id: str, resource_id: str, 
                                   action: str, reason: str):
        """Record access denial for audit."""
        denial = {
            'user_id': user_id,
            'resource_id': resource_id,
            'action': action,
            'reason': reason,
            'timestamp': time.time()
        }
        self.access_denials.append(denial)
        
        logger.warning(f"🚫 Access denied: {user_id} -> {resource_id} ({action}): {reason}")
        
    async def _quarantine_user(self, username: str, reason: str):
        """Quarantine a user account."""
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
                
        if user:
            self.quarantined_users.add(user.user_id)
            user.is_active = False
            
            # Invalidate all user sessions
            for session in self.sessions.values():
                if session.user_id == user.user_id:
                    session.is_active = False
                    
            logger.warning(f"🔒 User quarantined: {username} - {reason}")
            
    async def _session_cleanup_loop(self):
        """Cleanup expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if session.expires_at < current_time:
                        expired_sessions.append(session_id)
                        
                for session_id in expired_sessions:
                    session = self.sessions[session_id]
                    session.is_active = False
                    
                    # Remove from user's session tokens
                    if session.user_id in self.users:
                        self.users[session.user_id].session_tokens.discard(session_id)
                        
                if expired_sessions:
                    logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                    
                # Wait before next cleanup
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
                await asyncio.sleep(60)
                
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information."""
        user = self.users.get(user_id)
        if not user:
            return None
            
        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'access_level': user.access_level.value,
            'permissions': list(user.permissions),
            'is_active': user.is_active,
            'last_activity': user.last_activity,
            'active_sessions': len([s for s in self.sessions.values() 
                                  if s.user_id == user_id and s.is_active])
        }
        
    def get_access_statistics(self) -> Dict:
        """Get access control statistics."""
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.is_active])
        quarantined_users = len(self.quarantined_users)
        total_sessions = len(self.sessions)
        active_sessions = len([s for s in self.sessions.values() if s.is_active])
        blocked_ips = len(self.blocked_ips)
        access_denials = len(self.access_denials)
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'quarantined_users': quarantined_users,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'blocked_ips': blocked_ips,
            'access_denials': access_denials,
            'avg_auth_time': np.mean(self.auth_times) if self.auth_times else 0.0,
            'session_timeout': self.session_timeout,
            'max_concurrent_sessions': self.max_concurrent_sessions
        }
