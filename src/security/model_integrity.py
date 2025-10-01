"""
Model Artifact Signing and Attestation for EV Autonomous Driving.

Implements cryptographic signing and provenance tracking for all model artifacts
to prevent unauthorized adapter uploads and supply-chain attacks.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import torch
import numpy as np
from loguru import logger


@dataclass
class ModelAttestation:
    """Model attestation data structure."""
    model_id: str
    model_hash: str
    signature: str
    timestamp: float
    signer_id: str
    metadata: Dict[str, Any]
    is_valid: bool = False
    validation_reason: str = ""


@dataclass
class ModelRegistry:
    """Model registry entry."""
    model_id: str
    model_path: str
    model_hash: str
    attestation: ModelAttestation
    created_at: float
    last_updated: float
    access_level: str
    is_quarantined: bool = False


class ModelIntegrityManager:
    """
    Model Artifact Signing and Attestation Manager.
    
    Provides cryptographic signing and provenance tracking for all model artifacts
    to prevent unauthorized adapter uploads and supply-chain attacks.
    
    Key Features:
    - Cryptographic signing of model weights and adapters
    - Model registry with provenance tracking
    - Supply-chain integrity validation
    - Automated attestation verification
    """
    
    def __init__(self, config: Dict):
        """Initialize the model integrity manager."""
        self.config = config
        
        # Cryptographic keys
        self.private_key = None
        self.public_key = None
        self.signing_key_id = config.get('signing_key_id', 'ev_autonomous_driving')
        
        # Model registry
        self.model_registry: Dict[str, ModelRegistry] = {}
        self.trusted_signing_keys: Dict[str, str] = {}
        
        # Security parameters
        self.attestation_timeout = config.get('attestation_timeout', 30.0)
        self.require_signing = config.get('require_signing', True)
        self.auto_quarantine = config.get('auto_quarantine', True)
        
        # Performance tracking
        self.attestation_times = []
        self.validation_failures = []
        
        logger.info("🔐 Model integrity manager initialized")
        
    async def initialize(self):
        """Initialize the model integrity manager."""
        logger.info("🎯 Initializing model integrity manager...")
        
        # Generate or load cryptographic keys
        await self._initialize_cryptographic_keys()
        
        # Load trusted signing keys
        await self._load_trusted_keys()
        
        # Initialize model registry
        await self._initialize_model_registry()
        
        logger.info("✅ Model integrity manager initialized")
        
    async def _initialize_cryptographic_keys(self):
        """Initialize cryptographic keys for signing."""
        try:
            # Generate RSA key pair for signing
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            logger.info("🔑 Cryptographic keys generated")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cryptographic keys: {e}")
            raise
            
    async def _load_trusted_keys(self):
        """Load trusted signing keys from configuration."""
        # In production, load from secure key management system
        self.trusted_signing_keys = {
            'ev_autonomous_driving': self.signing_key_id,
            'trusted_vendor_1': 'vendor_1_key_id',
            'trusted_vendor_2': 'vendor_2_key_id'
        }
        
        logger.info(f"🔑 Loaded {len(self.trusted_signing_keys)} trusted signing keys")
        
    async def _initialize_model_registry(self):
        """Initialize the model registry."""
        # In production, load from persistent storage
        self.model_registry = {}
        logger.info("📋 Model registry initialized")
        
    async def sign_model_artifact(self, model_path: str, metadata: Dict[str, Any]) -> ModelAttestation:
        """
        Sign a model artifact with cryptographic signature.
        
        Args:
            model_path: Path to the model artifact
            metadata: Model metadata (version, author, etc.)
            
        Returns:
            Model attestation with signature
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"🔐 Signing model artifact: {model_path}")
            
            # Calculate model hash
            model_hash = await self._calculate_model_hash(model_path)
            
            # Create model ID
            model_id = f"model_{hashlib.sha256(model_path.encode()).hexdigest()[:16]}"
            
            # Create attestation data
            attestation_data = {
                'model_id': model_id,
                'model_hash': model_hash,
                'timestamp': asyncio.get_event_loop().time(),
                'signer_id': self.signing_key_id,
                'metadata': metadata
            }
            
            # Sign the attestation data
            signature = await self._sign_data(attestation_data)
            
            # Create attestation
            attestation = ModelAttestation(
                model_id=model_id,
                model_hash=model_hash,
                signature=signature,
                timestamp=attestation_data['timestamp'],
                signer_id=self.signing_key_id,
                metadata=metadata,
                is_valid=True,
                validation_reason="Self-signed"
            )
            
            # Record attestation time
            attestation_time = asyncio.get_event_loop().time() - start_time
            self.attestation_times.append(attestation_time)
            
            logger.info(f"✅ Model artifact signed: {model_id} in {attestation_time:.3f}s")
            return attestation
            
        except Exception as e:
            logger.error(f"❌ Failed to sign model artifact: {e}")
            raise
            
    async def validate_model(self, model_path: str) -> ModelAttestation:
        """
        Validate model artifact integrity and attestation.
        
        Args:
            model_path: Path to the model artifact
            
        Returns:
            Model attestation with validation results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"🔍 Validating model artifact: {model_path}")
            
            # Calculate current model hash
            current_hash = await self._calculate_model_hash(model_path)
            
            # Check if model is in registry
            model_id = f"model_{hashlib.sha256(model_path.encode()).hexdigest()[:16]}"
            
            if model_id not in self.model_registry:
                # Model not in registry - create invalid attestation
                attestation = ModelAttestation(
                    model_id=model_id,
                    model_hash=current_hash,
                    signature="",
                    timestamp=0.0,
                    signer_id="",
                    metadata={},
                    is_valid=False,
                    validation_reason="Model not found in registry"
                )
                
                if self.auto_quarantine:
                    await self._quarantine_model(model_id, "Not in registry")
                    
                return attestation
                
            # Get registry entry
            registry_entry = self.model_registry[model_id]
            
            # Validate hash integrity
            if current_hash != registry_entry.model_hash:
                attestation = ModelAttestation(
                    model_id=model_id,
                    model_hash=current_hash,
                    signature=registry_entry.attestation.signature,
                    timestamp=registry_entry.attestation.timestamp,
                    signer_id=registry_entry.attestation.signer_id,
                    metadata=registry_entry.attestation.metadata,
                    is_valid=False,
                    validation_reason="Hash mismatch - model may be corrupted"
                )
                
                if self.auto_quarantine:
                    await self._quarantine_model(model_id, "Hash mismatch")
                    
                return attestation
                
            # Validate signature
            signature_valid = await self._verify_signature(registry_entry.attestation)
            
            if not signature_valid:
                attestation = ModelAttestation(
                    model_id=model_id,
                    model_hash=current_hash,
                    signature=registry_entry.attestation.signature,
                    timestamp=registry_entry.attestation.timestamp,
                    signer_id=registry_entry.attestation.signer_id,
                    metadata=registry_entry.attestation.metadata,
                    is_valid=False,
                    validation_reason="Invalid signature"
                )
                
                if self.auto_quarantine:
                    await self._quarantine_model(model_id, "Invalid signature")
                    
                return attestation
                
            # Model is valid
            attestation = registry_entry.attestation
            attestation.is_valid = True
            attestation.validation_reason = "Validation passed"
            
            # Record validation time
            validation_time = asyncio.get_event_loop().time() - start_time
            self.attestation_times.append(validation_time)
            
            logger.info(f"✅ Model validation passed: {model_id} in {validation_time:.3f}s")
            return attestation
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            self.validation_failures.append({
                'model_path': model_path,
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Return invalid attestation
            return ModelAttestation(
                model_id="unknown",
                model_hash="",
                signature="",
                timestamp=0.0,
                signer_id="",
                metadata={},
                is_valid=False,
                validation_reason=f"Validation error: {e}"
            )
            
    async def register_model(self, model_path: str, attestation: ModelAttestation, 
                           access_level: str = "standard") -> bool:
        """
        Register a model in the model registry.
        
        Args:
            model_path: Path to the model artifact
            attestation: Model attestation
            access_level: Access level for the model
            
        Returns:
            True if registration successful
        """
        try:
            logger.info(f"📋 Registering model: {attestation.model_id}")
            
            # Create registry entry
            registry_entry = ModelRegistry(
                model_id=attestation.model_id,
                model_path=model_path,
                model_hash=attestation.model_hash,
                attestation=attestation,
                created_at=asyncio.get_event_loop().time(),
                last_updated=asyncio.get_event_loop().time(),
                access_level=access_level
            )
            
            # Add to registry
            self.model_registry[attestation.model_id] = registry_entry
            
            logger.info(f"✅ Model registered: {attestation.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            return False
            
    async def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of model artifact."""
        try:
            # For PyTorch models, calculate hash of state dict
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, dict) and 'state_dict' in model:
                    state_dict = model['state_dict']
                else:
                    state_dict = model
                    
                # Convert to bytes for hashing
                model_bytes = torch.save(state_dict, io.BytesIO())
                return hashlib.sha256(model_bytes.getvalue()).hexdigest()
            else:
                # For other file types, read file directly
                with open(model_path, 'rb') as f:
                    file_data = f.read()
                return hashlib.sha256(file_data).hexdigest()
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate model hash: {e}")
            return ""
            
    async def _sign_data(self, data: Dict[str, Any]) -> str:
        """Sign data with private key."""
        try:
            # Convert data to JSON string
            data_json = json.dumps(data, sort_keys=True)
            data_bytes = data_json.encode('utf-8')
            
            # Sign the data
            signature = self.private_key.sign(
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Return base64 encoded signature
            import base64
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ Failed to sign data: {e}")
            return ""
            
    async def _verify_signature(self, attestation: ModelAttestation) -> bool:
        """Verify signature of attestation."""
        try:
            # Check if signer is trusted
            if attestation.signer_id not in self.trusted_signing_keys:
                logger.warning(f"⚠️ Untrusted signer: {attestation.signer_id}")
                return False
                
            # Recreate attestation data
            attestation_data = {
                'model_id': attestation.model_id,
                'model_hash': attestation.model_hash,
                'timestamp': attestation.timestamp,
                'signer_id': attestation.signer_id,
                'metadata': attestation.metadata
            }
            
            # Convert to JSON string
            data_json = json.dumps(attestation_data, sort_keys=True)
            data_bytes = data_json.encode('utf-8')
            
            # Decode signature
            import base64
            signature = base64.b64decode(attestation.signature.encode('utf-8'))
            
            # Verify signature
            self.public_key.verify(
                signature,
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signature verification failed: {e}")
            return False
            
    async def _quarantine_model(self, model_id: str, reason: str):
        """Quarantine a model due to security concerns."""
        if model_id in self.model_registry:
            self.model_registry[model_id].is_quarantined = True
            logger.warning(f"🔒 Model quarantined: {model_id} - {reason}")
            
    def get_model_registry_status(self) -> Dict:
        """Get model registry status and statistics."""
        total_models = len(self.model_registry)
        quarantined_models = len([m for m in self.model_registry.values() if m.is_quarantined])
        valid_models = len([m for m in self.model_registry.values() if not m.is_quarantined])
        
        return {
            'total_models': total_models,
            'valid_models': valid_models,
            'quarantined_models': quarantined_models,
            'avg_attestation_time': np.mean(self.attestation_times) if self.attestation_times else 0.0,
            'validation_failures': len(self.validation_failures),
            'trusted_signing_keys': len(self.trusted_signing_keys)
        }
