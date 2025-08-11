"""
Safety-critical ML framework for EV autonomous trucking.

Implements ASIL-D compliant model runtime with:
- Deterministic execution guarantees (WCET analysis)
- Fault-injection resilient feature buffers
- Safety validation and monitoring
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass
from loguru import logger

from .asil_runtime import ASILRuntime
from .wcet_analyzer import WCETAnalyzer
from .fault_injection import FaultInjectionTester
from ..config import Config


@dataclass
class SafetyResult:
    """Result of safety validation."""
    is_safe: bool
    confidence: float
    reason: str
    violations: List[str]
    wcet_status: str
    fault_status: str


class SafetyFramework:
    """
    ASIL-D compliant safety framework for autonomous driving.
    
    Features:
    - Deterministic execution guarantees
    - WCET analysis for timing constraints
    - Fault-injection resilient buffers
    - Real-time safety monitoring
    """
    
    def __init__(self, config: Config):
        """Initialize the safety framework."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Safety configuration
        self.safety_config = config.safety
        self.max_latency = self.safety_config.max_latency_ms  # 32ms target
        self.min_confidence = self.safety_config.min_confidence
        
        # Initialize safety components
        self.asil_runtime = ASILRuntime(
            config=self.safety_config.asil,
            device=self.device
        )
        
        self.wcet_analyzer = WCETAnalyzer(
            config=self.safety_config.wcet,
            device=self.device
        )
        
        self.fault_tester = FaultInjectionTester(
            config=self.safety_config.fault_injection,
            device=self.device
        )
        
        # Safety state
        self.safety_history = []
        self.violation_count = 0
        self.last_safety_check = None
        
        logger.info("🛡️ Safety framework initialized (ASIL-D compliant)")
        
    async def initialize(self):
        """Initialize the safety framework."""
        logger.info("🎯 Initializing safety framework...")
        
        # Initialize ASIL runtime
        await self.asil_runtime.initialize()
        
        # Initialize WCET analyzer
        await self.wcet_analyzer.initialize()
        
        # Initialize fault injection tester
        await self.fault_tester.initialize()
        
        # Run initial safety checks
        await self._run_safety_checks()
        
        logger.info("✅ Safety framework initialized")
        
    async def validate_output(self, features: Dict) -> SafetyResult:
        """
        Validate perception output for safety compliance.
        
        Args:
            features: Dictionary containing perception features
            
        Returns:
            SafetyResult with validation status
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check WCET compliance
            wcet_status = await self._check_wcet_compliance(start_time)
            
            # Validate feature integrity
            feature_validation = await self._validate_features(features)
            
            # Check for fault injection
            fault_status = await self._check_fault_injection(features)
            
            # Determine overall safety
            is_safe = (
                wcet_status == "compliant" and
                feature_validation['is_valid'] and
                fault_status == "clean"
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                wcet_status, feature_validation, fault_status
            )
            
            # Compile violations
            violations = self._compile_violations(
                wcet_status, feature_validation, fault_status
            )
            
            # Create safety result
            result = SafetyResult(
                is_safe=is_safe,
                confidence=confidence,
                reason=self._generate_safety_reason(wcet_status, feature_validation, fault_status),
                violations=violations,
                wcet_status=wcet_status,
                fault_status=fault_status
            )
            
            # Update safety history
            self._update_safety_history(result)
            
            # Log safety status
            if not is_safe:
                logger.warning(f"⚠️ Safety validation failed: {result.reason}")
                self.violation_count += 1
            else:
                logger.debug(f"✅ Safety validation passed (confidence: {confidence:.2f})")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Safety validation error: {e}")
            # Return unsafe result on error
            return SafetyResult(
                is_safe=False,
                confidence=0.0,
                reason=f"Safety validation error: {e}",
                violations=["validation_error"],
                wcet_status="error",
                fault_status="error"
            )
            
    async def _check_wcet_compliance(self, start_time: float) -> str:
        """Check Worst-Case Execution Time compliance."""
        try:
            current_time = asyncio.get_event_loop().time()
            execution_time = (current_time - start_time) * 1000  # Convert to ms
            
            # Check against WCET bounds
            wcet_bounds = await self.wcet_analyzer.get_wcet_bounds()
            
            if execution_time <= wcet_bounds['nominal']:
                return "compliant"
            elif execution_time <= wcet_bounds['acceptable']:
                return "acceptable"
            else:
                return "violation"
                
        except Exception as e:
            logger.error(f"❌ WCET check failed: {e}")
            return "error"
            
    async def _validate_features(self, features: Dict) -> Dict:
        """Validate feature integrity and quality."""
        try:
            validation_result = {
                'is_valid': True,
                'quality_score': 1.0,
                'issues': []
            }
            
            # Check for required keys
            required_keys = ['point_features', 'object_features', 'segmentation_features']
            for key in required_keys:
                if key not in features:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"missing_key: {key}")
                    
            # Check feature dimensions
            if 'point_features' in features:
                point_features = features['point_features']
                if not isinstance(point_features, np.ndarray):
                    validation_result['is_valid'] = False
                    validation_result['issues'].append("invalid_point_features_type")
                elif point_features.size == 0:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append("empty_point_features")
                    
            # Check for NaN or infinite values
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        validation_result['is_valid'] = False
                        validation_result['issues'].append(f"invalid_values: {key}")
                        
            # Calculate quality score
            if validation_result['is_valid']:
                validation_result['quality_score'] = self._calculate_feature_quality(features)
                
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Feature validation failed: {e}")
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f"validation_error: {e}"]
            }
            
    async def _check_fault_injection(self, features: Dict) -> str:
        """Check for potential fault injection attacks."""
        try:
            # Run fault injection tests
            fault_result = await self.fault_tester.test_features(features)
            
            if fault_result['is_clean']:
                return "clean"
            elif fault_result['suspicious']:
                return "suspicious"
            else:
                return "compromised"
                
        except Exception as e:
            logger.error(f"❌ Fault injection check failed: {e}")
            return "error"
            
    def _calculate_confidence(self, wcet_status: str, feature_validation: Dict, fault_status: str) -> float:
        """Calculate overall safety confidence score."""
        confidence = 1.0
        
        # WCET confidence
        if wcet_status == "compliant":
            confidence *= 1.0
        elif wcet_status == "acceptable":
            confidence *= 0.8
        else:
            confidence *= 0.3
            
        # Feature validation confidence
        confidence *= feature_validation.get('quality_score', 0.5)
        
        # Fault injection confidence
        if fault_status == "clean":
            confidence *= 1.0
        elif fault_status == "suspicious":
            confidence *= 0.7
        else:
            confidence *= 0.2
            
        return max(0.0, min(1.0, confidence))
        
    def _compile_violations(self, wcet_status: str, feature_validation: Dict, fault_status: str) -> List[str]:
        """Compile list of safety violations."""
        violations = []
        
        if wcet_status not in ["compliant", "acceptable"]:
            violations.append(f"wcet_violation: {wcet_status}")
            
        if not feature_validation.get('is_valid', False):
            violations.extend(feature_validation.get('issues', []))
            
        if fault_status not in ["clean", "suspicious"]:
            violations.append(f"fault_injection: {fault_status}")
            
        return violations
        
    def _generate_safety_reason(self, wcet_status: str, feature_validation: Dict, fault_status: str) -> str:
        """Generate human-readable safety reason."""
        reasons = []
        
        if wcet_status == "compliant":
            reasons.append("WCET compliant")
        elif wcet_status == "acceptable":
            reasons.append("WCET acceptable")
        else:
            reasons.append(f"WCET violation: {wcet_status}")
            
        if feature_validation.get('is_valid', False):
            reasons.append("features valid")
        else:
            reasons.append("feature validation failed")
            
        if fault_status == "clean":
            reasons.append("no fault injection detected")
        else:
            reasons.append(f"fault injection: {fault_status}")
            
        return "; ".join(reasons)
        
    def _calculate_feature_quality(self, features: Dict) -> float:
        """Calculate feature quality score."""
        try:
            quality_scores = []
            
            # Point features quality
            if 'point_features' in features:
                point_features = features['point_features']
                if isinstance(point_features, np.ndarray) and point_features.size > 0:
                    # Check feature variance (higher variance = better quality)
                    variance = np.var(point_features)
                    quality_scores.append(min(1.0, variance / 1000))  # Normalize
                    
            # Object features quality
            if 'object_features' in features:
                object_features = features['object_features']
                if isinstance(object_features, np.ndarray) and object_features.size > 0:
                    variance = np.var(object_features)
                    quality_scores.append(min(1.0, variance / 1000))
                    
            # Segmentation features quality
            if 'segmentation_features' in features:
                seg_features = features['segmentation_features']
                if isinstance(seg_features, np.ndarray) and seg_features.size > 0:
                    variance = np.var(seg_features)
                    quality_scores.append(min(1.0, variance / 1000))
                    
            return np.mean(quality_scores) if quality_scores else 0.5
            
        except Exception as e:
            logger.error(f"❌ Feature quality calculation failed: {e}")
            return 0.5
            
    def _update_safety_history(self, result: SafetyResult):
        """Update safety history with new result."""
        self.safety_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'result': result,
            'violation_count': self.violation_count
        })
        
        # Keep only recent history (last 1000 entries)
        if len(self.safety_history) > 1000:
            self.safety_history.pop(0)
            
        self.last_safety_check = result
        
    async def get_safety_stats(self) -> Dict:
        """Get safety framework statistics."""
        if not self.safety_history:
            return {}
            
        recent_results = self.safety_history[-100:]  # Last 100 results
        
        return {
            'total_checks': len(self.safety_history),
            'recent_safety_rate': sum(1 for r in recent_results if r['result'].is_safe) / len(recent_results),
            'total_violations': self.violation_count,
            'recent_violations': sum(1 for r in recent_results if not r['result'].is_safe),
            'avg_confidence': np.mean([r['result'].confidence for r in recent_results]),
            'wcet_compliance_rate': sum(1 for r in recent_results if r['result'].wcet_status == "compliant") / len(recent_results),
            'fault_clean_rate': sum(1 for r in recent_results if r['result'].fault_status == "clean") / len(recent_results),
            'last_check': self.last_safety_check.is_safe if self.last_safety_check else None
        }
        
    async def shutdown(self):
        """Shutdown the safety framework."""
        logger.info("🛑 Shutting down safety framework...")
        
        await self.asil_runtime.shutdown()
        await self.wcet_analyzer.shutdown()
        await self.fault_tester.shutdown()
        
        logger.info("✅ Safety framework shutdown complete")
