"""
Worst-Case Execution Time (WCET) analyzer for EV autonomous trucking.

Implements WCET analysis for timing constraint validation in safety-critical
autonomous driving applications.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass
from loguru import logger


@dataclass
class WCETConfig:
    """Configuration for WCET analysis."""
    analysis_enabled: bool = True
    nominal_bound_ms: float = 25.0
    acceptable_bound_ms: float = 32.0
    critical_bound_ms: float = 50.0
    analysis_window: int = 100  # frames
    confidence_threshold: float = 0.95


@dataclass
class WCETResult:
    """Result of WCET analysis."""
    estimated_wcet: float
    confidence: float
    bound_status: str  # nominal, acceptable, critical, violation
    analysis_quality: float
    recommendations: List[str]


class WCETAnalyzer:
    """
    Worst-Case Execution Time analyzer for safety-critical systems.
    
    Features:
    - Real-time WCET estimation
    - Statistical analysis of execution times
    - Bound violation detection
    - Performance recommendations
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """Initialize the WCET analyzer."""
        self.config = WCETConfig(**config)
        self.device = device
        
        # Execution time history
        self.execution_times = []
        self.operation_times = {}
        
        # Statistical models
        self.statistical_model = None
        self.wcet_estimates = {}
        
        # Analysis state
        self.analysis_running = False
        self.last_analysis_time = 0.0
        
        logger.info("⏱️ WCET analyzer initialized")
        
    async def initialize(self):
        """Initialize the WCET analyzer."""
        logger.info("🎯 Initializing WCET analyzer...")
        
        # Initialize statistical models
        await self._init_statistical_models()
        
        # Start background analysis
        if self.config.analysis_enabled:
            asyncio.create_task(self._background_analysis())
            
        logger.info("✅ WCET analyzer initialized")
        
    async def _init_statistical_models(self):
        """Initialize statistical models for WCET analysis."""
        logger.info("📊 Initializing statistical models...")
        
        # Initialize execution time distributions
        self.execution_distributions = {
            'perception': {'mean': 0.0, 'std': 0.0, 'samples': []},
            'fusion': {'mean': 0.0, 'std': 0.0, 'samples': []},
            'safety': {'mean': 0.0, 'std': 0.0, 'samples': []},
            'total': {'mean': 0.0, 'std': 0.0, 'samples': []}
        }
        
        # Initialize WCET estimation models
        self.wcet_models = {
            'gaussian': self._init_gaussian_model(),
            'extreme_value': self._init_extreme_value_model(),
            'empirical': self._init_empirical_model()
        }
        
        logger.info("✅ Statistical models initialized")
        
    def _init_gaussian_model(self) -> Dict:
        """Initialize Gaussian model for WCET estimation."""
        return {
            'type': 'gaussian',
            'parameters': {'mean': 0.0, 'std': 0.0},
            'confidence_intervals': {}
        }
        
    def _init_extreme_value_model(self) -> Dict:
        """Initialize extreme value model for WCET estimation."""
        return {
            'type': 'extreme_value',
            'parameters': {'location': 0.0, 'scale': 0.0, 'shape': 0.0},
            'confidence_intervals': {}
        }
        
    def _init_empirical_model(self) -> Dict:
        """Initialize empirical model for WCET estimation."""
        return {
            'type': 'empirical',
            'percentiles': {},
            'confidence_intervals': {}
        }
        
    async def _background_analysis(self):
        """Background WCET analysis task."""
        while True:
            try:
                await self._perform_wcet_analysis()
                await asyncio.sleep(1.0)  # Analyze every second
            except Exception as e:
                logger.error(f"❌ Background WCET analysis error: {e}")
                await asyncio.sleep(1.0)
                
    async def _perform_wcet_analysis(self):
        """Perform WCET analysis on current data."""
        if len(self.execution_times) < 10:  # Need minimum samples
            return
            
        # Update statistical models
        await self._update_statistical_models()
        
        # Estimate WCET for each operation
        for operation in self.execution_distributions:
            wcet_estimate = await self._estimate_wcet(operation)
            self.wcet_estimates[operation] = wcet_estimate
            
        # Check for bound violations
        await self._check_bound_violations()
        
        self.last_analysis_time = asyncio.get_event_loop().time()
        
    async def _update_statistical_models(self):
        """Update statistical models with new execution times."""
        for operation, distribution in self.execution_distributions.items():
            if distribution['samples']:
                # Update mean and standard deviation
                samples = np.array(distribution['samples'])
                distribution['mean'] = float(np.mean(samples))
                distribution['std'] = float(np.std(samples))
                
                # Update WCET models
                await self._update_wcet_models(operation, samples)
                
    async def _update_wcet_models(self, operation: str, samples: np.ndarray):
        """Update WCET estimation models."""
        # Update Gaussian model
        gaussian_model = self.wcet_models['gaussian']
        gaussian_model['parameters']['mean'] = float(np.mean(samples))
        gaussian_model['parameters']['std'] = float(np.std(samples))
        
        # Update extreme value model
        extreme_model = self.wcet_models['extreme_value']
        # Simple extreme value parameter estimation
        extreme_model['parameters']['location'] = float(np.mean(samples))
        extreme_model['parameters']['scale'] = float(np.std(samples))
        extreme_model['parameters']['shape'] = 0.1  # Default shape parameter
        
        # Update empirical model
        empirical_model = self.wcet_models['empirical']
        empirical_model['percentiles'] = {
            'p50': float(np.percentile(samples, 50)),
            'p90': float(np.percentile(samples, 90)),
            'p95': float(np.percentile(samples, 95)),
            'p99': float(np.percentile(samples, 99)),
            'p99.9': float(np.percentile(samples, 99.9))
        }
        
    async def _estimate_wcet(self, operation: str) -> WCETResult:
        """Estimate WCET for a specific operation."""
        distribution = self.execution_distributions[operation]
        samples = distribution['samples']
        
        if len(samples) < 5:
            return WCETResult(
                estimated_wcet=0.0,
                confidence=0.0,
                bound_status="insufficient_data",
                analysis_quality=0.0,
                recommendations=["Need more execution samples"]
            )
            
        # Use multiple models for WCET estimation
        wcet_estimates = []
        
        # Gaussian model estimate (99.9th percentile)
        gaussian_model = self.wcet_models['gaussian']
        mean = gaussian_model['parameters']['mean']
        std = gaussian_model['parameters']['std']
        gaussian_wcet = mean + 3.09 * std  # 99.9th percentile
        wcet_estimates.append(gaussian_wcet)
        
        # Empirical model estimate (99.9th percentile)
        empirical_model = self.wcet_models['empirical']
        empirical_wcet = empirical_model['percentiles'].get('p99.9', 0.0)
        wcet_estimates.append(empirical_wcet)
        
        # Extreme value model estimate
        extreme_model = self.wcet_models['extreme_value']
        extreme_wcet = self._estimate_extreme_value_wcet(extreme_model, samples)
        wcet_estimates.append(extreme_wcet)
        
        # Use maximum estimate for safety
        estimated_wcet = max(wcet_estimates)
        
        # Determine bound status
        bound_status = self._determine_bound_status(estimated_wcet)
        
        # Calculate confidence
        confidence = self._calculate_confidence(samples, estimated_wcet)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(estimated_wcet, bound_status)
        
        return WCETResult(
            estimated_wcet=estimated_wcet,
            confidence=confidence,
            bound_status=bound_status,
            analysis_quality=self._calculate_analysis_quality(samples),
            recommendations=recommendations
        )
        
    def _estimate_extreme_value_wcet(self, model: Dict, samples: np.ndarray) -> float:
        """Estimate WCET using extreme value theory."""
        # Simple extreme value estimation
        # In production, use proper extreme value distribution fitting
        location = model['parameters']['location']
        scale = model['parameters']['scale']
        shape = model['parameters']['shape']
        
        # Gumbel distribution approximation
        if shape == 0:
            # Gumbel distribution
            wcet = location + scale * (-np.log(-np.log(0.999)))
        else:
            # Generalized extreme value distribution
            wcet = location + (scale / shape) * (np.power(-np.log(0.999), -shape) - 1)
            
        return float(wcet)
        
    def _determine_bound_status(self, wcet: float) -> str:
        """Determine WCET bound status."""
        if wcet <= self.config.nominal_bound_ms:
            return "nominal"
        elif wcet <= self.config.acceptable_bound_ms:
            return "acceptable"
        elif wcet <= self.config.critical_bound_ms:
            return "critical"
        else:
            return "violation"
            
    def _calculate_confidence(self, samples: np.ndarray, wcet: float) -> float:
        """Calculate confidence in WCET estimate."""
        # Simple confidence calculation based on sample size and variance
        n_samples = len(samples)
        variance = np.var(samples)
        
        # Confidence increases with sample size and decreases with variance
        confidence = min(1.0, n_samples / 100.0) * (1.0 / (1.0 + variance))
        
        return float(confidence)
        
    def _calculate_analysis_quality(self, samples: np.ndarray) -> float:
        """Calculate quality of WCET analysis."""
        n_samples = len(samples)
        
        # Quality based on sample size and distribution properties
        if n_samples < 10:
            return 0.0
        elif n_samples < 50:
            return 0.5
        elif n_samples < 100:
            return 0.8
        else:
            return 1.0
            
    def _generate_recommendations(self, wcet: float, bound_status: str) -> List[str]:
        """Generate recommendations based on WCET analysis."""
        recommendations = []
        
        if bound_status == "violation":
            recommendations.append("WCET exceeds critical bound - immediate optimization required")
            recommendations.append("Consider reducing model complexity or using faster hardware")
        elif bound_status == "critical":
            recommendations.append("WCET approaching critical bound - optimization recommended")
            recommendations.append("Monitor execution times closely")
        elif bound_status == "acceptable":
            recommendations.append("WCET within acceptable bounds")
            recommendations.append("Continue monitoring for performance degradation")
        else:
            recommendations.append("WCET within nominal bounds - good performance")
            
        return recommendations
        
    async def _check_bound_violations(self):
        """Check for WCET bound violations."""
        for operation, wcet_result in self.wcet_estimates.items():
            if wcet_result.bound_status in ["critical", "violation"]:
                logger.warning(f"⚠️ WCET violation in {operation}: {wcet_result.estimated_wcet:.2f}ms "
                             f"(bound: {wcet_result.bound_status})")
                
    async def analyze_wcet(self, operation: str) -> WCETResult:
        """Analyze WCET for a specific operation."""
        if operation not in self.wcet_estimates:
            return WCETResult(
                estimated_wcet=0.0,
                confidence=0.0,
                bound_status="no_data",
                analysis_quality=0.0,
                recommendations=["No execution data available"]
            )
            
        return self.wcet_estimates[operation]
        
    async def get_wcet_bounds(self) -> Dict:
        """Get WCET bounds for the system."""
        return {
            'nominal': self.config.nominal_bound_ms,
            'acceptable': self.config.acceptable_bound_ms,
            'critical': self.config.critical_bound_ms,
            'current_estimates': self.wcet_estimates
        }
        
    def record_execution_time(self, operation: str, execution_time_ms: float):
        """Record execution time for WCET analysis."""
        # Add to execution times
        self.execution_times.append(execution_time_ms)
        
        # Add to operation-specific times
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(execution_time_ms)
        
        # Add to distribution samples
        if operation in self.execution_distributions:
            self.execution_distributions[operation]['samples'].append(execution_time_ms)
            
            # Keep only recent samples
            max_samples = self.config.analysis_window
            if len(self.execution_distributions[operation]['samples']) > max_samples:
                self.execution_distributions[operation]['samples'].pop(0)
                
        # Keep only recent execution times
        if len(self.execution_times) > self.config.analysis_window:
            self.execution_times.pop(0)
            
    async def get_wcet_statistics(self) -> Dict:
        """Get WCET analysis statistics."""
        stats = {
            'total_executions': len(self.execution_times),
            'analysis_window': self.config.analysis_window,
            'last_analysis_time': self.last_analysis_time,
            'wcet_estimates': self.wcet_estimates,
            'execution_distributions': self.execution_distributions,
            'bound_violations': self._count_bound_violations()
        }
        
        return stats
        
    def _count_bound_violations(self) -> Dict:
        """Count WCET bound violations."""
        violations = {
            'nominal': 0,
            'acceptable': 0,
            'critical': 0,
            'violation': 0
        }
        
        for wcet_result in self.wcet_estimates.values():
            violations[wcet_result.bound_status] += 1
            
        return violations
        
    async def shutdown(self):
        """Shutdown the WCET analyzer."""
        logger.info("🛑 Shutting down WCET analyzer...")
        
        # Save final statistics
        final_stats = await self.get_wcet_statistics()
        logger.info(f"📊 Final WCET statistics: {final_stats}")
        
        logger.info("✅ WCET analyzer shutdown complete")
