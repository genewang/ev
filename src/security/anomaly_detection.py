"""
Anomaly Detection and Telemetry System for EV Autonomous Driving.

Monitors runtime model signals and detects deviations from baseline behavior
to identify potential security threats and system anomalies.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import torch
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    is_anomaly: bool
    severity: str
    anomaly_score: float
    anomaly_type: str
    confidence: float
    details: Dict[str, Any]
    timestamp: float


@dataclass
class TelemetryData:
    """Telemetry data structure."""
    component: str
    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class AnomalyDetector:
    """
    Anomaly Detection and Telemetry System.
    
    Monitors runtime behavior and detects:
    - Token distribution shifts
    - Perplexity changes
    - Output latency anomalies
    - Unusual API call patterns
    - Model behavior drift
    - Resource usage anomalies
    
    Key Features:
    - Real-time anomaly detection
    - Baseline behavior learning
    - Multi-metric analysis
    - Adaptive thresholds
    - Telemetry correlation
    """
    
    def __init__(self, config: Dict):
        """Initialize the anomaly detector."""
        self.config = config
        
        # Detection parameters
        self.baseline_window = config.get('baseline_window', 1000)
        self.anomaly_threshold = config.get('anomaly_threshold', 2.0)
        self.alert_threshold = config.get('alert_threshold', 3.0)
        self.update_frequency = config.get('update_frequency', 60.0)  # seconds
        
        # Telemetry storage
        self.telemetry_data: deque = deque(maxlen=10000)
        self.baseline_metrics = {}
        self.anomaly_models = {}
        
        # Detection state
        self.is_learning = True
        self.learning_samples = 0
        self.anomaly_count = 0
        self.last_anomaly_time = 0.0
        
        # Performance tracking
        self.detection_times = []
        self.false_positive_rate = 0.0
        
        logger.info("📊 Anomaly detector initialized")
        
    async def initialize(self):
        """Initialize the anomaly detection system."""
        logger.info("🎯 Initializing anomaly detector...")
        
        # Initialize baseline learning
        await self._initialize_baseline_learning()
        
        # Start continuous monitoring
        asyncio.create_task(self._continuous_monitoring_loop())
        
        logger.info("✅ Anomaly detector initialized")
        
    async def _initialize_baseline_learning(self):
        """Initialize baseline behavior learning."""
        # Initialize baseline metrics for different components
        self.baseline_metrics = {
            'perception_system': {
                'latency': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'confidence': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'throughput': {'mean': 0.0, 'std': 0.0, 'samples': []}
            },
            'fusion_system': {
                'latency': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'attention_weights': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'sensor_confidence': {'mean': 0.0, 'std': 0.0, 'samples': []}
            },
            'safety_framework': {
                'validation_time': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'threat_detection_rate': {'mean': 0.0, 'std': 0.0, 'samples': []},
                'quarantine_events': {'mean': 0.0, 'std': 0.0, 'samples': []}
            }
        }
        
        # Initialize anomaly detection models
        for component in self.baseline_metrics:
            self.anomaly_models[component] = {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'scaler': StandardScaler(),
                'is_trained': False
            }
            
    async def detect_anomalies(self, component: str, data: Dict[str, Any]) -> AnomalyResult:
        """
        Detect anomalies in component behavior.
        
        Args:
            component: Component being monitored
            data: Runtime data to analyze
            
        Returns:
            Anomaly detection result
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.debug(f"📊 Detecting anomalies in {component}")
            
            # Extract metrics from data
            metrics = self._extract_metrics(component, data)
            
            # Record telemetry
            telemetry = TelemetryData(
                component=component,
                timestamp=start_time,
                metrics=metrics,
                metadata=data.get('metadata', {})
            )
            self.telemetry_data.append(telemetry)
            
            # Update baseline if still learning
            if self.is_learning:
                await self._update_baseline(component, metrics)
                
            # Detect anomalies
            anomaly_result = await self._analyze_anomalies(component, metrics)
            
            # Record detection time
            detection_time = asyncio.get_event_loop().time() - start_time
            self.detection_times.append(detection_time)
            
            # Update anomaly count
            if anomaly_result.is_anomaly:
                self.anomaly_count += 1
                self.last_anomaly_time = start_time
                
            return anomaly_result
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return AnomalyResult(
                is_anomaly=True,
                severity='high',
                anomaly_score=1.0,
                anomaly_type='detection_error',
                confidence=1.0,
                details={'error': str(e)},
                timestamp=asyncio.get_event_loop().time()
            )
            
    def _extract_metrics(self, component: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant metrics from component data."""
        metrics = {}
        
        # Common metrics
        if 'latency' in data:
            metrics['latency'] = float(data['latency'])
        if 'confidence' in data:
            metrics['confidence'] = float(data['confidence'])
        if 'timestamp' in data:
            metrics['timestamp'] = float(data['timestamp'])
            
        # Component-specific metrics
        if component == 'perception_system':
            if 'num_detections' in data:
                metrics['num_detections'] = float(data['num_detections'])
            if 'feature_quality' in data:
                metrics['feature_quality'] = float(data['feature_quality'])
            if 'processing_time' in data:
                metrics['processing_time'] = float(data['processing_time'])
                
        elif component == 'fusion_system':
            if 'attention_weights' in data:
                weights = data['attention_weights']
                if isinstance(weights, (list, np.ndarray)):
                    metrics['attention_variance'] = float(np.var(weights))
                    metrics['attention_mean'] = float(np.mean(weights))
            if 'sensor_confidence' in data:
                conf = data['sensor_confidence']
                if isinstance(conf, dict):
                    metrics['min_sensor_confidence'] = float(min(conf.values()))
                    metrics['max_sensor_confidence'] = float(max(conf.values()))
                    
        elif component == 'safety_framework':
            if 'validation_time' in data:
                metrics['validation_time'] = float(data['validation_time'])
            if 'threats_detected' in data:
                metrics['threats_detected'] = float(data['threats_detected'])
            if 'quarantine_events' in data:
                metrics['quarantine_events'] = float(data['quarantine_events'])
                
        # Calculate derived metrics
        metrics.update(self._calculate_derived_metrics(component, data))
        
        return metrics
        
    def _calculate_derived_metrics(self, component: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate derived metrics for anomaly detection."""
        derived = {}
        
        # Throughput calculation
        if 'timestamp' in data and 'latency' in data:
            current_time = data['timestamp']
            latency = data['latency']
            
            # Calculate throughput (requests per second)
            if hasattr(self, '_last_timestamp') and hasattr(self, '_last_latency'):
                time_diff = current_time - self._last_timestamp
                if time_diff > 0:
                    derived['throughput'] = 1.0 / time_diff
                    
            self._last_timestamp = current_time
            self._last_latency = latency
            
        # Resource utilization
        if 'memory_usage' in data:
            derived['memory_utilization'] = float(data['memory_usage'])
        if 'cpu_usage' in data:
            derived['cpu_utilization'] = float(data['cpu_usage'])
            
        # Quality metrics
        if 'accuracy' in data:
            derived['accuracy'] = float(data['accuracy'])
        if 'precision' in data:
            derived['precision'] = float(data['precision'])
        if 'recall' in data:
            derived['recall'] = float(data['recall'])
            
        return derived
        
    async def _update_baseline(self, component: str, metrics: Dict[str, float]):
        """Update baseline metrics for a component."""
        if component not in self.baseline_metrics:
            return
            
        baseline = self.baseline_metrics[component]
        
        for metric_name, value in metrics.items():
            if metric_name in baseline:
                baseline[metric_name]['samples'].append(value)
                
                # Keep only recent samples
                if len(baseline[metric_name]['samples']) > self.baseline_window:
                    baseline[metric_name]['samples'] = baseline[metric_name]['samples'][-self.baseline_window:]
                    
                # Update statistics
                samples = baseline[metric_name]['samples']
                if len(samples) > 0:
                    baseline[metric_name]['mean'] = np.mean(samples)
                    baseline[metric_name]['std'] = np.std(samples)
                    
        # Check if we have enough samples to stop learning
        self.learning_samples += 1
        if self.learning_samples >= self.baseline_window:
            self.is_learning = False
            await self._train_anomaly_models()
            logger.info("📊 Baseline learning completed, anomaly detection active")
            
    async def _train_anomaly_models(self):
        """Train anomaly detection models on baseline data."""
        for component, baseline in self.baseline_metrics.items():
            # Prepare training data
            training_data = []
            for metric_name, stats in baseline.items():
                if len(stats['samples']) > 10:  # Need minimum samples
                    training_data.extend(stats['samples'])
                    
            if len(training_data) > 10:
                # Reshape for sklearn
                X = np.array(training_data).reshape(-1, 1)
                
                # Scale the data
                X_scaled = self.anomaly_models[component]['scaler'].fit_transform(X)
                
                # Train isolation forest
                self.anomaly_models[component]['isolation_forest'].fit(X_scaled)
                self.anomaly_models[component]['is_trained'] = True
                
                logger.info(f"📊 Anomaly model trained for {component}")
                
    async def _analyze_anomalies(self, component: str, metrics: Dict[str, float]) -> AnomalyResult:
        """Analyze metrics for anomalies."""
        if component not in self.baseline_metrics:
            return AnomalyResult(
                is_anomaly=False,
                severity='safe',
                anomaly_score=0.0,
                anomaly_type='none',
                confidence=1.0,
                details={},
                timestamp=asyncio.get_event_loop().time()
            )
            
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(component, metrics)
        
        # ML-based anomaly detection
        ml_anomalies = self._detect_ml_anomalies(component, metrics)
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(component, metrics)
        
        # Combine anomaly detection results
        combined_result = self._combine_anomaly_results(
            statistical_anomalies, ml_anomalies, pattern_anomalies
        )
        
        return combined_result
        
    def _detect_statistical_anomalies(self, component: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        anomalies = {}
        baseline = self.baseline_metrics.get(component, {})
        
        for metric_name, value in metrics.items():
            if metric_name in baseline and len(baseline[metric_name]['samples']) > 10:
                mean = baseline[metric_name]['mean']
                std = baseline[metric_name]['std']
                
                if std > 0:
                    # Z-score calculation
                    z_score = abs(value - mean) / std
                    
                    if z_score > self.anomaly_threshold:
                        anomalies[metric_name] = {
                            'z_score': z_score,
                            'value': value,
                            'mean': mean,
                            'std': std,
                            'severity': 'high' if z_score > self.alert_threshold else 'medium'
                        }
                        
        return anomalies
        
    def _detect_ml_anomalies(self, component: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies using ML models."""
        anomalies = {}
        
        if component in self.anomaly_models and self.anomaly_models[component]['is_trained']:
            try:
                # Prepare data for ML model
                values = list(metrics.values())
                if len(values) > 0:
                    X = np.array(values).reshape(1, -1)
                    X_scaled = self.anomaly_models[component]['scaler'].transform(X)
                    
                    # Predict anomaly
                    anomaly_score = self.anomaly_models[component]['isolation_forest'].decision_function(X_scaled)[0]
                    is_anomaly = self.anomaly_models[component]['isolation_forest'].predict(X_scaled)[0] == -1
                    
                    if is_anomaly:
                        anomalies['ml_anomaly'] = {
                            'anomaly_score': float(anomaly_score),
                            'is_anomaly': bool(is_anomaly),
                            'severity': 'high' if anomaly_score < -0.5 else 'medium'
                        }
                        
            except Exception as e:
                logger.error(f"❌ ML anomaly detection failed: {e}")
                
        return anomalies
        
    def _detect_pattern_anomalies(self, component: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies using pattern analysis."""
        anomalies = {}
        
        # Latency spike detection
        if 'latency' in metrics:
            latency = metrics['latency']
            if latency > 100.0:  # 100ms threshold
                anomalies['latency_spike'] = {
                    'latency': latency,
                    'threshold': 100.0,
                    'severity': 'high'
                }
                
        # Confidence drop detection
        if 'confidence' in metrics:
            confidence = metrics['confidence']
            if confidence < 0.5:  # 50% confidence threshold
                anomalies['confidence_drop'] = {
                    'confidence': confidence,
                    'threshold': 0.5,
                    'severity': 'medium'
                }
                
        # Throughput anomaly detection
        if 'throughput' in metrics:
            throughput = metrics['throughput']
            if throughput < 0.1:  # Very low throughput
                anomalies['throughput_drop'] = {
                    'throughput': throughput,
                    'threshold': 0.1,
                    'severity': 'medium'
                }
                
        return anomalies
        
    def _combine_anomaly_results(self, statistical: Dict, ml: Dict, pattern: Dict) -> AnomalyResult:
        """Combine different anomaly detection results."""
        all_anomalies = {**statistical, **ml, **pattern}
        
        if not all_anomalies:
            return AnomalyResult(
                is_anomaly=False,
                severity='safe',
                anomaly_score=0.0,
                anomaly_type='none',
                confidence=1.0,
                details={},
                timestamp=asyncio.get_event_loop().time()
            )
            
        # Determine overall severity
        severities = [anomaly.get('severity', 'low') for anomaly in all_anomalies.values()]
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        max_severity_level = max(severity_levels.get(s, 1) for s in severities)
        severity = [s for s, level in severity_levels.items() if level == max_severity_level][0]
        
        # Calculate anomaly score
        anomaly_scores = []
        for anomaly in all_anomalies.values():
            if 'z_score' in anomaly:
                anomaly_scores.append(min(anomaly['z_score'] / 5.0, 1.0))
            elif 'anomaly_score' in anomaly:
                anomaly_scores.append(abs(anomaly['anomaly_score']))
            else:
                anomaly_scores.append(0.5)
                
        avg_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        
        # Determine anomaly type
        anomaly_types = list(all_anomalies.keys())
        primary_type = anomaly_types[0] if anomaly_types else 'unknown'
        
        # Calculate confidence
        confidence = min(avg_anomaly_score * 1.2, 1.0)
        
        return AnomalyResult(
            is_anomaly=True,
            severity=severity,
            anomaly_score=avg_anomaly_score,
            anomaly_type=primary_type,
            confidence=confidence,
            details=all_anomalies,
            timestamp=asyncio.get_event_loop().time()
        )
        
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for system-wide anomalies."""
        while True:
            try:
                # Analyze recent telemetry data
                recent_data = list(self.telemetry_data)[-100:]  # Last 100 samples
                
                if len(recent_data) > 10:
                    # Detect system-wide anomalies
                    system_anomalies = await self._detect_system_anomalies(recent_data)
                    
                    if system_anomalies:
                        logger.warning(f"🚨 System-wide anomalies detected: {system_anomalies}")
                        
                # Update models periodically
                if not self.is_learning and len(self.telemetry_data) % 1000 == 0:
                    await self._retrain_models()
                    
                # Wait before next check
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"❌ Continuous monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _detect_system_anomalies(self, recent_data: List[TelemetryData]) -> List[str]:
        """Detect system-wide anomalies across components."""
        anomalies = []
        
        # Check for correlation between components
        components = set(data.component for data in recent_data)
        
        if len(components) > 1:
            # Check for simultaneous anomalies across components
            component_anomalies = {}
            for data in recent_data:
                if data.component not in component_anomalies:
                    component_anomalies[data.component] = []
                component_anomalies[data.component].append(data.metrics)
                
            # Analyze correlations
            for comp1, metrics1 in component_anomalies.items():
                for comp2, metrics2 in component_anomalies.items():
                    if comp1 != comp2 and len(metrics1) > 5 and len(metrics2) > 5:
                        correlation = self._calculate_correlation(metrics1, metrics2)
                        if abs(correlation) > 0.8:  # High correlation
                            anomalies.append(f"High correlation between {comp1} and {comp2}")
                            
        return anomalies
        
    def _calculate_correlation(self, metrics1: List[Dict], metrics2: List[Dict]) -> float:
        """Calculate correlation between two sets of metrics."""
        try:
            # Extract common metrics
            common_metrics = set(metrics1[0].keys()) & set(metrics2[0].keys())
            
            if not common_metrics:
                return 0.0
                
            # Calculate correlation for each common metric
            correlations = []
            for metric in common_metrics:
                values1 = [m.get(metric, 0) for m in metrics1]
                values2 = [m.get(metric, 0) for m in metrics2]
                
                if len(values1) == len(values2) and len(values1) > 1:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation failed: {e}")
            return 0.0
            
    async def _retrain_models(self):
        """Retrain anomaly detection models with recent data."""
        logger.info("🔄 Retraining anomaly detection models...")
        
        # Clear old models
        for component in self.anomaly_models:
            self.anomaly_models[component]['is_trained'] = False
            
        # Retrain with recent data
        await self._train_anomaly_models()
        
        logger.info("✅ Anomaly detection models retrained")
        
    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics."""
        total_detections = len(self.detection_times)
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0.0
        
        return {
            'total_detections': total_detections,
            'anomaly_count': self.anomaly_count,
            'avg_detection_time': avg_detection_time,
            'is_learning': self.is_learning,
            'learning_samples': self.learning_samples,
            'baseline_window': self.baseline_window,
            'anomaly_threshold': self.anomaly_threshold,
            'alert_threshold': self.alert_threshold,
            'telemetry_samples': len(self.telemetry_data),
            'trained_models': len([m for m in self.anomaly_models.values() if m['is_trained']])
        }
