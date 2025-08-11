"""
Configuration system for EV autonomous trucking perception system.

Manages all system parameters, sensor configurations, and runtime settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SensorConfig:
    """Configuration for individual sensor types."""
    num_sensors: int
    enabled: bool = True
    
    
@dataclass
class LiDARConfig(SensorConfig):
    """LiDAR sensor configuration."""
    lines: int = 128
    range_meters: float = 100.0
    frequency_hz: float = 10.0
    resolution_degrees: float = 0.1
    
    
@dataclass
class CameraConfig(SensorConfig):
    """Camera sensor configuration."""
    resolution: str = "8MP"
    fps: int = 30
    field_of_view: float = 120.0
    spectral_bands: int = 3
    
    
@dataclass
class RadarConfig(SensorConfig):
    """Radar sensor configuration."""
    frequency_ghz: float = 77.0
    range_meters: float = 200.0
    velocity_range: float = 50.0
    
    
@dataclass
class SensorsConfig:
    """Overall sensors configuration."""
    lidar: LiDARConfig = field(default_factory=lambda: LiDARConfig(num_sensors=4))
    camera: CameraConfig = field(default_factory=lambda: CameraConfig(num_sensors=8))
    radar: RadarConfig = field(default_factory=lambda: RadarConfig(num_sensors=6))
    
    
@dataclass
class PerceptionConfig:
    """Perception system configuration."""
    num_classes: int = 20
    num_features: int = 64
    batch_size: int = 1
    model_path: str = "models/pointnet_plus_plus.pth"
    
    object_detection: Dict[str, Any] = field(default_factory=lambda: {
        'confidence_threshold': 0.7,
        'nms_threshold': 0.3,
        'max_detections': 100
    })
    
    segmentation: Dict[str, Any] = field(default_factory=lambda: {
        'num_classes': 20,
        'confidence_threshold': 0.5
    })
    
    
@dataclass
class FusionConfig:
    """Multi-modal fusion configuration."""
    temporal_window: int = 10
    attention_mechanism: str = "llm_guided"
    
    llm_attention: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'qwen-7b',
        'max_length': 512,
        'temperature': 0.7
    })
    
    temporal: Dict[str, Any] = field(default_factory=lambda: {
        'filter_type': 'kalman',
        'smoothing_factor': 0.8
    })
    
    
@dataclass
class SafetyConfig:
    """Safety framework configuration."""
    max_latency_ms: float = 32.0
    min_confidence: float = 0.8
    asil_level: str = "ASIL-D"
    
    asil: Dict[str, Any] = field(default_factory=lambda: {
        'deterministic_execution': True,
        'fault_tolerance': True,
        'redundancy_level': 2
    })
    
    wcet: Dict[str, Any] = field(default_factory=lambda: {
        'analysis_enabled': True,
        'nominal_bound_ms': 25.0,
        'acceptable_bound_ms': 32.0,
        'critical_bound_ms': 50.0
    })
    
    fault_injection: Dict[str, Any] = field(default_factory=lambda: {
        'testing_enabled': True,
        'injection_rate': 0.01,
        'detection_threshold': 0.95
    })
    
    
@dataclass
class AIToolsConfig:
    """AI development tools configuration."""
    simulation: Dict[str, Any] = field(default_factory=lambda: {
        'llm_model': 'gpt-4',
        'edge_case_generation': True,
        'synthetic_data_generation': True
    })
    
    ticket_triage: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'llama-2-7b',
        'fine_tuned': True,
        'target_f1': 0.92,
        'inference_latency_ms': 50
    })
    
    llm_integration: Dict[str, Any] = field(default_factory=lambda: {
        'planning_interface': True,
        'natural_language_commands': True,
        'auto_summarization': True
    })
    
    
@dataclass
class ROS2Config:
    """ROS2 integration configuration."""
    enabled: bool = True
    node_name: str = "ev_perception"
    namespace: str = "/ev"
    
    topics: Dict[str, str] = field(default_factory=lambda: {
        'pointcloud': 'perception/pointcloud',
        'detections': 'perception/detections',
        'segmentation': 'perception/segmentation'
    })
    
    
@dataclass
class CloudConfig:
    """Cloud deployment configuration."""
    provider: str = "aws"
    containerization: bool = True
    auto_scaling: bool = True
    
    aws: Dict[str, Any] = field(default_factory=lambda: {
        'region': 'us-west-2',
        'ecs_cluster': 'ev-perception',
        'spot_instances': True,
        'cost_optimization': True
    })
    
    mlflow: Dict[str, Any] = field(default_factory=lambda: {
        'tracking_uri': 'http://localhost:5000',
        'experiment_name': 'ev-perception'
    })
    
    
@dataclass
class Config:
    """Main configuration class for the EV perception system."""
    
    # System configuration
    system_name: str = "EV Autonomous Trucking Perception System"
    version: str = "1.0.0"
    device: str = "auto"  # auto, cuda, cpu, neural_engine
    
    # Component configurations
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ai_tools: AIToolsConfig = field(default_factory=AIToolsConfig)
    ros2: ROS2Config = field(default_factory=ROS2Config)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    
    # Performance targets
    performance_targets: Dict[str, Any] = field(default_factory=lambda: {
        'end_to_end_latency_ms': 32.0,
        'throughput_boost': 4.0,
        'training_efficiency': 3.0,
        'manual_effort_reduction': 0.4
    })
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"⚠️ Configuration file not found: {config_path}, using defaults")
                return cls()
                
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Create config instance with loaded data
            config = cls()
            config._update_from_dict(config_data)
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}, using defaults")
            return cls()
            
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_data.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dataclass_fields__'):
                    # Update nested dataclass
                    nested_obj = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_obj, nested_key):
                            setattr(nested_obj, nested_key, nested_value)
                else:
                    # Update simple attribute
                    setattr(self, key, value)
                    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            
            if hasattr(value, '__dataclass_fields__'):
                # Convert nested dataclass to dict
                config_dict[field_name] = {
                    nested_field: getattr(value, nested_field)
                    for nested_field in value.__dataclass_fields__
                }
            else:
                # Simple value
                config_dict[field_name] = value
                
        return config_dict
        
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        try:
            config_dict = self.to_dict()
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            
    def validate(self) -> bool:
        """Validate configuration for consistency."""
        try:
            # Check performance targets
            if self.performance_targets['end_to_end_latency_ms'] > 50:
                logger.warning("⚠️ End-to-end latency target may be too high")
                
            # Check sensor configuration
            if self.sensors.lidar.num_sensors < 1:
                logger.error("❌ At least one LiDAR sensor required")
                return False
                
            if self.sensors.camera.num_sensors < 1:
                logger.error("❌ At least one camera sensor required")
                return False
                
            # Check safety configuration
            if self.safety.max_latency_ms > 50:
                logger.warning("⚠️ Safety latency limit may be too high")
                
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
            
    def get_sensor_summary(self) -> str:
        """Get human-readable sensor configuration summary."""
        return (
            f"Sensors: {self.sensors.lidar.num_sensors}x {self.sensors.lidar.lines}-line LiDAR, "
            f"{self.sensors.camera.num_sensors}x {self.sensors.camera.resolution} cameras, "
            f"{self.sensors.radar.num_sensors}x mmWave radars"
        )
        
    def get_performance_summary(self) -> str:
        """Get human-readable performance targets summary."""
        targets = self.performance_targets
        return (
            f"Targets: {targets['end_to_end_latency_ms']}ms latency, "
            f"{targets['throughput_boost']}x throughput, "
            f"{targets['training_efficiency']}x training efficiency, "
            f"{targets['manual_effort_reduction']:.0%} effort reduction"
        )
